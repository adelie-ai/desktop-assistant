//! D-Bus adapter for `/org/desktopAssistant/Conversations`.
//!
//! Mirrors `crates/dbus-interface/src/conversation.rs` method-for-method.
//! Translations of method args → `api::Command`, and of
//! `api::CommandResult` → method return values, match the WS adapter
//! semantics so existing TUI/GTK/KDE clients keep working unchanged.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

fn to_fdo<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

/// Translate a transport-level error to a D-Bus error. Daemon-level
/// errors propagate verbatim; everything else gets a `Failed` with a
/// descriptive prefix.
fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// D-Bus adapter for conversation management. The non-streaming
/// methods translate into `api::Command` dispatches; `SendPrompt`
/// triggers a `SendMessage` command which the daemon streams back as
/// `AssistantDelta` / `AssistantCompleted` / `AssistantError` events —
/// those are translated to `ResponseChunk` / `ResponseComplete` /
/// `ResponseError` signals by [`super::event_forwarder`].
pub struct DbusConversationsAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusConversationsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }
}

#[interface(name = "org.desktopAssistant.Conversations")]
impl<T: BridgeTransport + 'static> DbusConversationsAdapter<T> {
    /// Create a new conversation and return its ID.
    async fn create_conversation(&self, title: &str) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::CreateConversation {
                title: title.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::ConversationId { id } => Ok(id),
            other => Err(fdo::Error::Failed(format!(
                "unexpected CreateConversation result: {other:?}"
            ))),
        }
    }

    /// List conversations. Wire shape matches the in-process adapter:
    /// `(id, title, message_count, updated_at, archived)` tuples.
    async fn list_conversations(
        &self,
        max_age_days: i32,
        include_archived: bool,
    ) -> fdo::Result<Vec<(String, String, u32, String, bool)>> {
        let max_age_days = u32::try_from(max_age_days).ok().filter(|d| *d > 0);
        let result = self
            .dispatch(api::Command::ListConversations {
                max_age_days,
                include_archived,
            })
            .await?;
        match result {
            api::CommandResult::Conversations(summaries) => Ok(summaries
                .into_iter()
                .map(|s| (s.id, s.title, s.message_count, s.updated_at, s.archived))
                .collect()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListConversations result: {other:?}"
            ))),
        }
    }

    /// Archive a conversation by ID.
    async fn archive_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::ArchiveConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ArchiveConversation result: {other:?}"
            ))),
        }
    }

    /// Unarchive a conversation by ID.
    async fn unarchive_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::UnarchiveConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected UnarchiveConversation result: {other:?}"
            ))),
        }
    }

    /// Get a conversation by ID: `(id, title, messages)` where
    /// `messages` is `(role, content)` tuples.
    async fn get_conversation(
        &self,
        id: &str,
    ) -> fdo::Result<(String, String, Vec<(String, String)>)> {
        let result = self
            .dispatch(api::Command::GetConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Conversation(conv) => {
                let messages = conv
                    .messages
                    .into_iter()
                    .map(|m| (m.role, m.content))
                    .collect();
                Ok((conv.id, conv.title, messages))
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetConversation result: {other:?}"
            ))),
        }
    }

    /// Get messages with pagination + role filter. Returns
    /// `(total_raw_count, truncated, messages)`. Slicing is performed
    /// on the bridge side using the daemon's full conversation view
    /// because the daemon's command set does not (yet) expose this
    /// exact pagination shape on the wire.
    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> fdo::Result<(u32, bool, Vec<(String, String)>)> {
        let result = self
            .dispatch(api::Command::GetConversation { id: id.to_string() })
            .await?;
        let conv = match result {
            api::CommandResult::Conversation(c) => c,
            other => {
                return Err(fdo::Error::Failed(format!(
                    "unexpected GetConversation result for get_messages: {other:?}"
                )));
            }
        };

        let total = conv.messages.len() as u32;
        let all: Vec<(String, String)> = conv
            .messages
            .into_iter()
            .map(|m| (m.role, m.content))
            .collect();

        let use_after = after_count >= 0;
        let sliced: Vec<(String, String)> = if use_after {
            let start = (after_count as usize).min(all.len());
            all[start..].to_vec()
        } else {
            all
        };

        let filtered: Vec<(String, String)> = sliced
            .into_iter()
            .filter(|(role, _)| include_roles.is_empty() || include_roles.contains(role))
            .collect();

        let (truncated, messages) = if !use_after && tail > 0 && filtered.len() > tail as usize {
            let start = filtered.len() - tail as usize;
            (true, filtered[start..].to_vec())
        } else {
            (false, filtered)
        };

        Ok((total, truncated, messages))
    }

    /// Delete a conversation by ID.
    async fn delete_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::DeleteConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected DeleteConversation result: {other:?}"
            ))),
        }
    }

    /// Rename a conversation.
    async fn rename_conversation(&self, id: &str, title: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::RenameConversation {
                id: id.to_string(),
                title: title.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected RenameConversation result: {other:?}"
            ))),
        }
    }

    /// Delete every conversation; returns the count.
    async fn clear_all_history(&self) -> fdo::Result<u32> {
        let result = self.dispatch(api::Command::ClearAllHistory).await?;
        match result {
            api::CommandResult::Cleared { deleted_count } => Ok(deleted_count),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ClearAllHistory result: {other:?}"
            ))),
        }
    }

    /// Send a prompt; daemon streams back via `AssistantDelta` events
    /// which the event forwarder turns into `ResponseChunk` /
    /// `ResponseComplete` / `ResponseError` signals.
    ///
    /// Returns the `request_id` the daemon will use for event
    /// correlation. The daemon currently echoes the request id via
    /// the streamed event payloads, so the bridge picks its own UUID
    /// and the daemon's id is what shows up on the signal — same as
    /// the in-process adapter where the dbus-interface created its
    /// own request id.
    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
                override_selection: None,
                system_refinement: String::new(),
            })
            .await
            .map_err(|e| {
                // SendMessage returns an immediate Ack on success; if
                // the daemon refused the request before streaming, we
                // surface the error directly.
                to_fdo(e)
            })?;

        match result {
            api::CommandResult::Ack => {
                // Daemon hasn't told us the request id yet — events
                // will carry it. Use a placeholder so clients have
                // something to log against; the events flowing through
                // the forwarder carry the daemon's correlation id.
                Ok(uuid::Uuid::new_v4().to_string())
            }
            api::CommandResult::SendMessageAck { task_id } => Ok(task_id),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SendMessage result: {other:?}"
            ))),
        }
    }

    /// Signal emitted for each chunk of a streaming response.
    /// Body is forwarded by [`super::event_forwarder`] from
    /// `Event::AssistantDelta`.
    #[zbus(signal)]
    async fn response_chunk(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response is complete.
    /// Forwarded from `Event::AssistantCompleted`.
    #[zbus(signal)]
    async fn response_complete(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted on streaming failure. Forwarded from
    /// `Event::AssistantError`.
    #[zbus(signal)]
    async fn response_error(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::Result<()>;
}
