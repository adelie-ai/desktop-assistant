use std::sync::Arc;

use desktop_assistant_core::domain::ConversationId;
use desktop_assistant_core::ports::inbound::ConversationService;
use tokio::sync::mpsc;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

/// D-Bus adapter for the ConversationService.
///
/// Exposes conversation management and streaming prompt/response
/// over D-Bus signals.
pub struct DbusConversationAdapter<S: ConversationService + 'static> {
    service: Arc<S>,
}

impl<S: ConversationService + 'static> DbusConversationAdapter<S> {
    pub fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

/// Messages sent from the streaming task to the signal emitter.
enum StreamEvent {
    Chunk(String),
    Complete(String),
    Error(String),
}

#[interface(name = "org.desktopAssistant.Conversations")]
impl<S: ConversationService + 'static> DbusConversationAdapter<S> {
    /// Create a new conversation and return its ID.
    async fn create_conversation(&self, title: &str) -> fdo::Result<String> {
        let conv = self
            .service
            .create_conversation(title.to_string())
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;
        Ok(conv.id.0)
    }

    /// List conversations as an array of (id, title, message_count, updated_at),
    /// optionally filtered by max age in days (0 means no filtering).
    async fn list_conversations(
        &self,
        max_age_days: i32,
    ) -> fdo::Result<Vec<(String, String, u32, String)>> {
        let max_age = u32::try_from(max_age_days).ok().filter(|days| *days > 0);
        let summaries = self
            .service
            .list_conversations(max_age)
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;
        Ok(summaries
            .into_iter()
            .map(|s| (s.id.0, s.title, s.message_count as u32, s.updated_at))
            .collect())
    }

    /// Get a conversation by ID, returns (id, title, messages) where
    /// messages is an array of (role, content).
    async fn get_conversation(
        &self,
        id: &str,
    ) -> fdo::Result<(String, String, Vec<(String, String)>)> {
        let conv = self
            .service
            .get_conversation(&ConversationId::from(id))
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;
        let messages: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    desktop_assistant_core::domain::Role::User => "user",
                    desktop_assistant_core::domain::Role::Assistant => "assistant",
                    desktop_assistant_core::domain::Role::System => "system",
                    desktop_assistant_core::domain::Role::Tool => "tool",
                };
                (role.to_string(), m.content.clone())
            })
            .collect();
        Ok((conv.id.0, conv.title, messages))
    }

    /// Get messages from a conversation with optional pagination and role filtering.
    ///
    /// - `tail`: max messages to return from the *filtered* set (0 = unlimited).
    ///   Ignored when `after_count` >= 0.
    /// - `after_count`: skip the first N raw (pre-filter) messages; -1 means unused.
    /// - `include_roles`: allowlist of roles to return, e.g. `["user", "assistant"]`.
    ///   An empty list disables filtering and returns all roles.
    ///
    /// Returns `(total_raw_count, truncated, messages)`.
    /// `total_raw_count` always reflects the unfiltered length so callers can
    /// use it as the next `after_count` for incremental fetches.
    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> fdo::Result<(u32, bool, Vec<(String, String)>)> {
        let conv = self
            .service
            .get_conversation(&ConversationId::from(id))
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;

        let total = conv.messages.len() as u32;

        let all: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    desktop_assistant_core::domain::Role::User => "user",
                    desktop_assistant_core::domain::Role::Assistant => "assistant",
                    desktop_assistant_core::domain::Role::System => "system",
                    desktop_assistant_core::domain::Role::Tool => "tool",
                };
                (role.to_string(), m.content.clone())
            })
            .collect();

        // Slice by raw position first so after_count is always a stable index.
        let use_after = after_count >= 0;
        let sliced: Vec<(String, String)> = if use_after {
            let start = (after_count as usize).min(all.len());
            all[start..].to_vec()
        } else {
            all
        };

        // Apply role allowlist (empty = no filtering).
        let filtered: Vec<(String, String)> = sliced
            .into_iter()
            .filter(|(role, _)| include_roles.is_empty() || include_roles.contains(role))
            .collect();

        // Apply tail limit to the filtered set (tail mode only).
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
        self.service
            .delete_conversation(&ConversationId::from(id))
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    /// Delete every conversation and return how many were removed.
    async fn clear_all_history(&self) -> fdo::Result<u32> {
        self.service
            .clear_all_history()
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    /// Send a prompt and stream the response via signals.
    /// Returns a request_id that correlates the signals.
    async fn send_prompt(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        conversation_id: &str,
        prompt: &str,
    ) -> fdo::Result<String> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let conv_id = conversation_id.to_string();
        let prompt = prompt.to_string();
        let service = Arc::clone(&self.service);
        let req_id = request_id.clone();

        let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();

        // Spawn the LLM call task
        let llm_conv_id = conv_id.clone();
        tokio::spawn(async move {
            let tx_chunk = tx.clone();
            let callback: desktop_assistant_core::ports::llm::ChunkCallback =
                Box::new(move |chunk| tx_chunk.send(StreamEvent::Chunk(chunk)).is_ok());

            match service
                .send_prompt(
                    &ConversationId::from(llm_conv_id.as_str()),
                    prompt,
                    callback,
                )
                .await
            {
                Ok(full_response) => {
                    let _ = tx.send(StreamEvent::Complete(full_response));
                }
                Err(e) => {
                    tracing::error!(
                        conversation_id = %llm_conv_id,
                        "I hit an LLM backend error and could not complete this request. Details: {e}"
                    );
                    let _ = tx.send(StreamEvent::Error(e.to_string()));
                }
            }
        });

        // Spawn the signal emitter task
        let emitter = emitter.to_owned();
        let signal_conv_id = conv_id.clone();
        let signal_req_id = req_id.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Chunk(chunk) => {
                        if let Err(e) =
                            Self::response_chunk(&emitter, &signal_conv_id, &signal_req_id, &chunk)
                                .await
                        {
                            tracing::error!("failed to emit ResponseChunk signal: {e}");
                        }
                    }
                    StreamEvent::Complete(full) => {
                        if let Err(e) = Self::response_complete(
                            &emitter,
                            &signal_conv_id,
                            &signal_req_id,
                            &full,
                        )
                        .await
                        {
                            tracing::error!("failed to emit ResponseComplete signal: {e}");
                        }
                        break;
                    }
                    StreamEvent::Error(err) => {
                        if let Err(e) =
                            Self::response_error(&emitter, &signal_conv_id, &signal_req_id, &err)
                                .await
                        {
                            tracing::error!("failed to emit ResponseError signal: {e}");
                        }
                        break;
                    }
                }
            }
        });

        Ok(request_id)
    }

    /// Signal emitted for each chunk of a streaming response.
    #[zbus(signal)]
    async fn response_chunk(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response is complete.
    #[zbus(signal)]
    async fn response_complete(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response encounters an error.
    #[zbus(signal)]
    async fn response_error(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::{Conversation, ConversationSummary, Message, Role};
    use desktop_assistant_core::ports::llm::ChunkCallback;

    struct FakeConversationService;

    impl ConversationService for FakeConversationService {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("test-id", title))
        }

        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![ConversationSummary {
                id: ConversationId::from("test-id"),
                title: "Test".to_string(),
                created_at: "2026-02-16 00:00:00".to_string(),
                updated_at: "2026-02-16 00:00:00".to_string(),
                message_count: 0,
            }])
        }

        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            let mut conv = Conversation::new(id.as_str(), "Test");
            conv.messages.push(Message::new(Role::User, "hi"));
            Ok(conv)
        }

        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(1)
        }

        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
        ) -> Result<String, CoreError> {
            on_chunk("hello ".to_string());
            on_chunk("world".to_string());
            Ok("hello world".to_string())
        }
    }

    #[test]
    fn adapter_construction() {
        let service = Arc::new(FakeConversationService);
        let _adapter = DbusConversationAdapter::new(service);
    }

    #[tokio::test]
    async fn adapter_create_conversation() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        // We can't test D-Bus methods directly without a bus connection,
        // but we can verify the service is accessible.
        let conv = adapter
            .service
            .create_conversation("Test".into())
            .await
            .unwrap();
        assert_eq!(conv.id.as_str(), "test-id");
    }

    #[tokio::test]
    async fn adapter_list_conversations() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        let summaries = adapter.service.list_conversations(None).await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].title, "Test");
    }

    #[tokio::test]
    async fn adapter_get_conversation() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("test-id"))
            .await
            .unwrap();
        assert_eq!(conv.messages.len(), 1);
    }

    #[tokio::test]
    async fn get_messages_empty_include_returns_all() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(Arc::clone(&service));
        // Use the service directly since we can't call D-Bus methods in unit tests.
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("test-id"))
            .await
            .unwrap();
        // Empty include_roles → no filtering, all roles returned.
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn get_messages_include_filters_to_allowlist() {
        use desktop_assistant_core::domain::Message;
        struct MultiRoleService;
        impl ConversationService for MultiRoleService {
            async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
                Ok(Conversation::new("id", title))
            }
            async fn list_conversations(
                &self,
                _: Option<u32>,
            ) -> Result<Vec<ConversationSummary>, CoreError> {
                Ok(vec![])
            }
            async fn get_conversation(
                &self,
                id: &ConversationId,
            ) -> Result<Conversation, CoreError> {
                let mut conv = Conversation::new(id.as_str(), "T");
                conv.messages.push(Message::new(Role::User, "hello"));
                conv.messages
                    .push(Message::new(Role::Assistant, "response"));
                conv.messages
                    .push(Message::tool_result("c1", "tool output"));
                conv.messages.push(Message::new(Role::User, "follow-up"));
                Ok(conv)
            }
            async fn delete_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn clear_all_history(&self) -> Result<u32, CoreError> {
                Ok(0)
            }
            async fn send_prompt(
                &self,
                _: &ConversationId,
                _: String,
                _: ChunkCallback,
            ) -> Result<String, CoreError> {
                Ok(String::new())
            }
        }

        let adapter = DbusConversationAdapter::new(Arc::new(MultiRoleService));
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("id"))
            .await
            .unwrap();

        // Raw: 4 messages (user, assistant, tool, user)
        assert_eq!(conv.messages.len(), 4);

        // Simulate GetMessages with include_roles=["user", "assistant"].
        let total = conv.messages.len() as u32;
        let include = vec!["user".to_string(), "assistant".to_string()];
        let all: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                (role.to_string(), m.content.clone())
            })
            .collect();
        let filtered: Vec<_> = all
            .iter()
            .filter(|(r, _)| include.is_empty() || include.contains(r))
            .collect();
        assert_eq!(total, 4);
        assert_eq!(filtered.len(), 3); // user, assistant, user — tool excluded
        assert!(filtered.iter().all(|(r, _)| r != "tool"));
    }

    #[tokio::test]
    async fn get_messages_after_count_slices_raw() {
        use desktop_assistant_core::domain::Message;
        struct SeqService;
        impl ConversationService for SeqService {
            async fn create_conversation(&self, t: String) -> Result<Conversation, CoreError> {
                Ok(Conversation::new("id", t))
            }
            async fn list_conversations(
                &self,
                _: Option<u32>,
            ) -> Result<Vec<ConversationSummary>, CoreError> {
                Ok(vec![])
            }
            async fn get_conversation(
                &self,
                id: &ConversationId,
            ) -> Result<Conversation, CoreError> {
                let mut conv = Conversation::new(id.as_str(), "T");
                conv.messages.push(Message::new(Role::User, "u1"));
                conv.messages.push(Message::tool_result("c1", "t1"));
                conv.messages.push(Message::new(Role::Assistant, "a1"));
                conv.messages.push(Message::new(Role::User, "u2"));
                Ok(conv)
            }
            async fn delete_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn clear_all_history(&self) -> Result<u32, CoreError> {
                Ok(0)
            }
            async fn send_prompt(
                &self,
                _: &ConversationId,
                _: String,
                _: ChunkCallback,
            ) -> Result<String, CoreError> {
                Ok(String::new())
            }
        }

        let svc = Arc::new(SeqService);
        let conv = svc
            .get_conversation(&ConversationId::from("id"))
            .await
            .unwrap();
        let total = conv.messages.len() as u32; // 4 raw
        let include = vec!["user".to_string(), "assistant".to_string()];
        let all: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let r = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                (r.to_string(), m.content.clone())
            })
            .collect();

        // after_count=2 -> skip first 2 raw messages (user, tool)
        let sliced: Vec<_> = all[2..].to_vec();
        let filtered: Vec<_> = sliced
            .into_iter()
            .filter(|(r, _)| include.is_empty() || include.contains(r))
            .collect();
        assert_eq!(total, 4);
        // sliced: [assistant, user]; include=[user,assistant] → both pass
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, "assistant");
        assert_eq!(filtered[1].0, "user");
    }
}
