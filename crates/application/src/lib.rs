//! Application layer for protocol-neutral API handling.
//!
//! This crate maps canonical API [`desktop_assistant_api_model::Command`] values
//! to calls into the existing inbound ports in `desktop-assistant-core`.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_core::ports::inbound::{
    AssistantService, ConversationService, SettingsService,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("core error: {0}")]
    Core(String),

    #[error("unsupported command")]
    Unsupported,
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Protocol-neutral handler for the assistant API.
///
/// Adapters (D-Bus, WebSocket, etc.) should depend on this trait rather than
/// reaching into core services directly.
#[async_trait::async_trait]
pub trait AssistantApiHandler: Send + Sync {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult>;

    /// Handle a streaming command.
    ///
    /// For v1 we only stream assistant response chunks for `SendMessage`.
    async fn handle_send_message(
        &self,
        conversation_id: String,
        content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()>;
}

/// Minimal sink for emitting canonical events.
///
/// Implemented by protocol adapters to forward events to connected clients.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: api::Event);
}

pub struct DefaultAssistantApiHandler<A, C, S>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
{
    assistant: Arc<A>,
    conversations: Arc<C>,
    settings: Arc<S>,
}

impl<A, C, S> DefaultAssistantApiHandler<A, C, S>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
{
    pub fn new(assistant: Arc<A>, conversations: Arc<C>, settings: Arc<S>) -> Self {
        Self {
            assistant,
            conversations,
            settings,
        }
    }

    fn map_core_err<E: ToString>(e: E) -> ApiError {
        ApiError::Core(e.to_string())
    }
}

#[async_trait::async_trait]
impl<A, C, S> AssistantApiHandler for DefaultAssistantApiHandler<A, C, S>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
{
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::Ping => Ok(api::CommandResult::Pong {
                value: self.assistant.ping().to_string(),
            }),

            api::Command::GetStatus => Ok(api::CommandResult::Status(api::Status {
                version: self.assistant.version().to_string(),
            })),

            api::Command::CreateConversation { title } => {
                let conv = self
                    .conversations
                    .create_conversation(title)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::ConversationId { id: conv.id.0 })
            }

            api::Command::ListConversations { max_age_days } => {
                let list = self
                    .conversations
                    .list_conversations(max_age_days)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Conversations(
                    list.into_iter()
                        .map(|s| api::ConversationSummary {
                            id: s.id.0,
                            title: s.title,
                            message_count: s.message_count as u32,
                            updated_at: s.updated_at,
                        })
                        .collect(),
                ))
            }

            api::Command::GetConversation { id } => {
                let conv = self
                    .conversations
                    .get_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;

                Ok(api::CommandResult::Conversation(api::ConversationView {
                    id: conv.id.0,
                    title: conv.title,
                    messages: conv
                        .messages
                        .into_iter()
                        .map(|m| api::MessageView {
                            role: format!("{:?}", m.role).to_lowercase(),
                            content: m.content,
                        })
                        .collect(),
                }))
            }

            api::Command::DeleteConversation { id } => {
                self.conversations
                    .delete_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::ClearAllHistory => {
                let n = self
                    .conversations
                    .clear_all_history()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Cleared { deleted_count: n })
            }

            // Settings
            api::Command::GetLlmSettings => {
                let s = self
                    .settings
                    .get_llm_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::LlmSettings(api::LlmSettingsView {
                    connector: s.connector,
                    model: s.model,
                    base_url: s.base_url,
                    has_api_key: s.has_api_key,
                }))
            }

            api::Command::SetLlmSettings {
                connector,
                model,
                base_url,
            } => {
                self.settings
                    .set_llm_settings(connector, model, base_url)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::SetApiKey { api_key } => {
                self.settings
                    .set_api_key(api_key)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetEmbeddingsSettings => {
                let s = self
                    .settings
                    .get_embeddings_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::EmbeddingsSettings(
                    api::EmbeddingsSettingsView {
                        connector: s.connector,
                        model: s.model,
                        base_url: s.base_url,
                        has_api_key: s.has_api_key,
                        available: s.available,
                        is_default: s.is_default,
                    },
                ))
            }

            api::Command::SetEmbeddingsSettings {
                connector,
                model,
                base_url,
            } => {
                self.settings
                    .set_embeddings_settings(connector, model, base_url)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetConnectorDefaults { connector } => {
                let d = self
                    .settings
                    .get_connector_defaults(connector)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::ConnectorDefaults(
                    api::ConnectorDefaultsView {
                        llm_model: d.llm_model,
                        llm_base_url: d.llm_base_url,
                        embeddings_model: d.embeddings_model,
                        embeddings_base_url: d.embeddings_base_url,
                        embeddings_available: d.embeddings_available,
                    },
                ))
            }

            api::Command::GetPersistenceSettings => {
                let p = self
                    .settings
                    .get_persistence_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::PersistenceSettings(
                    api::PersistenceSettingsView {
                        enabled: p.enabled,
                        remote_url: p.remote_url,
                        remote_name: p.remote_name,
                        push_on_update: p.push_on_update,
                    },
                ))
            }

            api::Command::SetPersistenceSettings {
                enabled,
                remote_url,
                remote_name,
                push_on_update,
            } => {
                self.settings
                    .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // Streamed commands are handled elsewhere.
            api::Command::SendMessage { .. } => Err(ApiError::Unsupported),
        }
    }

    async fn handle_send_message(
        &self,
        conversation_id: String,
        content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<api::Event>();

        // Bridge chunks from core callback -> canonical events.
        let conv_id_for_cb = conversation_id.clone();
        let req_id_for_cb = request_id.clone();
        let callback: desktop_assistant_core::ports::llm::ChunkCallback = Box::new(move |chunk| {
            tx.send(api::Event::AssistantDelta {
                conversation_id: conv_id_for_cb.clone(),
                request_id: req_id_for_cb.clone(),
                chunk,
            })
            .is_ok()
        });

        let full = self
            .conversations
            .send_prompt(
                &desktop_assistant_core::domain::ConversationId::from(conversation_id.as_str()),
                content,
                callback,
            )
            .await;

        // Drain emitted deltas while the call runs (they're queued).
        while let Ok(ev) = rx.try_recv() {
            sink.emit(ev).await;
        }

        match full {
            Ok(full_response) => {
                sink.emit(api::Event::AssistantCompleted {
                    conversation_id,
                    request_id,
                    full_response,
                })
                .await;
                Ok(())
            }
            Err(e) => {
                sink.emit(api::Event::AssistantError {
                    conversation_id,
                    request_id,
                    error: e.to_string(),
                })
                .await;
                Err(Self::map_core_err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::{
        Conversation, ConversationId, ConversationSummary, Message, Role,
    };
    use desktop_assistant_core::ports::inbound::{
        ConnectorDefaultsView, EmbeddingsSettingsView, LlmSettingsView, PersistenceSettingsView,
    };
    use desktop_assistant_core::ports::llm::ChunkCallback;

    struct FakeAssistant;
    impl AssistantService for FakeAssistant {
        fn version(&self) -> &str {
            "0.0.0-test"
        }
        fn ping(&self) -> &str {
            "pong"
        }
    }

    struct FakeConversations;
    impl ConversationService for FakeConversations {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            let mut c = Conversation::new(id.as_str(), "t");
            c.messages.push(Message::new(Role::User, "hi"));
            Ok(c)
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
        ) -> Result<String, CoreError> {
            on_chunk("a".into());
            on_chunk("b".into());
            Ok("ab".into())
        }
    }

    struct FakeSettings;
    impl SettingsService for FakeSettings {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(LlmSettingsView {
                connector: "x".into(),
                model: "y".into(),
                base_url: "z".into(),
                has_api_key: false,
            })
        }
        async fn set_llm_settings(
            &self,
            _connector: String,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(EmbeddingsSettingsView {
                connector: "x".into(),
                model: "y".into(),
                base_url: "z".into(),
                has_api_key: false,
                available: false,
                is_default: true,
            })
        }
        async fn set_embeddings_settings(
            &self,
            _connector: Option<String>,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
            Ok(ConnectorDefaultsView {
                llm_model: "m".into(),
                llm_base_url: "u".into(),
                embeddings_model: "em".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: false,
            })
        }
        async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
            Ok(PersistenceSettingsView {
                enabled: false,
                remote_url: "".into(),
                remote_name: "origin".into(),
                push_on_update: false,
            })
        }
        async fn set_persistence_settings(
            &self,
            _enabled: bool,
            _remote_url: Option<String>,
            _remote_name: Option<String>,
            _push_on_update: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

    struct CollectSink(tokio::sync::Mutex<Vec<api::Event>>);
    #[async_trait::async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, event: api::Event) {
            self.0.lock().await.push(event);
        }
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
        );

        let res = h.handle_command(api::Command::Ping).await.unwrap();
        assert_eq!(
            res,
            api::CommandResult::Pong {
                value: "pong".into()
            }
        );
    }

    #[tokio::test]
    async fn send_message_emits_events_and_completes() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
        );

        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message("c1".into(), "hi".into(), "r1".into(), sink.clone())
            .await
            .unwrap();

        let evs = sink.0.lock().await.clone();
        assert!(matches!(evs[0], api::Event::AssistantDelta { .. }));
        assert!(matches!(evs[1], api::Event::AssistantDelta { .. }));
        assert!(matches!(evs[2], api::Event::AssistantCompleted { .. }));
    }
}
