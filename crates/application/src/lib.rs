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
use tracing::warn;

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
    /// Returns `false` when the sink is no longer available (e.g. disconnected client).
    async fn emit(&self, event: api::Event) -> bool;
}

const STREAM_EVENT_BUFFER: usize = 64;

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

    fn normalize_optional_string(value: Option<String>) -> Option<String> {
        value.and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
    }

    async fn get_config(&self) -> ApiResult<api::Config> {
        let llm = self
            .settings
            .get_llm_settings()
            .await
            .map_err(Self::map_core_err)?;
        let embeddings = self
            .settings
            .get_embeddings_settings()
            .await
            .map_err(Self::map_core_err)?;
        let persistence = self
            .settings
            .get_persistence_settings()
            .await
            .map_err(Self::map_core_err)?;

        Ok(api::Config {
            llm: api::LlmSettingsView {
                connector: llm.connector,
                model: llm.model,
                base_url: llm.base_url,
                has_api_key: llm.has_api_key,
            },
            embeddings: api::EmbeddingsSettingsView {
                connector: embeddings.connector,
                model: embeddings.model,
                base_url: embeddings.base_url,
                has_api_key: embeddings.has_api_key,
                available: embeddings.available,
                is_default: embeddings.is_default,
            },
            persistence: api::PersistenceSettingsView {
                enabled: persistence.enabled,
                remote_url: persistence.remote_url,
                remote_name: persistence.remote_name,
                push_on_update: persistence.push_on_update,
            },
        })
    }

    async fn set_config(&self, changes: api::ConfigChanges) -> ApiResult<api::Config> {
        let current = self.get_config().await?;
        let api::ConfigChanges {
            llm_connector,
            llm_model,
            llm_base_url,
            llm_api_key,
            embeddings_connector,
            embeddings_model,
            embeddings_base_url,
            persistence_enabled,
            persistence_remote_url,
            persistence_remote_name,
            persistence_push_on_update,
        } = changes;

        let llm_changed = llm_connector.is_some() || llm_model.is_some() || llm_base_url.is_some();
        if llm_changed {
            let connector = Self::normalize_optional_string(llm_connector)
                .unwrap_or_else(|| current.llm.connector.clone());
            let model = if llm_model.is_some() {
                Self::normalize_optional_string(llm_model)
            } else {
                Some(current.llm.model.clone())
            };
            let base_url = if llm_base_url.is_some() {
                Self::normalize_optional_string(llm_base_url)
            } else {
                Some(current.llm.base_url.clone())
            };

            self.settings
                .set_llm_settings(connector, model, base_url)
                .await
                .map_err(Self::map_core_err)?;
        }

        if let Some(api_key) = Self::normalize_optional_string(llm_api_key) {
            self.settings
                .set_api_key(api_key)
                .await
                .map_err(Self::map_core_err)?;
        }

        let embeddings_changed = embeddings_connector.is_some()
            || embeddings_model.is_some()
            || embeddings_base_url.is_some();
        if embeddings_changed {
            let connector = if embeddings_connector.is_some() {
                Self::normalize_optional_string(embeddings_connector)
            } else if current.embeddings.is_default {
                None
            } else {
                Some(current.embeddings.connector.clone())
            };

            let model = if embeddings_model.is_some() {
                Self::normalize_optional_string(embeddings_model)
            } else {
                Some(current.embeddings.model.clone())
            };

            let base_url = if embeddings_base_url.is_some() {
                Self::normalize_optional_string(embeddings_base_url)
            } else {
                Some(current.embeddings.base_url.clone())
            };

            self.settings
                .set_embeddings_settings(connector, model, base_url)
                .await
                .map_err(Self::map_core_err)?;
        }

        let persistence_changed = persistence_enabled.is_some()
            || persistence_remote_url.is_some()
            || persistence_remote_name.is_some()
            || persistence_push_on_update.is_some();
        if persistence_changed {
            let enabled = persistence_enabled.unwrap_or(current.persistence.enabled);
            let remote_url = if persistence_remote_url.is_some() {
                Self::normalize_optional_string(persistence_remote_url)
            } else {
                Some(current.persistence.remote_url.clone())
            };
            let remote_name = if persistence_remote_name.is_some() {
                Self::normalize_optional_string(persistence_remote_name)
            } else {
                Some(current.persistence.remote_name.clone())
            };
            let push_on_update =
                persistence_push_on_update.unwrap_or(current.persistence.push_on_update);

            self.settings
                .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                .await
                .map_err(Self::map_core_err)?;
        }

        self.get_config().await
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

            api::Command::GetConfig => {
                let config = self.get_config().await?;
                Ok(api::CommandResult::Config(config))
            }

            api::Command::SetConfig { changes } => {
                let config = self.set_config(changes).await?;
                Ok(api::CommandResult::Config(config))
            }

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
        let (tx, mut rx) = tokio::sync::mpsc::channel::<api::Event>(STREAM_EVENT_BUFFER);

        let sink_for_forwarder = Arc::clone(&sink);
        let forwarder = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if !sink_for_forwarder.emit(event).await {
                    break;
                }
            }
        });

        // Bridge chunks from core callback -> canonical events.
        let conv_id_for_cb = conversation_id.clone();
        let req_id_for_cb = request_id.clone();
        let callback: desktop_assistant_core::ports::llm::ChunkCallback = Box::new(move |chunk| {
            tx.try_send(api::Event::AssistantDelta {
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

        if let Err(e) = forwarder.await {
            warn!("stream forwarder task failed: {e}");
        }

        match full {
            Ok(full_response) => {
                let _ = sink
                    .emit(api::Event::AssistantCompleted {
                        conversation_id,
                        request_id,
                        full_response,
                    })
                    .await;
                Ok(())
            }
            Err(e) => {
                let _ = sink
                    .emit(api::Event::AssistantError {
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
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

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

    #[derive(Clone)]
    struct SettingsState {
        llm: LlmSettingsView,
        embeddings: EmbeddingsSettingsView,
        persistence: PersistenceSettingsView,
        api_key_set: bool,
    }

    struct ConfigurableSettings {
        state: Mutex<SettingsState>,
    }

    impl ConfigurableSettings {
        fn new() -> Self {
            Self {
                state: Mutex::new(SettingsState {
                    llm: LlmSettingsView {
                        connector: "openai".into(),
                        model: "gpt-5".into(),
                        base_url: "https://api.openai.com/v1".into(),
                        has_api_key: false,
                    },
                    embeddings: EmbeddingsSettingsView {
                        connector: "openai".into(),
                        model: "text-embedding-3-small".into(),
                        base_url: "https://api.openai.com/v1".into(),
                        has_api_key: false,
                        available: true,
                        is_default: true,
                    },
                    persistence: PersistenceSettingsView {
                        enabled: false,
                        remote_url: String::new(),
                        remote_name: "origin".into(),
                        push_on_update: true,
                    },
                    api_key_set: false,
                }),
            }
        }

        fn snapshot(&self) -> SettingsState {
            self.state.lock().unwrap().clone()
        }
    }

    impl SettingsService for ConfigurableSettings {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().llm.clone())
        }

        async fn set_llm_settings(
            &self,
            connector: String,
            model: Option<String>,
            base_url: Option<String>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.llm.connector = connector;
            if let Some(model) = model {
                state.llm.model = model;
            }
            if let Some(base_url) = base_url {
                state.llm.base_url = base_url;
            }
            Ok(())
        }

        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.api_key_set = true;
            state.llm.has_api_key = true;
            Ok(())
        }

        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().embeddings.clone())
        }

        async fn set_embeddings_settings(
            &self,
            connector: Option<String>,
            model: Option<String>,
            base_url: Option<String>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            if let Some(connector) = connector {
                state.embeddings.connector = connector;
                state.embeddings.is_default = false;
            }
            if let Some(model) = model {
                state.embeddings.model = model;
            }
            if let Some(base_url) = base_url {
                state.embeddings.base_url = base_url;
            }
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
            Ok(self.state.lock().unwrap().persistence.clone())
        }

        async fn set_persistence_settings(
            &self,
            enabled: bool,
            remote_url: Option<String>,
            remote_name: Option<String>,
            push_on_update: bool,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.persistence.enabled = enabled;
            if let Some(remote_url) = remote_url {
                state.persistence.remote_url = remote_url;
            }
            if let Some(remote_name) = remote_name {
                state.persistence.remote_name = remote_name;
            }
            state.persistence.push_on_update = push_on_update;
            Ok(())
        }
    }

    struct CollectSink(tokio::sync::Mutex<Vec<api::Event>>);
    #[async_trait::async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, event: api::Event) -> bool {
            self.0.lock().await.push(event);
            true
        }
    }

    struct DropSink;
    #[async_trait::async_trait]
    impl EventSink for DropSink {
        async fn emit(&self, _event: api::Event) -> bool {
            false
        }
    }

    struct AbortAwareConversations {
        aborted: Arc<AtomicBool>,
    }
    impl ConversationService for AbortAwareConversations {
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
            Ok(Conversation::new(id.as_str(), "t"))
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
            for _ in 0..10_000 {
                if !on_chunk("x".to_string()) {
                    self.aborted.store(true, Ordering::SeqCst);
                    return Ok("cancelled".to_string());
                }
                tokio::task::yield_now().await;
            }
            Ok("complete".to_string())
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

    #[tokio::test]
    async fn send_message_cancels_when_sink_disconnects() {
        let aborted = Arc::new(AtomicBool::new(false));
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(AbortAwareConversations {
                aborted: Arc::clone(&aborted),
            }),
            Arc::new(FakeSettings),
        );

        h.handle_send_message("c1".into(), "hi".into(), "r1".into(), Arc::new(DropSink))
            .await
            .unwrap();

        assert!(aborted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn get_config_returns_aggregated_settings() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
        );

        let res = h.handle_command(api::Command::GetConfig).await.unwrap();
        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };

        assert_eq!(config.llm.connector, "openai");
        assert_eq!(config.embeddings.model, "text-embedding-3-small");
        assert_eq!(config.persistence.remote_name, "origin");
    }

    #[tokio::test]
    async fn set_config_applies_changes_and_returns_updated_config() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
        );

        let res = h
            .handle_command(api::Command::SetConfig {
                changes: api::ConfigChanges {
                    llm_connector: Some("ollama".into()),
                    llm_model: Some("llama3.1:8b".into()),
                    llm_base_url: Some("http://localhost:11434".into()),
                    llm_api_key: Some("test-key".into()),
                    persistence_enabled: Some(true),
                    persistence_remote_url: Some("git@example.com/repo.git".into()),
                    persistence_remote_name: Some("upstream".into()),
                    persistence_push_on_update: Some(false),
                    ..Default::default()
                },
            })
            .await
            .unwrap();

        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(config.llm.connector, "ollama");
        assert_eq!(config.llm.model, "llama3.1:8b");
        assert_eq!(config.persistence.remote_name, "upstream");
        assert!(config.llm.has_api_key);

        let snapshot = settings.snapshot();
        assert!(snapshot.api_key_set);
    }
}
