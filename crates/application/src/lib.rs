//! Application layer for protocol-neutral API handling.
//!
//! This crate maps canonical API [`desktop_assistant_api_model::Command`] values
//! to calls into the existing inbound ports in `desktop-assistant-core`.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::inbound::{
    AssistantService, ConnectionAvailability, ConnectionConfigPayload, ConnectionsService,
    ConversationModelSelection, ConversationService, DispatchWarning, Effort, KnowledgeService,
    PromptSelectionOverride, PurposeConfigPayload, PurposeKind, SettingsService,
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

    /// Handle a streaming `SendMessage` with an optional per-send model
    /// override. The default implementation ignores the override and
    /// forwards to `handle_send_message`; the concrete handler overrides
    /// this to thread the override through.
    async fn handle_send_message_with_override(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        let _ = override_selection;
        self.handle_send_message(conversation_id, content, request_id, sink)
            .await
    }
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

pub struct DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    assistant: Arc<A>,
    conversations: Arc<C>,
    settings: Arc<S>,
    connections: Arc<N>,
    knowledge: Arc<K>,
}

impl<A, C, S, N, K> DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    pub fn new(
        assistant: Arc<A>,
        conversations: Arc<C>,
        settings: Arc<S>,
        connections: Arc<N>,
        knowledge: Arc<K>,
    ) -> Self {
        Self {
            assistant,
            conversations,
            settings,
            connections,
            knowledge,
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
            embeddings_connector,
            embeddings_model,
            embeddings_base_url,
            persistence_enabled,
            persistence_remote_url,
            persistence_remote_name,
            persistence_push_on_update,
        } = changes;

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

// ---- Conversion helpers between api-model wire types and core port types ----

fn api_connection_config_to_core(c: api::ConnectionConfigView) -> ConnectionConfigPayload {
    match c {
        api::ConnectionConfigView::Anthropic {
            base_url,
            api_key_env,
        } => ConnectionConfigPayload::Anthropic {
            base_url,
            api_key_env,
        },
        api::ConnectionConfigView::OpenAi {
            base_url,
            api_key_env,
        } => ConnectionConfigPayload::OpenAi {
            base_url,
            api_key_env,
        },
        api::ConnectionConfigView::Bedrock {
            aws_profile,
            region,
            base_url,
        } => ConnectionConfigPayload::Bedrock {
            aws_profile,
            region,
            base_url,
        },
        api::ConnectionConfigView::Ollama { base_url } => {
            ConnectionConfigPayload::Ollama { base_url }
        }
    }
}

fn core_connection_to_api_view(
    v: desktop_assistant_core::ports::inbound::ConnectionView,
) -> api::ConnectionView {
    api::ConnectionView {
        id: v.id,
        connector_type: v.connector_type,
        display_label: v.display_label,
        availability: match v.availability {
            ConnectionAvailability::Ok => api::ConnectionAvailability::Ok,
            ConnectionAvailability::Unavailable { reason } => {
                api::ConnectionAvailability::Unavailable { reason }
            }
        },
        has_credentials: v.has_credentials,
    }
}

fn core_model_listing_to_api(
    l: desktop_assistant_core::ports::inbound::ModelListing,
) -> api::ModelListing {
    api::ModelListing {
        connection_id: l.connection_id,
        connection_label: l.connection_label,
        model: api::ModelInfoView {
            id: l.model.id,
            display_name: l.model.display_name,
            context_limit: l.model.context_limit,
            capabilities: api::ModelCapabilitiesView {
                reasoning: l.model.capabilities.reasoning,
                vision: l.model.capabilities.vision,
                tools: l.model.capabilities.tools,
                embedding: l.model.capabilities.embedding,
            },
        },
    }
}

fn core_purpose_to_api(p: PurposeConfigPayload) -> api::PurposeConfigView {
    api::PurposeConfigView {
        connection: p.connection,
        model: p.model,
        effort: p.effort.map(effort_to_api),
        max_context_tokens: p.max_context_tokens,
    }
}

fn api_purpose_to_core(p: api::PurposeConfigView) -> PurposeConfigPayload {
    PurposeConfigPayload {
        connection: p.connection,
        model: p.model,
        effort: p.effort.map(effort_from_api),
        max_context_tokens: p.max_context_tokens,
    }
}

fn effort_to_api(e: Effort) -> api::EffortLevel {
    match e {
        Effort::Low => api::EffortLevel::Low,
        Effort::Medium => api::EffortLevel::Medium,
        Effort::High => api::EffortLevel::High,
    }
}

fn effort_from_api(e: api::EffortLevel) -> Effort {
    match e {
        api::EffortLevel::Low => Effort::Low,
        api::EffortLevel::Medium => Effort::Medium,
        api::EffortLevel::High => Effort::High,
    }
}

fn api_purpose_kind_to_core(k: api::PurposeKindApi) -> PurposeKind {
    match k {
        api::PurposeKindApi::Interactive => PurposeKind::Interactive,
        api::PurposeKindApi::Dreaming => PurposeKind::Dreaming,
        api::PurposeKindApi::Embedding => PurposeKind::Embedding,
        api::PurposeKindApi::Titling => PurposeKind::Titling,
    }
}

fn model_selection_to_view(sel: ConversationModelSelection) -> api::ConversationModelSelectionView {
    api::ConversationModelSelectionView {
        connection_id: sel.connection_id,
        model_id: sel.model_id,
        effort: sel.effort.map(|e| effort_to_api(Effort::from(e))),
    }
}

fn dispatch_warning_to_api(w: DispatchWarning) -> api::ConversationWarning {
    match w {
        DispatchWarning::DanglingModelSelection {
            previous,
            fallback_to,
        } => api::ConversationWarning::DanglingModelSelection {
            previous_selection: api::ConversationModelSelectionView {
                connection_id: previous.connection_id,
                model_id: previous.model_id,
                effort: previous.effort.map(|e| effort_to_api(Effort::from(e))),
            },
            fallback_to: api::ConversationModelSelectionView {
                connection_id: fallback_to.connection_id,
                model_id: fallback_to.model_id,
                effort: fallback_to.effort.map(|e| effort_to_api(Effort::from(e))),
            },
        },
    }
}

fn knowledge_entry_to_view(e: KnowledgeEntry) -> api::KnowledgeEntryView {
    api::KnowledgeEntryView {
        id: e.id,
        content: e.content,
        tags: e.tags,
        metadata: e.metadata,
        created_at: e.created_at,
        updated_at: e.updated_at,
    }
}

#[async_trait::async_trait]
impl<A, C, S, N, K> AssistantApiHandler for DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
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

            api::Command::ListConversations {
                max_age_days,
                include_archived,
            } => {
                let list = self
                    .conversations
                    .list_conversations(max_age_days, include_archived)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Conversations(
                    list.into_iter()
                        .map(|s| api::ConversationSummary {
                            id: s.id.0,
                            title: s.title,
                            message_count: s.message_count as u32,
                            updated_at: s.updated_at,
                            archived: s.archived,
                        })
                        .collect(),
                ))
            }

            api::Command::GetConversation { id } => {
                let conv_id = desktop_assistant_core::domain::ConversationId::from(id.as_str());
                let conv = self
                    .conversations
                    .get_conversation(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?;
                let model_selection = self
                    .conversations
                    .get_conversation_model_selection(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?
                    .map(model_selection_to_view);

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
                    warnings: Vec::new(),
                    model_selection,
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

            api::Command::RenameConversation { id, title } => {
                self.conversations
                    .rename_conversation(
                        &desktop_assistant_core::domain::ConversationId::from(id.as_str()),
                        title,
                    )
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::ArchiveConversation { id } => {
                self.conversations
                    .archive_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::UnarchiveConversation { id } => {
                self.conversations
                    .unarchive_conversation(&desktop_assistant_core::domain::ConversationId::from(
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
                        backend_llm_model: d.backend_llm_model,
                        embeddings_model: d.embeddings_model,
                        embeddings_base_url: d.embeddings_base_url,
                        embeddings_available: d.embeddings_available,
                        hosted_tool_search_available: d.hosted_tool_search_available,
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

            // Knowledge base management (issue #73)
            api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            } => {
                let entries = self
                    .knowledge
                    .list_entries(limit as usize, offset as usize, tag_filter)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntries(
                    entries.into_iter().map(knowledge_entry_to_view).collect(),
                ))
            }
            api::Command::GetKnowledgeEntry { id } => {
                let entry = self
                    .knowledge
                    .get_entry(id)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntry(
                    entry.map(knowledge_entry_to_view),
                ))
            }
            api::Command::SearchKnowledgeEntries {
                query,
                tag_filter,
                limit,
            } => {
                let entries = self
                    .knowledge
                    .search_entries(query, tag_filter, limit as usize)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntries(
                    entries.into_iter().map(knowledge_entry_to_view).collect(),
                ))
            }
            api::Command::CreateKnowledgeEntry {
                content,
                tags,
                metadata,
            } => {
                let entry = self
                    .knowledge
                    .create_entry(content, tags, metadata)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntryWritten(
                    knowledge_entry_to_view(entry),
                ))
            }
            api::Command::UpdateKnowledgeEntry {
                id,
                content,
                tags,
                metadata,
            } => {
                let entry = self
                    .knowledge
                    .update_entry(id, content, tags, metadata)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntryWritten(
                    knowledge_entry_to_view(entry),
                ))
            }
            api::Command::DeleteKnowledgeEntry { id } => {
                self.knowledge
                    .delete_entry(id)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // MCP server management
            api::Command::ListMcpServers => {
                let servers = self
                    .settings
                    .list_mcp_servers()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::McpServers(
                    servers
                        .into_iter()
                        .map(|s| api::McpServerView {
                            name: s.name,
                            command: s.command,
                            args: s.args,
                            namespace: s.namespace,
                            enabled: s.enabled,
                            status: s.status,
                            tool_count: s.tool_count,
                        })
                        .collect(),
                ))
            }

            api::Command::AddMcpServer {
                name,
                command,
                args,
                namespace,
                enabled,
            } => {
                self.settings
                    .add_mcp_server(name, command, args, namespace, enabled)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::RemoveMcpServer { name } => {
                self.settings
                    .remove_mcp_server(name)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::SetMcpServerEnabled { name, enabled } => {
                self.settings
                    .set_mcp_server_enabled(name, enabled)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::McpServerAction { action, server } => {
                let servers = self
                    .settings
                    .mcp_server_action(action, server)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::McpServers(
                    servers
                        .into_iter()
                        .map(|s| api::McpServerView {
                            name: s.name,
                            command: s.command,
                            args: s.args,
                            namespace: s.namespace,
                            enabled: s.enabled,
                            status: s.status,
                            tool_count: s.tool_count,
                        })
                        .collect(),
                ))
            }

            // Named connections (#11)
            api::Command::ListConnections => {
                let views = self
                    .connections
                    .list_connections()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Connections(
                    views.into_iter().map(core_connection_to_api_view).collect(),
                ))
            }

            api::Command::CreateConnection { id, config } => {
                self.connections
                    .create_connection(id, api_connection_config_to_core(config))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::UpdateConnection { id, config } => {
                self.connections
                    .update_connection(id, api_connection_config_to_core(config))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::DeleteConnection { id, force } => {
                self.connections
                    .delete_connection(id, force)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::ListAvailableModels {
                connection_id,
                refresh,
            } => {
                let listings = self
                    .connections
                    .list_available_models(connection_id, refresh)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Models(
                    listings
                        .into_iter()
                        .map(core_model_listing_to_api)
                        .collect(),
                ))
            }

            api::Command::GetPurposes => {
                let p = self
                    .connections
                    .get_purposes()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Purposes(api::PurposesView {
                    interactive: p.interactive.map(core_purpose_to_api),
                    dreaming: p.dreaming.map(core_purpose_to_api),
                    embedding: p.embedding.map(core_purpose_to_api),
                    titling: p.titling.map(core_purpose_to_api),
                }))
            }

            api::Command::SetPurpose { purpose, config } => {
                self.connections
                    .set_purpose(
                        api_purpose_kind_to_core(purpose),
                        api_purpose_to_core(config),
                    )
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
        self.handle_send_message_with_override(conversation_id, content, None, request_id, sink)
            .await
    }

    async fn handle_send_message_with_override(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
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

        // Bridge status updates from core callback -> canonical events.
        let status_tx = sink.clone();
        let conv_id_for_status = conversation_id.clone();
        let req_id_for_status = request_id.clone();
        let on_status: desktop_assistant_core::ports::llm::StatusCallback =
            Box::new(move |message| {
                let sink = Arc::clone(&status_tx);
                let conv_id = conv_id_for_status.clone();
                let req_id = req_id_for_status.clone();
                // Fire-and-forget: status messages are best-effort.
                tokio::spawn(async move {
                    sink.emit(api::Event::AssistantStatus {
                        conversation_id: conv_id,
                        request_id: req_id,
                        message,
                    })
                    .await;
                });
            });

        let override_for_core = override_selection.map(|o| PromptSelectionOverride {
            connection_id: o.connection_id,
            model_id: o.model_id,
            effort: o.effort.map(effort_from_api),
        });

        let outcome = self
            .conversations
            .send_prompt_with_override(
                &desktop_assistant_core::domain::ConversationId::from(conversation_id.as_str()),
                content,
                override_for_core,
                callback,
                on_status,
            )
            .await;

        if let Err(e) = forwarder.await {
            warn!("stream forwarder task failed: {e}");
        }

        match outcome {
            Ok(outcome) => {
                // Emit any one-time advisory warnings before the completion frame.
                for w in outcome.warnings {
                    let _ = sink
                        .emit(api::Event::ConversationWarningEmitted {
                            conversation_id: conversation_id.clone(),
                            warning: dispatch_warning_to_api(w),
                        })
                        .await;
                }

                let full_response = outcome.response;
                let _ = sink
                    .emit(api::Event::AssistantCompleted {
                        conversation_id: conversation_id.clone(),
                        request_id,
                        full_response,
                    })
                    .await;

                // Emit title change event so clients can update their UI
                // (the core service may generate a title after the first message).
                if let Ok(conv) = self
                    .conversations
                    .get_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        conversation_id.as_str(),
                    ))
                    .await
                {
                    let _ = sink
                        .emit(api::Event::ConversationTitleChanged {
                            conversation_id,
                            title: conv.title,
                        })
                        .await;
                }

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
        BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView,
        EmbeddingsSettingsView, LlmSettingsView, ModelListing as CoreModelListing,
        PersistenceSettingsView, PurposesView as CorePurposesView,
    };
    use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct FakeKnowledge;
    impl desktop_assistant_core::ports::inbound::KnowledgeService for FakeKnowledge {
        async fn list_entries(
            &self,
            _limit: usize,
            _offset: usize,
            _tag_filter: Option<Vec<String>>,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn get_entry(&self, _id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
            Ok(None)
        }
        async fn search_entries(
            &self,
            _query: String,
            _tag_filter: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn create_entry(
            &self,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<KnowledgeEntry, CoreError> {
            let mut e = KnowledgeEntry::new("kb-test", content, tags);
            e.metadata = metadata;
            Ok(e)
        }
        async fn update_entry(
            &self,
            id: String,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<KnowledgeEntry, CoreError> {
            let mut e = KnowledgeEntry::new(id, content, tags);
            e.metadata = metadata;
            Ok(e)
        }
        async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
            Ok(())
        }
    }

    struct FakeConnections;
    impl ConnectionsService for FakeConnections {
        async fn list_connections(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::ConnectionView>, CoreError>
        {
            Ok(vec![])
        }
        async fn create_connection(
            &self,
            _id: String,
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn update_connection(
            &self,
            _id: String,
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn delete_connection(&self, _id: String, _force: bool) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_available_models(
            &self,
            _connection_id: Option<String>,
            _refresh: bool,
        ) -> Result<Vec<CoreModelListing>, CoreError> {
            Ok(vec![])
        }
        async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
            Ok(CorePurposesView::default())
        }
        async fn set_purpose(
            &self,
            _purpose: PurposeKind,
            _config: PurposeConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

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
            _include_archived: bool,
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
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
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
            _on_status: StatusCallback,
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
                temperature: None,
                top_p: None,
                max_tokens: None,
                hosted_tool_search: None,
            })
        }
        async fn set_llm_settings(
            &self,
            _connector: String,
            _model: Option<String>,
            _base_url: Option<String>,
            _temperature: Option<f64>,
            _top_p: Option<f64>,
            _max_tokens: Option<u32>,
            _hosted_tool_search: Option<bool>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
            Ok(format!(
                "jwt-for-{}",
                subject.unwrap_or_else(|| "desktop-client".to_string())
            ))
        }
        async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
            Ok(token.starts_with("jwt-for-"))
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
                backend_llm_model: "bm".into(),
                embeddings_model: "em".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: false,
                hosted_tool_search_available: false,
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
        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(DatabaseSettingsView {
                url: String::new(),
                max_connections: 5,
            })
        }
        async fn set_database_settings(
            &self,
            _url: Option<String>,
            _max_connections: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
            Ok(BackendTasksSettingsView {
                has_separate_llm: false,
                llm_connector: "openai".into(),
                llm_model: "gpt-5".into(),
                llm_base_url: "https://api.openai.com/v1".into(),
                dreaming_enabled: false,
                dreaming_interval_secs: 3600,
                archive_after_days: 0,
            })
        }
        async fn set_backend_tasks_settings(
            &self,
            _llm_connector: Option<String>,
            _llm_model: Option<String>,
            _llm_base_url: Option<String>,
            _dreaming_enabled: bool,
            _dreaming_interval_secs: u64,
            _archive_after_days: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_mcp_servers(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn add_mcp_server(
            &self,
            _name: String,
            _command: String,
            _args: Vec<String>,
            _namespace: Option<String>,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            _name: String,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            _action: String,
            _server: Option<String>,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn get_ws_auth_settings(
            &self,
        ) -> Result<desktop_assistant_core::ports::inbound::WsAuthSettingsView, CoreError> {
            Ok(desktop_assistant_core::ports::inbound::WsAuthSettingsView {
                methods: vec![],
                oidc_issuer: String::new(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            })
        }
        async fn set_ws_auth_settings(
            &self,
            _methods: Vec<String>,
            _oidc_issuer: String,
            _oidc_auth_endpoint: String,
            _oidc_token_endpoint: String,
            _oidc_client_id: String,
            _oidc_scopes: String,
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
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        hosted_tool_search: None,
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

        #[allow(dead_code)]
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
            temperature: Option<f64>,
            top_p: Option<f64>,
            max_tokens: Option<u32>,
            hosted_tool_search: Option<bool>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.llm.connector = connector;
            if let Some(model) = model {
                state.llm.model = model;
            }
            if let Some(base_url) = base_url {
                state.llm.base_url = base_url;
            }
            state.llm.temperature = temperature;
            state.llm.top_p = top_p;
            state.llm.max_tokens = max_tokens;
            state.llm.hosted_tool_search = hosted_tool_search;
            Ok(())
        }

        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.api_key_set = true;
            state.llm.has_api_key = true;
            Ok(())
        }

        async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
            Ok(format!(
                "jwt-for-{}",
                subject.unwrap_or_else(|| "desktop-client".to_string())
            ))
        }

        async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
            Ok(token.starts_with("jwt-for-"))
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
                backend_llm_model: "bm".into(),
                embeddings_model: "em".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: false,
                hosted_tool_search_available: false,
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

        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(DatabaseSettingsView {
                url: String::new(),
                max_connections: 5,
            })
        }

        async fn set_database_settings(
            &self,
            _url: Option<String>,
            _max_connections: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
            Ok(BackendTasksSettingsView {
                has_separate_llm: false,
                llm_connector: "openai".into(),
                llm_model: "gpt-5".into(),
                llm_base_url: "https://api.openai.com/v1".into(),
                dreaming_enabled: false,
                dreaming_interval_secs: 3600,
                archive_after_days: 0,
            })
        }
        async fn set_backend_tasks_settings(
            &self,
            _llm_connector: Option<String>,
            _llm_model: Option<String>,
            _llm_base_url: Option<String>,
            _dreaming_enabled: bool,
            _dreaming_interval_secs: u64,
            _archive_after_days: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_mcp_servers(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn add_mcp_server(
            &self,
            _name: String,
            _command: String,
            _args: Vec<String>,
            _namespace: Option<String>,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            _name: String,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            _action: String,
            _server: Option<String>,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn get_ws_auth_settings(
            &self,
        ) -> Result<desktop_assistant_core::ports::inbound::WsAuthSettingsView, CoreError> {
            Ok(desktop_assistant_core::ports::inbound::WsAuthSettingsView {
                methods: vec![],
                oidc_issuer: String::new(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            })
        }
        async fn set_ws_auth_settings(
            &self,
            _methods: Vec<String>,
            _oidc_issuer: String,
            _oidc_auth_endpoint: String,
            _oidc_token_endpoint: String,
            _oidc_client_id: String,
            _oidc_scopes: String,
        ) -> Result<(), CoreError> {
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
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
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
            _on_status: StatusCallback,
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
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
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
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
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
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
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
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h.handle_command(api::Command::GetConfig).await.unwrap();
        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };

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
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h
            .handle_command(api::Command::SetConfig {
                changes: api::ConfigChanges {
                    embeddings_connector: Some("openai".into()),
                    embeddings_model: Some("text-embedding-3-large".into()),
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
        assert_eq!(config.embeddings.model, "text-embedding-3-large");
        assert_eq!(config.persistence.remote_name, "upstream");
        assert!(!config.persistence.push_on_update);
    }
}
