use anyhow::Result;
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use tokio::sync::mpsc;

use crate::auth::resolve_ws_bearer_token;
use crate::commands::AssistantCommands;
use crate::config::{ConnectionConfig, TransportMode, default_desktop_socket_path};
use crate::signal::SignalEvent;
use crate::types::{ConversationDetail, ConversationSummary};
use crate::uds_client::UdsClient;
use crate::ws_client::WsClient;

#[async_trait]
pub trait AssistantClient: Send + Sync {
    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>>;
    async fn list_conversations_with_archived(&self) -> Result<Vec<ConversationSummary>>;
    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail>;
    async fn create_conversation(&self, title: &str) -> Result<String>;
    async fn delete_conversation(&self, id: &str) -> Result<()>;
    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()>;
    async fn archive_conversation(&self, id: &str) -> Result<()>;
    async fn unarchive_conversation(&self, id: &str) -> Result<()>;
    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String>;

    // Knowledge management (#73)
    async fn list_knowledge_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<api::KnowledgeEntryView>>;
    async fn get_knowledge_entry(&self, id: &str) -> Result<Option<api::KnowledgeEntryView>>;
    async fn search_knowledge_entries(
        &self,
        query: &str,
        tag_filter: Option<Vec<String>>,
        limit: u32,
    ) -> Result<Vec<api::KnowledgeEntryView>>;
    async fn create_knowledge_entry(
        &self,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView>;
    async fn update_knowledge_entry(
        &self,
        id: &str,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView>;
    async fn delete_knowledge_entry(&self, id: &str) -> Result<()>;
}

pub enum TransportClient {
    #[cfg(feature = "dbus")]
    Dbus(crate::dbus_client::DbusClient),
    Ws(WsClient),
    Uds(UdsClient),
}

impl TransportClient {
    /// Access the underlying WebSocket client when the transport is WS, so
    /// callers can issue commands that aren't exposed on the shared
    /// `AssistantClient` trait (e.g. named-connection management).
    pub fn as_ws(&self) -> Option<&WsClient> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(_) => None,
            Self::Ws(client) => Some(client),
            Self::Uds(_) => None,
        }
    }

    /// Access the transport-agnostic command channel, so callers can issue
    /// commands that aren't exposed on the high-level [`AssistantClient`]
    /// facade (config Settings, per-conversation model override, background
    /// tasks) over *any* socket transport — WebSocket or local UDS.
    ///
    /// Returns `Some` for the `Ws` and `Uds` variants, which both speak the
    /// shared `WsRequest`/`WsFrame` protocol via [`AssistantCommands`], and
    /// `None` for `Dbus`, which talks a separate typed zbus interface and so
    /// does not implement that trait. This supersedes [`as_ws`](Self::as_ws)
    /// for command-channel access (adele-gtk#49); `as_ws` is retained until
    /// downstream callers migrate.
    pub fn as_commands(&self) -> Option<&(dyn AssistantCommands + '_)> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(_) => None,
            Self::Ws(client) => Some(client),
            Self::Uds(client) => Some(client),
        }
    }
}

#[async_trait]
impl AssistantClient for TransportClient {
    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.list_conversations().await,
            Self::Ws(client) => client.list_conversations().await,
            Self::Uds(client) => client.list_conversations().await,
        }
    }

    async fn list_conversations_with_archived(&self) -> Result<Vec<ConversationSummary>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.list_conversations_with_archived().await,
            Self::Ws(client) => client.list_conversations_with_archived().await,
            Self::Uds(client) => client.list_conversations_with_archived().await,
        }
    }

    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.get_conversation(id).await,
            Self::Ws(client) => client.get_conversation(id).await,
            Self::Uds(client) => client.get_conversation(id).await,
        }
    }

    async fn create_conversation(&self, title: &str) -> Result<String> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.create_conversation(title).await,
            Self::Ws(client) => client.create_conversation(title).await,
            Self::Uds(client) => client.create_conversation(title).await,
        }
    }

    async fn delete_conversation(&self, id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.delete_conversation(id).await,
            Self::Ws(client) => client.delete_conversation(id).await,
            Self::Uds(client) => client.delete_conversation(id).await,
        }
    }

    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.rename_conversation(id, title).await,
            Self::Ws(client) => client.rename_conversation(id, title).await,
            Self::Uds(client) => client.rename_conversation(id, title).await,
        }
    }

    async fn archive_conversation(&self, id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.archive_conversation(id).await,
            Self::Ws(client) => client.archive_conversation(id).await,
            Self::Uds(client) => client.archive_conversation(id).await,
        }
    }

    async fn unarchive_conversation(&self, id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.unarchive_conversation(id).await,
            Self::Ws(client) => client.unarchive_conversation(id).await,
            Self::Uds(client) => client.unarchive_conversation(id).await,
        }
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.send_prompt(conversation_id, prompt).await,
            Self::Ws(client) => client.send_prompt(conversation_id, prompt).await,
            Self::Uds(client) => client.send_prompt(conversation_id, prompt).await,
        }
    }

    async fn list_knowledge_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => {
                client
                    .list_knowledge_entries(limit, offset, tag_filter)
                    .await
            }
            Self::Ws(client) => {
                client
                    .list_knowledge_entries(limit, offset, tag_filter)
                    .await
            }
            Self::Uds(client) => {
                client
                    .list_knowledge_entries(limit, offset, tag_filter)
                    .await
            }
        }
    }

    async fn get_knowledge_entry(&self, id: &str) -> Result<Option<api::KnowledgeEntryView>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.get_knowledge_entry(id).await,
            Self::Ws(client) => client.get_knowledge_entry(id).await,
            Self::Uds(client) => client.get_knowledge_entry(id).await,
        }
    }

    async fn search_knowledge_entries(
        &self,
        query: &str,
        tag_filter: Option<Vec<String>>,
        limit: u32,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => {
                client
                    .search_knowledge_entries(query, tag_filter, limit)
                    .await
            }
            Self::Ws(client) => {
                client
                    .search_knowledge_entries(query, tag_filter, limit)
                    .await
            }
            Self::Uds(client) => {
                client
                    .search_knowledge_entries(query, tag_filter, limit)
                    .await
            }
        }
    }

    async fn create_knowledge_entry(
        &self,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.create_knowledge_entry(content, tags, metadata).await,
            Self::Ws(client) => client.create_knowledge_entry(content, tags, metadata).await,
            Self::Uds(client) => client.create_knowledge_entry(content, tags, metadata).await,
        }
    }

    async fn update_knowledge_entry(
        &self,
        id: &str,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => {
                client
                    .update_knowledge_entry(id, content, tags, metadata)
                    .await
            }
            Self::Ws(client) => {
                client
                    .update_knowledge_entry(id, content, tags, metadata)
                    .await
            }
            Self::Uds(client) => {
                client
                    .update_knowledge_entry(id, content, tags, metadata)
                    .await
            }
        }
    }

    async fn delete_knowledge_entry(&self, id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.delete_knowledge_entry(id).await,
            Self::Ws(client) => client.delete_knowledge_entry(id).await,
            Self::Uds(client) => client.delete_knowledge_entry(id).await,
        }
    }
}

pub fn transport_label(config: &ConnectionConfig) -> String {
    match config.transport_mode {
        TransportMode::Dbus => "Connected via D-Bus".to_string(),
        TransportMode::Ws => format!("Connected to {}", config.ws_url),
        TransportMode::Uds => match &config.socket_path {
            Some(path) => format!("Connected via local socket {}", path.display()),
            None => "Connected via local socket".to_string(),
        },
    }
}

pub async fn connect_transport(
    config: &ConnectionConfig,
) -> Result<(TransportClient, mpsc::UnboundedReceiver<SignalEvent>)> {
    match config.transport_mode {
        #[cfg(feature = "dbus")]
        TransportMode::Dbus => {
            let client = crate::dbus_client::DbusClient::connect().await?;
            let signal_rx = client.subscribe_signals().await?;
            Ok((TransportClient::Dbus(client), signal_rx))
        }
        #[cfg(not(feature = "dbus"))]
        TransportMode::Dbus => Err(anyhow::anyhow!(
            "D-Bus transport is not available (compiled without dbus feature)"
        )),
        TransportMode::Ws => {
            let token = resolve_ws_bearer_token(config).await?;
            let (client, signal_rx) =
                WsClient::connect(&config.ws_url, &token, config.tls_ca_cert.as_deref()).await?;
            Ok((TransportClient::Ws(client), signal_rx))
        }
        TransportMode::Uds => {
            // The local minter issues the same JWT the UDS server's handshake
            // expects, so the existing resolver is reused unchanged.
            let token = resolve_ws_bearer_token(config).await?;
            let path = config
                .socket_path
                .clone()
                .or_else(default_desktop_socket_path)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no UDS socket path: set ConnectionConfig.socket_path or XDG_RUNTIME_DIR"
                    )
                })?;
            let (client, signal_rx) = UdsClient::connect(&path, &token).await?;
            Ok((TransportClient::Uds(client), signal_rx))
        }
    }
}
