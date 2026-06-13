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
    /// Windowed message fetch (CC-5 / #361): a slice of a conversation instead
    /// of the full transcript. `after_count >= 0` = from that raw index; else
    /// `tail > 0` = the last `tail`; `include_roles` empty = all roles.
    async fn get_messages(
        &self,
        conversation_id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> Result<api::MessagesView>;
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
    /// Access the transport-agnostic command channel, so callers can issue
    /// commands that aren't exposed on the high-level [`AssistantClient`]
    /// facade (config Settings, per-conversation model override / model
    /// selection, background tasks, purposes, named-connection management)
    /// over *any* transport.
    ///
    /// Returns `Some` for **all three** transports (#213): `Ws` and `Uds`
    /// speak the shared `WsRequest`/`WsFrame` protocol, and `Dbus` round-trips
    /// the same `api::Command`/`api::CommandResult` as JSON over the
    /// `org.desktopAssistant.Commands` interface — all via the single
    /// [`AssistantCommands`] trait. This replaced the WS-only `as_ws`
    /// accessor (adele-gtk#49 / #213): there is no longer any
    /// connector-specific command surface.
    pub fn as_commands(&self) -> Option<&(dyn AssistantCommands + '_)> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => Some(client),
            Self::Ws(client) => Some(client),
            Self::Uds(client) => Some(client),
        }
    }

    /// Re-establish the underlying connection in place after a drop (#246),
    /// re-running the handshake (re-auth via the credential in `config`) and
    /// re-binding the persistent signal/drop channels — so the *same*
    /// `TransportClient` (and any `&TransportClient` a caller holds) keeps
    /// working without the Connector swapping the client. The socket transports
    /// (UDS/WS) support this; the D-Bus transport doesn't reconnect this way and
    /// returns an error (its clients don't use the Connector). Resolves the
    /// bearer token from `config` on each attempt so a freshly-minted local JWT
    /// is used (a long outage can still outlive a token's validity — surfaced as
    /// an auth error the supervisor retries with backoff).
    pub async fn reconnect(&self, config: &ConnectionConfig) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(_) => Err(anyhow::anyhow!(
                "the D-Bus transport does not support Connector auto-reconnect"
            )),
            Self::Ws(client) => {
                let token = resolve_ws_bearer_token(config).await?;
                // Re-send the #248 system id + host label from the stored config
                // so co-location survives a reconnect (the supervisor re-reads
                // this same config).
                client
                    .reconnect(
                        &config.ws_url,
                        &token,
                        config.tls_ca_cert.as_deref(),
                        config.system_id.as_deref(),
                        config.host_label.as_deref(),
                    )
                    .await
            }
            Self::Uds(client) => {
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
                client
                    .reconnect(
                        &path,
                        &token,
                        config.system_id.as_deref(),
                        config.host_label.as_deref(),
                    )
                    .await
            }
        }
    }
}

/// A receiver that fires once per underlying-socket close, so the
/// [`Connector`](crate::Connector) reconnect supervisor can react (#246). `None`
/// for the D-Bus transport, which doesn't auto-reconnect.
pub type DropNotifier = mpsc::UnboundedReceiver<()>;

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

    async fn get_messages(
        &self,
        conversation_id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> Result<api::MessagesView> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => {
                client
                    .get_messages(conversation_id, tail, after_count, include_roles)
                    .await
            }
            Self::Ws(client) => {
                client
                    .get_messages(conversation_id, tail, after_count, include_roles)
                    .await
            }
            Self::Uds(client) => {
                client
                    .get_messages(conversation_id, tail, after_count, include_roles)
                    .await
            }
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

/// Connect over the transport named by `config`. Returns the client, the signal
/// stream, and a [`DropNotifier`] that fires once per underlying-socket close
/// for the socket transports (#246) — `None` for D-Bus, which doesn't
/// auto-reconnect. The [`Connector`](crate::Connector) uses the notifier to
/// drive its reconnect supervisor.
pub async fn connect_transport(
    config: &ConnectionConfig,
) -> Result<(
    TransportClient,
    mpsc::UnboundedReceiver<SignalEvent>,
    Option<DropNotifier>,
)> {
    match config.transport_mode {
        #[cfg(feature = "dbus")]
        TransportMode::Dbus => {
            let client = crate::dbus_client::DbusClient::connect().await?;
            let signal_rx = client.subscribe_signals().await?;
            Ok((TransportClient::Dbus(client), signal_rx, None))
        }
        #[cfg(not(feature = "dbus"))]
        TransportMode::Dbus => Err(anyhow::anyhow!(
            "D-Bus transport is not available (compiled without dbus feature)"
        )),
        TransportMode::Ws => {
            let token = resolve_ws_bearer_token(config).await?;
            // Carry the #248 system id + host label from the config into the
            // handshake (custom upgrade headers); the Connector stamps them on.
            let (client, signal_rx, drop_rx) = WsClient::connect(
                &config.ws_url,
                &token,
                config.tls_ca_cert.as_deref(),
                config.system_id.as_deref(),
                config.host_label.as_deref(),
            )
            .await?;
            Ok((TransportClient::Ws(client), signal_rx, Some(drop_rx)))
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
            // Carry the #248 system id + host label from the config into the
            // JWT handshake frame; the Connector stamps them on.
            let (client, signal_rx, drop_rx) = UdsClient::connect(
                &path,
                &token,
                config.system_id.as_deref(),
                config.host_label.as_deref(),
            )
            .await?;
            Ok((TransportClient::Uds(client), signal_rx, Some(drop_rx)))
        }
    }
}
