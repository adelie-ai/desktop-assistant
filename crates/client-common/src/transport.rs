use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::auth::resolve_ws_bearer_token;
use crate::config::{ConnectionConfig, TransportMode};
use crate::signal::SignalEvent;
use crate::types::{ConversationDetail, ConversationSummary};
use crate::ws_client::WsClient;

#[async_trait]
pub trait AssistantClient: Send + Sync {
    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>>;
    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail>;
    async fn create_conversation(&self, title: &str) -> Result<String>;
    async fn delete_conversation(&self, id: &str) -> Result<()>;
    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()>;
    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String>;
}

pub enum TransportClient {
    #[cfg(feature = "dbus")]
    Dbus(crate::dbus_client::DbusClient),
    Ws(WsClient),
}

#[async_trait]
impl AssistantClient for TransportClient {
    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.list_conversations().await,
            Self::Ws(client) => client.list_conversations().await,
        }
    }

    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.get_conversation(id).await,
            Self::Ws(client) => client.get_conversation(id).await,
        }
    }

    async fn create_conversation(&self, title: &str) -> Result<String> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.create_conversation(title).await,
            Self::Ws(client) => client.create_conversation(title).await,
        }
    }

    async fn delete_conversation(&self, id: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.delete_conversation(id).await,
            Self::Ws(client) => client.delete_conversation(id).await,
        }
    }

    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.rename_conversation(id, title).await,
            Self::Ws(client) => client.rename_conversation(id, title).await,
        }
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        match self {
            #[cfg(feature = "dbus")]
            Self::Dbus(client) => client.send_prompt(conversation_id, prompt).await,
            Self::Ws(client) => client.send_prompt(conversation_id, prompt).await,
        }
    }
}

pub fn transport_label(mode: TransportMode) -> &'static str {
    match mode {
        TransportMode::Dbus => "Connected via D-Bus",
        TransportMode::Ws => "Connected via WebSocket",
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
            let (client, signal_rx) = WsClient::connect(&config.ws_url, &token).await?;
            Ok((TransportClient::Ws(client), signal_rx))
        }
    }
}
