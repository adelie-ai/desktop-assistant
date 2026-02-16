use anyhow::Result;
use futures::StreamExt;
use tokio::sync::mpsc;
use zbus::Connection;

use crate::app::{ChatMessage, ConversationDetail, ConversationSummary};

const DEFAULT_DBUS_SERVICE: &str = "org.desktopAssistant";
const DBUS_CONVERSATIONS_PATH: &str = "/org/desktopAssistant/Conversations";

#[zbus::proxy(interface = "org.desktopAssistant.Conversations")]
trait Conversations {
    async fn create_conversation(&self, title: &str) -> zbus::fdo::Result<String>;

    async fn list_conversations(&self) -> zbus::fdo::Result<Vec<(String, String, u32)>>;

    async fn get_conversation(
        &self,
        id: &str,
    ) -> zbus::fdo::Result<(String, String, Vec<(String, String)>)>;

    async fn delete_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> zbus::fdo::Result<String>;

    #[zbus(signal)]
    fn response_chunk(
        &self,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_complete(
        &self,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_error(
        &self,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::fdo::Result<()>;
}

/// Signal event received from D-Bus.
#[derive(Debug)]
pub enum SignalEvent {
    Chunk {
        request_id: String,
        chunk: String,
    },
    Complete {
        request_id: String,
        full_response: String,
    },
    Error {
        request_id: String,
        error: String,
    },
}

pub struct DbusClient {
    proxy: ConversationsProxy<'static>,
}

impl DbusClient {
    pub async fn connect() -> Result<Self> {
        let connection = Connection::session().await?;
        let service_name = std::env::var("DESKTOP_ASSISTANT_DBUS_SERVICE")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_DBUS_SERVICE.to_string());
        let proxy = ConversationsProxy::builder(&connection)
            .destination(service_name)?
            .path(DBUS_CONVERSATIONS_PATH)?
            .build()
            .await?;
        Ok(Self { proxy })
    }

    pub async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let raw = self.proxy.list_conversations().await?;
        Ok(raw
            .into_iter()
            .map(|(id, title, message_count)| ConversationSummary {
                id,
                title,
                message_count,
            })
            .collect())
    }

    pub async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        let (conv_id, title, messages) = self.proxy.get_conversation(id).await?;
        Ok(ConversationDetail {
            id: conv_id,
            title,
            messages: messages
                .into_iter()
                .map(|(role, content)| ChatMessage { role, content })
                .collect(),
        })
    }

    pub async fn create_conversation(&self, title: &str) -> Result<String> {
        let id = self.proxy.create_conversation(title).await?;
        Ok(id)
    }

    pub async fn delete_conversation(&self, id: &str) -> Result<()> {
        self.proxy.delete_conversation(id).await?;
        Ok(())
    }

    pub async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        let request_id = self.proxy.send_prompt(conversation_id, prompt).await?;
        Ok(request_id)
    }

    /// Subscribe to all response signals and return a receiver for signal events.
    /// Spawns background tasks that forward signals to the channel.
    pub async fn subscribe_signals(&self) -> Result<mpsc::UnboundedReceiver<SignalEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut chunk_stream = self.proxy.receive_response_chunk().await?;
        let tx_chunk = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = chunk_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_chunk.send(SignalEvent::Chunk {
                        request_id: args.request_id.to_string(),
                        chunk: args.chunk.to_string(),
                    });
                }
            }
        });

        let mut complete_stream = self.proxy.receive_response_complete().await?;
        let tx_complete = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = complete_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_complete.send(SignalEvent::Complete {
                        request_id: args.request_id.to_string(),
                        full_response: args.full_response.to_string(),
                    });
                }
            }
        });

        let mut error_stream = self.proxy.receive_response_error().await?;
        tokio::spawn(async move {
            while let Some(signal) = error_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx.send(SignalEvent::Error {
                        request_id: args.request_id.to_string(),
                        error: args.error.to_string(),
                    });
                }
            }
        });

        Ok(rx)
    }
}
