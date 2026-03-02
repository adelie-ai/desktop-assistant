use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use desktop_assistant_api_model as api;
use desktop_assistant_ws::{WsFrame, WsRequest};
use futures::{SinkExt, StreamExt};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use crate::signal::SignalEvent;
use crate::types::{ConversationDetail, ConversationSummary};

type PendingResult = Result<api::CommandResult, String>;

pub struct WsClient {
    outbound_tx: mpsc::UnboundedSender<Message>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<PendingResult>>>>,
}

impl WsClient {
    pub async fn connect(
        ws_url: &str,
        bearer_token: &str,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SignalEvent>)> {
        let mut request = ws_url.into_client_request()?;
        request.headers_mut().insert(
            tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
            format!("Bearer {bearer_token}").parse()?,
        );

        let (socket, _response) = tokio_tungstenite::connect_async(request).await?;
        let (mut ws_tx, mut ws_rx) = socket.split();

        let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
        tokio::spawn(async move {
            while let Some(message) = outbound_rx.recv().await {
                if ws_tx.send(message).await.is_err() {
                    break;
                }
            }
        });

        let pending = Arc::new(Mutex::new(
            HashMap::<String, oneshot::Sender<PendingResult>>::new(),
        ));
        let pending_for_reader = Arc::clone(&pending);

        let (signal_tx, signal_rx) = mpsc::unbounded_channel::<SignalEvent>();
        tokio::spawn(async move {
            while let Some(Ok(message)) = ws_rx.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Ok(frame) = serde_json::from_str::<WsFrame>(&text) else {
                    continue;
                };

                match frame {
                    WsFrame::Result { id, result } => {
                        if let Some(tx) = pending_for_reader.lock().await.remove(&id) {
                            let _ = tx.send(Ok(result));
                        }
                    }
                    WsFrame::Error { id, error } => {
                        if let Some(tx) = pending_for_reader.lock().await.remove(&id) {
                            let _ = tx.send(Err(error));
                        }
                    }
                    WsFrame::Event { event } => {
                        if let Some(signal) = map_event_to_signal(event) {
                            let _ = signal_tx.send(signal);
                        }
                    }
                }
            }

            let mut pending = pending_for_reader.lock().await;
            for (_id, tx) in pending.drain() {
                let _ = tx.send(Err("websocket disconnected".to_string()));
            }
        });

        Ok((
            Self {
                outbound_tx,
                pending,
            },
            signal_rx,
        ))
    }

    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = WsRequest {
            id: id.clone(),
            command,
        };
        let payload = serde_json::to_string(&request)?;

        let (tx, rx) = oneshot::channel::<PendingResult>();
        self.pending.lock().await.insert(id.clone(), tx);

        if self
            .outbound_tx
            .send(Message::Text(payload.into()))
            .is_err()
        {
            self.pending.lock().await.remove(&id);
            return Err(anyhow!("failed to send websocket request"));
        }

        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(_closed) => Err(anyhow!("websocket response channel closed")),
        }
    }

    pub async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let result = self
            .send_command(api::Command::ListConversations { max_age_days: None })
            .await?;

        let api::CommandResult::Conversations(items) = result else {
            return Err(anyhow!(
                "unexpected websocket response for list_conversations"
            ));
        };

        Ok(items.into_iter().map(ConversationSummary::from).collect())
    }

    pub async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        let result = self
            .send_command(api::Command::GetConversation { id: id.to_string() })
            .await?;

        let api::CommandResult::Conversation(conversation) = result else {
            return Err(anyhow!(
                "unexpected websocket response for get_conversation"
            ));
        };

        Ok(ConversationDetail::from(conversation))
    }

    pub async fn create_conversation(&self, title: &str) -> Result<String> {
        let result = self
            .send_command(api::Command::CreateConversation {
                title: title.to_string(),
            })
            .await?;

        let api::CommandResult::ConversationId { id } = result else {
            return Err(anyhow!(
                "unexpected websocket response for create_conversation"
            ));
        };
        Ok(id)
    }

    pub async fn delete_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!(
                "unexpected websocket response for delete_conversation"
            ));
        };
        Ok(())
    }

    pub async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::RenameConversation {
                id: id.to_string(),
                title: title.to_string(),
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!(
                "unexpected websocket response for rename_conversation"
            ));
        };
        Ok(())
    }

    pub async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        let result = self
            .send_command(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected websocket response for send_prompt"));
        };

        // WS send-message ack does not include request id; first stream event carries it.
        Ok(String::new())
    }
}

pub fn map_event_to_signal(event: api::Event) -> Option<SignalEvent> {
    match event {
        api::Event::AssistantDelta {
            request_id, chunk, ..
        } => Some(SignalEvent::Chunk { request_id, chunk }),
        api::Event::AssistantCompleted {
            request_id,
            full_response,
            ..
        } => Some(SignalEvent::Complete {
            request_id,
            full_response,
        }),
        api::Event::AssistantError {
            request_id, error, ..
        } => Some(SignalEvent::Error { request_id, error }),
        api::Event::ConversationTitleChanged {
            conversation_id,
            title,
        } => Some(SignalEvent::TitleChanged {
            conversation_id,
            title,
        }),
        api::Event::ConfigChanged { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_stream_events_to_signal_events() {
        let delta = map_event_to_signal(api::Event::AssistantDelta {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            chunk: "he".to_string(),
        });
        assert!(matches!(delta, Some(SignalEvent::Chunk { .. })));

        let complete = map_event_to_signal(api::Event::AssistantCompleted {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            full_response: "hello".to_string(),
        });
        assert!(matches!(complete, Some(SignalEvent::Complete { .. })));

        let error = map_event_to_signal(api::Event::AssistantError {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            error: "oops".to_string(),
        });
        assert!(matches!(error, Some(SignalEvent::Error { .. })));
    }

    #[test]
    fn maps_title_changed_event() {
        let event = map_event_to_signal(api::Event::ConversationTitleChanged {
            conversation_id: "c1".to_string(),
            title: "New Title".to_string(),
        });
        assert!(matches!(event, Some(SignalEvent::TitleChanged { .. })));
    }

    #[test]
    fn ignores_non_stream_config_events() {
        let event = map_event_to_signal(api::Event::ConfigChanged {
            config: api::Config {
                llm: api::LlmSettingsView {
                    connector: "openai".to_string(),
                    model: "gpt-5.2".to_string(),
                    base_url: "https://api.openai.com/v1".to_string(),
                    has_api_key: true,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                },
                embeddings: api::EmbeddingsSettingsView {
                    connector: "openai".to_string(),
                    model: "text-embedding-3-small".to_string(),
                    base_url: "https://api.openai.com/v1".to_string(),
                    has_api_key: true,
                    available: true,
                    is_default: true,
                },
                persistence: api::PersistenceSettingsView {
                    enabled: false,
                    remote_url: String::new(),
                    remote_name: "origin".to_string(),
                    push_on_update: true,
                },
            },
        });
        assert!(event.is_none());
    }
}
