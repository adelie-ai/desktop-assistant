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
                    tracing::error!("LLM error for conversation {llm_conv_id}: {e}");
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
}
