use std::net::SocketAddr;
use std::sync::Arc;

use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_ws::{WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
use desktop_assistant_core::ports::inbound::{
    AssistantService, ConnectorDefaultsView, ConversationService, EmbeddingsSettingsView,
    LlmSettingsView, PersistenceSettingsView, SettingsService,
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
        _on_chunk: ChunkCallback,
    ) -> Result<String, CoreError> {
        Ok("".into())
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

#[tokio::test]
async fn ws_ping_roundtrip() {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(FakeSettings),
    ));

    let app = router(handler);

    // bind ephemeral port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let req = WsRequest {
        id: "1".into(),
        command: desktop_assistant_api_model::Command::Ping,
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();

    let msg = ws.next().await.unwrap().unwrap();
    let text = msg.into_text().unwrap();
    let frame: WsFrame = serde_json::from_str(&text).unwrap();

    match frame {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "1");
            assert_eq!(
                result,
                desktop_assistant_api_model::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    server.abort();
}
