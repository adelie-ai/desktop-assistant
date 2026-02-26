use std::net::SocketAddr;
use std::sync::Arc;

use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_ws::{WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::{Duration, timeout};

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
        mut on_chunk: ChunkCallback,
    ) -> Result<String, CoreError> {
        on_chunk("he".into());
        on_chunk("llo".into());
        Ok("hello".into())
    }
}

struct CancelAwareConversations {
    cancelled: Arc<AtomicBool>,
}
impl ConversationService for CancelAwareConversations {
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
            if !on_chunk("x".repeat(512)) {
                self.cancelled.store(true, Ordering::SeqCst);
                return Ok("cancelled".to_string());
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Ok("done".to_string())
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

struct StatefulSettings {
    state: Mutex<SettingsState>,
}

impl StatefulSettings {
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
}

impl SettingsService for StatefulSettings {
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

#[tokio::test]
async fn ws_get_status_roundtrip() {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(FakeSettings),
    ));

    let app = router(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let req = WsRequest {
        id: "2".into(),
        command: desktop_assistant_api_model::Command::GetStatus,
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
            assert_eq!(id, "2");
            assert_eq!(
                result,
                desktop_assistant_api_model::CommandResult::Status(
                    desktop_assistant_api_model::Status {
                        version: "0.0.0-test".into()
                    }
                )
            );
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    server.abort();
}

#[tokio::test]
async fn ws_set_config_roundtrip_emits_config_changed() {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(StatefulSettings::new()),
    ));

    let app = router(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let req = WsRequest {
        id: "cfg-1".into(),
        command: desktop_assistant_api_model::Command::SetConfig {
            changes: desktop_assistant_api_model::ConfigChanges {
                llm_connector: Some("ollama".into()),
                llm_model: Some("llama3.1:8b".into()),
                llm_base_url: Some("http://localhost:11434".into()),
                llm_api_key: Some("abc123".into()),
                persistence_enabled: Some(true),
                persistence_remote_name: Some("upstream".into()),
                ..Default::default()
            },
        },
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();

    let result_frame =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    let config_from_result = match result_frame {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "cfg-1");
            match result {
                desktop_assistant_api_model::CommandResult::Config(config) => config,
                other => panic!("unexpected result payload: {other:?}"),
            }
        }
        other => panic!("unexpected frame: {other:?}"),
    };

    assert_eq!(config_from_result.llm.connector, "ollama");
    assert_eq!(config_from_result.llm.model, "llama3.1:8b");
    assert!(config_from_result.llm.has_api_key);
    assert_eq!(config_from_result.persistence.remote_name, "upstream");

    let event_frame =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    match event_frame {
        WsFrame::Event {
            event: desktop_assistant_api_model::Event::ConfigChanged { config },
        } => {
            assert_eq!(config, config_from_result);
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    server.abort();
}

#[tokio::test]
async fn ws_send_message_ack_then_streaming_events() {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(FakeSettings),
    ));

    let app = router(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let req = WsRequest {
        id: "3".into(),
        command: desktop_assistant_api_model::Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hello".into(),
        },
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();

    let first =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    assert_eq!(
        first,
        WsFrame::Result {
            id: "3".into(),
            result: desktop_assistant_api_model::CommandResult::Ack
        }
    );

    let second =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    let request_id = match second {
        WsFrame::Event {
            event:
                desktop_assistant_api_model::Event::AssistantDelta {
                    conversation_id,
                    request_id,
                    chunk,
                },
        } => {
            assert_eq!(conversation_id, "c1");
            assert_eq!(chunk, "he");
            request_id
        }
        other => panic!("unexpected frame: {other:?}"),
    };

    let third =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    match third {
        WsFrame::Event {
            event:
                desktop_assistant_api_model::Event::AssistantDelta {
                    conversation_id,
                    request_id: next_request_id,
                    chunk,
                },
        } => {
            assert_eq!(conversation_id, "c1");
            assert_eq!(chunk, "llo");
            assert_eq!(next_request_id, request_id);
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    let fourth =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    match fourth {
        WsFrame::Event {
            event:
                desktop_assistant_api_model::Event::AssistantCompleted {
                    conversation_id,
                    request_id: completed_request_id,
                    full_response,
                },
        } => {
            assert_eq!(conversation_id, "c1");
            assert_eq!(completed_request_id, request_id);
            assert_eq!(full_response, "hello");
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    server.abort();
}

#[tokio::test]
async fn ws_send_message_cancels_when_client_disconnects() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(CancelAwareConversations {
            cancelled: Arc::clone(&cancelled),
        }),
        Arc::new(FakeSettings),
    ));

    let app = router(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let req = WsRequest {
        id: "4".into(),
        command: desktop_assistant_api_model::Command::SendMessage {
            conversation_id: "c1".into(),
            content: "cancel-me".into(),
        },
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();

    let ack =
        serde_json::from_str::<WsFrame>(&ws.next().await.unwrap().unwrap().into_text().unwrap())
            .unwrap();
    assert_eq!(
        ack,
        WsFrame::Result {
            id: "4".into(),
            result: desktop_assistant_api_model::CommandResult::Ack
        }
    );

    ws.close(None).await.unwrap();
    drop(ws);

    timeout(Duration::from_secs(2), async {
        while !cancelled.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("stream was not cancelled after disconnect");

    server.abort();
}

#[tokio::test]
async fn ws_serve_with_shutdown_exits() {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(FakeSettings),
    ));
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let server = tokio::spawn(desktop_assistant_ws::serve_with_shutdown(
        handler,
        addr,
        async {
            tokio::time::sleep(Duration::from_millis(20)).await;
        },
    ));

    let join = timeout(Duration::from_secs(2), server).await.unwrap();
    let result = join.unwrap();
    assert!(
        result.is_ok(),
        "server should shut down cleanly: {result:?}"
    );
}
