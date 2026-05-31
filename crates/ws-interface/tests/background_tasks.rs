//! End-to-end WebSocket smoke tests for the background-task command
//! surface (#114). The dispatcher-level contract is covered by tests in
//! `desktop-assistant-transport-dispatch`; this file verifies the wiring
//! survives the full WS upgrade + JSON framing round-trip.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::background_tasks::BackgroundTaskRegistry;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink, UserId};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_ws::{WsAuthValidator, WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

/// Handler exposing a real `BackgroundTaskRegistry` for the new
/// background-task arms. The other arms are not exercised here.
struct RegistryHandler {
    registry: Arc<BackgroundTaskRegistry>,
}

#[async_trait::async_trait]
impl AssistantApiHandler for RegistryHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        let user_id = current_user_id();
        match cmd {
            api::Command::ListBackgroundTasks {
                include_finished,
                limit,
            } => Ok(api::CommandResult::BackgroundTasks(self.registry.list(
                &user_id,
                include_finished,
                limit,
            ))),
            api::Command::SubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
            _ => Err(ApiError::Unsupported),
        }
    }
    async fn handle_send_message(
        &self,
        _c: String,
        _t: String,
        _r: String,
        _s: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        Ok(())
    }
    async fn subscribe_user_events(&self) -> Option<broadcast::Receiver<api::Event>> {
        Some(self.registry.subscribe(&current_user_id()))
    }
}

struct StaticJwtAuth;

#[async_trait::async_trait]
impl WsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        token == "test-jwt"
    }
    async fn extract_user_id(&self, _token: &str) -> Option<UserId> {
        Some(UserId::new("alice"))
    }
}

fn ws_request(
    url: &str,
    bearer: Option<&str>,
) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = url.into_client_request().unwrap();
    if let Some(token) = bearer {
        request.headers_mut().insert(
            tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
    }
    request
}

async fn start_server(
    handler: Arc<dyn AssistantApiHandler>,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app = router(handler, Arc::new(StaticJwtAuth));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, server)
}

/// Acceptance: ListBackgroundTasks round-trips over a real WS upgrade.
#[tokio::test]
async fn ws_list_background_tasks_roundtrip() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(RegistryHandler {
        registry: Arc::clone(&registry),
    });
    let alice = UserId::new("alice");

    // Pre-populate two tasks under alice.
    for i in 0..2 {
        registry.spawn(
            alice.clone(),
            api::TaskKind::Standalone {
                name: format!("t-{i}"),
                conversation_id: "c".into(),
            },
            format!("t-{i}"),
            |_ctx| async move {
                std::future::pending::<()>().await;
                Ok(())
            },
        );
    }

    let (addr, server) = start_server(handler).await;
    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, Some("test-jwt")))
        .await
        .unwrap();

    let req = WsRequest {
        id: "list-1".into(),
        command: api::Command::ListBackgroundTasks {
            include_finished: false,
            limit: None,
        },
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();

    let msg = timeout(Duration::from_secs(2), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let frame: WsFrame = serde_json::from_str(&msg.into_text().unwrap()).unwrap();
    match frame {
        WsFrame::Result {
            id,
            result: api::CommandResult::BackgroundTasks(tasks),
        } => {
            assert_eq!(id, "list-1");
            assert_eq!(tasks.len(), 2);
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    server.abort();
}

/// Acceptance: subscribe over WS, then trigger a registry spawn — the
/// connection observes `TaskStarted` (and eventually `TaskCompleted`)
/// for the new task. This is the end-to-end equivalent of the
/// dispatcher-level test in `transport-dispatch`.
#[tokio::test]
async fn ws_subscribe_streams_task_events() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(RegistryHandler {
        registry: Arc::clone(&registry),
    });
    let (addr, server) = start_server(handler).await;
    let url = format!("ws://{}/ws", addr);
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, Some("test-jwt")))
        .await
        .unwrap();

    // Subscribe.
    let sub = WsRequest {
        id: "sub-1".into(),
        command: api::Command::SubscribeBackgroundTasks,
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&sub).unwrap().into(),
    ))
    .await
    .unwrap();
    let ack = timeout(Duration::from_secs(2), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let frame: WsFrame = serde_json::from_str(&ack.into_text().unwrap()).unwrap();
    assert!(matches!(
        frame,
        WsFrame::Result {
            result: api::CommandResult::Ack,
            ..
        }
    ));

    // Trigger a spawn on the SAME user (the WS validator pins alice).
    let _id = registry.spawn(
        UserId::new("alice"),
        api::TaskKind::Standalone {
            name: "spawned-after-subscribe".into(),
            conversation_id: "c".into(),
        },
        "spawned-after-subscribe".into(),
        |_ctx| async move { Ok(()) },
    );

    // Look for a TaskStarted event in the next few frames.
    let mut saw_started = false;
    for _ in 0..10 {
        let next = timeout(Duration::from_millis(500), ws.next()).await;
        let Ok(Some(Ok(msg))) = next else {
            continue;
        };
        let frame: WsFrame = match msg {
            tokio_tungstenite::tungstenite::Message::Text(t) => serde_json::from_str(&t).unwrap(),
            _ => continue,
        };
        if let WsFrame::Event {
            event: api::Event::TaskStarted { task },
        } = frame
        {
            assert_eq!(task.title, "spawned-after-subscribe");
            saw_started = true;
            break;
        }
    }
    assert!(saw_started, "expected TaskStarted event on the connection");

    server.abort();
}
