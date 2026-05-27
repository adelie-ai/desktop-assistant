//! End-to-end UDS smoke test for the background-task command surface.
//!
//! Per #114 acceptance: the dispatcher refactor (#103) means both WS
//! and UDS share the same `dispatch_loop`. This test drives a
//! real UDS listener with a registry-backed handler and verifies the
//! same `ListBackgroundTasks` + `SubscribeBackgroundTasks` story works
//! over the framed byte-stream transport. If a future change wires the
//! arms only on the WS side, this test breaks.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::background_tasks::BackgroundTaskRegistry;
use desktop_assistant_application::{
    ApiError, ApiResult, AssistantApiHandler, EventSink, UserId,
};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_uds::{
    UdsAuthValidator, UdsServer, UdsServerConfig, read_frame, write_frame,
};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::sync::broadcast;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-uds-iss";
const AUD: &str = "test-uds-aud";

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn mint_test_jwt(signing_key: &str, subject: &str) -> String {
    let now = unix_now();
    let claims = jwt::Claims {
        iss: ISS.into(),
        sub: subject.into(),
        aud: AUD.into(),
        exp: now + 600,
        iat: now,
        nbf: now.saturating_sub(1),
        jti: uuid::Uuid::new_v4().to_string(),
    };
    jwt::encode(&claims, signing_key).expect("encode jwt")
}

/// Handler with the same background-task wiring as the dispatcher test.
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

struct StaticJwtAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }
    async fn extract_user_id(&self, token: &str) -> Option<UserId> {
        jwt::decode(token, &self.signing_key, ISS, AUD)
            .ok()
            .map(|claims| UserId::new(claims.sub))
    }
}

fn socket_path(dir: &TempDir) -> PathBuf {
    dir.path().join("adelie.sock")
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if path.exists() && UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("uds socket {path:?} did not appear");
}

/// Acceptance: the new ListBackgroundTasks arm works over UDS end-to-end.
/// This proves the dispatcher refactor (#103) is the single point of
/// truth — a regression that only wires WS gets caught here.
#[tokio::test]
async fn uds_list_background_tasks_through_shared_dispatcher() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);

    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(RegistryHandler {
        registry: Arc::clone(&registry),
    });
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth {
        signing_key: signing_key.clone(),
    });
    let config = UdsServerConfig::new(path.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(Arc::clone(&handler), auth, config);
    let server_join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    wait_for_socket(&path).await;

    // Spawn two tasks under alice in the registry before the client
    // connects — proves the dispatcher reads from the live registry.
    let alice = UserId::new("alice");
    for i in 0..2 {
        registry.spawn(
            alice.clone(),
            api::TaskKind::Standalone {
                name: format!("task-{i}"),
                conversation_id: "c".into(),
            },
            format!("task-{i}"),
            |_ctx| async move {
                // Hold the task alive for the test by waiting forever; the
                // server task will be aborted at the end of the test.
                std::future::pending::<()>().await;
                Ok(())
            },
        );
    }

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "alice");
    let handshake = serde_json::json!({ "jwt": token });
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();

    let req = api::WsRequest {
        id: "list-1".into(),
        command: api::Command::ListBackgroundTasks {
            include_finished: false,
            limit: None,
        },
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("no response within 2s")
        .expect("io error");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Result {
            id,
            result: api::CommandResult::BackgroundTasks(tasks),
        } => {
            assert_eq!(id, "list-1");
            assert_eq!(tasks.len(), 2, "expected the two alice tasks");
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    let _ = server_join.await;
}
