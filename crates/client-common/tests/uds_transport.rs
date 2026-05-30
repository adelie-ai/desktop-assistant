//! Integration tests for the Unix-domain-socket transport (#159).
//!
//! These spin up the real `desktop-assistant-uds` server in-process with a
//! tiny `AssistantApiHandler` double, mint a JWT with `auth-jwt`, and then
//! connect through `client-common`'s public `connect_transport` over the new
//! `TransportMode::Uds`. The point is to exercise the actual client framing +
//! handshake against the actual server, not a hand-rolled stand-in.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_client_common::{
    AssistantClient, ConnectionConfig, TransportMode, connect_transport,
};
use desktop_assistant_uds::{UdsAuthValidator, UdsServer, UdsServerConfig};
use tempfile::TempDir;
use tokio::net::UnixStream;
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

/// Minimal handler: enough command coverage to round-trip two distinct
/// request/response shapes through the transport.
struct TestHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for TestHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::ListConversations { .. } => {
                Ok(api::CommandResult::Conversations(Vec::new()))
            }
            api::Command::CreateConversation { title } => Ok(api::CommandResult::ConversationId {
                id: format!("id-for-{title}"),
            }),
            _ => Err(ApiError::Unsupported),
        }
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        Ok(())
    }
}

/// JWT auth validator backed by an in-memory signing key.
struct StaticJwtAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }
}

fn start_server(
    socket_path: PathBuf,
    signing_key: String,
) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(TestHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
    let config = UdsServerConfig::new(socket_path);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, config);
    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    tx
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

fn uds_config(socket_path: PathBuf, jwt: String) -> ConnectionConfig {
    ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(socket_path),
        ws_jwt: Some(jwt),
        ..ConnectionConfig::default()
    }
}

#[tokio::test]
async fn uds_transport_round_trips_commands() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let config = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let (client, _signals) = connect_transport(&config)
        .await
        .expect("connect over uds");

    // Empty-list round-trip: confirms request/response correlation + framing.
    let conversations = timeout(Duration::from_secs(2), client.list_conversations())
        .await
        .expect("no response within 2s")
        .expect("list_conversations over uds");
    assert!(conversations.is_empty());

    // Non-trivial value round-trip.
    let id = timeout(Duration::from_secs(2), client.create_conversation("hello"))
        .await
        .expect("no response within 2s")
        .expect("create_conversation over uds");
    assert_eq!(id, "id-for-hello");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_transport_rejects_invalid_jwt() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key);
    wait_for_socket(&path).await;

    // A token the server's validator will reject. The handshake is written at
    // connect time; the rejection surfaces as a failed command (the server
    // writes an error frame and closes).
    let config = uds_config(path, "not-a-real-jwt".to_string());
    let (client, _signals) = connect_transport(&config)
        .await
        .expect("socket connect itself succeeds");

    let result = timeout(Duration::from_secs(2), client.list_conversations())
        .await
        .expect("auth rejection should resolve the call, not hang");
    let err = result.expect_err("invalid jwt must fail the command");
    assert!(
        err.to_string().to_lowercase().contains("auth"),
        "expected an auth error, got: {err}"
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_transport_connect_fails_when_socket_missing() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("does-not-exist.sock");
    let config = uds_config(missing, "irrelevant".to_string());

    let result = connect_transport(&config).await;
    assert!(
        result.is_err(),
        "connecting to a missing socket path must error"
    );
}
