//! Regression test for voice#49: streamed response events must reach a UDS
//! client AND be correlatable to the send.
//!
//! The bug: over the socket transports (UDS / WS) the dispatcher generates a
//! turn `request_id` (which every streamed `AssistantDelta` / `AssistantCompleted`
//! event is stamped with) that was *different* from the id the `SendMessage`
//! ack returned (the registry `task_id`). A streaming client (voice) filters
//! response events by the id its `send_prompt` returned, so with mismatched ids
//! every event was dropped and the turn hung in `Processing`. The D-Bus path
//! worked because its `SendPrompt` reply returns the same id its signals carry.
//!
//! The fix makes `SendMessageAck` carry the turn `request_id`, and the client's
//! `send_prompt*` returns it — so `returned_id == streamed_event.request_id`.
//!
//! These tests spin up the real `desktop-assistant-uds` server in-process with
//! a handler that mimics the production registry path (returns `Some(task_id)`,
//! streams events from a background task through the per-connection sink), then
//! connect through `client-common`'s public `connect_transport`.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_client_common::{
    AssistantClient, ConnectionConfig, Connector, SignalEvent, TransportMode, connect_transport,
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

/// Mimics the production registry path: returns `Some(task_id)` so the
/// dispatcher replies with `SendMessageAck`, then streams events from a
/// background task through the per-connection sink — stamping each event with
/// the dispatcher-supplied `request_id` (NOT the task_id), exactly as the real
/// `run_send_turn` does.
struct RegistryStreamingHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for RegistryStreamingHandler {
    async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
        Ok(api::CommandResult::Ack)
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

    async fn start_send_message(
        &self,
        conversation_id: String,
        _content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        request_id: String,
        _idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        // The task_id is deliberately DIFFERENT from the request_id, just like
        // production — this is what regresses voice#49 if the ack returned the
        // task_id instead of the request_id.
        let task_id = api::TaskId(uuid::Uuid::new_v4().to_string());
        tokio::spawn(async move {
            sink.emit(api::Event::AssistantDelta {
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                chunk: "hel".into(),
            })
            .await;
            sink.emit(api::Event::AssistantDelta {
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                chunk: "lo".into(),
            })
            .await;
            sink.emit(api::Event::AssistantCompleted {
                conversation_id,
                request_id,
                full_response: "hello".into(),
            })
            .await;
        });
        Ok(Some(task_id))
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

fn start_server(socket_path: PathBuf, signing_key: String) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(RegistryStreamingHandler);
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

/// The core voice#49 assertion at the raw-transport level: streamed response
/// events reach a UDS client, AND the id `send_prompt` returns is the SAME id
/// those events carry (so a client filtering by that id sees its response).
#[tokio::test]
async fn uds_streamed_response_is_correlatable_to_the_send() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let (client, mut signals, _drop) = connect_transport(&cfg).await.expect("connect over uds");

    let returned_id = timeout(Duration::from_secs(2), client.send_prompt("conv-1", "hi"))
        .await
        .expect("send_prompt should ack")
        .expect("ack ok");
    assert!(
        !returned_id.is_empty(),
        "send_prompt must return the turn request_id, not an empty string"
    );

    // Read events; every response event's request_id MUST equal the id the
    // send returned — otherwise a client filtering by it (voice) drops them.
    let mut chunks = String::new();
    let mut got_complete = false;
    for _ in 0..10 {
        match timeout(Duration::from_secs(2), signals.recv()).await {
            Ok(Some(SignalEvent::Chunk {
                request_id, chunk, ..
            })) => {
                assert_eq!(
                    request_id, returned_id,
                    "chunk request_id must match the id send_prompt returned (voice#49)"
                );
                chunks.push_str(&chunk);
            }
            Ok(Some(SignalEvent::Complete {
                request_id,
                full_response,
                ..
            })) => {
                assert_eq!(
                    request_id, returned_id,
                    "complete request_id must match the id send_prompt returned (voice#49)"
                );
                assert_eq!(full_response, "hello");
                got_complete = true;
                break;
            }
            Ok(Some(_other)) => {}
            Ok(None) => panic!("signal stream closed before completion"),
            Err(_) => panic!("timed out waiting for a response event (chunks so far: {chunks:?})"),
        }
    }
    assert_eq!(chunks, "hello", "expected streamed chunks");
    assert!(got_complete, "expected the Complete event");

    let _ = shutdown.send(());
}

/// Same assertion exercised through the high-level [`Connector`] — the exact
/// path the voice service uses (subscribe before send, then read the fanned-out
/// signal stream and filter by the returned id).
#[tokio::test]
async fn connector_over_uds_delivers_correlated_response() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let connector = Connector::connect(&cfg).await.expect("connector over uds");

    // Subscribe BEFORE sending (the voice pipeline's ordering) so no early
    // chunk is lost.
    let mut events = connector.subscribe();

    let returned_id = timeout(
        Duration::from_secs(2),
        connector.send_prompt("conv-1", "hi"),
    )
    .await
    .expect("send_prompt should ack")
    .expect("ack ok");

    let mut chunks = String::new();
    let mut got_complete = false;
    for _ in 0..10 {
        match timeout(Duration::from_secs(2), events.recv()).await {
            Ok(Some(SignalEvent::Chunk {
                request_id, chunk, ..
            })) if request_id == returned_id => chunks.push_str(&chunk),
            Ok(Some(SignalEvent::Complete {
                request_id,
                full_response,
                ..
            })) if request_id == returned_id => {
                assert_eq!(full_response, "hello");
                got_complete = true;
                break;
            }
            // Events for a different request id would be dropped by a real
            // client's filter — the regression we're guarding against.
            Ok(Some(_)) => {}
            Ok(None) => panic!("connector signal stream closed before completion"),
            Err(_) => panic!("timed out waiting for a correlated response event"),
        }
    }
    assert_eq!(
        chunks, "hello",
        "connector must deliver the streamed chunks"
    );
    assert!(got_complete, "connector must deliver the Complete event");

    let _ = shutdown.send(());
}
