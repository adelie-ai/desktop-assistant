//! #231: the client half of the client-tool protocol must work end-to-end over
//! the live UDS transport — the same path the voice service uses.
//!
//! Three things are exercised against a real in-process `desktop-assistant-uds`
//! server reached through `client-common`'s public `Connector`:
//!
//! 1. `register_client_tools` sends `Command::RegisterClientTools` and returns
//!    the count the daemon acked (`CommandResult::ClientToolsRegistered`).
//! 2. A turn emits `Event::ClientToolCall` through the per-connection sink; the
//!    client receives it on the signal stream as `SignalEvent::ClientToolCall`
//!    (it used to be silently dropped — `=> None`).
//! 3. `submit_client_tool_result` sends `Command::ClientToolResult` back; the
//!    handler records it and acks.
//!
//! The handler is a recording double (no real LLM/turn machinery): it stands in
//! for the "server half" that already exists in the daemon, so this test pins
//! the wire shapes the client emits/consumes, not the daemon's resumption logic.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_client_common::{ConnectionConfig, Connector, SignalEvent, TransportMode};
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

/// Records the client-tool commands the client sends, and — for a turn — emits a
/// `ClientToolCall` event through the per-connection sink so the client receives
/// it on its signal stream.
#[derive(Default)]
struct ClientToolHandler {
    /// Captured `RegisterClientTools` payload(s).
    registered: Mutex<Vec<api::ClientToolRegistration>>,
    /// Captured `ClientToolResult` command(s).
    results: Mutex<Vec<api::Command>>,
}

#[async_trait::async_trait]
impl AssistantApiHandler for ClientToolHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::RegisterClientTools { tools } => {
                let count = tools.len() as u32;
                *self.registered.lock().unwrap() = tools;
                Ok(api::CommandResult::ClientToolsRegistered { count })
            }
            cmd @ api::Command::ClientToolResult { .. } => {
                self.results.lock().unwrap().push(cmd);
                Ok(api::CommandResult::Ack)
            }
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

    /// Use the registry path so the dispatcher hands us the per-connection sink
    /// (like the real turn body), through which we emit a single
    /// `ClientToolCall` event for the client to react to.
    #[allow(clippy::too_many_arguments)]
    async fn start_send_message(
        &self,
        conversation_id: String,
        _content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        _request_id: String,
        _idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        let task_id = api::TaskId(uuid::Uuid::new_v4().to_string());
        let emitted_task_id = task_id.clone();
        tokio::spawn(async move {
            sink.emit(api::Event::ClientToolCall {
                task_id: emitted_task_id,
                conversation_id,
                tool_call_id: "call-1".into(),
                tool_name: "weather".into(),
                arguments: serde_json::json!({ "city": "Boston" }),
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

fn start_server(
    socket_path: PathBuf,
    signing_key: String,
    handler: Arc<ClientToolHandler>,
) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn AssistantApiHandler> = handler;
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

/// Full client-tool round trip over UDS: register, receive the tool-call event,
/// post the result back.
#[tokio::test]
async fn uds_client_tools_round_trip_register_call_result() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let handler = Arc::new(ClientToolHandler::default());
    let shutdown = start_server(path.clone(), signing_key.clone(), Arc::clone(&handler));
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let connector = Connector::connect(&cfg).await.expect("connector over uds");

    // Subscribe BEFORE sending so the emitted ClientToolCall isn't lost.
    let mut events = connector.subscribe();

    // 1. Register two client tools and confirm the daemon-acked count.
    let tools = vec![
        api::ClientToolRegistration {
            name: "weather".into(),
            description: "look up the weather".into(),
            input_schema: serde_json::json!({ "type": "object" }),
        },
        api::ClientToolRegistration {
            name: "calendar".into(),
            description: String::new(),
            input_schema: serde_json::Value::Null,
        },
    ];
    let count = timeout(
        Duration::from_secs(2),
        connector.register_client_tools(tools.clone()),
    )
    .await
    .expect("register_client_tools should resolve")
    .expect("registration ok");
    assert_eq!(count, 2, "daemon must ack the registered tool count");
    assert_eq!(
        *handler.registered.lock().unwrap(),
        tools,
        "the handler must receive the full RegisterClientTools payload"
    );

    // 2. Kick off a turn; the handler emits a ClientToolCall through the sink.
    let _request_id = timeout(
        Duration::from_secs(2),
        connector.send_prompt("conv-1", "hi"),
    )
    .await
    .expect("send_prompt should ack")
    .expect("ack ok");

    // The client must SEE the tool call on its signal stream (it used to be
    // dropped). Skip any unrelated events (e.g. a status heartbeat).
    let (task_id, tool_call_id, tool_name, arguments) = loop {
        match timeout(Duration::from_secs(2), events.recv()).await {
            Ok(Some(SignalEvent::ClientToolCall {
                task_id,
                conversation_id,
                tool_call_id,
                tool_name,
                arguments,
            })) => {
                assert_eq!(conversation_id, "conv-1");
                break (task_id, tool_call_id, tool_name, arguments);
            }
            Ok(Some(_other)) => continue,
            Ok(None) => panic!("signal stream closed before the tool call arrived"),
            Err(_) => panic!("timed out waiting for SignalEvent::ClientToolCall"),
        }
    };
    assert_eq!(tool_call_id, "call-1");
    assert_eq!(tool_name, "weather");
    assert_eq!(arguments, serde_json::json!({ "city": "Boston" }));
    assert!(!task_id.is_empty(), "task_id must be carried through");

    // 3. Post the tool result back; the handler records the ClientToolResult.
    timeout(
        Duration::from_secs(2),
        connector.submit_client_tool_result(&task_id, &tool_call_id, Ok("sunny, 72F".into())),
    )
    .await
    .expect("submit_client_tool_result should resolve")
    .expect("result ack ok");

    let results = handler.results.lock().unwrap().clone();
    assert_eq!(results.len(), 1, "exactly one ClientToolResult expected");
    match &results[0] {
        api::Command::ClientToolResult {
            task_id: got_task,
            tool_call_id: got_call,
            result,
            error,
        } => {
            assert_eq!(got_task.0, task_id);
            assert_eq!(got_call, "call-1");
            assert_eq!(result.as_deref(), Some("sunny, 72F"));
            assert!(error.is_none());
        }
        other => panic!("expected ClientToolResult, got {other:?}"),
    }

    let _ = shutdown.send(());
}
