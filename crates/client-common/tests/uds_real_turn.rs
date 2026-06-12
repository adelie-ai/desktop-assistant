//! Real-path regression test for voice#49 (reopened): a `SendMessage` that
//! arrives over UDS must stream ALL its events back to the *same* UDS
//! connection.
//!
//! ## Why this test exists (the gap #218's test had)
//!
//! `tests/uds_streaming.rs` (the #218 regression test) used a hand-written
//! `start_send_message` that emitted directly to the per-connection sink. It
//! proved the *correlation* fix (ack `request_id` == streamed events'
//! `request_id`) but it never drove the REAL application turn pipeline:
//! `DefaultAssistantApiHandler::start_send_message` → registry spawn → the
//! in-flight/TeeSink path (taken whenever the send carries an
//! `idempotency_key`) → `run_send_turn` → the per-connection sink.
//!
//! Crucially, the voice client sends with an **idempotency key** on every turn
//! (`assistant-connector`'s `send_prompt_with_system_refinement`), which routes
//! the real handler through the keyed branch that #218 never exercised. This
//! test drives a SendMessage with a key through the *real* handler (wired like
//! the daemon: registry + client-tool coordinator) standing behind the *real*
//! UDS server, and asserts the connecting client receives every streamed event
//! correlated to the id `send_prompt` returned.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_application::{
    DefaultAssistantApiHandler,
    background_tasks::BackgroundTaskRegistry,
    client_tools::{ClientToolCoordinator, InMemoryTurnStateStore},
};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_client_common::{ConnectionConfig, Connector, SignalEvent, TransportMode};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, KnowledgeEntry, Message, Role,
};
use desktop_assistant_core::ports::inbound::{
    AssistantService, BackendTasksSettingsView, ConnectionConfigPayload, ConnectionView,
    ConnectionsService, ConnectorDefaultsView, ConversationService, DatabaseSettingsView,
    EmbeddingsSettingsView, KnowledgeService, LlmSettingsView, McpServerView, ModelListing,
    PersistenceSettingsView, PurposeConfigPayload, PurposeKind, PurposesView, SettingsService,
    WsAuthSettingsView,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
use desktop_assistant_uds::{UdsAuthValidator, UdsServer, UdsServerConfig};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-real-iss";
const AUD: &str = "test-real-aud";

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

// ---------------------------------------------------------------------------
// Minimal real services. The send-turn path only touches the conversation
// service; the other four exist solely to satisfy the handler's generics.
// ---------------------------------------------------------------------------

/// A conversation service whose `send_prompt` streams two chunks plus a status
/// message, then returns a final response — exactly the shape `run_send_turn`
/// bridges into `AssistantStatus` / `AssistantDelta` / `AssistantCompleted`.
struct StreamingConversations;

#[async_trait::async_trait]
impl ConversationService for StreamingConversations {
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        Ok(Conversation::new("conv-1", title))
    }
    async fn list_conversations(
        &self,
        _max_age_days: Option<u32>,
        _include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(vec![])
    }
    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let mut c = Conversation::new(id.as_str(), "Real Turn");
        c.messages.push(Message::new(Role::User, "hi"));
        Ok(c)
    }
    async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn rename_conversation(
        &self,
        _id: &ConversationId,
        _title: String,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
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
        mut on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        // A status update (turn-start narration / heartbeat), then streamed
        // chunks, then the final reply — mirroring a real LLM turn.
        on_status("thinking".into());
        on_chunk("hel".into());
        on_chunk("lo".into());
        Ok("hello".into())
    }
}

struct FakeAssistant;
impl AssistantService for FakeAssistant {
    fn version(&self) -> &str {
        "0.0.0-test"
    }
    fn ping(&self) -> &str {
        "pong"
    }
}

struct FakeConnections;
impl ConnectionsService for FakeConnections {
    async fn list_connections(&self) -> Result<Vec<ConnectionView>, CoreError> {
        Ok(vec![])
    }
    async fn create_connection(
        &self,
        _id: String,
        _config: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn update_connection(
        &self,
        _id: String,
        _config: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn delete_connection(&self, _id: String, _force: bool) -> Result<(), CoreError> {
        Ok(())
    }
    async fn list_available_models(
        &self,
        _connection_id: Option<String>,
        _refresh: bool,
    ) -> Result<Vec<ModelListing>, CoreError> {
        Ok(vec![])
    }
    async fn get_purposes(&self) -> Result<PurposesView, CoreError> {
        Ok(PurposesView::default())
    }
    async fn set_purpose(
        &self,
        _purpose: PurposeKind,
        _config: PurposeConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
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
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
        })
    }
    async fn set_llm_settings(
        &self,
        _connector: String,
        _model: Option<String>,
        _base_url: Option<String>,
        _temperature: Option<f64>,
        _top_p: Option<f64>,
        _max_tokens: Option<u32>,
        _hosted_tool_search: Option<bool>,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
        Ok(())
    }
    async fn generate_ws_jwt(&self, _subject: Option<String>) -> Result<String, CoreError> {
        Ok("jwt".into())
    }
    async fn validate_ws_jwt(&self, _token: String) -> Result<bool, CoreError> {
        Ok(true)
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
            backend_llm_model: "bm".into(),
            embeddings_model: "em".into(),
            embeddings_base_url: "eu".into(),
            embeddings_available: false,
            hosted_tool_search_available: false,
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
    async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
        Ok(DatabaseSettingsView {
            url: String::new(),
            max_connections: 5,
        })
    }
    async fn set_database_settings(
        &self,
        _url: Option<String>,
        _max_connections: u32,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
        Ok(BackendTasksSettingsView {
            has_separate_llm: false,
            llm_connector: "openai".into(),
            llm_model: "gpt-5".into(),
            llm_base_url: "https://api.openai.com/v1".into(),
            dreaming_enabled: false,
            dreaming_interval_secs: 3600,
            archive_after_days: 0,
        })
    }
    async fn set_backend_tasks_settings(
        &self,
        _llm_connector: Option<String>,
        _llm_model: Option<String>,
        _llm_base_url: Option<String>,
        _dreaming_enabled: bool,
        _dreaming_interval_secs: u64,
        _archive_after_days: u32,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn list_mcp_servers(&self) -> Result<Vec<McpServerView>, CoreError> {
        Ok(vec![])
    }
    async fn add_mcp_server(
        &self,
        _name: String,
        _command: String,
        _args: Vec<String>,
        _namespace: Option<String>,
        _enabled: bool,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
        Ok(())
    }
    async fn set_mcp_server_enabled(&self, _name: String, _enabled: bool) -> Result<(), CoreError> {
        Ok(())
    }
    async fn mcp_server_action(
        &self,
        _action: String,
        _server: Option<String>,
    ) -> Result<Vec<McpServerView>, CoreError> {
        Ok(vec![])
    }
    async fn get_ws_auth_settings(&self) -> Result<WsAuthSettingsView, CoreError> {
        Ok(WsAuthSettingsView {
            methods: vec![],
            oidc_issuer: String::new(),
            oidc_auth_endpoint: String::new(),
            oidc_token_endpoint: String::new(),
            oidc_client_id: String::new(),
            oidc_scopes: String::new(),
        })
    }
    async fn set_ws_auth_settings(
        &self,
        _methods: Vec<String>,
        _oidc_issuer: String,
        _oidc_auth_endpoint: String,
        _oidc_token_endpoint: String,
        _oidc_client_id: String,
        _oidc_scopes: String,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}

struct FakeKnowledge;
impl KnowledgeService for FakeKnowledge {
    async fn list_entries(
        &self,
        _limit: usize,
        _offset: usize,
        _tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Ok(vec![])
    }
    async fn get_entry(&self, _id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
        Ok(None)
    }
    async fn search_entries(
        &self,
        _query: String,
        _tag_filter: Option<Vec<String>>,
        _limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Ok(vec![])
    }
    async fn create_entry(
        &self,
        content: String,
        tags: Vec<String>,
        _metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        Ok(KnowledgeEntry::new("kb-test", content, tags))
    }
    async fn update_entry(
        &self,
        _id: String,
        content: String,
        tags: Vec<String>,
        _metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        Ok(KnowledgeEntry::new("kb-test", content, tags))
    }
    async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
        Ok(())
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

/// Build the *real* `DefaultAssistantApiHandler` wired exactly like the daemon:
/// a `BackgroundTaskRegistry` (so `start_send_message` takes the registry path)
/// plus the client-tool coordinator (#234 — installed for every live turn).
fn real_handler() -> Arc<dyn desktop_assistant_application::AssistantApiHandler> {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn desktop_assistant_core::ports::store::TurnStateStore> =
        Arc::new(InMemoryTurnStateStore::new());
    let handler = DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(StreamingConversations),
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    )
    .with_registry(registry)
    .with_client_tool_coordinator(coord, store);
    Arc::new(handler)
}

fn start_server(socket_path: PathBuf, signing_key: String) -> tokio::sync::oneshot::Sender<()> {
    let handler = real_handler();
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

// ---------------------------------------------------------------------------
// #246 reconnect harness: a server whose entire runtime can be dropped to
// simulate a daemon *restart* (which closes every live connection — unlike a
// graceful shutdown, whose spawned per-connection tasks would linger), then a
// fresh server stood up on the same socket path.
// ---------------------------------------------------------------------------

/// Like [`real_handler`] but bound to a caller-supplied coordinator, so a test
/// can inspect the tools registered against the *new* daemon after reconnect.
fn real_handler_with_coord(
    coord: Arc<ClientToolCoordinator>,
) -> Arc<dyn desktop_assistant_application::AssistantApiHandler> {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let store: Arc<dyn desktop_assistant_core::ports::store::TurnStateStore> =
        Arc::new(InMemoryTurnStateStore::new());
    let handler = DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(StreamingConversations),
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    )
    .with_registry(registry)
    .with_client_tool_coordinator(coord, store);
    Arc::new(handler)
}

/// A server instance running on its own dedicated runtime thread. Dropping it
/// shuts the runtime down, aborting the accept loop **and** every spawned
/// per-connection task — so any connected client sees its socket close, exactly
/// like a daemon binary restart. The coordinator is shared so the test can
/// assert what the (post-restart) daemon has registered.
struct ServerInstance {
    runtime: Option<tokio::runtime::Runtime>,
    coord: Arc<ClientToolCoordinator>,
}

impl ServerInstance {
    fn start(socket_path: PathBuf, signing_key: String) -> Self {
        let coord = Arc::new(ClientToolCoordinator::new());
        let coord_for_server = Arc::clone(&coord);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build server runtime");
        runtime.spawn(async move {
            let handler = real_handler_with_coord(coord_for_server);
            let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
            let server = UdsServer::new(handler, auth, UdsServerConfig::new(socket_path));
            // Serve until the runtime is dropped (no graceful shutdown — a
            // restart is abrupt).
            let _ = server
                .serve_with_shutdown(std::future::pending::<()>())
                .await;
        });
        Self {
            runtime: Some(runtime),
            coord,
        }
    }
}

impl Drop for ServerInstance {
    fn drop(&mut self) {
        // Tear the runtime down without blocking the async test thread on its
        // worker join (which would deadlock): hand the shutdown to a scratch
        // thread.
        if let Some(rt) = self.runtime.take() {
            std::thread::spawn(move || drop(rt));
        }
    }
}

/// #246: after the daemon *restarts* (its runtime — and every live connection —
/// is dropped, then a fresh daemon binds the same socket), the Connector must
/// transparently reconnect: a `send_prompt` issued after the drop streams its
/// turn on the new connection, and the client tools registered before the drop
/// are **replayed** so the new daemon knows about them again.
#[tokio::test]
async fn connector_reconnects_and_replays_tools_after_daemon_restart() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");

    // Daemon #1.
    let server1 = ServerInstance::start(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path.clone(), mint_test_jwt(&signing_key, "dave"));
    // Short backoff is built into the connector; we drive the timing with a
    // generous wait loop below rather than tuning constants here.
    let connector = Connector::connect(&cfg).await.expect("connector over uds");

    // Register a client tool against daemon #1 (voice's stop_listening shape).
    let tool = desktop_assistant_api_model::ClientToolRegistration {
        name: "stop_listening".into(),
        description: "stop the microphone".into(),
        input_schema: serde_json::json!({ "type": "object" }),
    };
    let count = connector
        .register_client_tools(vec![tool.clone()])
        .await
        .expect("initial register");
    assert_eq!(count, 1, "daemon #1 accepted the tool");

    // A turn works on the original connection.
    drive_one_turn(&connector).await;

    // --- Daemon restart: drop #1 (closing the connection), bind #2. ---
    drop(server1);
    let server2 = ServerInstance::start(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    // The fresh daemon starts with NO client tools — until the Connector
    // replays the registration. Poll until the reconnect+replay lands.
    let replayed = wait_until(Duration::from_secs(10), || {
        let coord = Arc::clone(&server2.coord);
        // The replay lands under the reconnected connection's server-minted
        // `(user, session_id)` bucket (#261), which the test can't name from
        // out here; assert across all sessions instead.
        async move { coord.is_registered_in_any_session("stop_listening").await }
    })
    .await;
    assert!(
        replayed,
        "the Connector must replay the client-tool registration to the restarted daemon (#246)"
    );

    // And a brand-new turn must stream on the reconnected transport. The send
    // itself may transiently fail while the socket is between connections, so
    // retry briefly until the reconnected transport accepts it.
    drive_one_turn_with_retry(&connector, Duration::from_secs(10)).await;

    drop(server2);
}

/// Drive a single SendMessage turn and assert it streams to completion on the
/// connector's current connection.
async fn drive_one_turn(connector: &Connector) {
    let mut events = connector.subscribe();
    let returned_id = timeout(
        Duration::from_secs(5),
        connector.send_prompt_with_system_refinement_idempotent(
            "conv-1",
            "hi",
            "",
            Some(uuid::Uuid::new_v4().to_string()),
        ),
    )
    .await
    .expect("send acks")
    .expect("ack ok");
    assert!(
        collect_turn(&mut events, &returned_id).await,
        "turn completed"
    );
}

/// Like [`drive_one_turn`] but tolerant of the brief window right after a drop
/// where the send may hit the dead socket before the reconnect lands: retries
/// the send until it acks (or the deadline), then asserts the turn streams.
async fn drive_one_turn_with_retry(connector: &Connector, within: Duration) {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let mut events = connector.subscribe();
        let send = connector.send_prompt_with_system_refinement_idempotent(
            "conv-1",
            "hi again",
            "",
            Some(uuid::Uuid::new_v4().to_string()),
        );
        match timeout(Duration::from_secs(5), send).await {
            Ok(Ok(returned_id)) => {
                if collect_turn(&mut events, &returned_id).await {
                    return;
                }
            }
            _ => {
                if tokio::time::Instant::now() >= deadline {
                    panic!("send never succeeded on the reconnected transport within {within:?}");
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("turn never completed on the reconnected transport within {within:?}");
        }
    }
}

/// Read from `events` until the turn for `request_id` completes. Returns `false`
/// if the connection drops or stalls first (so a caller can retry).
async fn collect_turn(
    events: &mut tokio::sync::mpsc::UnboundedReceiver<SignalEvent>,
    request_id: &str,
) -> bool {
    for _ in 0..30 {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Some(SignalEvent::Complete {
                request_id: rid,
                full_response,
            })) if rid == request_id => {
                return full_response == "hello";
            }
            Ok(Some(SignalEvent::Disconnected { .. })) => return false,
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return false,
        }
    }
    false
}

/// Poll `cond` until it returns true or the deadline elapses.
async fn wait_until<F, Fut>(within: Duration, mut cond: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + within;
    while tokio::time::Instant::now() < deadline {
        if cond().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

/// #246 invariant guard: a per-turn **stall** (a waiter on an open-but-silent
/// connection) must surface the stall `Disconnected` to the waiter (#221)
/// WITHOUT tearing the transport down or reconnecting. A reconnect would emit a
/// *second*, "signal stream closed" Disconnected and would replay the tool
/// registration; neither must happen. After the stall, the SAME connection must
/// still stream the next turn.
#[tokio::test]
async fn per_turn_stall_does_not_trigger_a_reconnect() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let server = ServerInstance::start(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path.clone(), mint_test_jwt(&signing_key, "dave"));
    // A short stall window. The server stays up the whole time; the connection
    // is open and healthy — it just has a waiter with no in-flight events.
    let stall = Duration::from_millis(200);
    let connector = Connector::connect_with_stall_timeout(&cfg, stall)
        .await
        .expect("connector over uds");

    // A waiter on a silent connection (subscribe, but send nothing) trips the
    // per-turn stall. It must get a "stalled" Disconnected, NOT a close.
    let mut waiter = connector.subscribe();
    let event = timeout(stall * 5, waiter.recv())
        .await
        .expect("the stall must unstick the waiter")
        .expect("a terminal event");
    match event {
        SignalEvent::Disconnected { reason } => assert!(
            reason.contains("stalled"),
            "a per-turn stall must surface a 'stalled' Disconnected, not a close: {reason}"
        ),
        other => panic!("expected a stall Disconnected, got {other:?}"),
    }

    // No reconnect happened, so there must be NO follow-up "signal stream
    // closed" Disconnected on a fresh subscriber, and the SAME connection must
    // still stream the next turn (the transport was never torn down).
    let mut probe = connector.subscribe();
    // Give any (erroneous) reconnect a window to fire its close-Disconnected.
    if let Ok(Some(SignalEvent::Disconnected { reason })) = timeout(stall, probe.recv()).await {
        panic!("a stall must NOT trigger a transport close/reconnect, but got: {reason}");
    }
    drop(probe);

    drive_one_turn(&connector).await;

    drop(server);
}

/// #246 invariant guard: an **idle** Connector (connected, but no turns, no
/// traffic) must NOT spuriously reconnect or tear its transport down — even
/// well past the event-stall window. After sitting idle, a turn must still
/// stream on the *original* connection (no reconnect happened).
#[tokio::test]
async fn idle_connector_does_not_spuriously_reconnect() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let server = ServerInstance::start(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path.clone(), mint_test_jwt(&signing_key, "dave"));
    // A short stall window; the connection is healthy and open the whole time.
    let stall = Duration::from_millis(150);
    let connector = Connector::connect_with_stall_timeout(&cfg, stall)
        .await
        .expect("connector over uds");

    // Register a tool, then sit idle WELL past the stall window. If an idle
    // connection wrongly reconnected, the replay would re-run; more importantly,
    // a spurious teardown would break the next turn.
    connector
        .register_client_tools(vec![desktop_assistant_api_model::ClientToolRegistration {
            name: "noop".into(),
            description: String::new(),
            input_schema: serde_json::Value::Null,
        }])
        .await
        .expect("register");
    tokio::time::sleep(stall * 5).await;

    // The tool is still registered on the SAME daemon (no restart happened);
    // and a turn still streams on the original, never-dropped connection.
    assert!(
        // Registered under the connection's server-minted `(user, session_id)`
        // bucket (#261); the test can't name that session from out here, so
        // assert across all sessions.
        server.coord.is_registered_in_any_session("noop").await,
        "an idle connection must not have dropped/reconnected (tool would survive either way, \
         but a teardown would break the turn below)"
    );
    drive_one_turn(&connector).await;

    drop(server);
}

/// The real-path voice#49 assertion: a SendMessage carrying an idempotency key
/// (the voice client's exact behaviour) driven through the REAL handler must
/// stream Status + Delta + Completed events back to the connecting UDS client,
/// each correlated to the id `send_prompt` returned.
#[tokio::test]
async fn keyed_send_over_uds_streams_real_turn_events() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let connector = Connector::connect(&cfg).await.expect("connector over uds");

    // Subscribe BEFORE sending (the voice pipeline's ordering).
    let mut events = connector.subscribe();

    // Send WITH an idempotency key — this is what voice's
    // `send_prompt_with_system_refinement` does on every turn, and it routes
    // the real handler through the keyed in-flight/TeeSink branch that #218's
    // hand-emitting stub never touched.
    let returned_id = timeout(
        Duration::from_secs(5),
        connector.send_prompt_with_system_refinement_idempotent(
            "conv-1",
            "hi",
            "",
            Some(uuid::Uuid::new_v4().to_string()),
        ),
    )
    .await
    .expect("send should ack within the window")
    .expect("ack ok");
    assert!(
        !returned_id.is_empty(),
        "send must return the turn request_id"
    );

    let mut chunks = String::new();
    let mut got_status = false;
    let mut got_complete = false;
    for _ in 0..20 {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Some(SignalEvent::Status { request_id, .. })) if request_id == returned_id => {
                got_status = true;
            }
            Ok(Some(SignalEvent::Chunk { request_id, chunk })) if request_id == returned_id => {
                chunks.push_str(&chunk);
            }
            Ok(Some(SignalEvent::Complete {
                request_id,
                full_response,
            })) if request_id == returned_id => {
                assert_eq!(full_response, "hello");
                got_complete = true;
                break;
            }
            // An event correlated to a DIFFERENT id would be dropped by the
            // real client's filter — that is precisely the bug (zero events
            // reaching voice) we are guarding against.
            Ok(Some(SignalEvent::Disconnected { reason })) => {
                panic!("connection dropped before completion: {reason}")
            }
            Ok(Some(_other)) => {}
            Ok(None) => panic!("signal stream closed before completion"),
            Err(_) => panic!(
                "timed out waiting for a correlated response event \
                 (got status={got_status}, chunks so far={chunks:?})"
            ),
        }
    }

    assert!(
        got_status,
        "the turn's AssistantStatus must reach the UDS client"
    );
    assert_eq!(
        chunks, "hello",
        "the streamed chunks must reach the UDS client"
    );
    assert!(
        got_complete,
        "the AssistantCompleted must reach the UDS client"
    );

    let _ = shutdown.send(());
}

/// The reopened-voice#49 root cause: a Connector that sits IDLE past its
/// event-stall window — exactly what a voice service does between connecting at
/// startup and the first "Hey Adele" — must still deliver a turn's events.
///
/// Before the fix, the fan-out task tripped the stall on an idle-but-healthy
/// connection and EXITED permanently (UDS has no keepalive to reset the clock,
/// unlike WS), so the subscriber registered for the first turn was attached to a
/// dead pump and received ZERO events. The send still acked (the command
/// channel is independent), the turn completed server-side — but the response
/// stream never reached the client. This drives that exact sequence: connect,
/// stay idle past the stall window, THEN subscribe + send a real turn.
#[tokio::test]
async fn idle_past_stall_then_turn_still_streams_events() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let cfg = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    // A short stall window so the test doesn't wait the production 90s. The
    // connection is healthy and open the whole time — just idle (no turn yet).
    let stall = Duration::from_millis(150);
    let connector = Connector::connect_with_stall_timeout(&cfg, stall)
        .await
        .expect("connector over uds");

    // Sit idle WELL past the stall window, no subscribers, no events — the
    // between-connect-and-first-turn gap a voice service always has.
    tokio::time::sleep(stall * 4).await;

    // Now the first turn arrives. Subscribe, then send (voice's ordering).
    let mut events = connector.subscribe();
    let returned_id = timeout(
        Duration::from_secs(5),
        connector.send_prompt_with_system_refinement_idempotent(
            "conv-1",
            "hi",
            "",
            Some(uuid::Uuid::new_v4().to_string()),
        ),
    )
    .await
    .expect("send should ack after an idle period")
    .expect("ack ok");

    let mut chunks = String::new();
    let mut got_complete = false;
    for _ in 0..20 {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Some(SignalEvent::Chunk { request_id, chunk })) if request_id == returned_id => {
                chunks.push_str(&chunk);
            }
            Ok(Some(SignalEvent::Complete {
                request_id,
                full_response,
            })) if request_id == returned_id => {
                assert_eq!(full_response, "hello");
                got_complete = true;
                break;
            }
            Ok(Some(SignalEvent::Disconnected { reason })) => {
                panic!(
                    "fan-out died on an idle-but-healthy connection and tore down \
                     the first turn's stream (reopened voice#49): {reason}"
                )
            }
            Ok(Some(_other)) => {}
            Ok(None) => panic!("signal stream closed before completion"),
            Err(_) => panic!(
                "timed out waiting for the turn's events after an idle period \
                 — the fan-out stalled out the healthy connection (chunks={chunks:?})"
            ),
        }
    }

    assert_eq!(
        chunks, "hello",
        "a turn after an idle period must still stream its chunks"
    );
    assert!(
        got_complete,
        "a turn after an idle period must still complete"
    );

    let _ = shutdown.send(());
}
