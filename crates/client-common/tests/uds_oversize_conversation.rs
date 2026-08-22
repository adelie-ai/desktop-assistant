//! A conversation larger than one transport frame must still open, and must
//! not take the connection down with it (issue #1303).
//!
//! `GetConversation` used to map every message into the response with content
//! whole. Every transport caps one message at 4 MiB, nothing bounded the
//! response against that cap, and the client's reader loop broke on the
//! rejected frame rather than failing the one request. The conversation was
//! then permanently unopenable: each attempt repeated the disconnect.
//!
//! These tests stand the REAL `DefaultAssistantApiHandler` behind the REAL UDS
//! server and reach it with the REAL `client-common` client, so what they
//! prove is the path a desktop client actually takes.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_client_common::{
    AssistantClient, ConnectionConfig, TransportMode, connect_transport,
};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, KnowledgeEntry, Message, Role,
};
use desktop_assistant_core::ports::inbound::{
    AssistantService, BackendTasksSettingsView, ConnectionConfigPayload, ConnectionView,
    ConnectionsService, ConnectorDefaultsView, ConversationService, DatabaseSettingsView,
    EmbeddingsSettingsView, KnowledgeService, LlmSettingsView, McpServerView, ModelListing,
    PurposeConfigPayload, PurposeKind, PurposesView, SettingsService, WsAuthSettingsView,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
use desktop_assistant_uds::{UdsAuthValidator, UdsServer, UdsServerConfig};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-oversize-iss";
const AUD: &str = "test-oversize-aud";

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

/// A conversation service that answers with messages of a fixed size, so a
/// test can build a conversation past the transport frame cap.
struct BigConversations {
    count: usize,
    bytes_each: usize,
}

#[async_trait::async_trait]
impl ConversationService for BigConversations {
    async fn create_conversation(
        &self,
        title: String,
        _tags: Vec<String>,
    ) -> Result<Conversation, CoreError> {
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
        let mut c = Conversation::new(id.as_str(), "big");
        for i in 0..self.count {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            c.messages
                .push(Message::new(role, "x".repeat(self.bytes_each)));
        }
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
        _on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        on_chunk("ok".into());
        Ok("ok".into())
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
            health: Default::default(),
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
    async fn trash_count(&self) -> Result<usize, CoreError> {
        Ok(0)
    }
    async fn empty_trash(&self) -> Result<usize, CoreError> {
        Ok(0)
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

    /// Identity is part of acceptance (#807): a validator that accepts a token
    /// must name the subject it belongs to, or the connection is refused.
    async fn extract_user_id(&self, token: &str) -> Option<jwt::UserId> {
        jwt::decode(token, &self.signing_key, ISS, AUD)
            .ok()
            .map(|claims| jwt::UserId::new(claims.sub))
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
    conversations: BigConversations,
) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn desktop_assistant_application::AssistantApiHandler> =
        Arc::new(DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(conversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        ));
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

/// Stand up the server and connect the real client to it.
async fn connected(
    dir: &TempDir,
    conversations: BigConversations,
) -> (
    Box<dyn AssistantClient>,
    tokio::sync::oneshot::Sender<()>,
    Box<dyn std::any::Any + Send>,
) {
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone(), conversations);
    wait_for_socket(&path).await;
    let config = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let (client, _signals, guard) = connect_transport(&config).await.expect("connect over uds");
    (Box::new(client), shutdown, Box::new(guard))
}

/// Acceptance (#1303): a conversation whose messages exceed the frame cap
/// still opens. Both `GetConversation` and `GetMessages` come back with a
/// usable answer, and the connection is still serving afterwards.
#[tokio::test]
async fn a_conversation_past_the_frame_cap_still_opens_over_uds_and_keeps_the_connection() {
    let dir = TempDir::new().unwrap();
    // Eight messages of 512 KiB is 4 MiB of content, past the 4 MiB frame cap
    // once the JSON envelope and escaping are added.
    let (client, shutdown, _guard) = connected(
        &dir,
        BigConversations {
            count: 8,
            bytes_each: 512 * 1024,
        },
    )
    .await;

    let detail = timeout(Duration::from_secs(10), client.get_conversation("c1"))
        .await
        .expect("GetConversation must answer within 10s, not hang on a dead socket")
        .expect("GetConversation must succeed, not fail the request");
    assert!(
        !detail.messages.is_empty(),
        "a partial answer must still carry messages"
    );

    let window = timeout(
        Duration::from_secs(10),
        client.get_messages("c1", 0, -1, vec![]),
    )
    .await
    .expect("GetMessages must answer within 10s")
    .expect("GetMessages must succeed");
    assert!(
        !window.messages.is_empty(),
        "a partial window must still carry messages"
    );
    assert!(
        window.size_capped,
        "this conversation is past the budget, so the window must say it was cut"
    );

    // The connection must still be serving: a third request on the SAME
    // client proves the reader loop did not break.
    let conversations = timeout(Duration::from_secs(10), client.list_conversations())
        .await
        .expect("the connection must still serve after a partial answer")
        .expect("list_conversations must succeed");
    assert!(conversations.is_empty());

    let _ = shutdown.send(());
}

/// Acceptance (#1303): the single-oversize-message case on its own. One
/// message larger than the whole budget must still yield a usable response -
/// the conversation is never empty because one row was too large.
#[tokio::test]
async fn a_single_message_past_the_frame_cap_still_opens_over_uds() {
    let dir = TempDir::new().unwrap();
    let (client, shutdown, _guard) = connected(
        &dir,
        BigConversations {
            count: 1,
            bytes_each: 6 * 1024 * 1024,
        },
    )
    .await;

    let detail = timeout(Duration::from_secs(10), client.get_conversation("c1"))
        .await
        .expect("GetConversation must answer within 10s, not hang on a dead socket")
        .expect("GetConversation must succeed, not fail the request");
    assert_eq!(
        detail.messages.len(),
        1,
        "the one message must come back headed, not dropped"
    );
    assert!(
        !detail.messages[0].content.is_empty(),
        "a headed message must still carry content"
    );

    let window = timeout(
        Duration::from_secs(10),
        client.get_messages("c1", 0, -1, vec![]),
    )
    .await
    .expect("GetMessages must answer within 10s")
    .expect("GetMessages must succeed");
    assert_eq!(window.messages.len(), 1);
    assert_eq!(
        window.messages[0].content_total_bytes,
        Some(6 * 1024 * 1024),
        "a headed row must report the true size of what is stored"
    );

    let conversations = timeout(Duration::from_secs(10), client.list_conversations())
        .await
        .expect("the connection must still serve after a headed answer")
        .expect("list_conversations must succeed");
    assert!(conversations.is_empty());

    let _ = shutdown.send(());
}

/// A handler that answers `ListConversations` with a payload past the frame
/// cap, and everything else with a small one. The daemon bounds a real
/// response where it is built, so this stands in for the backstop under that:
/// a reply the transport cannot carry must fail the ONE request, never the
/// connection.
struct OversizeHandler;

#[async_trait::async_trait]
impl desktop_assistant_application::AssistantApiHandler for OversizeHandler {
    async fn handle_command(
        &self,
        cmd: desktop_assistant_api_model::Command,
    ) -> desktop_assistant_application::ApiResult<desktop_assistant_api_model::CommandResult> {
        use desktop_assistant_api_model as api;
        match cmd {
            api::Command::ListConversations { .. } => Ok(api::CommandResult::Conversations(
                (0..64)
                    .map(|i| api::ConversationSummary {
                        id: format!("c{i}"),
                        title: "t".repeat(128 * 1024),
                        message_count: 0,
                        updated_at: "2026-01-01 00:00:00".into(),
                        archived: false,
                        tags: vec![],
                        title_total_bytes: None,
                        omitted_trailing_conversations: 0,
                        omitted_tags: 0,
                    })
                    .collect(),
            )),
            _ => Ok(api::CommandResult::ConversationId { id: "small".into() }),
        }
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn desktop_assistant_application::EventSink>,
    ) -> desktop_assistant_application::ApiResult<()> {
        Ok(())
    }
}

/// Acceptance (#1303): over UDS, a reply the frame codec refuses becomes a
/// failure of the one request. The caller's call returns an error rather than
/// hanging or disconnecting, and the connection keeps serving - which is the
/// difference between one failed read and an unopenable conversation.
#[tokio::test]
async fn an_oversize_reply_over_uds_fails_the_one_request_and_keeps_the_connection() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");

    let handler: Arc<dyn desktop_assistant_application::AssistantApiHandler> =
        Arc::new(OversizeHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth {
        signing_key: signing_key.clone(),
    });
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, UdsServerConfig::new(path.clone()));
    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    wait_for_socket(&path).await;

    let config = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let (client, _signals, _guard) = connect_transport(&config).await.expect("connect over uds");

    let failed = timeout(Duration::from_secs(10), client.list_conversations())
        .await
        .expect("the request must fail within 10s, not hang on a torn-down socket");
    assert!(
        failed.is_err(),
        "an unsendable reply must fail the request it answers"
    );

    // The connection must still be serving.
    let id = timeout(Duration::from_secs(10), client.create_conversation("hi"))
        .await
        .expect("the connection must still serve after an unsendable reply")
        .expect("a small reply must still succeed");
    assert_eq!(id, "small");

    let _ = tx.send(());
}

/// Acceptance (#1303): an oversize OUTBOUND request costs the client that one
/// request, and nothing else.
///
/// `write_frame` refuses a body past the cap, and the writer loop breaks on any
/// error - which shuts the write half down, so the daemon reads EOF and closes,
/// and the client's reader fires the drop notifier. The connection recovers by
/// reconnecting, so this was never a wedge; but one refused request should not
/// cost the connection at all. The client checks the size before it enqueues
/// and fails that request on the spot instead.
///
/// Reachable rather than theoretical: a client-side MCP tool result is capped
/// at 1 MiB of RAW bytes, and JSON escaping multiplies a control byte by six.
#[tokio::test]
async fn an_oversize_request_fails_that_one_request_and_keeps_the_connection() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(
        path.clone(),
        signing_key.clone(),
        BigConversations {
            count: 0,
            bytes_each: 0,
        },
    );
    wait_for_socket(&path).await;

    let config = uds_config(path, mint_test_jwt(&signing_key, "dave"));
    let (client, _signals, drop_rx) = connect_transport(&config).await.expect("connect over uds");
    let mut drop_rx = drop_rx.expect("uds reports socket drops");

    // 5 MiB of title, so the serialized request is past the 4 MiB frame cap.
    let oversize = "t".repeat(5 * 1024 * 1024);
    let failed = timeout(
        Duration::from_secs(10),
        client.create_conversation(&oversize),
    )
    .await
    .expect("an oversize request must fail promptly, not wait for a response that cannot come");
    let err = failed.expect_err("an oversize request must fail");
    let text = format!("{err:#}");
    assert!(
        text.contains("4194304"),
        "the failure must name the cap the request is past: {text}"
    );

    // The connection must still be serving: there is no reconnect supervisor
    // behind `connect_transport`, so a second request only succeeds if the
    // first one left the socket alone.
    let id = timeout(Duration::from_secs(10), client.create_conversation("hi"))
        .await
        .expect("the connection must still serve after a refused request")
        .expect("a request inside the cap must still succeed");
    assert_eq!(id, "conv-1");

    assert!(
        drop_rx.try_recv().is_err(),
        "a refused request must not drop the socket"
    );

    let _ = shutdown.send(());
}
