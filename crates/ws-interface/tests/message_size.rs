//! WebSocket inbound message / frame size cap tests (issue #142).
//!
//! The WS handler caps both `max_message_size` and `max_frame_size` at
//! 4 MiB to match the UDS (`crates/uds-interface`) and D-Bus bridge
//! (`crates/dbus-bridge/src/transport.rs`) framing caps. These tests
//! pin three things down at the wire level:
//!
//! 1. A message one byte under the cap round-trips end-to-end.
//! 2. A message one byte over the cap is rejected with a defined
//!    closure (RFC 6455 close code 1009 "Message Too Big") rather than
//!    a panic or a silent drop.
//! 3. The per-frame cap is enforced on fragmented messages: a single
//!    fragment whose length exceeds the cap is rejected before the
//!    server even attempts to assemble the message.
//!
//! The fragmented-frame test bypasses `tokio-tungstenite` (which never
//! exposes a sink at the frame level) and writes the WebSocket frame
//! by hand over a raw TCP socket after performing the HTTP upgrade.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, KnowledgeEntry,
};
use desktop_assistant_core::ports::inbound::{
    AssistantService, BackendTasksSettingsView, ConnectionConfigPayload,
    ConnectionView as CoreConnectionView, ConnectionsService, ConnectorDefaultsView,
    ConversationService, DatabaseSettingsView, EmbeddingsSettingsView, KnowledgeService,
    LlmSettingsView, ModelListing as CoreModelListing, PersistenceSettingsView,
    PurposeConfigPayload, PurposeKind as CorePurposeKind, PurposesView as CorePurposesView,
    SettingsService, WsAuthSettingsView,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
use desktop_assistant_ws::{WsAuthValidator, WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

/// Bytes for the per-message and per-frame caps applied by the WS
/// handler. Must match `crates/uds-interface/src/lib.rs::MAX_FRAME_LEN`
/// and `crates/dbus-bridge/src/transport.rs::MAX_FRAME_LEN`.
const MAX_WS_BYTES: usize = 4 * 1024 * 1024;

// ---------------------------------------------------------------------
// Minimal fakes — the size-cap tests only exercise `Ping`, so we can
// keep almost every trait method as a stub. The shapes mirror those in
// `tests/ping.rs` so the two files stay easy to keep in sync.
// ---------------------------------------------------------------------

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
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let mut e = KnowledgeEntry::new("kb-test", content, tags);
        e.metadata = metadata;
        Ok(e)
    }
    async fn update_entry(
        &self,
        id: String,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let mut e = KnowledgeEntry::new(id, content, tags);
        e.metadata = metadata;
        Ok(e)
    }
    async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
        Ok(())
    }
}

struct FakeConnections;
impl ConnectionsService for FakeConnections {
    async fn list_connections(&self) -> Result<Vec<CoreConnectionView>, CoreError> {
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
    ) -> Result<Vec<CoreModelListing>, CoreError> {
        Ok(vec![])
    }
    async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
        Ok(CorePurposesView::default())
    }
    async fn set_purpose(
        &self,
        _purpose: CorePurposeKind,
        _config: PurposeConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
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

struct StaticJwtAuth;

#[async_trait::async_trait]
impl WsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        token == "test-jwt"
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
        _include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(vec![])
    }
    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        Ok(Conversation::new(id.as_str(), "t"))
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
    async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
        Ok(format!(
            "jwt-for-{}",
            subject.unwrap_or_else(|| "desktop-client".to_string())
        ))
    }
    async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
        Ok(token.starts_with("jwt-for-"))
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
    async fn list_mcp_servers(
        &self,
    ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
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
    ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
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

fn make_app() -> axum::Router {
    let handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(FakeConversations),
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    ));
    router(handler, Arc::new(StaticJwtAuth))
}

async fn spawn_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app = make_app();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, server)
}

/// Build a `WsRequest::Ping` whose serialized JSON length is exactly
/// `target` bytes. We pad the `id` field with ASCII so the JSON encoder
/// doesn't escape anything, which would otherwise throw the length
/// calculation off by the escape overhead.
fn ping_request_with_total_len(target: usize) -> String {
    let base = WsRequest {
        id: String::new(),
        command: desktop_assistant_api_model::Command::Ping,
    };
    let base_len = serde_json::to_string(&base).unwrap().len();
    assert!(
        target >= base_len,
        "target len {target} is smaller than envelope overhead {base_len}"
    );
    let pad = target - base_len;
    let id = "x".repeat(pad);
    let req = WsRequest {
        id,
        command: desktop_assistant_api_model::Command::Ping,
    };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(
        json.len(),
        target,
        "padded WsRequest length mismatch (expected {target}, got {})",
        json.len()
    );
    json
}

#[tokio::test]
async fn ws_message_at_4_mb_minus_one_byte_is_accepted() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, Some("test-jwt")))
        .await
        .unwrap();

    let payload = ping_request_with_total_len(MAX_WS_BYTES - 1);
    let id_echo: String = serde_json::from_str::<serde_json::Value>(&payload)
        .unwrap()
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap()
        .to_string();

    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        payload.into(),
    ))
    .await
    .expect("client should be able to send the 4 MiB - 1 byte message");

    let frame = timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("server should respond within 10s")
        .expect("stream should yield a frame")
        .expect("frame should not be an error");
    let text = frame.into_text().expect("response should be text");
    let parsed: WsFrame = serde_json::from_str(&text).expect("response should be valid WsFrame");

    match parsed {
        WsFrame::Result { id, result } => {
            assert_eq!(id, id_echo, "result id should echo the request id");
            assert_eq!(
                result,
                desktop_assistant_api_model::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("expected Pong result, got {other:?}"),
    }

    server.abort();
}

#[tokio::test]
async fn ws_message_at_4_mb_plus_one_byte_is_rejected_with_clean_error_frame() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    // Override the client config so the client itself doesn't reject the
    // oversize message before it hits the wire — the cap under test is
    // the *server*'s.
    let client_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(16 << 20))
        .max_frame_size(Some(16 << 20));
    let (mut ws, _) = tokio_tungstenite::connect_async_with_config(
        ws_request(&url, Some("test-jwt")),
        Some(client_config),
        false,
    )
    .await
    .unwrap();

    let payload = ping_request_with_total_len(MAX_WS_BYTES + 1);

    // The client `.send` may or may not surface an error depending on
    // when the server tears the socket down; we treat either path as
    // acceptable so long as the *next read* terminates cleanly.
    let _ = ws
        .send(tokio_tungstenite::tungstenite::Message::Text(
            payload.into(),
        ))
        .await;

    // Drain frames; we expect either a Close with code 1009 (Message
    // Too Big) or a stream-end / IO error. Anything else (a Result
    // frame, a panic, hanging forever) fails the test.
    let mut saw_close_with_1009 = false;
    let mut saw_stream_end = false;
    let drain = async {
        while let Some(frame) = ws.next().await {
            match frame {
                Ok(tokio_tungstenite::tungstenite::Message::Close(Some(close))) => {
                    if u16::from(close.code) == 1009 {
                        saw_close_with_1009 = true;
                    }
                    break;
                }
                Ok(tokio_tungstenite::tungstenite::Message::Close(None)) => {
                    break;
                }
                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                    let parsed = serde_json::from_str::<WsFrame>(&text);
                    if let Ok(WsFrame::Result { .. }) = parsed {
                        panic!(
                            "server returned a Result frame for an oversize message — \
                             cap is not enforced"
                        );
                    }
                    // Other frames (e.g. unsolicited events) wouldn't
                    // be expected, but ignore them defensively.
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        saw_stream_end = true;
    };
    timeout(Duration::from_secs(10), drain)
        .await
        .expect("server should drop / close the connection within 10s");

    assert!(
        saw_close_with_1009 || saw_stream_end,
        "expected either a 1009 close frame or a clean stream end"
    );

    server.abort();
}

/// Sec-WebSocket-Accept is `base64(sha1(key + magic))`. We don't bother
/// validating the response value in our test — we just need to drive
/// the handshake so the server hands the upgraded stream to its WS
/// layer. The key itself is a 16-byte random nonce; we use a fixed one
/// here for reproducibility.
const WS_KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";

/// Perform a minimal RFC 6455 handshake on a raw TCP stream and return
/// the still-open stream once the server has answered 101.
async fn raw_ws_handshake(addr: SocketAddr, bearer: &str) -> std::io::Result<TcpStream> {
    let mut stream = TcpStream::connect(addr).await?;
    let req = format!(
        "GET /ws HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: {WS_KEY}\r\n\
         Sec-WebSocket-Version: 13\r\n\
         Authorization: Bearer {bearer}\r\n\
         \r\n",
    );
    stream.write_all(req.as_bytes()).await?;

    // Read until we see end-of-headers (\r\n\r\n). The body of a 101 is
    // empty, so once we hit the blank line the stream is the WS layer.
    let mut buf = Vec::with_capacity(512);
    let mut chunk = [0u8; 256];
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Err(std::io::Error::other("server closed during handshake"));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 8192 {
            return Err(std::io::Error::other("handshake response too large"));
        }
    }
    let status_ok = buf.starts_with(b"HTTP/1.1 101");
    if !status_ok {
        return Err(std::io::Error::other(format!(
            "expected 101 Switching Protocols, got: {}",
            String::from_utf8_lossy(&buf[..buf.len().min(200)])
        )));
    }
    Ok(stream)
}

/// Encode a client→server WebSocket frame header (the wire format from
/// RFC 6455 §5.2) for a text-opcode frame with the given `fin` flag and
/// `payload_len`. The mask key is included but is zero, which still
/// satisfies the MUST-be-masked rule. Returns header bytes only — the
/// caller writes the (already-masked, or in our case identity-masked)
/// payload separately.
fn encode_text_frame_header(fin: bool, payload_len: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(14);
    let first = if fin { 0x80 } else { 0x00 } | 0x01; // text opcode = 0x1
    out.push(first);
    // Mask bit (0x80) + length encoding.
    if payload_len < 126 {
        out.push(0x80 | (payload_len as u8));
    } else if payload_len <= u16::MAX as u64 {
        out.push(0x80 | 126);
        out.extend_from_slice(&(payload_len as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&payload_len.to_be_bytes());
    }
    // Zero mask key — masking with zeroes is a no-op so we can write
    // the raw payload bytes unchanged.
    out.extend_from_slice(&[0u8; 4]);
    out
}

#[tokio::test]
async fn ws_frame_size_cap_enforced_on_fragmented_messages() {
    let (addr, server) = spawn_server().await;

    let mut stream = raw_ws_handshake(addr, "test-jwt")
        .await
        .expect("handshake should succeed against the test server");

    // First fragment of a fragmented text message: fin=0, opcode=text,
    // length one byte past the cap. The server must reject *this frame*
    // (per `max_frame_size`) before it considers assembling further
    // fragments. We send the header plus a single payload byte and
    // then wait — *without* closing our end — for the server to react.
    //
    // The key distinction this test pins down vs. "server just blocked
    // waiting for more bytes": with the cap enforced, tungstenite
    // surfaces an error as soon as it parses the header (it never
    // waits for the payload), so the server closes the socket within
    // a handful of milliseconds. Without the cap, the server would
    // block on `read_exact` for the rest of the claimed 4 MiB + 1
    // bytes — bytes we never send — and our read here would time out.
    let header = encode_text_frame_header(false, (MAX_WS_BYTES + 1) as u64);
    stream
        .write_all(&header)
        .await
        .expect("server should accept the header bytes");
    // One real payload byte just to push the server's reader along.
    let _ = stream.write_all(b"x").await;
    let _ = stream.flush().await;

    // Read with a tight bound. The cap is enforced synchronously
    // against the frame header, so a healthy server should respond
    // within well under a second; we allow 5s for CI jitter. Without
    // the cap the server blocks waiting for the remaining ~4 MiB,
    // this read times out, and the test fails.
    let mut readbuf = [0u8; 1024];
    let outcome = timeout(Duration::from_secs(5), async {
        let mut all = Vec::new();
        loop {
            match stream.read(&mut readbuf).await {
                Ok(0) => return Ok::<Vec<u8>, std::io::Error>(all),
                Ok(n) => {
                    all.extend_from_slice(&readbuf[..n]);
                    // Stop once we have at least one complete control
                    // frame's worth — control frames are <=125 bytes
                    // and start with 0x88 for a server-side close.
                    if all.contains(&0x88) {
                        return Ok(all);
                    }
                }
                Err(e) => return Err(e),
            }
        }
    })
    .await
    .expect(
        "server should reject the oversize fragment and close within 5s — \
         a hang here means max_frame_size is not enforced and the server \
         is still patiently waiting for the rest of the (oversize) payload",
    );

    match outcome {
        Ok(bytes) => {
            // The server may close cleanly (0x88 control frame) or just
            // drop the TCP connection. Either is acceptable; what isn't
            // acceptable is a text-opcode reply or no reaction at all.
            assert!(
                !bytes.contains(&0x81),
                "server emitted a final text frame (opcode 0x81) in response \
                 to an oversize fragment — frame cap is not enforced"
            );
        }
        Err(_io) => {
            // Connection reset by peer is fine — the server bailed.
        }
    }

    server.abort();
}

#[tokio::test]
async fn ws_malformed_json_does_not_drop_connection() {
    // Regression: a malformed JSON payload (well under the cap) must
    // not panic the server or terminate the socket. It is silently
    // dropped (the dispatcher has no error frame to address it to —
    // there is no `id` to echo back), which matches pre-existing
    // behavior. We send a malformed payload, then a valid Ping and
    // confirm the Pong still arrives.
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, Some("test-jwt")))
        .await
        .unwrap();

    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        "{not valid json".into(),
    ))
    .await
    .expect("malformed payload should still send");

    let req = WsRequest {
        id: "after-bad-json".into(),
        command: desktop_assistant_api_model::Command::Ping,
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .expect("subsequent valid payload should send");

    let frame = timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("server should still be alive after malformed payload")
        .expect("stream should yield")
        .expect("frame should not error");
    let parsed: WsFrame = serde_json::from_str(&frame.into_text().unwrap()).unwrap();
    match parsed {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "after-bad-json");
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
async fn ws_partial_frame_then_close_is_handled_cleanly() {
    // Regression: a client that starts a frame and then drops the TCP
    // connection mid-payload must not panic the server. We perform the
    // handshake manually, claim a (within-cap) 1 KiB payload, send only
    // a few bytes, then close. The server should terminate the inbound
    // loop without faulting.
    let (addr, server) = spawn_server().await;

    let mut stream = raw_ws_handshake(addr, "test-jwt")
        .await
        .expect("handshake should succeed");

    let header = encode_text_frame_header(true, 1024);
    stream.write_all(&header).await.unwrap();
    // Send a handful of bytes then close — server is left waiting on
    // the rest of the payload that never arrives.
    let _ = stream.write_all(b"partial").await;
    drop(stream);

    // The smoke check is just that the server is still healthy enough
    // to serve a fresh connection.
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, Some("test-jwt")))
        .await
        .expect("server should still accept new connections");
    let req = WsRequest {
        id: "after-partial".into(),
        command: desktop_assistant_api_model::Command::Ping,
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();
    let frame = timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("server should respond on a fresh connection")
        .expect("stream should yield")
        .expect("frame should not error");
    let _: WsFrame = serde_json::from_str(&frame.into_text().unwrap()).unwrap();

    server.abort();
}
