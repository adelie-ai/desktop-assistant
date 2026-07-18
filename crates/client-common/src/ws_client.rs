use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use api::{WsFrame, WsRequest};
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use futures::{SinkExt, StreamExt};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use crate::commands::{AssistantCommands, PendingResult};
use crate::signal::SignalEvent;
use crate::timeouts::{DISPATCH_TIMEOUT, WS_PING_INTERVAL};

/// In-flight request correlation map plus a terminal "closed" marker. Mirrors
/// the UDS client's `PendingState` so a reconnect (#246) can re-arm the WS
/// client after the reader drains it on a drop.
struct PendingState {
    map: HashMap<String, oneshot::Sender<PendingResult>>,
    closed: Option<String>,
}

impl PendingState {
    fn close(&mut self, reason: &str) {
        if self.closed.is_none() {
            self.closed = Some(reason.to_string());
        }
        for (_id, tx) in self.map.drain() {
            let _ = tx.send(Err(reason.to_string()));
        }
    }

    fn reopen(&mut self) {
        self.closed = None;
    }
}

/// The live connection's write handle, swapped on reconnect (#246).
struct ConnState {
    outbound_tx: mpsc::UnboundedSender<Message>,
}

pub struct WsClient {
    /// The live writer, replaced in place by [`reconnect`](Self::reconnect).
    conn: Arc<Mutex<ConnState>>,
    pending: Arc<Mutex<PendingState>>,
    /// The persistent signal stream every reader (across reconnects) feeds.
    signal_tx: mpsc::UnboundedSender<SignalEvent>,
    /// Fires once per underlying-socket close so the Connector's reconnect
    /// supervisor knows to back off and reconnect (#246).
    drop_tx: mpsc::UnboundedSender<()>,
    /// Per-command response deadline (#221). Defaults to
    /// [`DISPATCH_TIMEOUT`]; tunable via [`set_dispatch_timeout`].
    dispatch_timeout: std::time::Duration,
}

impl WsClient {
    /// Override the per-command dispatch timeout (#221). See
    /// [`UdsClient::set_dispatch_timeout`](crate::uds_client::UdsClient::set_dispatch_timeout).
    pub fn set_dispatch_timeout(&mut self, timeout: std::time::Duration) {
        self.dispatch_timeout = timeout;
    }

    /// Connect a WebSocket transport. Returns the client, the persistent signal
    /// stream, and a drop-notifier receiver that fires once per underlying
    /// socket close (#246) — the Connector uses the latter to drive reconnect.
    ///
    /// `system_id` / `host_label` (#248) ride custom upgrade headers (WS
    /// authenticates via the `Authorization` header at upgrade time, not an
    /// in-band frame) so the daemon can compute exact co-location even over
    /// WebSocket; `None`/`None` simply omits them.
    pub async fn connect(
        ws_url: &str,
        bearer_token: &str,
        tls_ca_cert: Option<&Path>,
        system_id: Option<&str>,
        host_label: Option<&str>,
    ) -> Result<(
        Self,
        mpsc::UnboundedReceiver<SignalEvent>,
        mpsc::UnboundedReceiver<()>,
    )> {
        let pending = Arc::new(Mutex::new(PendingState {
            map: HashMap::new(),
            closed: None,
        }));
        let (signal_tx, signal_rx) = mpsc::unbounded_channel::<SignalEvent>();
        let (drop_tx, drop_rx) = mpsc::unbounded_channel::<()>();

        let outbound_tx = Self::spawn_connection(
            ws_url,
            bearer_token,
            tls_ca_cert,
            system_id,
            host_label,
            Arc::clone(&pending),
            signal_tx.clone(),
            drop_tx.clone(),
        )
        .await?;

        Ok((
            Self {
                conn: Arc::new(Mutex::new(ConnState { outbound_tx })),
                pending,
                signal_tx,
                drop_tx,
                dispatch_timeout: DISPATCH_TIMEOUT,
            },
            signal_rx,
            drop_rx,
        ))
    }

    /// Connect the socket, spawn the writer/keepalive/reader tasks bound to the
    /// **persistent** `pending` / `signal_tx` / `drop_tx`, and return the new
    /// writer handle. Shared by [`connect`](Self::connect) and
    /// [`reconnect`](Self::reconnect) (#246).
    // Each argument is a distinct connection input (endpoint, credential, TLS,
    // the #248 id/label, and the three persistent channels); bundling them into
    // a struct would just relocate the same fan-out without making it clearer.
    #[allow(clippy::too_many_arguments)]
    async fn spawn_connection(
        ws_url: &str,
        bearer_token: &str,
        tls_ca_cert: Option<&Path>,
        system_id: Option<&str>,
        host_label: Option<&str>,
        pending: Arc<Mutex<PendingState>>,
        signal_tx: mpsc::UnboundedSender<SignalEvent>,
        drop_tx: mpsc::UnboundedSender<()>,
    ) -> Result<mpsc::UnboundedSender<Message>> {
        use tokio_tungstenite::tungstenite::http::HeaderName;
        let mut request = ws_url.into_client_request()?;
        request.headers_mut().insert(
            tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
            format!("Bearer {bearer_token}").parse()?,
        );
        // System-id co-location handshake (#248): the daemon reads these custom
        // headers and compares the id to its own. Optional — omitted when the
        // client has no id, so the daemon falls back to the transport heuristic.
        // A header value that won't parse (non-ASCII control bytes) is silently
        // skipped — the id is a hint, not load-bearing, so we degrade rather
        // than fail the connect.
        if let Some(id) = system_id.filter(|s| !s.trim().is_empty())
            && let Ok(value) = id.parse()
        {
            request
                .headers_mut()
                .insert(HeaderName::from_static(api::WS_SYSTEM_ID_HEADER), value);
        }
        if let Some(label) = host_label.filter(|s| !s.trim().is_empty())
            && let Ok(value) = label.parse()
        {
            request
                .headers_mut()
                .insert(HeaderName::from_static(api::WS_HOST_LABEL_HEADER), value);
        }

        let connector = if ws_url.starts_with("wss://") {
            Some(build_tls_connector(tls_ca_cert)?)
        } else {
            None
        };

        let (socket, _response) =
            tokio_tungstenite::connect_async_tls_with_config(request, None, false, connector)
                .await?;
        let (mut ws_tx, mut ws_rx) = socket.split();

        let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
        tokio::spawn(async move {
            while let Some(message) = outbound_rx.recv().await {
                if ws_tx.send(message).await.is_err() {
                    break;
                }
            }
        });

        // Keepalive (#221): periodically push a `Ping` through the same writer
        // so a dead-but-open socket is detected. The server's matching `Pong`
        // (and any other inbound traffic) resets the reader/connector stall
        // clock; if the socket is dead the ping write fails, the writer task
        // breaks and drops its receiver, and this ticker exits on the next send
        // error. Cheap and self-terminating — no extra teardown wiring needed.
        let ping_tx = outbound_tx.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(WS_PING_INTERVAL);
            ticker.tick().await; // first tick fires immediately; skip it
            loop {
                ticker.tick().await;
                if ping_tx
                    .send(Message::Ping(tokio_tungstenite::tungstenite::Bytes::new()))
                    .is_err()
                {
                    break; // writer gone -> connection torn down
                }
            }
        });

        let pending_for_reader = Arc::clone(&pending);
        tokio::spawn(async move {
            while let Some(Ok(message)) = ws_rx.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Ok(frame) = serde_json::from_str::<WsFrame>(&text) else {
                    continue;
                };

                match frame {
                    WsFrame::Result { id, result } => {
                        if let Some(tx) = pending_for_reader.lock().await.map.remove(&id) {
                            let _ = tx.send(Ok(result));
                        }
                    }
                    WsFrame::Error { id, error } => {
                        if let Some(tx) = pending_for_reader.lock().await.map.remove(&id) {
                            let _ = tx.send(Err(error));
                        }
                    }
                    WsFrame::Event { event } => {
                        if let Some(signal) = map_event_to_signal(event) {
                            let _ = signal_tx.send(signal);
                        }
                    }
                }
            }

            // The socket closed. As with UDS (#246), do NOT emit a
            // `Disconnected` on the persistent signal stream — fail any
            // outstanding requests and notify the reconnect supervisor via
            // `drop_tx`, which owns the terminal-Disconnected + reconnect.
            pending_for_reader
                .lock()
                .await
                .close("websocket disconnected");
            let _ = drop_tx.send(());
        });

        Ok(outbound_tx)
    }

    /// Re-establish the underlying WebSocket after a drop (#246): reconnect,
    /// spawn fresh writer/keepalive/reader tasks bound to the persistent
    /// channels, and swap in the new writer. On failure the error is returned so
    /// the supervisor can back off and retry.
    ///
    /// The system id + host label (#248) are re-sent on every reconnect — the
    /// caller (`TransportClient::reconnect`) re-reads them from the stored
    /// `ConnectionConfig`, so the co-location handshake field survives a daemon
    /// restart exactly like the bearer token does.
    pub(crate) async fn reconnect(
        &self,
        ws_url: &str,
        bearer_token: &str,
        tls_ca_cert: Option<&Path>,
        system_id: Option<&str>,
        host_label: Option<&str>,
    ) -> Result<()> {
        let outbound_tx = Self::spawn_connection(
            ws_url,
            bearer_token,
            tls_ca_cert,
            system_id,
            host_label,
            Arc::clone(&self.pending),
            self.signal_tx.clone(),
            self.drop_tx.clone(),
        )
        .await?;
        self.pending.lock().await.reopen();
        self.conn.lock().await.outbound_tx = outbound_tx;
        Ok(())
    }

    /// Send a prompt with an optional per-message model/connection override.
    ///
    /// Backward-compatibility shim: the implementation now lives on the
    /// transport-agnostic [`AssistantCommands`] trait (so `UdsClient` gets it
    /// too — adele-gtk#49). This inherent delegator is kept so existing
    /// `ws.send_prompt_with_override(...)` call sites in downstream repos
    /// (adele-tui, adele-kde) keep compiling whether or not they have the
    /// trait in scope.
    pub async fn send_prompt_with_override(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
    ) -> Result<String> {
        AssistantCommands::send_prompt_with_override(
            self,
            conversation_id,
            prompt,
            override_selection,
        )
        .await
    }

    /// List models across every healthy connection. Pass `connection_id =
    /// Some(_)` to scope to a single connection. `refresh = true` bypasses
    /// connector caches (e.g. Bedrock).
    ///
    /// Backward-compatibility shim delegating to the [`AssistantCommands`]
    /// trait default (see `send_prompt_with_override` above).
    pub async fn list_available_models(
        &self,
        connection_id: Option<&str>,
        refresh: bool,
    ) -> Result<Vec<api::ModelListing>> {
        AssistantCommands::list_available_models(self, connection_id, refresh).await
    }
}

#[async_trait]
impl AssistantCommands for WsClient {
    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = WsRequest {
            id: id.clone(),
            command,
        };
        let payload = serde_json::to_string(&request)?;

        let (tx, rx) = oneshot::channel::<PendingResult>();
        {
            let mut state = self.pending.lock().await;
            if let Some(reason) = &state.closed {
                return Err(anyhow!("websocket connection closed: {reason}"));
            }
            state.map.insert(id.clone(), tx);
        }

        if self
            .conn
            .lock()
            .await
            .outbound_tx
            .send(Message::Text(payload.into()))
            .is_err()
        {
            self.pending.lock().await.map.remove(&id);
            return Err(anyhow!("failed to send websocket request"));
        }

        // Bound the wait for the response frame (#221), mirroring the UDS path:
        // a silent server must not hang the caller. Drop the pending slot on
        // expiry so it can't leak, and return a clear transport error.
        match tokio::time::timeout(self.dispatch_timeout, rx).await {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(error))) => Err(anyhow!(error)),
            Ok(Err(_closed)) => Err(anyhow!("websocket response channel closed")),
            Err(_elapsed) => {
                self.pending.lock().await.map.remove(&id);
                Err(anyhow!(
                    "websocket command timed out after {:?} with no response from the server",
                    self.dispatch_timeout
                ))
            }
        }
    }
}

/// Builds the trust anchors for an outbound `wss://` connection: the public
/// webpki roots, plus any certificate in `ca_cert_path`.
///
/// Why the public roots are the floor: a remote brain normally sits behind an
/// ingress that terminates TLS with a publicly-signed certificate, while
/// `ca_cert_path` defaults to the *daemon's* self-signed local CA. Treating the
/// configured CA as a replacement rather than an addition left clients trusting
/// exactly one private CA and unable to verify any public endpoint (#521). This
/// also matches the reqwest path in `auth::request_ws_login_token`, whose
/// `add_root_certificate` has always layered onto the built-in roots — the two
/// halves of the connect flow previously disagreed.
///
/// A `ca_cert_path` that does not exist is a warning, not an error: clients
/// populate the default path unconditionally, so a machine that has never run a
/// local daemon has no file there and still needs to reach public endpoints. A
/// file that exists but yields no certificate *is* an error — the operator
/// pointed at it deliberately, and silently ignoring it would downgrade trust
/// without saying so.
pub(crate) fn build_root_store(ca_cert_path: Option<&Path>) -> Result<rustls::RootCertStore> {
    let mut root_store = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };

    let Some(ca_path) = ca_cert_path else {
        return Ok(root_store);
    };

    let pem_bytes = match std::fs::read(ca_path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(
                path = %ca_path.display(),
                "no local CA certificate; trusting the public roots only"
            );
            return Ok(root_store);
        }
        Err(e) => return Err(anyhow!("reading CA cert {}: {e}", ca_path.display())),
    };

    use rustls::pki_types::pem::PemObject;
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls::pki_types::CertificateDer::pem_reader_iter(&mut std::io::BufReader::new(
            pem_bytes.as_slice(),
        ))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("parsing CA cert {}: {e}", ca_path.display()))?;

    if certs.is_empty() {
        return Err(anyhow!(
            "CA cert {} contains no certificates",
            ca_path.display()
        ));
    }

    let added = certs.len();
    for cert in certs {
        root_store
            .add(cert)
            .map_err(|e| anyhow!("trusting CA cert {}: {e}", ca_path.display()))?;
    }
    tracing::debug!(
        path = %ca_path.display(),
        added,
        "trusting local CA certificate(s) alongside the public roots"
    );

    Ok(root_store)
}

fn build_tls_connector(ca_cert_path: Option<&Path>) -> Result<tokio_tungstenite::Connector> {
    let root_store = build_root_store(ca_cert_path)?;

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    Ok(tokio_tungstenite::Connector::Rustls(Arc::new(config)))
}

// `SignalEvent` and the `api::Event` → `SignalEvent` projection now live in
// `api-model` (so the wasm client cores can use them without this crate's native
// transport tail, #377). Re-exported here so `crate::ws_client::map_event_to_signal`
// callers (uds_client, the streaming loop above) are unchanged.
pub use desktop_assistant_api_model::signal::map_event_to_signal;

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `pem` to a scratch file and returns the handle. Kept alive by the
    /// caller so the path stays valid for the duration of the assertion.
    fn ca_file(pem: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("create temp CA file");
        f.write_all(pem.as_bytes()).expect("write temp CA file");
        f.flush().expect("flush temp CA file");
        f
    }

    /// A freshly minted self-signed certificate, PEM-encoded.
    fn self_signed_pem() -> String {
        rcgen::generate_simple_self_signed(vec!["ca.test".to_string()])
            .expect("generate self-signed cert")
            .cert
            .pem()
    }

    /// The public trust anchors are the floor: a client with no extra CA
    /// configured must still be able to verify a publicly-signed endpoint.
    #[test]
    fn root_store_without_custom_ca_trusts_public_roots() {
        let store = build_root_store(None).expect("build root store with no custom CA");
        assert!(
            !store.is_empty(),
            "expected the public webpki roots to be trusted by default, got an empty store"
        );
    }

    /// The regression this issue is about: a configured CA must be *additional*
    /// trust, never a replacement for the public roots.
    #[test]
    fn root_store_with_custom_ca_keeps_public_roots() {
        let baseline = build_root_store(None).expect("baseline root store").len();
        let ca = ca_file(&self_signed_pem());

        let store = build_root_store(Some(ca.path())).expect("build root store with custom CA");

        assert_eq!(
            store.len(),
            baseline + 1,
            "custom CA should be added to the public roots, not replace them"
        );
    }

    /// The default CA path is populated unconditionally, so a machine that has
    /// never run a local daemon points at a file that does not exist. That must
    /// degrade to "public roots only", not fail the connection.
    #[test]
    fn missing_ca_file_warns_and_keeps_public_roots() {
        let baseline = build_root_store(None).expect("baseline root store").len();
        let missing = std::path::Path::new("/nonexistent/desktop-assistant/tls/ca.pem");

        let store = build_root_store(Some(missing)).expect("missing CA file must not be fatal");

        assert_eq!(
            store.len(),
            baseline,
            "a missing CA file should leave the public roots intact"
        );
    }

    /// An explicitly configured path that holds no certificate is operator
    /// error and must be loud — unlike the absent-file case above.
    #[test]
    fn ca_file_without_certificates_is_an_error() {
        let junk = ca_file("this is not a certificate\n");

        let err = build_root_store(Some(junk.path()))
            .expect_err("a CA file with no certificates must be rejected");

        let msg = err.to_string();
        assert!(
            msg.contains("no certificates"),
            "error should name the empty-bundle cause, got: {msg}"
        );
    }

    /// Concatenated bundles are the normal way to trust an internal CA and a
    /// public root at once; every certificate in the file must land.
    #[test]
    fn multi_certificate_ca_bundle_adds_every_certificate() {
        let baseline = build_root_store(None).expect("baseline root store").len();
        let bundle = ca_file(&format!("{}{}", self_signed_pem(), self_signed_pem()));

        let store = build_root_store(Some(bundle.path())).expect("build root store from bundle");

        assert_eq!(
            store.len(),
            baseline + 2,
            "both certificates in the bundle should be trusted"
        );
    }

    #[test]
    fn maps_stream_events_to_signal_events() {
        let delta = map_event_to_signal(api::Event::AssistantDelta {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            chunk: "he".to_string(),
        });
        assert!(matches!(delta, Some(SignalEvent::Chunk { .. })));

        let complete = map_event_to_signal(api::Event::AssistantCompleted {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            full_response: "hello".to_string(),
        });
        assert!(matches!(complete, Some(SignalEvent::Complete { .. })));

        let error = map_event_to_signal(api::Event::AssistantError {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            error: "oops".to_string(),
        });
        assert!(matches!(error, Some(SignalEvent::Error { .. })));
    }

    #[test]
    fn maps_context_usage_event() {
        // Issue #341: the per-turn fill report crosses to the signal stream
        // with its counts and flag intact so clients can render the indicator.
        let signal = map_event_to_signal(api::Event::ContextUsage {
            conversation_id: "c1".to_string(),
            request_id: "r1".to_string(),
            used_tokens: 12_000,
            budget_tokens: 32_000,
            compaction_active: true,
        });
        match signal {
            Some(SignalEvent::ContextUsage {
                conversation_id,
                request_id,
                used_tokens,
                budget_tokens,
                compaction_active,
            }) => {
                assert_eq!(conversation_id, "c1");
                assert_eq!(request_id, "r1");
                assert_eq!(used_tokens, 12_000);
                assert_eq!(budget_tokens, 32_000);
                assert!(compaction_active);
            }
            other => panic!("expected SignalEvent::ContextUsage, got {other:?}"),
        }
    }

    #[test]
    fn maps_title_changed_event() {
        let event = map_event_to_signal(api::Event::ConversationTitleChanged {
            conversation_id: "c1".to_string(),
            title: "New Title".to_string(),
        });
        assert!(matches!(event, Some(SignalEvent::TitleChanged { .. })));
    }

    #[test]
    fn maps_task_started_event() {
        let task = api::TaskView {
            id: api::TaskId("t-1".into()),
            kind: api::TaskKind::Standalone {
                name: "researcher".into(),
                conversation_id: "c-1".into(),
            },
            status: api::TaskStatus::Running,
            started_at: 1,
            ended_at: None,
            last_error: None,
            parent: None,
            children: Vec::new(),
            title: "Researcher: pricing data".into(),
            progress_hint: None,
        };
        let signal = map_event_to_signal(api::Event::TaskStarted { task });
        match signal {
            Some(SignalEvent::TaskStarted { task }) => {
                assert_eq!(task.id, api::TaskId("t-1".into()));
                assert_eq!(task.title, "Researcher: pricing data");
            }
            other => panic!("expected SignalEvent::TaskStarted, got {other:?}"),
        }
    }

    #[test]
    fn maps_task_progress_event() {
        let signal = map_event_to_signal(api::Event::TaskProgress {
            id: "t-1".into(),
            progress_hint: Some("step 2/5".into()),
        });
        match signal {
            Some(SignalEvent::TaskProgress { id, progress_hint }) => {
                assert_eq!(id, "t-1");
                assert_eq!(progress_hint.as_deref(), Some("step 2/5"));
            }
            other => panic!("expected SignalEvent::TaskProgress, got {other:?}"),
        }
    }

    #[test]
    fn maps_task_log_appended_event() {
        let entry = api::TaskLogEntry {
            seq: 7,
            timestamp: 1_700_000_000,
            level: api::LogLevel::Info,
            category: api::LogCategory::Status,
            message: "fetching".into(),
            data: None,
        };
        let signal = map_event_to_signal(api::Event::TaskLogAppended {
            id: "t-1".into(),
            entry,
        });
        match signal {
            Some(SignalEvent::TaskLogAppended { id, entry }) => {
                assert_eq!(id, "t-1");
                assert_eq!(entry.seq, 7);
                assert_eq!(entry.message, "fetching");
            }
            other => panic!("expected SignalEvent::TaskLogAppended, got {other:?}"),
        }
    }

    #[test]
    fn maps_task_completed_event() {
        let signal = map_event_to_signal(api::Event::TaskCompleted {
            id: "t-1".into(),
            status: api::TaskStatus::Failed,
            last_error: Some("LLM rate limit".into()),
        });
        match signal {
            Some(SignalEvent::TaskCompleted {
                id,
                status,
                last_error,
            }) => {
                assert_eq!(id, "t-1");
                assert!(matches!(status, api::TaskStatus::Failed));
                assert_eq!(last_error.as_deref(), Some("LLM rate limit"));
            }
            other => panic!("expected SignalEvent::TaskCompleted, got {other:?}"),
        }
    }

    #[test]
    fn maps_scratchpad_changed_event() {
        let signal = map_event_to_signal(api::Event::ScratchpadChanged {
            conversation_id: "c-1".into(),
        });
        match signal {
            Some(SignalEvent::ScratchpadChanged { conversation_id }) => {
                assert_eq!(conversation_id, "c-1");
            }
            other => panic!("expected SignalEvent::ScratchpadChanged, got {other:?}"),
        }
    }

    #[test]
    fn maps_client_tool_call_event() {
        // #231: a `ClientToolCall` event used to be dropped (`=> None`); it now
        // surfaces on the signal stream so a client that advertised tools can
        // run the call and post a result back. The `TaskId` newtype is unwrapped
        // to its inner string.
        let signal = map_event_to_signal(api::Event::ClientToolCall {
            task_id: api::TaskId("task-1".into()),
            conversation_id: "conv-1".into(),
            tool_call_id: "call-1".into(),
            tool_name: "weather".into(),
            arguments: serde_json::json!({ "city": "Boston" }),
        });
        match signal {
            Some(SignalEvent::ClientToolCall {
                task_id,
                conversation_id,
                tool_call_id,
                tool_name,
                arguments,
            }) => {
                assert_eq!(task_id, "task-1");
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(tool_call_id, "call-1");
                assert_eq!(tool_name, "weather");
                assert_eq!(arguments, serde_json::json!({ "city": "Boston" }));
            }
            other => panic!("expected SignalEvent::ClientToolCall, got {other:?}"),
        }
    }

    #[test]
    fn ignores_non_stream_config_events() {
        let event = map_event_to_signal(api::Event::ConfigChanged {
            config: api::Config {
                embeddings: api::EmbeddingsSettingsView {
                    connector: "openai".to_string(),
                    model: "text-embedding-3-small".to_string(),
                    base_url: "https://api.openai.com/v1".to_string(),
                    has_api_key: true,
                    available: true,
                    is_default: true,
                    health: Default::default(),
                },
                persistence: api::PersistenceSettingsView {
                    enabled: false,
                    remote_url: String::new(),
                    remote_name: "origin".to_string(),
                    push_on_update: true,
                },
                personality: api::PersonalitySettingsView::default(),
            },
        });
        assert!(event.is_none());
    }
}
