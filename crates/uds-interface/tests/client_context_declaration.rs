//! What reaches the system prompt when a client declares whether it shares its
//! local environment (#549/#558/#783).
//!
//! The daemon grounds the prompt from the kernel-attested peer identity when a
//! local client reports no client context (#558). A client that connects on
//! behalf of somebody else - the web BFF serves browsers, and its own process
//! environment is the server's, not the browser user's - must be able to refuse
//! that substitution. These tests drive a real turn over a real Unix socket and
//! read the rendered client-context block, which is the exact text
//! `assemble_system_instruction` puts in the system prompt.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_core::ports::transport::current_client_context;
use desktop_assistant_core::prompts::render_client_context;
use desktop_assistant_peer_cred::PeerIdentity;
use desktop_assistant_uds::{UdsAuth, UdsAuthValidator, UdsServer, UdsServerConfig, write_frame};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-uds-ctx-iss";
const AUD: &str = "test-uds-ctx-aud";

fn mint_test_jwt(signing_key: &str, subject: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock is after the unix epoch")
        .as_secs();
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

/// Handler that reports, for each turn, the client-context block the model
/// would read - `None` when the prompt carries no such block at all.
struct PromptSectionCapture {
    rendered: mpsc::UnboundedSender<Option<String>>,
}

#[async_trait::async_trait]
impl AssistantApiHandler for PromptSectionCapture {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::Ping => Ok(api::CommandResult::Pong {
                value: "pong".into(),
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
        let section = current_client_context()
            .as_ref()
            .and_then(render_client_context);
        let _ = self.rendered.send(section);
        Ok(())
    }
}

/// Accepts a bearer token, exactly like the daemon's remote-door validator.
struct JwtOnlyAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for JwtOnlyAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }

    async fn extract_user_id(&self, token: &str) -> Option<jwt::UserId> {
        jwt::decode(token, &self.signing_key, ISS, AUD)
            .ok()
            .map(|claims| jwt::UserId::new(claims.sub))
    }
}

/// Authenticates from the kernel peer identity alone (#407): no token is sent
/// and none is accepted, so a connection this validator admits proves peer-cred
/// authentication ran.
struct PeerCredOnlyAuth;

#[async_trait::async_trait]
impl UdsAuthValidator for PeerCredOnlyAuth {
    async fn validate_bearer_token(&self, _token: &str) -> bool {
        false
    }

    async fn authenticate(&self, _token: Option<&str>, peer: Option<&PeerIdentity>) -> UdsAuth {
        match peer {
            Some(p) => UdsAuth::allow_tenant(jwt::UserId::new(p.username.clone())),
            None => UdsAuth::Reject("auth: no kernel peer identity".to_string()),
        }
    }
}

/// A listener on a fresh socket, plus the turn-report receiver and the shutdown
/// trigger. Keep the `TempDir` alive for the life of the test.
struct Harness {
    _dir: TempDir,
    path: std::path::PathBuf,
    rendered: mpsc::UnboundedReceiver<Option<String>>,
    shutdown: tokio::sync::oneshot::Sender<()>,
    _join: tokio::task::JoinHandle<anyhow::Result<()>>,
}

fn start_server(auth: Arc<dyn UdsAuthValidator>) -> Harness {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("adelie.sock");
    let (rendered_tx, rendered_rx) = mpsc::unbounded_channel();
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PromptSectionCapture {
        rendered: rendered_tx,
    });
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, UdsServerConfig::new(path.clone()));
    let join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Harness {
        _dir: dir,
        path,
        rendered: rendered_rx,
        shutdown: shutdown_tx,
        _join: join,
    }
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("uds socket did not appear");
}

/// Connect, send `handshake` verbatim, drive one turn, and return the
/// client-context block the model would have read.
async fn client_context_block_for_one_turn(
    harness: &mut Harness,
    handshake: serde_json::Value,
) -> Option<String> {
    wait_for_socket(&harness.path).await;
    let mut stream = UnixStream::connect(&harness.path)
        .await
        .expect("connect to the listener");
    write_frame(
        &mut stream,
        &serde_json::to_vec(&handshake).expect("encode"),
    )
    .await
    .expect("write handshake");

    let req = api::WsRequest {
        id: "turn-1".into(),
        command: api::Command::SendMessage {
            conversation_id: "conv-1".into(),
            content: "hello".into(),
            override_selection: None,
            system_refinement: String::new(),
            client_context: None,
            idempotency_key: None,
            turn_id: None,
            traceparent: None,
        },
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).expect("encode"))
        .await
        .expect("write send-message");

    timeout(Duration::from_secs(5), harness.rendered.recv())
        .await
        .expect("the turn did not reach the handler within 5s")
        .expect("the handler reported no turn")
}

/// #783 acceptance: a client that declared `share_client_context = false` gets
/// no client-context block in the prompt, even though the kernel peer identity
/// of the connecting process is available and the client reported no context of
/// its own. The declaration is the client's, so it wins over the daemon's
/// peer-cred inference.
#[tokio::test]
async fn a_declined_client_context_puts_no_client_block_in_the_prompt() {
    let signing_key = "deadbeef".repeat(8);
    let mut harness = start_server(Arc::new(JwtOnlyAuth {
        signing_key: signing_key.clone(),
    }));
    let handshake = serde_json::json!({
        "jwt": mint_test_jwt(&signing_key, "web-bff"),
        "share_client_context": false,
    });

    let section = client_context_block_for_one_turn(&mut harness, handshake).await;

    assert_eq!(
        section, None,
        "a client that declined to share its environment must leave the prompt with no \
         client-context block; the daemon must not substitute the connecting process's own \
         identity for it"
    );
    let _ = harness.shutdown.send(());
}

/// #558 regression: a client that made no declaration at all - every client
/// that predates #783, and every local desktop client that cannot build a
/// context - still has its prompt grounded from the kernel peer identity.
#[tokio::test]
async fn an_undeclared_client_still_grounds_the_prompt_from_the_peer_identity() {
    let signing_key = "deadbeef".repeat(8);
    let mut harness = start_server(Arc::new(JwtOnlyAuth {
        signing_key: signing_key.clone(),
    }));
    let handshake = serde_json::json!({ "jwt": mint_test_jwt(&signing_key, "desktop") });

    let section = client_context_block_for_one_turn(&mut harness, handshake).await;

    assert!(
        section.is_some(),
        "a client that neither reported a context nor declined one must keep the #558 \
         peer-identity grounding"
    );
    let _ = harness.shutdown.send(());
}

/// #783 acceptance, second half: declining the client context must not cost the
/// connection its authentication. This connection sends no token at all, so it
/// is admitted only by the kernel peer identity (#407) - and that same identity
/// still supplies no prompt grounding.
#[tokio::test]
async fn declining_the_client_context_keeps_peer_cred_authentication() {
    let mut harness = start_server(Arc::new(PeerCredOnlyAuth));
    let handshake = serde_json::json!({ "share_client_context": false });

    let section = client_context_block_for_one_turn(&mut harness, handshake).await;

    assert_eq!(
        section, None,
        "peer-cred authenticated the connection - the turn was served - and the same peer \
         identity must still not reach the prompt"
    );
    let _ = harness.shutdown.send(());
}

/// The guard against a gate that withholds everything (#782). The two cases
/// above both assert an ABSENCE, so a daemon that dropped every client context
/// unconditionally would satisfy them and break the feature. This one can only
/// pass when a reported context actually survives to the prompt, and survives
/// as the client's own values rather than anything the daemon substituted.
#[tokio::test]
async fn a_reported_context_reaches_the_prompt_as_the_clients_own_values() {
    let signing_key = "deadbeef".repeat(8);
    let mut harness = start_server(Arc::new(JwtOnlyAuth {
        signing_key: signing_key.clone(),
    }));
    let handshake = serde_json::json!({
        "jwt": mint_test_jwt(&signing_key, "desktop"),
        "client_context": {
            "real_name": "Ada Lovelace",
            "username": "ada",
            "home_dir": "/home/ada",
            "hostname": "analytical-engine",
            "timezone": "Europe/London",
            "os": "TestOS 9000",
        },
    });

    let section = client_context_block_for_one_turn(&mut harness, handshake)
        .await
        .expect("a reported context must render a client-context block");

    // The client's values, not the connecting process's. `ada` is nobody on this
    // machine, so a peer-cred substitution could not have produced it.
    assert!(section.contains("Ada Lovelace"), "section: {section}");
    assert!(section.contains("analytical-engine"), "section: {section}");
    assert!(section.contains("Europe/London"), "section: {section}");
    let _ = harness.shutdown.send(());
}

/// `share_client_context: true` is documented to behave exactly as an absent
/// field, and only the absent case was covered. This pins the explicit arm, so
/// the equivalence is enforced rather than merely stated - and so that a later
/// change making an absent field fail closed cannot silently take explicit
/// consent down with it.
#[tokio::test]
async fn an_explicit_consent_grounds_the_prompt_exactly_as_an_absent_field_does() {
    let signing_key = "deadbeef".repeat(8);
    let mut harness = start_server(Arc::new(JwtOnlyAuth {
        signing_key: signing_key.clone(),
    }));
    let handshake = serde_json::json!({
        "jwt": mint_test_jwt(&signing_key, "desktop"),
        "share_client_context": true,
    });

    let section = client_context_block_for_one_turn(&mut harness, handshake).await;

    assert!(
        section.is_some(),
        "an explicitly consenting client that reported no context must keep the #558 \
         peer-identity grounding, exactly as an undeclared one does"
    );
    let _ = harness.shutdown.send(());
}
