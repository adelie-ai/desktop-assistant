//! Hardening of the remote WebSocket door's login and identity paths
//! (#806, #807, #808).
//!
//! Two separate properties are pinned here, both of which the door got wrong:
//!
//! - **Identity is part of acceptance.** A bearer token the validator accepts
//!   but whose subject it cannot name must not upgrade. The old handler
//!   collapsed that case to the schema sentinel `"default"`, which is the
//!   primary user's whole data partition, and - since the authorization tier
//!   landed - a subject an operator can put on the administrator allowlist.
//! - **`POST /login` is rate limited.** Repeated failures from one source, or
//!   against one username, lock the door for a growing window, so the endpoint
//!   cannot be used as an unthrottled password oracle.

use std::net::SocketAddr;
use std::sync::Arc;

use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::connect_info::MockConnectInfo;
use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink, UserId};
use desktop_assistant_ws::{WsAuthValidator, WsFrame, WsLoginService, WsRequest, router_full};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tower::ServiceExt;

// --- test doubles ----------------------------------------------------------

/// The smallest handler the dispatcher will run: `Ping` answers, everything
/// else is unsupported. None of these tests exercise the command surface.
struct PingHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for PingHandler {
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
        Ok(())
    }
}

/// A validator that accepts every token and names the subject it is given.
/// `subject == None` reproduces the production OIDC shape this work closes: a
/// token that validates (the RS256 `Validation` only requires `exp`) but that
/// carries no `sub` claim.
struct SubjectAuth {
    subject: Option<&'static str>,
}

#[async_trait::async_trait]
impl WsAuthValidator for SubjectAuth {
    async fn validate_bearer_token(&self, _token: &str) -> bool {
        true
    }

    async fn extract_user_id(&self, _token: &str) -> Option<UserId> {
        self.subject.map(UserId::from)
    }
}

/// The `/login` door under test: one account, one password.
struct StaticLogin;

/// A door that counts how many requests reached the credential check, and stays
/// inside it long enough for the rest to race the budget. Standing in for the
/// real slow half: a PAM call parks for the whole of libpam's fail delay.
struct SlowCountingLogin {
    reached: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl WsLoginService for SlowCountingLogin {
    async fn authenticate_basic(&self, _username: &str, _password: &str) -> bool {
        self.reached.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        false
    }

    async fn issue_token_for_subject(&self, subject: &str) -> Result<String, String> {
        Ok(format!("jwt-for-{subject}"))
    }
}

#[async_trait::async_trait]
impl WsLoginService for StaticLogin {
    async fn authenticate_basic(&self, username: &str, password: &str) -> bool {
        username == "alice" && password == "s3cr3t"
    }

    async fn issue_token_for_subject(&self, subject: &str) -> Result<String, String> {
        Ok(format!("jwt-for-{subject}"))
    }
}

// --- helpers ---------------------------------------------------------------

fn basic_auth_header(username: &str, password: &str) -> String {
    let encoded = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        format!("{username}:{password}").as_bytes(),
    );
    format!("Basic {encoded}")
}

/// A `/login` router whose requests all appear to come from one source
/// address, so the per-source counter is exercised deterministically.
fn login_app() -> axum::Router {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    router_full(
        handler,
        Arc::new(SubjectAuth {
            subject: Some("alice"),
        }),
        Some(Arc::new(StaticLogin)),
        None,
        Vec::new(),
    )
    .layer(MockConnectInfo(SocketAddr::from(([192, 0, 2, 10], 4242))))
}

async fn post_login(
    app: &axum::Router,
    username: &str,
    password: &str,
) -> axum::http::Response<axum::body::Body> {
    app.clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/login")
                .header("Authorization", basic_auth_header(username, password))
                .body(axum::body::Body::empty())
                .expect("build the login request"),
        )
        .await
        .expect("the router must answer a login request")
}

/// Guess until the door refuses, and return that refusal. Bounded, so a
/// regression names the requirement instead of hanging the suite.
async fn lock_the_door(app: &axum::Router) -> axum::http::Response<axum::body::Body> {
    for _ in 0..50 {
        let response = post_login(app, "alice", "wrong").await;
        if response.status() == axum::http::StatusCode::TOO_MANY_REQUESTS {
            return response;
        }
    }
    panic!("the door answered 50 wrong passwords without ever refusing one");
}

/// The `Retry-After` a refusal carries, in whole seconds.
fn retry_after_seconds(response: &axum::http::Response<axum::body::Body>) -> u64 {
    response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .expect("a lockout must carry Retry-After")
        .to_str()
        .expect("Retry-After must be ASCII")
        .parse::<u64>()
        .expect("Retry-After must be whole seconds")
}

fn ws_request(url: &str, bearer: &str) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = url.into_client_request().expect("valid ws url");
    request.headers_mut().insert(
        tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
        format!("Bearer {bearer}").parse().expect("valid header"),
    );
    request
}

/// Serve `validator` on an ephemeral loopback port.
async fn spawn_ws(
    validator: Arc<dyn WsAuthValidator>,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let app = router_full(handler, validator, None, None, Vec::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral port");
    let addr = listener.local_addr().expect("read the bound address");
    let server = tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    (addr, server)
}

/// The upgrade's HTTP status, for a handshake that is expected to fail.
async fn upgrade_status(
    addr: SocketAddr,
    token: &str,
) -> tokio_tungstenite::tungstenite::http::StatusCode {
    let url = format!("ws://{addr}/ws");
    match tokio_tungstenite::connect_async(ws_request(&url, token)).await {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => response.status(),
        Err(other) => panic!("unexpected handshake error: {other:?}"),
        Ok(_) => panic!("the handshake was accepted, but it must be refused"),
    }
}

// --- #807: identity resolution is part of acceptance -----------------------

/// A token that authenticates but names no subject must not upgrade. Before
/// this, `extract_user_id(..).unwrap_or_default()` silently authorized it as
/// `UserId("default")` - the primary partition - with no log line.
#[tokio::test]
async fn ws_upgrade_is_refused_when_the_accepted_token_carries_no_subject() {
    let (addr, server) = spawn_ws(Arc::new(SubjectAuth { subject: None })).await;
    assert_eq!(
        upgrade_status(addr, "any-token").await,
        tokio_tungstenite::tungstenite::http::StatusCode::UNAUTHORIZED,
        "a token with no extractable subject must be refused, not collapsed to the sentinel"
    );
    server.abort();
}

/// A blank or whitespace-only subject is no subject. It would otherwise reach
/// storage as an empty `user_id` and the allowlist as an empty string.
#[tokio::test]
async fn ws_upgrade_is_refused_when_the_subject_is_blank() {
    let (addr, server) = spawn_ws(Arc::new(SubjectAuth {
        subject: Some("   "),
    }))
    .await;
    assert_eq!(
        upgrade_status(addr, "any-token").await,
        tokio_tungstenite::tungstenite::http::StatusCode::UNAUTHORIZED,
        "a blank subject must be refused"
    );
    server.abort();
}

/// The path that must keep working: a token whose subject the validator can
/// name upgrades and serves commands as before.
#[tokio::test]
async fn ws_upgrade_succeeds_when_the_token_names_a_subject() {
    let (addr, server) = spawn_ws(Arc::new(SubjectAuth {
        subject: Some("alice"),
    }))
    .await;

    let url = format!("ws://{addr}/ws");
    let (mut socket, _) = tokio_tungstenite::connect_async(ws_request(&url, "any-token"))
        .await
        .expect("a token naming a subject must upgrade");

    let request = WsRequest {
        id: "req-1".into(),
        command: api::Command::Ping,
    };
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::to_string(&request).expect("serialize").into(),
        ))
        .await
        .expect("send ping");

    let reply = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
        .await
        .expect("a reply must arrive")
        .expect("the socket must stay open")
        .expect("the frame must decode");
    let frame: WsFrame = serde_json::from_str(reply.to_text().expect("text frame"))
        .expect("the reply must be a WsFrame");
    assert!(
        matches!(frame, WsFrame::Result { .. }),
        "an authenticated subject must still get its result frame, got {frame:?}"
    );

    server.abort();
}

// --- #808: /login is rate limited ------------------------------------------

/// Repeated wrong passwords stop being answered. Before this the endpoint
/// answered at full request rate forever, which is what made it a usable
/// password oracle against a real OS account.
///
/// The exact budget is the throttle's business, so this asserts the shape: a
/// person who mistypes gets several tries, and a guessing loop is cut off long
/// before it is useful.
#[tokio::test]
async fn login_locks_out_after_repeated_failures_from_one_source() {
    let app = login_app();
    let mut answered = 0;
    let mut refused = false;
    for _ in 0..50 {
        match post_login(&app, "alice", "wrong").await.status() {
            axum::http::StatusCode::UNAUTHORIZED => answered += 1,
            axum::http::StatusCode::TOO_MANY_REQUESTS => {
                refused = true;
                break;
            }
            other => panic!("unexpected status {other} from /login"),
        }
    }

    assert!(
        refused,
        "guessing must stop being answered; {answered} guesses were all answered"
    );
    assert!(
        (3..=10).contains(&answered),
        "the budget must leave room for a mistyped password without answering a \
         guessing loop; {answered} guesses were answered"
    );
}

/// The budget has to hold when the requests arrive together, which is the case
/// it exists for: checking the lockout and then counting the failure afterwards
/// leaves the whole credential check as a window in which every concurrent
/// request passes a check against counters nothing has recorded yet. One budget
/// then buys as many guesses as an attacker can open sockets.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_guesses_cannot_outrun_the_budget() {
    let reached = Arc::new(AtomicUsize::new(0));
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let app = router_full(
        handler,
        Arc::new(SubjectAuth {
            subject: Some("alice"),
        }),
        Some(Arc::new(SlowCountingLogin {
            reached: Arc::clone(&reached),
        })),
        None,
        Vec::new(),
    )
    .layer(MockConnectInfo(SocketAddr::from(([192, 0, 2, 10], 4242))));

    let attempts = 50;
    let statuses = futures_util::future::join_all((0..attempts).map(|_| {
        let app = app.clone();
        async move { post_login(&app, "alice", "wrong").await.status() }
    }))
    .await;

    let checked = reached.load(Ordering::SeqCst);
    assert!(
        checked >= 1,
        "at least one request must have reached the credential check"
    );
    assert!(
        checked <= 10,
        "{checked} of {attempts} concurrent guesses reached the password check; the \
         budget must bound them however many arrive at once"
    );
    assert!(
        statuses.contains(&axum::http::StatusCode::TOO_MANY_REQUESTS),
        "the requests past the budget must be refused, not answered"
    );
}

/// A refused request costs the caller nothing, so polling a locked door cannot
/// lengthen the lockout. Without this a client that retries on a schedule turns
/// its own stale password into a permanent outage.
#[tokio::test]
async fn polling_a_locked_door_does_not_lengthen_the_lockout() {
    let app = login_app();
    lock_the_door(&app).await;

    let first = retry_after_seconds(&post_login(&app, "alice", "wrong").await);
    for _ in 0..25 {
        let _ = post_login(&app, "alice", "wrong").await;
    }
    let last = retry_after_seconds(&post_login(&app, "alice", "wrong").await);

    assert!(
        last <= first,
        "polling grew the wait from {first} s to {last} s"
    );
}

/// A refusal names a wait a client can sit out, not one that reads as an
/// outage. Guessing in a tight loop cannot make it grow, because a refused
/// request spends nothing.
///
/// This cannot observe the escalation itself - the wait only grows for a caller
/// that waits each lockout out, and this test's clock never moves. The unit
/// test `waiting_out_a_lockout_and_trying_again_climbs_to_the_ceiling_and_stops`
/// drives that with a clock it controls.
#[tokio::test]
async fn a_refusal_names_a_wait_a_client_can_sit_out() {
    let app = login_app();
    let mut longest = 0;
    for _ in 0..200 {
        let response = post_login(&app, "alice", "wrong").await;
        if response.status() == axum::http::StatusCode::TOO_MANY_REQUESTS {
            longest = longest.max(retry_after_seconds(&response));
        }
    }
    assert!(
        longest > 0,
        "the door must have refused at least once in 200 guesses"
    );
    assert!(
        longest <= 60,
        "a lockout of {longest} s is an outage, not a delay"
    );
}

/// A locked-out caller is told when to come back, so a legitimate client that
/// mistyped can recover without guessing.
#[tokio::test]
async fn login_lockout_reports_retry_after() {
    let app = login_app();
    let response = lock_the_door(&app).await;

    assert!(
        retry_after_seconds(&response) > 0,
        "Retry-After must name a real wait"
    );
}

/// The lockout must not strand the legitimate user: one success clears the
/// counter, so an ordinary mistyped password costs nothing lasting.
#[tokio::test]
async fn successful_login_clears_the_failure_count() {
    let app = login_app();
    for _ in 0..3 {
        let _ = post_login(&app, "alice", "wrong").await;
    }

    assert_eq!(
        post_login(&app, "alice", "s3cr3t").await.status(),
        axum::http::StatusCode::OK,
        "the correct password must still work while the budget is unspent"
    );

    for _ in 0..3 {
        assert_eq!(
            post_login(&app, "alice", "wrong").await.status(),
            axum::http::StatusCode::UNAUTHORIZED,
            "the counter must have been cleared by the success"
        );
    }
}

/// Counting per username only would let one source walk a user list at full
/// rate. The source address is its own counter, so failures against different
/// usernames still add up.
#[tokio::test]
async fn login_lockout_counts_failures_across_usernames_from_one_source() {
    let app = login_app();
    for _ in 0..3 {
        let _ = post_login(&app, "alice", "wrong").await;
    }
    for _ in 0..3 {
        let _ = post_login(&app, "bob", "wrong").await;
    }

    assert_eq!(
        post_login(&app, "alice", "s3cr3t").await.status(),
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        "failures from one source must add up even though no single \
         username reached the limit"
    );
}

/// An `Authorization` header the door cannot parse is refused, and - because
/// there is no credential in it to guess with - it is not counted against the
/// failure budget.
#[tokio::test]
async fn login_rejects_a_malformed_authorization_header() {
    let app = login_app();
    for _ in 0..20 {
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/login")
                    .header("Authorization", "Basic not-valid-base64!!")
                    .body(axum::body::Body::empty())
                    .expect("build the login request"),
            )
            .await
            .expect("the router must answer");
        assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    assert_eq!(
        post_login(&app, "alice", "s3cr3t").await.status(),
        axum::http::StatusCode::OK,
        "a malformed header carries no credential, so it must not spend the budget"
    );
}
