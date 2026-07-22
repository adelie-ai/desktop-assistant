//! Unix-domain-socket frontend for the assistant API.
//!
//! Local clients (D-Bus bridge, CLI, future minter shim) talk to the
//! daemon over a single UDS path that speaks the same JSON wire format
//! as the WebSocket adapter. Per `docs/architecture-evolution.md` rule
//! #5, the transport is pluggable and the daemon does **not** auth
//! local clients by peer-cred — every connection still presents a JWT,
//! issued by the local minter (#100 / `auth-jwt`).
//!
//! ## Framing
//!
//! UDS is a byte stream, not message-oriented, so each frame is sent
//! as a 4-byte little-endian `u32` length prefix followed by the JSON
//! body. The first frame on every connection is a handshake of the
//! shape `{"jwt": "<token>"}`. If the token validates we enter the
//! shared [`desktop_assistant_transport_dispatch::dispatch_loop`];
//! otherwise we write a `WsFrame::Error` describing the failure and
//! close the socket.
//!
//! ## Auth
//!
//! Authentication is delegated to a [`UdsAuthValidator`] trait so the
//! daemon can plug in whichever JWT validator it already uses (local
//! HS256, OIDC RS256, or both). This crate stays policy-free and
//! ships a thin [`JwtUdsAuth`] convenience that wraps any
//! `async`-callable validator.
//!
//! ## Lifecycle
//!
//! [`UdsServer::serve_with_shutdown`] binds the socket (unlinking any
//! stale file at the same path), serves connections until the
//! provided shutdown future resolves, and removes the socket file on
//! exit. Both behaviors are covered by tests in
//! `tests/uds.rs`.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{AssistantApiHandler, UserId};
use desktop_assistant_peer_cred::extract_peer_identity;
use desktop_assistant_transport_dispatch::{AuthContext, TransportKind, dispatch_loop};
use futures::stream;
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio_util::sync::PollSender;
use tracing::{debug, info, warn};

pub use api::{WsFrame, WsRequest};
pub use desktop_assistant_peer_cred::PeerIdentity;

/// Default desktop path. The system-service flavor (`/run/adelie/sock`)
/// is the daemon's responsibility — it picks the right default based
/// on its deployment mode.
pub fn default_desktop_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("sock"))
}

/// Outcome of authenticating a UDS connection (#407).
///
/// The listener owns the wire framing; the validator only renders a verdict.
pub enum UdsAuth {
    /// Authenticated as this user — enter the dispatcher loop.
    Allow(UserId),
    /// Rejected — the listener writes `reason` as an error frame and closes.
    Reject(String),
}

/// Result of the connection handshake.
///
/// The validator only owns the auth decision; the listener owns the wire
/// framing. This means the validator can be implemented against any JWT
/// library / claim shape without dragging `auth-jwt` into this crate's public
/// API.
///
/// Authentication has two inputs: an optional handshake bearer token and the
/// kernel-attested peer credentials of the connecting process. The default
/// policy is **token-only** (back-compat with the uniform-JWT model): a valid
/// token is required and peer-cred is ignored. A daemon that trusts local peers
/// (issue #407) overrides [`Self::authenticate`] to accept the peer identity
/// without a token.
#[async_trait::async_trait]
pub trait UdsAuthValidator: Send + Sync {
    /// Validate a bearer token. Returning `true` accepts it; `false` rejects.
    async fn validate_bearer_token(&self, token: &str) -> bool;

    /// Extract the user id ([JWT `sub`]) from a bearer token that
    /// [`Self::validate_bearer_token`] already accepted (#105).
    ///
    /// Default returns `None`, which collapses the connection to the
    /// schema sentinel `UserId::default()`. Single-tenant desktop
    /// installs that don't care about identity can keep the default;
    /// multi-tenant or multi-user-host deploys override this method
    /// to return the JWT subject so storage queries scope per-user.
    async fn extract_user_id(&self, token: &str) -> Option<UserId> {
        let _ = token;
        None
    }

    /// Authenticate a connection from its handshake `token` (if any) and the
    /// kernel-attested `peer` credentials (if the OS provided them).
    ///
    /// The default implementation is the historical token-only policy: require
    /// a valid bearer token and ignore peer credentials. Local-trust daemons
    /// override this to authenticate by `peer` instead (#407).
    async fn authenticate(&self, token: Option<&str>, peer: Option<&PeerIdentity>) -> UdsAuth {
        let _ = peer;
        match token {
            Some(t) if self.validate_bearer_token(t).await => {
                UdsAuth::Allow(self.extract_user_id(t).await.unwrap_or_default())
            }
            Some(_) => UdsAuth::Reject("auth: invalid jwt".to_string()),
            None => UdsAuth::Reject("auth: missing jwt in handshake".to_string()),
        }
    }
}

/// Convenience JWT validator: holds the signing key + expected
/// issuer/audience and decodes via `auth-jwt`. Daemons that already
/// have an OIDC-aware validator can implement [`UdsAuthValidator`]
/// directly and skip this.
pub struct JwtUdsAuth<F>
where
    F: for<'a> Fn(
            &'a str,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
        + Send
        + Sync,
{
    validate: F,
}

impl<F> JwtUdsAuth<F>
where
    F: for<'a> Fn(
            &'a str,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
        + Send
        + Sync,
{
    pub fn new(validate: F) -> Self {
        Self { validate }
    }
}

#[async_trait::async_trait]
impl<F> UdsAuthValidator for JwtUdsAuth<F>
where
    F: for<'a> Fn(
            &'a str,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
        + Send
        + Sync,
{
    async fn validate_bearer_token(&self, token: &str) -> bool {
        (self.validate)(token).await
    }
}

/// Configuration for the UDS listener.
#[derive(Debug, Clone)]
pub struct UdsServerConfig {
    /// Filesystem path the listener binds. Existing files at this
    /// path are unlinked before bind.
    pub socket_path: PathBuf,
    /// The daemon's own per-machine system id (#248), read once at startup.
    /// Compared against each client's reported id in the handshake to decide
    /// co-location exactly. `None` ⇒ the daemon couldn't resolve its own id, so
    /// co-location falls back to the transport heuristic (UDS ⇒ co-located).
    pub daemon_system_id: Option<String>,
    /// How long a freshly-accepted connection may take to present its JWT
    /// handshake frame before the server closes it (review finding DT-7).
    /// Without this a client that connects and sends nothing pins a
    /// connection task forever.
    pub handshake_timeout: std::time::Duration,
}

impl UdsServerConfig {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            daemon_system_id: None,
            handshake_timeout: std::time::Duration::from_secs(10),
        }
    }

    /// Set the daemon's own system id for the #248 co-location handshake.
    pub fn with_daemon_system_id(mut self, id: Option<String>) -> Self {
        self.daemon_system_id = id;
        self
    }

    /// Override the handshake timeout (DT-7); mainly for tests.
    pub fn with_handshake_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.handshake_timeout = timeout;
        self
    }
}

/// UDS frontend server.
///
/// Construct with [`UdsServer::new`], then call
/// [`UdsServer::serve_with_shutdown`] to bind + serve. The server
/// removes the socket file when the shutdown future resolves so the
/// daemon can restart cleanly on systemd `Restart=on-failure`.
pub struct UdsServer {
    handler: Arc<dyn AssistantApiHandler>,
    auth: Arc<dyn UdsAuthValidator>,
    config: UdsServerConfig,
}

impl UdsServer {
    pub fn new(
        handler: Arc<dyn AssistantApiHandler>,
        auth: Arc<dyn UdsAuthValidator>,
        config: UdsServerConfig,
    ) -> Self {
        Self {
            handler,
            auth,
            config,
        }
    }

    /// Bind the socket and serve until `shutdown` resolves. The
    /// socket file is removed on exit (graceful or otherwise) so a
    /// restart can re-bind without an `EADDRINUSE` recovery dance.
    ///
    /// Bind behavior: any existing file at `socket_path` is unlinked
    /// first. This is the friendlier choice for desktop installs —
    /// after an unclean shutdown a stale socket file is the most
    /// common reason a daemon fails to come up, and "refuse to start"
    /// puts the recovery burden on operators every time. Multi-tenant
    /// deployments that need a hard refuse-to-overwrite policy can
    /// add a config flag later (separate issue).
    pub async fn serve_with_shutdown<F>(&self, shutdown: F) -> anyhow::Result<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        // Create the parent directory if missing. Tighten to 0700 on
        // Unix so other users on the host can't even stat the socket.
        //
        // DT-13: the 0700 parent is enforced *before* bind and chmod
        // failures are fatal, not best-effort. The freshly-bound socket
        // file briefly carries umask-derived perms, but with the parent
        // already 0700 no other user can traverse to it, so there is no
        // pre-chmod connect window.
        if let Some(parent) = self.config.socket_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!("failed to create uds parent dir {}: {e}", parent.display())
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).map_err(
                    |e| {
                        anyhow::anyhow!(
                            "failed to tighten uds parent dir {} to 0700: {e}",
                            parent.display()
                        )
                    },
                )?;
            }
        }

        // Unlink any stale file at the socket path; ignore "doesn't
        // exist" but propagate other errors so an operator who points
        // the daemon at `/etc/passwd` finds out immediately.
        match std::fs::remove_file(&self.config.socket_path) {
            Ok(()) => debug!(
                path = %self.config.socket_path.display(),
                "removed existing socket file before bind"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "failed to remove existing uds path {}: {e}",
                    self.config.socket_path.display()
                ));
            }
        }

        let listener = UnixListener::bind(&self.config.socket_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to bind uds {}: {e}",
                self.config.socket_path.display()
            )
        })?;

        // Tighten the socket file's perms to 0600 so only the daemon's
        // own user can connect. Multi-user hosts that want the minter
        // group to connect should adjust this in a follow-up. Failure is
        // fatal (DT-13): silently serving a group/world-connectable
        // socket is worse than refusing to start.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                &self.config.socket_path,
                std::fs::Permissions::from_mode(0o600),
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to tighten uds socket {} to 0600: {e}",
                    self.config.socket_path.display()
                )
            })?;
        }

        info!(
            path = %self.config.socket_path.display(),
            "uds listener bound"
        );

        let handler = Arc::clone(&self.handler);
        let auth = Arc::clone(&self.auth);
        // The daemon's own system id (#248), shared by every connection's
        // co-location comparison. `Arc` so the spawn per connection is cheap.
        let daemon_system_id = Arc::new(self.config.daemon_system_id.clone());
        let accept_loop = async {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let handler = Arc::clone(&handler);
                        let auth = Arc::clone(&auth);
                        let daemon_system_id = Arc::clone(&daemon_system_id);
                        let handshake_timeout = self.config.handshake_timeout;
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(
                                stream,
                                handler,
                                auth,
                                daemon_system_id,
                                handshake_timeout,
                            )
                            .await
                            {
                                debug!("uds connection ended: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        warn!("uds accept error: {e}");
                        // Avoid spinning hot on a transient error.
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
        };

        tokio::select! {
            _ = accept_loop => {}
            _ = shutdown => {
                info!(
                    path = %self.config.socket_path.display(),
                    "uds listener shutting down"
                );
            }
        }

        // Remove the socket file so a restart can re-bind cleanly.
        // Best-effort: a missing file is fine, anything else is a
        // diagnostic.
        if let Err(e) = std::fs::remove_file(&self.config.socket_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            warn!(
                path = %self.config.socket_path.display(),
                "failed to remove uds socket on shutdown: {e}"
            );
        }

        Ok(())
    }
}

/// Resolve the client context to attach for a UDS connection (#549/#558).
///
/// A client that reported a non-empty context keeps it verbatim. Otherwise the
/// local, co-located client sent none (e.g. the KDE FFI client, which cannot
/// build one): fall back to the kernel-attested peer identity so the prompt is
/// still grounded with the connecting user's real name / login / home. The
/// fallback fills ONLY those user-identity fields — peer-cred does not attest a
/// device hostname / OS / timezone, so those stay absent rather than borrowing
/// the daemon host's. Returns `None` when nothing at all is known.
fn resolve_local_client_context(
    reported: Option<api::ClientContext>,
    peer: Option<&PeerIdentity>,
) -> Option<api::ClientContext> {
    // Spec stub — the peer-cred fallback lands in the implementation commit.
    let _ = peer;
    reported.filter(|c| !c.is_empty())
}

/// Per-connection lifecycle: handshake + dispatcher loop.
async fn handle_connection(
    stream: UnixStream,
    handler: Arc<dyn AssistantApiHandler>,
    auth: Arc<dyn UdsAuthValidator>,
    daemon_system_id: Arc<Option<String>>,
    handshake_timeout: std::time::Duration,
) -> anyhow::Result<()> {
    // Read the kernel-attested peer identity before splitting the stream
    // (`peer_cred` is a `UnixStream` method). On local transports this is the
    // authentication (#407); `None` if the OS couldn't supply it (the auth
    // policy then falls back to the bearer token, if any).
    let peer = extract_peer_identity(&stream).ok();

    let (mut read_half, mut write_half) = stream.into_split();

    // Handshake: first frame may carry a JWT plus, optionally, the client's
    // per-machine system id + host label for co-location (#248). Anything that
    // isn't valid JSON is rejected with an explicit error frame so clients can
    // tell auth from framing problems. The `jwt` field is optional — a
    // local-trust daemon authenticates by peer-cred (above) and needs no token;
    // an older client sends the bare `{"jwt": "<token>"}`, which still parses.
    //
    // The read is bounded by `handshake_timeout` (DT-7): a client that
    // connects and sends nothing must not pin a connection task forever.
    let handshake_raw =
        match tokio::time::timeout(handshake_timeout, read_frame(&mut read_half)).await {
            Ok(result) => result?,
            Err(_) => {
                let _ = write_error(&mut write_half, "", "handshake timed out").await;
                return Ok(());
            }
        };
    let handshake: api::UdsHandshake = match serde_json::from_slice(&handshake_raw) {
        Ok(v) => v,
        Err(e) => {
            // Surface the parse error to the client before closing so a
            // misconfigured client doesn't silently hang.
            let _ = write_error(&mut write_half, "", &format!("invalid handshake json: {e}")).await;
            return Ok(());
        }
    };

    let token = handshake
        .jwt
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // System-id co-location (#248): compare the client's reported id to the
    // daemon's own. `None` (older client / unresolved daemon id) defers to the
    // transport heuristic. The id is a routing HINT, not a trust boundary — it
    // is self-reported and no privilege is gated on it (auth is below).
    let co_located = desktop_assistant_core::system_id::co_location_from_ids(
        daemon_system_id.as_deref(),
        handshake.system_id.as_deref(),
    );
    let client_label = handshake.host_label;
    // The self-reported client context (#549) rides the handshake alongside the
    // #248 system-id fields. Best-effort display data, not a trust boundary. When
    // a local client sent none (e.g. the KDE FFI client), fall back to the kernel
    // peer identity so the user's name / login / home still ground the prompt
    // (#558).
    let client_context = resolve_local_client_context(handshake.client_context, peer.as_ref());

    // Authenticate from the (optional) token and the kernel peer-cred. The
    // default validator requires a valid token; a local-trust daemon (#407)
    // accepts the peer identity. Identity (#105): the resolved `UserId` is
    // installed into the per-task task-local before each command runs.
    let user_id = match auth.authenticate(token.as_deref(), peer.as_ref()).await {
        UdsAuth::Allow(user_id) => user_id,
        UdsAuth::Reject(reason) => {
            write_error(&mut write_half, "", &reason).await?;
            return Ok(());
        }
    };

    // Auth passed; enter the shared dispatcher.
    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::channel::<anyhow::Result<WsRequest>>(16);
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::channel::<WsFrame>(64);

    // Reader: pulls length-prefixed frames off the socket and pushes
    // parsed requests into the inbound channel.
    let reader_task = tokio::spawn(async move {
        loop {
            let frame = match read_frame(&mut read_half).await {
                Ok(b) => b,
                Err(_) => break,
            };
            if frame.is_empty() {
                break;
            }
            match serde_json::from_slice::<WsRequest>(&frame) {
                Ok(req) => {
                    if inbound_tx.send(Ok(req)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if inbound_tx
                        .send(Err(anyhow::anyhow!("invalid request json: {e}")))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    // Writer: pulls outbound frames and writes them length-prefixed.
    let writer_task = tokio::spawn(async move {
        while let Some(frame) = outbound_rx.recv().await {
            let body = match serde_json::to_vec(&frame) {
                Ok(b) => b,
                Err(_) => continue,
            };
            if write_frame(&mut write_half, &body).await.is_err() {
                break;
            }
        }
    });

    // Bridge the inbound mpsc into a `Stream` and the outbound mpsc
    // into a `Sink<WsFrame>` so we can call into the shared dispatcher.
    let inbound = stream::unfold(inbound_rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    futures::pin_mut!(inbound);

    let sink = PollSender::new(outbound_tx.clone());

    // Per-connection identity resolved above (#105). The dispatcher
    // installs this into the `with_user_id` task-local around each
    // command so storage queries scope to the right partition. A UDS
    // connection is local, so the transport heuristic already treats its tools
    // as co-located (#243); when the client also reported a system id we attach
    // the authoritative match result + an optional host label (#248).
    let auth_ctx = AuthContext::new(user_id.into_inner(), TransportKind::Uds)
        .with_co_location(co_located)
        .with_client_label(client_label)
        .with_client_context(client_context);

    dispatch_loop(handler, auth_ctx, inbound, sink).await;

    // Dispatcher returned (client disconnected, parse error, or outbound
    // closed). Teardown (DT-6): replies still queued on the outbound
    // channel must reach a client that half-closed and is waiting for
    // them — the old `writer_task.abort()` could drop queued frames or
    // tear one mid-`write_all`. `dispatch_loop` has already awaited its
    // internal writer, so its `PollSender` clone of `outbound_tx` is
    // gone; dropping ours leaves the channel senderless and the writer
    // drains to completion. Only the reader is aborted. A bounded grace
    // period guards against a peer that stops reading entirely.
    reader_task.abort();
    drop(outbound_tx);
    let mut writer_task = writer_task;
    if tokio::time::timeout(std::time::Duration::from_secs(5), &mut writer_task)
        .await
        .is_err()
    {
        warn!("uds outbound drain timed out; aborting writer");
        writer_task.abort();
    }

    Ok(())
}

/// Write a single `WsFrame::Error` with the given id (or empty string
/// if this is a pre-dispatch handshake error) and flush.
async fn write_error<W>(write_half: &mut W, id: &str, error: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let frame = WsFrame::Error {
        id: id.to_string(),
        error: error.to_string(),
    };
    let body = serde_json::to_vec(&frame)?;
    write_frame(write_half, &body).await?;
    write_half.flush().await?;
    Ok(())
}

// The length-prefixed frame codec is shared with the D-Bus bridge and the
// clients via `desktop-assistant-frame-codec` so the 4 MB frame cap and the
// framing rules can never drift between transports (#279/#280). Re-exported
// here under the historical `read_frame`/`write_frame` names this crate's
// integration tests import.
pub use desktop_assistant_frame_codec::{read_frame, write_frame};

#[cfg(test)]
mod tests {
    use super::*;

    fn peer_with(real_name: Option<&str>, home_dir: Option<&str>) -> PeerIdentity {
        PeerIdentity {
            uid: 1000,
            username: "ada".to_string(),
            real_name: real_name.map(str::to_string),
            home_dir: home_dir.map(str::to_string),
        }
    }

    #[test]
    fn reported_client_context_is_kept_verbatim() {
        // #558: when the client reported a non-empty context the peer identity is
        // ignored — the client's self-report wins.
        let reported = api::ClientContext {
            real_name: Some("Reported Name".into()),
            timezone: Some("Europe/London".into()),
            ..api::ClientContext::default()
        };
        let resolved = resolve_local_client_context(
            Some(reported.clone()),
            Some(&peer_with(Some("Peer Name"), Some("/home/peer"))),
        );
        assert_eq!(resolved, Some(reported));
    }

    #[test]
    fn missing_client_context_falls_back_to_peer_identity() {
        // #558: a local client that sent no context (None) still grounds the
        // prompt with the peer's name / login / home from kernel peer-cred.
        let resolved = resolve_local_client_context(
            None,
            Some(&peer_with(Some("Ada Lovelace"), Some("/home/ada"))),
        )
        .expect("peer fallback should produce a context");
        assert_eq!(resolved.real_name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(resolved.username.as_deref(), Some("ada"));
        assert_eq!(resolved.home_dir.as_deref(), Some("/home/ada"));
        // peer-cred does not attest a device hostname / OS / timezone.
        assert_eq!(resolved.hostname, None);
        assert_eq!(resolved.os, None);
        assert_eq!(resolved.timezone, None);
    }

    #[test]
    fn empty_reported_context_falls_back_to_peer_identity() {
        // An all-absent reported context ({}) counts as "sent none", so the peer
        // fallback still applies.
        let resolved = resolve_local_client_context(
            Some(api::ClientContext::default()),
            Some(&peer_with(None, Some("/home/ada"))),
        )
        .expect("empty reported context should fall back to peer");
        assert_eq!(resolved.username.as_deref(), Some("ada"));
        assert_eq!(resolved.home_dir.as_deref(), Some("/home/ada"));
        assert_eq!(resolved.real_name, None);
    }

    #[test]
    fn peer_without_name_or_home_still_yields_username() {
        // The peer identity always carries a username, so even with no GECOS name
        // / home dir the fallback is a non-empty context (username only).
        let resolved = resolve_local_client_context(None, Some(&peer_with(None, None)))
            .expect("username-only peer still yields a context");
        assert_eq!(resolved.username.as_deref(), Some("ada"));
        assert_eq!(resolved.real_name, None);
        assert_eq!(resolved.home_dir, None);
    }

    #[test]
    fn no_context_and_no_peer_is_none() {
        // A remote connection (no peer-cred) that sent no context yields nothing
        // to attach.
        assert_eq!(resolve_local_client_context(None, None), None);
    }
}
