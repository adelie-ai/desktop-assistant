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
use desktop_assistant_application::AssistantApiHandler;
use desktop_assistant_transport_dispatch::{AuthContext, dispatch_loop};
use futures::stream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::sync::PollSender;
use tracing::{debug, info, warn};

pub use api::{WsFrame, WsRequest};

/// Default desktop path. The system-service flavor (`/run/adelie/sock`)
/// is the daemon's responsibility — it picks the right default based
/// on its deployment mode.
pub fn default_desktop_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("sock"))
}

/// Result of the JWT handshake.
///
/// The validator only owns the bool/claims decision; the listener
/// owns the wire framing. This means the validator can be implemented
/// against any JWT library / claim shape without dragging
/// `auth-jwt` into this crate's public API.
#[async_trait::async_trait]
pub trait UdsAuthValidator: Send + Sync {
    /// Validate a bearer token. Returning `true` enters the dispatcher
    /// loop; returning `false` causes the listener to write an error
    /// frame and close.
    async fn validate_bearer_token(&self, token: &str) -> bool;

    /// Extract the user id ([JWT `sub`]) from a bearer token that
    /// [`Self::validate_bearer_token`] already accepted (#105).
    ///
    /// Default returns `None`, which collapses the connection to the
    /// schema sentinel `UserId::default()`. Single-tenant desktop
    /// installs that don't care about identity can keep the default;
    /// multi-tenant or multi-user-host deploys override this method
    /// to return the JWT subject so storage queries scope per-user.
    async fn extract_user_id(
        &self,
        token: &str,
    ) -> Option<desktop_assistant_application::UserId> {
        let _ = token;
        None
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
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
        + Send
        + Sync,
{
    validate: F,
}

impl<F> JwtUdsAuth<F>
where
    F: for<'a> Fn(
            &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
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
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>>
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
}

impl UdsServerConfig {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
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
        if let Some(parent) = self.config.socket_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!(
                    "failed to create uds parent dir {}: {e}",
                    parent.display()
                )
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(
                    parent,
                    std::fs::Permissions::from_mode(0o700),
                );
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
        // group to connect should adjust this in a follow-up.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(
                &self.config.socket_path,
                std::fs::Permissions::from_mode(0o600),
            );
        }

        info!(
            path = %self.config.socket_path.display(),
            "uds listener bound"
        );

        let handler = Arc::clone(&self.handler);
        let auth = Arc::clone(&self.auth);
        let accept_loop = async {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let handler = Arc::clone(&handler);
                        let auth = Arc::clone(&auth);
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, handler, auth).await {
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

/// Per-connection lifecycle: handshake + dispatcher loop.
async fn handle_connection(
    stream: UnixStream,
    handler: Arc<dyn AssistantApiHandler>,
    auth: Arc<dyn UdsAuthValidator>,
) -> anyhow::Result<()> {
    let (mut read_half, mut write_half) = stream.into_split();

    // Handshake: first frame must be `{"jwt": "<token>"}`. Anything
    // else (including non-JSON, an empty body, or a missing/blank
    // field) is rejected with an explicit error frame so clients can
    // tell auth from framing problems.
    let handshake_raw = read_frame(&mut read_half).await?;
    let handshake_json: serde_json::Value = match serde_json::from_slice(&handshake_raw) {
        Ok(v) => v,
        Err(e) => {
            // Surface the parse error to the client before closing so a
            // misconfigured client doesn't silently hang.
            let _ = write_error(
                &mut write_half,
                "",
                &format!("invalid handshake json: {e}"),
            )
            .await;
            return Ok(());
        }
    };

    let token = handshake_json
        .get("jwt")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let token = match token {
        Some(t) => t,
        None => {
            write_error(&mut write_half, "", "auth: missing jwt in handshake").await?;
            return Ok(());
        }
    };

    if !auth.validate_bearer_token(&token).await {
        write_error(&mut write_half, "", "auth: invalid jwt").await?;
        return Ok(());
    }

    // Identity (#105): the validator either returns the `sub` (multi-
    // tenant deploys) or `None` (single-tenant fallback, mapped to
    // the schema sentinel). The dispatcher installs this into the
    // per-task task-local before each command runs.
    let user_id = auth.extract_user_id(&token).await.unwrap_or_default();

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
    // command so storage queries scope to the right partition.
    let auth_ctx = AuthContext::new(user_id.into_inner());

    dispatch_loop(handler, auth_ctx, inbound, sink).await;

    // Dispatcher returned (client disconnected, parse error, or
    // outbound closed). Tear everything down.
    drop(outbound_tx);
    writer_task.abort();
    reader_task.abort();

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


/// Read one length-prefixed frame.
///
/// The header is a 4-byte little-endian `u32` length. We `read_exact`
/// the header, allocate the body, then `read_exact` the body. The
/// 4 MB cap (`MAX_FRAME_LEN`) keeps a hostile client from triggering
/// an OOM by claiming a multi-GB length.
pub async fn read_frame<R>(read_half: &mut R) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    const MAX_FRAME_LEN: u32 = 4 * 1024 * 1024;

    let mut len_buf = [0u8; 4];
    read_half.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    if len > 0 {
        read_half.read_exact(&mut body).await?;
    }
    Ok(body)
}

/// Write one length-prefixed frame.
pub async fn write_frame<W>(write_half: &mut W, body: &[u8]) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let len = body.len() as u32;
    write_half.write_all(&len.to_le_bytes()).await?;
    write_half.write_all(body).await?;
    write_half.flush().await?;
    Ok(())
}

