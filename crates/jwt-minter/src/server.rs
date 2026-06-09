//! UDS server loop.
//!
//! `serve` binds the socket (removing any stale file), then runs a select
//! loop accepting connections until `shutdown` resolves. On exit the socket
//! file is unlinked so the next launch starts fresh.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, anyhow};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use desktop_assistant_auth_jwt as auth_jwt;

use crate::config::MintConfig;
use crate::group::{self, GroupGate};
use crate::peer::{self, PeerIdentity};
use crate::request::handle_request;

/// Server-side options that aren't part of the JWT policy.
#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub socket_path: PathBuf,
    pub group_gate: Option<GroupGate>,
}

/// Per-connection wire limits (review finding DT-7): a request line is a
/// small JSON object, so anything beyond a few KiB is hostile or broken,
/// and a client that connects but never sends must not pin a connection
/// task forever.
#[derive(Debug, Clone, Copy)]
pub struct ConnectionLimits {
    /// Maximum accepted request-line length in bytes (newline included).
    pub max_line_bytes: usize,
    /// How long to wait for the request line before giving up.
    pub read_timeout: Duration,
}

impl Default for ConnectionLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: 8 * 1024,
            read_timeout: Duration::from_secs(5),
        }
    }
}

/// Source of accepted connections for the serve loop. Production passes the
/// bound `UnixListener`; tests inject accept errors to pin down the loop's
/// survival behaviour (review finding DT-8).
trait AcceptSource: Send {
    fn accept_stream(
        &mut self,
    ) -> impl std::future::Future<Output = std::io::Result<UnixStream>> + Send;
}

impl AcceptSource for UnixListener {
    async fn accept_stream(&mut self) -> std::io::Result<UnixStream> {
        self.accept().await.map(|(stream, _)| stream)
    }
}

/// Bind `socket_path` (removing any stale file first), accept until
/// `shutdown` resolves, then atomically unlink the socket file.
pub async fn serve(
    options: ServerOptions,
    config: MintConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let listener = bind_unix_listener(&options.socket_path)?;
    // Load the signing key once at startup — the daemon would do the
    // same. If it doesn't exist yet, generate it; the daemon will pick up
    // the same file on its next startup.
    let signing_key =
        auth_jwt::ensure_signing_key_at(&config.signing_key_path).with_context(|| {
            format!(
                "failed to load signing key from {}",
                config.signing_key_path.display()
            )
        })?;

    tracing::info!(
        socket = %options.socket_path.display(),
        group = ?options.group_gate.as_ref().map(|g| &g.name),
        "adelie-mint listening",
    );

    let result = serve_accept_loop(listener, &options, &config, &signing_key, shutdown).await;

    remove_socket(&options.socket_path);
    result
}

/// The accept loop, factored out of [`serve`] so its error handling is
/// unit-testable (DT-8). Consumes (and drops) the listener before returning
/// so the kernel side is closed prior to unlinking the socket file.
async fn serve_accept_loop<A: AcceptSource>(
    mut listener: A,
    options: &ServerOptions,
    config: &MintConfig,
    signing_key: &str,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            biased;
            () = &mut shutdown => {
                tracing::info!("shutdown requested");
                break Ok(());
            }
            accept = listener.accept_stream() => {
                match accept {
                    Ok(stream) => {
                        let cfg = config.clone();
                        let key = signing_key.to_string();
                        let gate = options.group_gate.clone();
                        tokio::spawn(async move {
                            if let Err(err) = serve_connection(
                                stream, &cfg, &key, gate.as_ref(),
                            ).await {
                                tracing::warn!(error = %err, "request failed");
                            }
                        });
                    }
                    Err(err) => {
                        tracing::error!(error = %err, "accept failed");
                        break Err(anyhow!("accept loop: {err}"));
                    }
                }
            }
        }
    }
}

/// Best-effort cleanup helper; safe to call multiple times.
pub fn remove_socket(path: &Path) {
    if path.exists()
        && let Err(err) = std::fs::remove_file(path)
    {
        tracing::warn!(
            path = %path.display(),
            error = %err,
            "failed to remove socket file"
        );
    }
}

/// Per-connection handler with default [`ConnectionLimits`]. Reads exactly
/// one JSON line, writes one JSON response, returns. Exposed so integration
/// tests can drive it directly.
pub async fn serve_connection(
    stream: UnixStream,
    config: &MintConfig,
    signing_key: &str,
    group_gate: Option<&GroupGate>,
) -> anyhow::Result<()> {
    serve_connection_with_limits(
        stream,
        config,
        signing_key,
        group_gate,
        ConnectionLimits::default(),
    )
    .await
}

/// Like [`serve_connection`] but with caller-supplied wire limits, so tests
/// can use short timeouts.
pub async fn serve_connection_with_limits(
    mut stream: UnixStream,
    config: &MintConfig,
    signing_key: &str,
    group_gate: Option<&GroupGate>,
    limits: ConnectionLimits,
) -> anyhow::Result<()> {
    let _ = limits;
    let identity =
        peer::extract_peer_identity(&stream).context("failed to extract peer credentials")?;

    if let Some(gate) = group_gate
        && !is_member(&identity, gate)?
    {
        let response = format!(
            r#"{{"error":"caller uid {} is not a member of group {}"}}"#,
            identity.uid, gate.name
        );
        stream.write_all(response.as_bytes()).await?;
        stream.write_all(b"\n").await?;
        stream.flush().await?;
        return Ok(());
    }

    let mut buf = String::new();
    let mut reader = BufReader::new(&mut stream);
    reader
        .read_line(&mut buf)
        .await
        .context("failed to read request line")?;
    drop(reader);

    let response = handle_request(&buf, &identity, config, signing_key);
    stream.write_all(response.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

fn is_member(identity: &PeerIdentity, gate: &GroupGate) -> anyhow::Result<bool> {
    let primary = group::primary_gid_for_uid(identity.uid)?
        .ok_or_else(|| anyhow!("no primary GID for uid {}", identity.uid))?;
    let groups = group::grouplist_for(&identity.username, primary)?;
    Ok(group::uid_in_groups(gate.gid, &groups))
}

/// Public binder used by both `serve` and tests. Removes a stale socket
/// file at `path` then binds.
pub fn bind_unix_listener(path: &Path) -> anyhow::Result<UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket parent dir {}", parent.display()))?;
    }
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove stale socket {}", path.display()))?;
    }
    UnixListener::bind(path).with_context(|| format!("failed to bind {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Accept source that replays a scripted sequence, then pends forever.
    struct ScriptedAccept {
        script: Mutex<VecDeque<std::io::Result<UnixStream>>>,
        calls: std::sync::Arc<AtomicUsize>,
    }

    impl AcceptSource for ScriptedAccept {
        async fn accept_stream(&mut self) -> std::io::Result<UnixStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let next = self.script.lock().unwrap().pop_front();
            match next {
                Some(item) => item,
                None => std::future::pending().await,
            }
        }
    }

    fn test_options() -> ServerOptions {
        ServerOptions {
            socket_path: PathBuf::from("/nonexistent/test.sock"),
            group_gate: None,
        }
    }

    fn test_config(dir: &tempfile::TempDir) -> (MintConfig, String) {
        let key_path = dir.path().join("key");
        let signing_key = auth_jwt::ensure_signing_key_at(&key_path).expect("ensure key");
        let mut cfg = MintConfig::with_default_paths();
        cfg.signing_key_path = key_path;
        (cfg, signing_key)
    }

    /// DT-8: a transient accept error must not stop the minter. The loop
    /// must log, keep accepting, and still serve the next connection.
    #[tokio::test]
    async fn accept_error_does_not_stop_the_minter() {
        let dir = tempfile::TempDir::new().unwrap();
        let (cfg, key) = test_config(&dir);

        // One transient error, then a real connection (the kernel half of a
        // socketpair), then pending.
        let (server_side, mut client_side) = UnixStream::pair().unwrap();
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let listener = ScriptedAccept {
            script: Mutex::new(VecDeque::from([
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionAborted,
                    "transient accept failure",
                )),
                Ok(server_side),
            ])),
            calls: std::sync::Arc::clone(&calls),
        };

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let options = test_options();
        let loop_task = tokio::spawn(async move {
            serve_accept_loop(listener, &options, &cfg, &key, async move {
                let _ = shutdown_rx.await;
            })
            .await
        });

        // Drive a request over the post-error connection: if the loop died
        // on the accept error this write is never answered.
        client_side.write_all(b"{}\n").await.unwrap();
        let mut response = String::new();
        let mut reader = BufReader::new(&mut client_side);
        let read = tokio::time::timeout(Duration::from_secs(2), reader.read_line(&mut response))
            .await
            .expect("minter must keep serving after a transient accept error")
            .expect("read response");
        assert!(read > 0, "expected a response line");
        assert!(
            response.contains("token") || response.contains("error"),
            "expected a mint response, got: {response}"
        );
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "the loop must call accept again after a transient error"
        );

        assert!(!loop_task.is_finished(), "loop must still be running");
        let _ = shutdown_tx.send(());
        let result = tokio::time::timeout(Duration::from_secs(2), loop_task)
            .await
            .expect("loop must exit on shutdown")
            .expect("loop must not panic");
        assert!(result.is_ok(), "shutdown exit must be Ok: {result:?}");
    }

    /// DT-7: a request line longer than the cap is rejected promptly —
    /// without waiting for a newline or EOF — so a hostile client can't make
    /// the minter buffer unbounded memory.
    #[tokio::test]
    async fn oversize_request_line_is_rejected_without_buffering() {
        let dir = tempfile::TempDir::new().unwrap();
        let (cfg, key) = test_config(&dir);
        let (server_side, mut client_side) = UnixStream::pair().unwrap();

        let limits = ConnectionLimits {
            max_line_bytes: 1024,
            read_timeout: Duration::from_secs(5),
        };
        let server = tokio::spawn(async move {
            serve_connection_with_limits(server_side, &cfg, &key, None, limits).await
        });

        // 4x the cap, no newline, and we do NOT close our end — only the cap
        // can end the read.
        let big = vec![b'x'; 4096];
        client_side.write_all(&big).await.unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .expect("server must reject an oversize line promptly, not wait for newline/EOF")
            .expect("server task must not panic");
        assert!(
            result.is_err(),
            "an oversize request line must be an error, got {result:?}"
        );
    }

    /// DT-7: a client that connects and never sends must not pin the
    /// connection task forever.
    #[tokio::test]
    async fn request_read_times_out_when_client_sends_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let (cfg, key) = test_config(&dir);
        let (server_side, client_side) = UnixStream::pair().unwrap();

        let limits = ConnectionLimits {
            max_line_bytes: 8 * 1024,
            read_timeout: Duration::from_millis(200),
        };
        let server = tokio::spawn(async move {
            serve_connection_with_limits(server_side, &cfg, &key, None, limits).await
        });

        let result = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .expect("connection must end after the read timeout")
            .expect("server task must not panic");
        assert!(
            result.is_err(),
            "a silent client must produce a timeout error, got {result:?}"
        );
        drop(client_side);
    }
}
