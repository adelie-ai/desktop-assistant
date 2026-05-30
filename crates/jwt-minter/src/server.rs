//! UDS server loop.
//!
//! `serve` binds the socket (removing any stale file), then runs a select
//! loop accepting connections until `shutdown` resolves. On exit — clean
//! or due to an accept error — the socket file is unlinked so the next
//! launch starts fresh.

use std::path::{Path, PathBuf};

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

    tokio::pin!(shutdown);
    let result = loop {
        tokio::select! {
            biased;
            () = &mut shutdown => {
                tracing::info!("shutdown requested");
                break Ok(());
            }
            accept = listener.accept() => {
                match accept {
                    Ok((stream, _)) => {
                        let cfg = config.clone();
                        let key = signing_key.clone();
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
    };

    // Drop the listener before unlinking so the kernel side is closed.
    drop(listener);
    remove_socket(&options.socket_path);
    result
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

/// Per-connection handler. Reads exactly one JSON line, writes one JSON
/// response, returns. Exposed so integration tests can drive it directly.
pub async fn serve_connection(
    mut stream: UnixStream,
    config: &MintConfig,
    signing_key: &str,
    group_gate: Option<&GroupGate>,
) -> anyhow::Result<()> {
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
