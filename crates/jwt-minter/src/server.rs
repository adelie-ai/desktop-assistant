//! UDS server loop.

use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::config::MintConfig;
use crate::group::GroupGate;
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
    let _ = (options, config, shutdown);
    Err(anyhow!("serve: not implemented"))
}

/// Best-effort cleanup helper; safe to call multiple times.
pub fn remove_socket(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Per-connection handler. Reads exactly one JSON line, writes one JSON
/// response, returns. Exposed so integration tests can drive it directly.
pub async fn serve_connection(
    mut stream: UnixStream,
    config: &MintConfig,
    signing_key: &str,
    group_gate: Option<&GroupGate>,
) -> anyhow::Result<()> {
    let identity = peer::extract_peer_identity(&stream)
        .context("failed to extract peer credentials")?;

    if let Some(gate) = group_gate
        && !is_member(&identity, gate)?
    {
        let response = r#"{"error":"caller is not a member of the required group"}"#;
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
    let primary = crate::group::primary_gid_for_uid(identity.uid)?
        .ok_or_else(|| anyhow!("no primary GID for uid {}", identity.uid))?;
    let groups = crate::group::grouplist_for(&identity.username, primary)?;
    Ok(crate::group::uid_in_groups(gate.gid, &groups))
}

/// Public binder used by both `serve` and tests. Removes a stale socket
/// file at `path` then binds.
pub fn bind_unix_listener(path: &Path) -> anyhow::Result<UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("failed to create socket parent dir {}", parent.display())
        })?;
    }
    if path.exists() {
        std::fs::remove_file(path).with_context(|| {
            format!("failed to remove stale socket {}", path.display())
        })?;
    }
    UnixListener::bind(path).with_context(|| format!("failed to bind {}", path.display()))
}
