//! Client for the local `adelie-mint` UDS.
//!
//! `adelie-mint` (issue #101) accepts one line of JSON
//! (`{"ttl_seconds": ..., "audience": ...}` — both fields optional;
//! `{}` is fine) and replies with one line of JSON
//! (`{"token": "...", "exp": ...}` or `{"error": "..."}`). The minter
//! authenticates the caller via `SO_PEERCRED`, so the bridge only has
//! to talk to it as the desktop user that ran `systemctl --user enable
//! adelie-dbus.service`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Default UDS path the minter binds. Mirrors the convention in
/// `crates/jwt-minter/src/main.rs::default_socket_path`.
///
/// Returns `None` when `XDG_RUNTIME_DIR` is unset; callers should
/// supply an explicit path in that case (e.g. via CLI flag).
pub fn default_minter_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(|p| PathBuf::from(p).join("adelie").join("mint.sock"))
}

/// Request payload — both fields optional. The minter clamps `ttl` and
/// fills in defaults when fields are omitted.
#[derive(Debug, Default, Clone, Serialize)]
pub struct MintRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
}

/// Response payload mirroring the minter's reply shape. Either `token`
/// is present (success) or `error` is (failure).
#[derive(Debug, Deserialize)]
pub struct MintResponse {
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub exp: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Fetch a JWT from the minter at `socket_path`.
///
/// Returns the token string on success. On any failure — socket
/// missing/unreachable, minter responded with an error, malformed
/// response, timeout — returns a descriptive `anyhow::Error` so the
/// caller (the bridge `main`) can log a clear "your minter is down,
/// run `systemctl --user start adelie-mint`" message and exit
/// non-zero rather than spin in a reconnect loop.
pub async fn fetch_jwt(
    socket_path: &Path,
    request: MintRequest,
    timeout: Duration,
) -> anyhow::Result<String> {
    let connect = UnixStream::connect(socket_path);
    let stream = tokio::time::timeout(timeout, connect)
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "timed out connecting to minter at {} after {timeout:?}",
                socket_path.display()
            )
        })?
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to connect to minter at {}: {e}",
                socket_path.display()
            )
        })?;

    let body = serde_json::to_string(&request)
        .map_err(|e| anyhow::anyhow!("failed to serialize mint request: {e}"))?;

    let (read_half, mut write_half) = stream.into_split();

    // Minter expects one line of JSON per request.
    write_half
        .write_all(body.as_bytes())
        .await
        .map_err(|e| anyhow::anyhow!("failed to write mint request: {e}"))?;
    write_half
        .write_all(b"\n")
        .await
        .map_err(|e| anyhow::anyhow!("failed to write mint request newline: {e}"))?;
    write_half
        .flush()
        .await
        .map_err(|e| anyhow::anyhow!("failed to flush mint request: {e}"))?;

    // Cap the reply so a misbehaving or compromised minter can't
    // exhaust memory. A real reply is ~250 bytes; 16 KiB is generous.
    const MAX_MINTER_REPLY: usize = 16 * 1024;
    let mut reader = BufReader::new(read_half).take(MAX_MINTER_REPLY as u64);
    let mut line = String::new();
    let read_fut = reader.read_line(&mut line);
    tokio::time::timeout(timeout, read_fut)
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for minter reply"))?
        .map_err(|e| anyhow::anyhow!("failed to read minter reply: {e}"))?;

    if line.trim().is_empty() {
        return Err(anyhow::anyhow!("minter closed connection without replying"));
    }

    let response: MintResponse = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("malformed minter reply {:?}: {e}", line.trim()))?;

    if let Some(error) = response.error {
        return Err(anyhow::anyhow!("minter rejected request: {error}"));
    }

    response
        .token
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("minter reply contained neither token nor error"))
}
