//! Client for the local `adelie-mint` UDS (issue #101).
//!
//! `adelie-mint` accepts one line of JSON (`{"ttl_seconds": ..., "audience":
//! ...}` — both fields optional; `{}` is fine) and replies with one line of
//! JSON (`{"token": "...", "exp": ...}` or `{"error": "..."}`). It
//! authenticates the caller via `SO_PEERCRED`, so a local desktop client only
//! has to talk to it as the user that runs the daemon's user services.
//!
//! This lives in `client-common` so the [`Connector`](crate::Connector) can mint
//! a **fresh** bearer token on every (re)connect via
//! [`resolve_ws_bearer_token`](crate::auth::resolve_ws_bearer_token) when a
//! [`ConnectionConfig.minter_socket`](crate::ConnectionConfig::minter_socket) is
//! set — the local minter, not the (retiring) D-Bus `GenerateWsJwt`, is the JWT
//! source for socket transports (#281: JWT off D-Bus).
//!
//! Kept derive-free (hand-rolled `serde_json`) so `client-common` doesn't take
//! on a `serde` derive dependency for two tiny payloads.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Default UDS path the minter binds. Mirrors `jwt-minter`'s
/// `default_socket_path`. Returns `None` when `XDG_RUNTIME_DIR` is unset;
/// callers should supply an explicit path in that case.
pub fn default_minter_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("mint.sock"))
}

/// Request payload — both fields optional. The minter clamps `ttl` and fills in
/// defaults when fields are omitted.
#[derive(Debug, Default, Clone)]
pub struct MintRequest {
    pub ttl_seconds: Option<u64>,
    pub audience: Option<String>,
}

impl MintRequest {
    /// One line of JSON with only the present fields (so `{}` when both absent,
    /// byte-identical to what the previous derive-based client sent).
    fn to_json_line(&self) -> String {
        let mut obj = serde_json::Map::new();
        if let Some(ttl) = self.ttl_seconds {
            obj.insert("ttl_seconds".to_string(), ttl.into());
        }
        if let Some(audience) = &self.audience {
            obj.insert("audience".to_string(), audience.clone().into());
        }
        serde_json::Value::Object(obj).to_string()
    }
}

/// Fetch a JWT from the minter at `socket_path`.
///
/// Returns the token string on success. On any failure — socket
/// missing/unreachable, minter error, malformed response, timeout — returns a
/// descriptive `anyhow::Error` so the caller can surface a clear "your minter is
/// down" message.
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

    let (read_half, mut write_half) = stream.into_split();

    // Minter expects one line of JSON per request.
    write_half
        .write_all(request.to_json_line().as_bytes())
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

    // Cap the reply so a misbehaving or compromised minter can't exhaust
    // memory. A real reply is ~250 bytes; 16 KiB is generous.
    const MAX_MINTER_REPLY: usize = 16 * 1024;
    let mut reader = BufReader::new(read_half).take(MAX_MINTER_REPLY as u64);
    let mut line = String::new();
    let read_fut = reader.read_line(&mut line);
    tokio::time::timeout(timeout, read_fut)
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for minter reply"))?
        .map_err(|e| anyhow::anyhow!("failed to read minter reply: {e}"))?;

    let line = line.trim();
    if line.is_empty() {
        return Err(anyhow::anyhow!("minter closed connection without replying"));
    }

    let reply: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| anyhow::anyhow!("malformed minter reply {line:?}: {e}"))?;

    if let Some(error) = reply.get("error").and_then(|v| v.as_str()) {
        return Err(anyhow::anyhow!("minter rejected request: {error}"));
    }

    reply
        .get("token")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("minter reply contained neither token nor error"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_request_omits_absent_fields() {
        assert_eq!(MintRequest::default().to_json_line(), "{}");
        assert_eq!(
            MintRequest {
                ttl_seconds: Some(3600),
                audience: None,
            }
            .to_json_line(),
            r#"{"ttl_seconds":3600}"#
        );
    }
}
