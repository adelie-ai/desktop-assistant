//! Client for the local `adelie-mint` UDS. STUB — bodies arrive in
//! the implementation commit.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default UDS path the minter binds; same convention as
/// `adelie-mint` itself.
pub fn default_minter_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(|p| PathBuf::from(p).join("adelie").join("mint.sock"))
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct MintRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MintResponse {
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub exp: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Stub: real impl arrives in the implementation commit.
pub async fn fetch_jwt(
    _socket_path: &Path,
    _request: MintRequest,
    _timeout: Duration,
) -> anyhow::Result<String> {
    Err(anyhow::anyhow!("fetch_jwt not implemented yet"))
}
