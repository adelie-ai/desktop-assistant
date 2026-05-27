//! Minter configuration knobs.

use std::path::PathBuf;
use std::time::Duration;

/// Defaults that match the daemon's expectations
/// (`crates/daemon/src/config/jwt.rs`).
pub const DEFAULT_ISSUER: &str = "org.desktopAssistant.local";
pub const DEFAULT_AUDIENCE: &str = "desktop-assistant-ws";
pub const DEFAULT_TTL_SECS: u64 = 15 * 60;
pub const MIN_TTL_SECS: u64 = 60;
pub const MAX_TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone)]
pub struct MintConfig {
    pub signing_key_path: PathBuf,
    pub issuer: String,
    pub default_audience: String,
    pub default_ttl: Duration,
    pub min_ttl: Duration,
    pub max_ttl: Duration,
}

impl MintConfig {
    /// Build a default config rooted at the daemon's conventional
    /// signing-key path.
    pub fn with_default_paths() -> Self {
        Self {
            signing_key_path: desktop_assistant_auth_jwt::default_signing_key_path(),
            issuer: DEFAULT_ISSUER.to_string(),
            default_audience: DEFAULT_AUDIENCE.to_string(),
            default_ttl: Duration::from_secs(DEFAULT_TTL_SECS),
            min_ttl: Duration::from_secs(MIN_TTL_SECS),
            max_ttl: Duration::from_secs(MAX_TTL_SECS),
        }
    }

    /// Clamp a caller-supplied TTL (in seconds) to `[min_ttl, max_ttl]`,
    /// falling back to `default_ttl` when the caller omits the field.
    pub fn clamp_ttl(&self, requested_seconds: Option<u64>) -> Duration {
        match requested_seconds {
            None => self.default_ttl,
            Some(secs) => {
                let secs = secs
                    .max(self.min_ttl.as_secs())
                    .min(self.max_ttl.as_secs());
                Duration::from_secs(secs)
            }
        }
    }
}
