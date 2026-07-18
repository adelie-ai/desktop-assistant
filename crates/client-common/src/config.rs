use std::path::{Path, PathBuf};

pub const DEFAULT_WS_URL: &str = "wss://127.0.0.1:11339/ws";
pub const DEFAULT_WS_SUBJECT: &str = "desktop-tui";

/// Default path to the daemon's auto-generated CA certificate.
pub fn default_ca_cert_path() -> PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join(".local")
                .join("share")
        });
    data_home
        .join("desktop-assistant")
        .join("tls")
        .join("ca.pem")
}

/// Reads an optional CA-certificate bundle from disk.
///
/// `Ok(None)` means "no extra CA to trust": either none was configured, or the
/// configured path does not exist. The latter is deliberately not an error —
/// clients populate the default path unconditionally, so a machine that has
/// never run a local daemon has no file there and must still be able to reach
/// endpoints that need no private CA at all (#521). Any other read failure
/// (permissions, a directory, I/O) is a real error and propagates.
pub fn read_optional_ca_pem(ca_cert_path: Option<&Path>) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(path) = ca_cert_path else {
        return Ok(None);
    };
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(
                path = %path.display(),
                "no local CA certificate; trusting the public roots only"
            );
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!("reading CA cert {}: {e}", path.display())),
    }
}

/// Default path to the daemon's local Unix domain socket, or `None` when
/// `XDG_RUNTIME_DIR` is unset (no sensible desktop default). Mirrors the
/// daemon-side `desktop_assistant_uds::default_desktop_socket_path` so local
/// clients resolve the same endpoint without linking the server crate.
pub fn default_desktop_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("sock"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Ws,
    Dbus,
    /// Local Unix domain socket. The concrete path lives on
    /// [`ConnectionConfig::socket_path`]; `None` there resolves to
    /// [`default_desktop_socket_path`].
    Uds,
}

#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub transport_mode: TransportMode,
    pub ws_url: String,
    pub ws_jwt: Option<String>,
    pub ws_login_username: Option<String>,
    pub ws_login_password: Option<String>,
    pub ws_subject: String,
    /// Path to a PEM CA certificate to trust for `wss://` connections.
    /// Defaults to the daemon's auto-generated CA at
    /// `$XDG_DATA_HOME/desktop-assistant/tls/ca.pem`.
    pub tls_ca_cert: Option<PathBuf>,
    /// Path to the daemon's local Unix domain socket. Only meaningful when
    /// `transport_mode == TransportMode::Uds`; `None` resolves to
    /// [`default_desktop_socket_path`].
    pub socket_path: Option<PathBuf>,
    /// The client's per-machine **system id** for tool-locality co-location
    /// (#248), sent in the connect handshake on every transport. Stored on the
    /// config so the Connector's reconnect path re-sends it (the supervisor
    /// re-reads the config on each reconnect). `None` ⇒ no id reported and the
    /// daemon falls back to the transport heuristic. The Connector fills this in
    /// from `system_id::local_system_id()` when it connects; callers normally
    /// leave it `None`.
    pub system_id: Option<String>,
    /// An optional friendly host label sent alongside [`Self::system_id`] (#248)
    /// to make the remote tool note nicer (e.g. the client's hostname). Stored
    /// on the config for the same reconnect reason.
    pub host_label: Option<String>,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            transport_mode: TransportMode::Ws,
            ws_url: DEFAULT_WS_URL.to_string(),
            ws_jwt: None,
            ws_login_username: None,
            ws_login_password: None,
            ws_subject: DEFAULT_WS_SUBJECT.to_string(),
            tls_ca_cert: Some(default_ca_cert_path()),
            socket_path: None,
            system_id: None,
            host_label: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_ws_and_no_socket_path() {
        let config = ConnectionConfig::default();
        assert_eq!(config.transport_mode, TransportMode::Ws);
        assert!(config.socket_path.is_none());
    }

    #[test]
    fn default_socket_path_joins_runtime_dir() {
        // SAFETY: no other test in this binary reads XDG_RUNTIME_DIR, so the
        // global mutation is observationally single-threaded here.
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", "/run/user/4242");
        }
        assert_eq!(
            default_desktop_socket_path(),
            Some(PathBuf::from("/run/user/4242/adelie/sock"))
        );
    }
}
