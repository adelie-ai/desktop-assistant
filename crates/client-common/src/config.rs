use std::path::PathBuf;

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
    data_home.join("desktop-assistant").join("tls").join("ca.pem")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Ws,
    Dbus,
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
        }
    }
}
