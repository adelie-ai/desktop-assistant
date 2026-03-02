pub const DEFAULT_WS_URL: &str = "ws://127.0.0.1:11339/ws";
pub const DEFAULT_WS_SUBJECT: &str = "desktop-tui";

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
        }
    }
}
