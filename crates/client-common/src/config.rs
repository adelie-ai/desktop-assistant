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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TransportMode {
    Ws,
    Dbus,
    /// Local Unix domain socket. The concrete path lives on
    /// [`ConnectionConfig::socket_path`]; `None` there resolves to
    /// [`default_desktop_socket_path`].
    Uds,
}

/// Runtime connection settings for a client.
///
/// Deserializable from a client's config file with **container-level
/// `#[serde(default)]`**: any field a config omits falls back to this struct's
/// [`Default`] impl, so an older config that predates a field keeps working and
/// gains the new default. Only `Deserialize` is derived, not `Serialize` — the
/// struct carries secrets (`ws_jwt`, `ws_login_password`) that must never be
/// written back out to disk.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
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
    /// Share basic device context (name, username, home dir, hostname,
    /// timezone, OS) with the assistant so it can personalize; unchecked, the
    /// client sends nothing (#549). Default **on**: an absent field in an
    /// existing config still means on (`#[serde(default)]` routes through
    /// [`Default`]).
    ///
    /// How the daemon learns of it differs by transport (#782/#783):
    ///
    /// - **UDS**: the resolved context rides the connect handshake, and a
    ///   REFUSAL is stated outright beside it - `share_client_context: false`
    ///   (#783). Only the refusal is stated, because it is the one case the
    ///   daemon cannot infer: this door can substitute the kernel peer identity
    ///   for a client that reported nothing, so it must be told when not to. A
    ///   sharing client sends no such field and its handshake stays
    ///   byte-identical to the pre-#783 shape. Both survive reconnect.
    /// - **WebSocket**: only the resolved context travels, in a base64 upgrade
    ///   header. That door substitutes nothing of its own, so an absent header is
    ///   already the whole refusal and a field would say nothing new.
    ///   `resolve_ws_client_context` in the ws-interface crate is where that
    ///   rule lives and is tested; a fallback added inside it fails those tests.
    ///   Keeping it the single route from the upgrade headers to the
    ///   connection's context is a convention, not something the compiler
    ///   enforces - a fallback bolted on at its call site would need such a
    ///   field.
    /// - **D-Bus**: the bridge holds the daemon connection, so neither the
    ///   context nor the refusal has a handshake to ride here. This client
    ///   declares the decision to the bridge
    ///   (`org.desktopAssistant.Commands.SetShareClientContext`) and the bridge
    ///   builds this caller's daemon session from it, which is where the
    ///   handshake above is then written (#782).
    ///
    /// When it is off, the machine's hostname is withheld too: it also rides the
    /// handshake as the `host_label` tool-note hint, resolved by the same
    /// function that fills the context's `hostname`. The per-machine `system_id`
    /// still travels, because tool routing depends on it; see `stamp_system_id`
    /// for what that does and does not cost.
    ///
    /// This governs the client's own self-report. It does not reach facts the
    /// daemon reports about the machine it runs on; those are governed
    /// separately, where they are produced.
    pub share_client_context: bool,
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
            share_client_context: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_client_context_defaults_true() {
        assert!(ConnectionConfig::default().share_client_context);
    }

    #[test]
    fn config_toml_without_share_client_context_key_parses_true() {
        // Back-compat: a config that predates the field (here, an empty
        // document) still deserializes with sharing ON.
        let cfg: ConnectionConfig = toml::from_str("").expect("empty config parses");
        assert!(cfg.share_client_context);
    }

    #[test]
    fn config_toml_can_disable_sharing() {
        let cfg: ConnectionConfig =
            toml::from_str("share_client_context = false").expect("config parses");
        assert!(!cfg.share_client_context);
    }

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

    /// Only *absence* is benign. A path that exists but cannot be read is a
    /// real fault (wrong permissions, a directory) and must surface rather than
    /// silently downgrading the connection's trust anchors.
    #[test]
    fn unreadable_ca_path_is_an_error() {
        let dir = tempfile::tempdir().expect("create temp dir");

        let err = read_optional_ca_pem(Some(dir.path()))
            .expect_err("an unreadable CA path must not be treated as absent");

        assert!(
            err.to_string().contains("reading CA cert"),
            "error should name the read failure, got: {err}"
        );
    }

    #[test]
    fn absent_ca_path_yields_none() {
        let missing = Path::new("/nonexistent/desktop-assistant/tls/ca.pem");

        let pem = read_optional_ca_pem(Some(missing)).expect("absent CA file must not be fatal");

        assert!(pem.is_none());
    }

    #[test]
    fn unconfigured_ca_path_yields_none() {
        let pem = read_optional_ca_pem(None).expect("no configured CA is not an error");

        assert!(pem.is_none());
    }
}
