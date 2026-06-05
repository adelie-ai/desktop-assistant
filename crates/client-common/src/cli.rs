//! Optional `clap` integration (behind the `clap` feature).
//!
//! [`TransportArgs`] is a flattenable argument group giving every CLI client the
//! same transport flags (`--transport`, `--service`, `--socket-path`, the WS
//! auth options) and a one-call path to a [`Connector`]. It defaults to the
//! local Unix socket, so a tool run with no flags talks to the local daemon.
//!
//! ```ignore
//! #[derive(clap::Parser)]
//! struct Cli {
//!     #[command(flatten)]
//!     transport: desktop_assistant_client_common::TransportArgs,
//! }
//! let cli = Cli::parse();
//! let connector = cli.transport.connect().await?;
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::config::{ConnectionConfig, TransportMode};
use crate::connector::Connector;

/// Standard transport flags for a CLI assistant client.
#[derive(Args, Debug, Clone, Default)]
pub struct TransportArgs {
    /// Transport to reach the assistant daemon: `dbus`, `uds` (local socket),
    /// or `ws`. Inferred from `--service` / `--socket-path`; defaults to `uds`.
    #[arg(long, value_name = "dbus|uds|ws")]
    pub transport: Option<String>,

    /// WebSocket URL of a (possibly remote) daemon, e.g. `wss://host:11339/ws`.
    /// Implies `--transport ws`.
    #[arg(long, value_name = "URL")]
    pub service: Option<String>,

    /// Path to the daemon's local Unix socket. Implies `--transport uds`.
    #[arg(long, value_name = "PATH")]
    pub socket_path: Option<PathBuf>,

    /// Bearer JWT for the WebSocket/UDS transport (skips token minting).
    #[arg(long, value_name = "JWT")]
    pub ws_jwt: Option<String>,

    /// Username for the `/login` token fallback (WebSocket).
    #[arg(long, value_name = "USER")]
    pub ws_login_username: Option<String>,

    /// Password for the `/login` token fallback (WebSocket).
    #[arg(long, value_name = "PASS")]
    pub ws_login_password: Option<String>,

    /// PEM CA certificate to trust for `wss://` (defaults to the daemon's CA).
    #[arg(long, value_name = "PATH")]
    pub tls_ca_cert: Option<PathBuf>,
}

impl TransportArgs {
    /// Resolve these flags into a [`ConnectionConfig`]. Precedence: an explicit
    /// `--transport` wins; otherwise `--service` selects WS and `--socket-path`
    /// selects UDS; with nothing given it defaults to **local UDS**. Unset
    /// fields fall back to [`ConnectionConfig`]'s defaults.
    pub fn connection_config(&self) -> Result<ConnectionConfig> {
        let mode = match self.transport.as_deref() {
            Some(raw) => match raw.to_ascii_lowercase().as_str() {
                "dbus" => TransportMode::Dbus,
                "uds" | "local" => TransportMode::Uds,
                "ws" | "websocket" => TransportMode::Ws,
                other => {
                    return Err(anyhow::anyhow!(
                        "unknown --transport '{other}' (expected dbus, uds, or ws)"
                    ));
                }
            },
            None if self.service.is_some() => TransportMode::Ws,
            None if self.socket_path.is_some() => TransportMode::Uds,
            None => TransportMode::Uds, // local-first default
        };

        let mut config = ConnectionConfig {
            transport_mode: mode,
            ..ConnectionConfig::default()
        };
        if let Some(url) = &self.service {
            config.ws_url = url.clone();
        }
        if self.socket_path.is_some() {
            config.socket_path = self.socket_path.clone();
        }
        if self.ws_jwt.is_some() {
            config.ws_jwt = self.ws_jwt.clone();
        }
        if self.ws_login_username.is_some() {
            config.ws_login_username = self.ws_login_username.clone();
        }
        if self.ws_login_password.is_some() {
            config.ws_login_password = self.ws_login_password.clone();
        }
        if self.tls_ca_cert.is_some() {
            config.tls_ca_cert = self.tls_ca_cert.clone();
        }
        Ok(config)
    }

    /// Resolve the config and [`Connector::connect`] in one call.
    pub async fn connect(&self) -> Result<Connector> {
        Connector::connect(&self.connection_config()?).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_local_uds() {
        let config = TransportArgs::default().connection_config().unwrap();
        assert_eq!(config.transport_mode, TransportMode::Uds);
    }

    #[test]
    fn service_url_implies_ws() {
        let args = TransportArgs {
            service: Some("wss://host:11339/ws".into()),
            ..Default::default()
        };
        let config = args.connection_config().unwrap();
        assert_eq!(config.transport_mode, TransportMode::Ws);
        assert_eq!(config.ws_url, "wss://host:11339/ws");
    }

    #[test]
    fn socket_path_implies_uds() {
        let args = TransportArgs {
            socket_path: Some("/run/user/1000/adelie/sock".into()),
            ..Default::default()
        };
        let config = args.connection_config().unwrap();
        assert_eq!(config.transport_mode, TransportMode::Uds);
        assert_eq!(
            config.socket_path.as_deref(),
            Some(std::path::Path::new("/run/user/1000/adelie/sock"))
        );
    }

    #[test]
    fn explicit_transport_overrides_inference() {
        // --transport wins even when a --service URL is also present.
        let args = TransportArgs {
            transport: Some("dbus".into()),
            service: Some("wss://host/ws".into()),
            ..Default::default()
        };
        assert_eq!(
            args.connection_config().unwrap().transport_mode,
            TransportMode::Dbus
        );
    }

    #[test]
    fn unknown_transport_is_an_error() {
        let args = TransportArgs {
            transport: Some("carrier-pigeon".into()),
            ..Default::default()
        };
        assert!(args.connection_config().is_err());
    }
}
