//! # DEPRECATED (#383, containerization epic #378)
//!
//! `adelie-mint` is **deprecated**. The decision in #383 makes an OIDC
//! provider the sole token issuer (default FOSS provider: Dex), and the
//! daemon already validates OIDC RS256 tokens over both UDS and WebSocket
//! with one shared validator — so this local HS256 minter is redundant.
//! See `docs/oidc-auth.md`. Removal is tracked once the client token path
//! (#384) lands (D-Bus bridge + `Connector` obtain OIDC tokens instead of
//! minting). See also `crates/jwt-minter/README.md`.
//!
//! `adelie-mint` — local JWT minter binary (issue #101).
//!
//! Listens on a Unix domain socket and mints short-lived HS256 JWTs for
//! the OS user identified by `SO_PEERCRED`. Desktop-dev convenience —
//! production deployments use an external OIDC IdP instead.

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::Parser;
use desktop_assistant_auth_jwt as auth_jwt;
use desktop_assistant_jwt_minter::config::{
    DEFAULT_AUDIENCE, DEFAULT_ISSUER, DEFAULT_TTL_SECS, MAX_TTL_SECS, MIN_TTL_SECS, MintConfig,
};
use desktop_assistant_jwt_minter::group::resolve_group;
use desktop_assistant_jwt_minter::server::{ServerOptions, serve};
use tokio::signal::unix::{SignalKind, signal};

/// Local JWT minter for the desktop assistant daemon.
#[derive(Debug, Parser)]
#[command(
    name = "adelie-mint",
    about = "Mint short-lived HS256 JWTs for local clients identified via SO_PEERCRED.",
    version
)]
struct Cli {
    /// Path to the UDS to listen on. Defaults to
    /// `$XDG_RUNTIME_DIR/adelie/mint.sock`, falling back to
    /// `/run/adelie/mint.sock` when `XDG_RUNTIME_DIR` is unset.
    #[arg(long, env = "ADELIE_MINT_SOCKET")]
    socket: Option<PathBuf>,

    /// Optional Unix group name; when set, only callers whose UID is a
    /// member of this group can mint tokens. Validated at startup.
    #[arg(long, env = "ADELIE_MINT_GROUP")]
    group: Option<String>,

    /// Path to the HS256 signing key file. Defaults to the same file the
    /// daemon uses (under `$XDG_DATA_HOME/desktop-assistant/secrets`).
    #[arg(long, env = "ADELIE_MINT_SIGNING_KEY")]
    signing_key: Option<PathBuf>,

    /// JWT `iss` claim. Defaults to the daemon's expected issuer.
    #[arg(long, default_value = DEFAULT_ISSUER)]
    issuer: String,

    /// Default JWT `aud` claim when the request omits one. Defaults to
    /// the daemon's expected audience.
    #[arg(long, default_value = DEFAULT_AUDIENCE)]
    audience: String,

    /// Default TTL in seconds when the request omits one.
    #[arg(long, default_value_t = DEFAULT_TTL_SECS)]
    default_ttl: u64,

    /// Minimum TTL clamp.
    #[arg(long, default_value_t = MIN_TTL_SECS)]
    min_ttl: u64,

    /// Maximum TTL clamp.
    #[arg(long, default_value_t = MAX_TTL_SECS)]
    max_ttl: u64,
}

fn default_socket_path() -> PathBuf {
    if let Ok(runtime) = std::env::var("XDG_RUNTIME_DIR")
        && !runtime.is_empty()
    {
        return PathBuf::from(runtime).join("adelie").join("mint.sock");
    }
    PathBuf::from("/run/adelie/mint.sock")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.min_ttl == 0 {
        return Err(anyhow!("--min-ttl must be > 0"));
    }
    if cli.max_ttl < cli.min_ttl {
        return Err(anyhow!(
            "--max-ttl ({}) must be >= --min-ttl ({})",
            cli.max_ttl,
            cli.min_ttl
        ));
    }

    let signing_key_path = cli
        .signing_key
        .unwrap_or_else(auth_jwt::default_signing_key_path);

    let config = MintConfig {
        signing_key_path,
        issuer: cli.issuer,
        default_audience: cli.audience,
        default_ttl: std::time::Duration::from_secs(cli.default_ttl),
        min_ttl: std::time::Duration::from_secs(cli.min_ttl),
        max_ttl: std::time::Duration::from_secs(cli.max_ttl),
    };

    let group_gate = match cli.group.as_deref() {
        None => None,
        Some(name) => {
            let resolved =
                resolve_group(name).with_context(|| format!("group lookup for {name:?} failed"))?;
            match resolved {
                Some(gate) => {
                    tracing::info!(group = %gate.name, gid = gate.gid, "group gate active");
                    Some(gate)
                }
                None => {
                    return Err(anyhow!(
                        "configured --group {name:?} does not exist on this system"
                    ));
                }
            }
        }
    };

    let socket_path = cli.socket.unwrap_or_else(default_socket_path);
    let options = ServerOptions {
        socket_path,
        group_gate,
    };

    let shutdown = build_shutdown_signal()?;
    serve(options, config, shutdown).await
}

fn build_shutdown_signal() -> anyhow::Result<impl std::future::Future<Output = ()> + Send + 'static>
{
    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let mut sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok(async move {
        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM");
            }
            _ = sigint.recv() => {
                tracing::info!("received SIGINT");
            }
        }
    })
}
