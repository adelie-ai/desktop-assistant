//! Bridge transport: a thin seam over the shared client-common [`Connector`].
//!
//! Before #316 this module held a bespoke UDS client — JWT handshake, length-
//! prefixed framing, frame demux, per-request reply correlation, all connected
//! **once per process lifetime** (a daemon restart left the bridge dark). All of
//! that is now the [`Connector`]'s job: the bridge connects to the daemon as an
//! ordinary authenticated UDS client and inherits, for free, the same hardened
//! path every other client uses — automatic reconnect with backoff, a fresh
//! minted JWT on every (re)connect (so a restart can't strand it on an expired
//! token), and the shared frame codec.
//!
//! What's left here is just the seam the D-Bus adapters dispatch through: the
//! [`BridgeTransport`] trait (so the adapters stay testable against an in-memory
//! fake) and [`ConnectorBridgeTransport`], which forwards each command to the
//! Connector's command channel. Event delivery is handled separately by
//! [`crate::adapter::event_forwarder`], which consumes the Connector's signal
//! stream directly.

use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_client_common::Connector;

/// Errors surfaced to the D-Bus adapters. The Connector reports failures as
/// `anyhow::Error`; the bridge collapses them into [`Self::Daemon`] (message
/// preserved), keeping the richer variants for the in-memory test fakes and any
/// future structured mapping.
#[derive(Debug, thiserror::Error)]
pub enum BridgeTransportError {
    /// Socket-level failure.
    #[error("transport error: {0}")]
    Io(#[from] std::io::Error),

    /// The daemon (or the Connector) returned an error. The message is
    /// surfaced verbatim so debugging stays one hop from the source.
    #[error("daemon error: {0}")]
    Daemon(String),

    /// Request was sent but the connection went away before a reply arrived.
    #[error("daemon connection closed while awaiting reply for {request_id}")]
    Disconnected { request_id: String },

    /// A request timed out waiting for a reply.
    #[error("daemon did not reply for {request_id} within {timeout:?}")]
    Timeout {
        request_id: String,
        timeout: Duration,
    },

    /// Wire payload could not be parsed.
    #[error("malformed daemon frame: {0}")]
    BadFrame(String),
}

/// The seam the D-Bus adapters dispatch through. Returning [`api::CommandResult`]
/// keeps adapters small — they pattern-match only the variant they expect. Event
/// delivery is out of band (the event forwarder consumes the Connector stream).
#[async_trait::async_trait]
pub trait BridgeTransport: Send + Sync {
    /// Send `command` and await the daemon's [`api::CommandResult`].
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError>;
}

/// Production transport: forwards each command to the daemon over the
/// [`Connector`]'s authenticated, auto-reconnecting UDS session.
///
/// The held `&TransportClient` is stable across reconnects (#246): the Connector
/// swaps its socket in place, so a command issued after a daemon restart simply
/// rides the freshly re-established connection.
pub struct ConnectorBridgeTransport {
    connector: Arc<Connector>,
}

impl ConnectorBridgeTransport {
    pub fn new(connector: Arc<Connector>) -> Self {
        Self { connector }
    }
}

#[async_trait::async_trait]
impl BridgeTransport for ConnectorBridgeTransport {
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        let commands = self.connector.client().as_commands().ok_or_else(|| {
            BridgeTransportError::Daemon(
                "the active transport has no command channel (a socket transport is required)"
                    .to_string(),
            )
        })?;
        commands
            .send_command(command)
            .await
            .map_err(|e| BridgeTransportError::Daemon(e.to_string()))
    }
}

// The length-prefixed frame codec, re-exported under the historical
// `read_frame`/`write_frame` names the integration tests' stub daemon imports.
pub use desktop_assistant_frame_codec::{read_frame, write_frame};

/// Create `path`'s parent directory if needed — a convenience for tests that
/// bind temp sockets.
pub fn ensure_parent_dir(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
