//! UDS transport client for the daemon's UDS frontend (issue #103).
//!
//! [`BridgeTransport`] is the trait the D-Bus adapters use to ship
//! requests to the daemon. The production impl ([`UdsBridgeTransport`])
//! opens the UDS, performs the JWT handshake, runs the
//! length-prefixed framing on both directions, and demuxes inbound
//! frames into per-request reply channels (keyed by `WsRequest::id`)
//! and a broadcast event channel.
//!
//! The trait abstraction is deliberate: tests can swap in an
//! in-memory `BridgeTransport` that asserts the request shape and
//! injects synthetic events without touching real sockets.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use serde::Serialize;
use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

pub use api::{WsFrame, WsRequest};

/// Errors the transport surfaces to callers (the D-Bus adapters and
/// the binary's `main`). Distinct from the wire-level
/// [`WsFrame::Error`] — those land as `BridgeTransportError::Daemon`.
#[derive(Debug, thiserror::Error)]
pub enum BridgeTransportError {
    /// Socket-level failure: not connected, peer closed, framing.
    #[error("transport error: {0}")]
    Io(#[from] std::io::Error),

    /// The daemon's handshake / dispatcher returned a wire-level error
    /// frame. The bridge surfaces the daemon's message verbatim so
    /// debugging stays one hop away from the source.
    #[error("daemon error: {0}")]
    Daemon(String),

    /// Request was sent but the connection went away before a reply
    /// arrived.
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

/// Trait the D-Bus adapters call to dispatch commands.
///
/// Returning [`api::CommandResult`] keeps adapters small: they only
/// pattern-match on the result variant they expect. Event delivery is
/// out-of-band via [`subscribe_events`](BridgeTransport::subscribe_events).
#[async_trait::async_trait]
pub trait BridgeTransport: Send + Sync {
    /// Send `command` and wait for the matching `WsFrame::Result` /
    /// `WsFrame::Error`.
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError>;

    /// Subscribe to inbound `WsFrame::Event` frames. Each subscriber
    /// gets a clone of every event; lagging subscribers drop the
    /// oldest items (broadcast channel semantics) which is fine for
    /// D-Bus signals — losing an old event matters less than blocking
    /// the demux task.
    fn subscribe_events(&self) -> broadcast::Receiver<api::Event>;
}

/// Tuning knobs.
#[derive(Debug, Clone)]
pub struct UdsBridgeConfig {
    /// Path to the daemon's UDS socket. Defaults to
    /// `$XDG_RUNTIME_DIR/adelie/sock`.
    pub socket_path: PathBuf,
    /// Per-request reply timeout. Keep generous — `SendMessage` only
    /// gets an immediate `Ack`, but other commands may stat the DB.
    pub request_timeout: Duration,
    /// Inbound event broadcast buffer.
    pub event_buffer: usize,
}

impl UdsBridgeConfig {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            request_timeout: Duration::from_secs(30),
            event_buffer: 256,
        }
    }
}

/// Default desktop path for the daemon's UDS socket. Mirrors
/// `desktop_assistant_uds::default_desktop_socket_path` so the bridge
/// and daemon agree on the same default without a cross-crate dep.
pub fn default_daemon_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("sock"))
}

type PendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<Result<api::CommandResult, String>>>>>;

/// UDS-backed implementation of [`BridgeTransport`].
///
/// Spawns two tokio tasks per connection — a reader that demuxes
/// inbound frames, and a writer that drains the outbound mpsc onto
/// the socket. The reader closes the [`CancellationToken`] when the
/// peer hangs up so the writer task exits cleanly.
pub struct UdsBridgeTransport {
    outbound_tx: mpsc::Sender<Vec<u8>>,
    pending: PendingMap,
    events_tx: broadcast::Sender<api::Event>,
    request_timeout: Duration,
    cancel: CancellationToken,
}

impl UdsBridgeTransport {
    /// Connect to `socket_path`, send the JWT handshake, and return a
    /// transport ready to dispatch requests. Returns an error when
    /// the socket is missing, the handshake is rejected, or the
    /// framing fails.
    pub async fn connect(config: UdsBridgeConfig, jwt: &str) -> Result<Self, BridgeTransportError> {
        let stream = UnixStream::connect(&config.socket_path).await?;
        Self::connect_on_stream(stream, jwt, config.event_buffer, config.request_timeout).await
    }

    /// Test-friendly: connect on a pre-made stream so integration
    /// tests can drive the bridge against an in-memory UDS pair
    /// without binding a path.
    pub async fn connect_on_stream(
        stream: UnixStream,
        jwt: &str,
        event_buffer: usize,
        request_timeout: Duration,
    ) -> Result<Self, BridgeTransportError> {
        let (read_half, mut write_half) = stream.into_split();

        // Send the handshake: `{"jwt": "<token>"}` length-prefixed.
        let handshake = serde_json::to_vec(&Handshake { jwt })
            .map_err(|e| BridgeTransportError::BadFrame(format!("serialize handshake: {e}")))?;
        write_frame(&mut write_half, &handshake).await?;

        let (outbound_tx, mut outbound_rx) = mpsc::channel::<Vec<u8>>(64);
        let (events_tx, _events_rx) = broadcast::channel::<api::Event>(event_buffer);
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let cancel = CancellationToken::new();

        // Writer task.
        let writer_cancel = cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = writer_cancel.cancelled() => break,
                    next = outbound_rx.recv() => {
                        let Some(payload) = next else { break };
                        if let Err(e) = write_frame(&mut write_half, &payload).await {
                            debug!("bridge outbound write failed: {e}");
                            break;
                        }
                    }
                }
            }
        });

        // Reader task: demux frames into pending result channels and
        // the broadcast event channel.
        let reader_pending = Arc::clone(&pending);
        let reader_events = events_tx.clone();
        let reader_cancel = cancel.clone();
        tokio::spawn(async move {
            let mut read_half = read_half;
            loop {
                let frame = tokio::select! {
                    biased;
                    _ = reader_cancel.cancelled() => break,
                    res = read_frame(&mut read_half) => match res {
                        Ok(b) => b,
                        Err(e) => {
                            debug!("bridge inbound read ended: {e}");
                            break;
                        }
                    }
                };
                if frame.is_empty() {
                    continue;
                }
                let frame: api::WsFrame = match serde_json::from_slice(&frame) {
                    Ok(f) => f,
                    Err(e) => {
                        warn!("bridge inbound frame parse error: {e}");
                        continue;
                    }
                };
                match frame {
                    api::WsFrame::Result { id, result } => {
                        if let Some(tx) = reader_pending.lock().await.remove(&id) {
                            let _ = tx.send(Ok(result));
                        } else {
                            debug!("bridge got result for unknown id {id}");
                        }
                    }
                    api::WsFrame::Error { id, error } => {
                        if id.is_empty() {
                            // Pre-dispatch error (auth, framing). Fail
                            // every in-flight request with the daemon
                            // message so callers don't hang.
                            let mut pending = reader_pending.lock().await;
                            for (_id, tx) in pending.drain() {
                                let _ = tx.send(Err(error.clone()));
                            }
                            warn!("bridge received pre-dispatch error: {error}");
                            break;
                        }
                        if let Some(tx) = reader_pending.lock().await.remove(&id) {
                            let _ = tx.send(Err(error));
                        } else {
                            debug!("bridge got error for unknown id {id}");
                        }
                    }
                    api::WsFrame::Event { event } => {
                        // Best-effort; if no subscribers, drop.
                        let _ = reader_events.send(event);
                    }
                }
            }
            // Connection torn down: fail every in-flight request.
            let mut pending = reader_pending.lock().await;
            for (_id, tx) in pending.drain() {
                let _ = tx.send(Err("connection closed".to_string()));
            }
            reader_cancel.cancel();
        });

        Ok(Self {
            outbound_tx,
            pending,
            events_tx,
            request_timeout,
            cancel,
        })
    }

    /// Trigger shutdown of the reader/writer tasks. Idempotent.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for UdsBridgeTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Serialize)]
struct Handshake<'a> {
    jwt: &'a str,
}

#[async_trait::async_trait]
impl BridgeTransport for UdsBridgeTransport {
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        let id = uuid::Uuid::new_v4().to_string();
        let envelope = api::WsRequest {
            id: id.clone(),
            command,
        };
        let body = serde_json::to_vec(&envelope)
            .map_err(|e| BridgeTransportError::BadFrame(format!("serialize request: {e}")))?;

        let (tx, rx) = oneshot::channel::<Result<api::CommandResult, String>>();
        self.pending.lock().await.insert(id.clone(), tx);

        if self.outbound_tx.send(body).await.is_err() {
            // Reader/writer task is gone; clear our pending entry.
            self.pending.lock().await.remove(&id);
            return Err(BridgeTransportError::Disconnected { request_id: id });
        }

        let reply = tokio::time::timeout(self.request_timeout, rx).await;
        match reply {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(msg))) => Err(BridgeTransportError::Daemon(msg)),
            Ok(Err(_)) => Err(BridgeTransportError::Disconnected { request_id: id }),
            Err(_) => {
                // Timed out — drop our pending entry so a late reply
                // can't poison the next request.
                self.pending.lock().await.remove(&id);
                Err(BridgeTransportError::Timeout {
                    request_id: id,
                    timeout: self.request_timeout,
                })
            }
        }
    }

    fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
        self.events_tx.subscribe()
    }
}

/// Read one length-prefixed frame; matches `uds-interface`'s framing.
pub async fn read_frame<R>(reader: &mut R) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    const MAX_FRAME_LEN: u32 = 4 * 1024 * 1024;

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    if len > 0 {
        reader.read_exact(&mut body).await?;
    }
    Ok(body)
}

/// Write one length-prefixed frame.
pub async fn write_frame<W>(writer: &mut W, body: &[u8]) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let len = body.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(body).await?;
    writer.flush().await?;
    Ok(())
}

/// Path helper for binding test sockets so tests don't have to inline
/// the same `XDG_RUNTIME_DIR` dance everywhere.
pub fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
