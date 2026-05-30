//! Unix-domain-socket client transport.
//!
//! The daemon's `uds-interface` serves the same `WsRequest`/`WsFrame` JSON
//! protocol as the WebSocket adapter, but over a local Unix socket framed with
//! a 4-byte little-endian length prefix — the first frame being the
//! `{"jwt": "<token>"}` handshake. `UdsClient` mirrors `WsClient`'s
//! request/response multiplexing and shares every typed command via
//! [`AssistantCommands`]; only the connect step and the framing differ.
//!
//! The framing functions below are a deliberate re-implementation of the
//! server's `read_frame`/`write_frame` (identical wire format — see
//! `crates/uds-interface/src/lib.rs`). Depending on that crate would drag the
//! entire daemon stack into every client binary, so the ~20 lines are
//! duplicated on purpose.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use api::{WsFrame, WsRequest};
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::commands::{AssistantCommands, PendingResult};
use crate::signal::SignalEvent;
use crate::ws_client::map_event_to_signal;

/// 4 MB cap, matching the server. Keeps a buggy/hostile peer from claiming a
/// multi-GB frame length and forcing an allocation blow-up.
const MAX_FRAME_LEN: u32 = 4 * 1024 * 1024;

/// In-flight requests plus a terminal "closed" marker, behind a single mutex.
///
/// The send path and the reader's teardown both synchronize here, so a request
/// can never be orphaned: it is either handed a response/error, drained at
/// close, or rejected up-front because the connection is already gone. This
/// matters because the UDS server rejects a bad handshake *in band* — after
/// `connect` has already returned — so, unlike the WS path (which fails during
/// the HTTP upgrade), this client must surface a post-connect auth failure
/// without hanging the first command.
struct PendingState {
    map: HashMap<String, oneshot::Sender<PendingResult>>,
    closed: Option<String>,
}

impl PendingState {
    /// Record the connection as closed (keeping the first reason) and fail
    /// every outstanding request with `reason`.
    fn close(&mut self, reason: &str) {
        if self.closed.is_none() {
            self.closed = Some(reason.to_string());
        }
        for (_id, tx) in self.map.drain() {
            let _ = tx.send(Err(reason.to_string()));
        }
    }
}

pub struct UdsClient {
    outbound_tx: mpsc::UnboundedSender<Vec<u8>>,
    pending: Arc<Mutex<PendingState>>,
}

impl UdsClient {
    pub async fn connect(
        socket_path: &Path,
        bearer_token: &str,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SignalEvent>)> {
        let stream = UnixStream::connect(socket_path)
            .await
            .map_err(|e| anyhow!("failed to connect uds {}: {e}", socket_path.display()))?;
        let (mut read_half, mut write_half) = stream.into_split();

        // Handshake: the first frame must be `{"jwt": "<token>"}`. On success
        // the server sends nothing back and proceeds straight to the
        // dispatcher; on failure it writes an error frame and closes, which
        // the reader below turns into a connection-level failure.
        let handshake = serde_json::to_vec(&serde_json::json!({ "jwt": bearer_token }))?;
        write_frame(&mut write_half, &handshake)
            .await
            .map_err(|e| anyhow!("uds handshake write failed: {e}"))?;

        // Writer: serialized request bodies -> length-prefixed frames.
        let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(async move {
            while let Some(body) = outbound_rx.recv().await {
                if write_frame(&mut write_half, &body).await.is_err() {
                    break;
                }
            }
        });

        let pending = Arc::new(Mutex::new(PendingState {
            map: HashMap::new(),
            closed: None,
        }));
        let pending_for_reader = Arc::clone(&pending);

        let (signal_tx, signal_rx) = mpsc::unbounded_channel::<SignalEvent>();
        tokio::spawn(async move {
            loop {
                let raw = match read_frame(&mut read_half).await {
                    Ok(buf) if buf.is_empty() => break, // 0-length frame == close
                    Ok(buf) => buf,
                    Err(_) => break, // EOF or io error
                };
                let Ok(frame) = serde_json::from_slice::<WsFrame>(&raw) else {
                    continue; // ignore a malformed frame rather than tearing down
                };
                match frame {
                    WsFrame::Result { id, result } => {
                        if let Some(tx) = pending_for_reader.lock().await.map.remove(&id) {
                            let _ = tx.send(Ok(result));
                        }
                    }
                    WsFrame::Error { id, error } => {
                        let mut state = pending_for_reader.lock().await;
                        if let Some(tx) = state.map.remove(&id) {
                            // Per-request error: fail just that call.
                            let _ = tx.send(Err(error));
                        } else {
                            // Unmatched id: the server's pre-dispatch errors
                            // (auth/handshake) carry an empty id and are
                            // followed by a close — connection-level failure.
                            state.close(&error);
                            drop(state);
                            break;
                        }
                    }
                    WsFrame::Event { event } => {
                        if let Some(signal) = map_event_to_signal(event) {
                            let _ = signal_tx.send(signal);
                        }
                    }
                }
            }

            // Teardown: mark closed (preserving a connection-level error reason
            // if one was already set), drain any stragglers, notify once.
            let reason = {
                let mut state = pending_for_reader.lock().await;
                state.close("uds connection closed");
                state
                    .closed
                    .clone()
                    .unwrap_or_else(|| "uds connection closed".to_string())
            };
            let _ = signal_tx.send(SignalEvent::Disconnected { reason });
        });

        Ok((
            Self {
                outbound_tx,
                pending,
            },
            signal_rx,
        ))
    }
}

#[async_trait]
impl AssistantCommands for UdsClient {
    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult> {
        let id = uuid::Uuid::new_v4().to_string();
        let request = WsRequest {
            id: id.clone(),
            command,
        };
        let body = serde_json::to_vec(&request)?;

        let (tx, rx) = oneshot::channel::<PendingResult>();
        {
            let mut state = self.pending.lock().await;
            if let Some(reason) = &state.closed {
                return Err(anyhow!("uds connection closed: {reason}"));
            }
            state.map.insert(id.clone(), tx);
        }

        if self.outbound_tx.send(body).is_err() {
            self.pending.lock().await.map.remove(&id);
            return Err(anyhow!("failed to send uds request: writer closed"));
        }

        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(_closed) => Err(anyhow!("uds response channel closed")),
        }
    }
}

/// Read one length-prefixed frame: a 4-byte little-endian `u32` length followed
/// by that many body bytes. A 0-length frame signals a clean close.
async fn read_frame<R>(read_half: &mut R) -> io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    read_half.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    if len > 0 {
        read_half.read_exact(&mut body).await?;
    }
    Ok(body)
}

/// Write one length-prefixed frame.
async fn write_frame<W>(write_half: &mut W, body: &[u8]) -> io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let len = body.len() as u32;
    write_half.write_all(&len.to_le_bytes()).await?;
    write_half.write_all(body).await?;
    write_half.flush().await?;
    Ok(())
}
