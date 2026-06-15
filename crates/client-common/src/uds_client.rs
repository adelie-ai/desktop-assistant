//! Unix-domain-socket client transport.
//!
//! The daemon's `uds-interface` serves the same `WsRequest`/`WsFrame` JSON
//! protocol as the WebSocket adapter, but over a local Unix socket framed with
//! a 4-byte little-endian length prefix — the first frame being the
//! `{"jwt": "<token>"}` handshake. `UdsClient` mirrors `WsClient`'s
//! request/response multiplexing and shares every typed command via
//! [`AssistantCommands`]; only the connect step and the framing differ.
//!
//! The framing is shared with the server via the dependency-light
//! [`desktop_assistant_frame_codec`] crate (`read_frame`/`write_frame`), so the
//! frame cap and framing rules can never drift between client and server.
//! That crate carries only the `tokio` async I/O traits, so it doesn't drag
//! the daemon stack into client binaries.
//!
//! ## Reconnect (#246)
//!
//! The live socket (the writer's `outbound_tx`) lives behind a swappable
//! [`ConnState`] cell, while the request-correlation map, the signal stream the
//! [`Connector`](crate::Connector) reads, and the drop-notification channel all
//! **persist across reconnects**. [`UdsClient::reconnect`] re-runs the handshake
//! and spawns fresh reader/writer tasks wired to those same persistent
//! channels, then swaps the cell — so the connection comes back transparently
//! under a stable `&TransportClient` without the Connector having to re-subscribe
//! the event stream.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use api::{WsFrame, WsRequest};
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use desktop_assistant_frame_codec::{read_frame, write_frame};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::commands::{AssistantCommands, PendingResult};
use crate::signal::SignalEvent;
use crate::timeouts::DISPATCH_TIMEOUT;
use crate::ws_client::map_event_to_signal;

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

    /// Re-arm the map for a fresh connection (#246): clear the closed marker so
    /// new commands are accepted again. Any stragglers from the old connection
    /// were already drained by `close`.
    fn reopen(&mut self) {
        self.closed = None;
    }
}

/// The live connection's write handle, swapped on reconnect (#246).
struct ConnState {
    outbound_tx: mpsc::UnboundedSender<Vec<u8>>,
}

pub struct UdsClient {
    /// The live writer, replaced in place by [`reconnect`](Self::reconnect).
    conn: Arc<Mutex<ConnState>>,
    pending: Arc<Mutex<PendingState>>,
    /// The persistent signal stream every reader (across reconnects) feeds. The
    /// Connector subscribes to its receiver once and keeps it forever.
    signal_tx: mpsc::UnboundedSender<SignalEvent>,
    /// Fires once per underlying-socket close so the Connector's reconnect
    /// supervisor knows to back off and reconnect (#246). Persistent across
    /// reconnects; each fresh reader clones it.
    drop_tx: mpsc::UnboundedSender<()>,
    /// Per-command response deadline (#221). Defaults to
    /// [`DISPATCH_TIMEOUT`]; tunable via [`set_dispatch_timeout`].
    dispatch_timeout: Duration,
}

impl UdsClient {
    /// Override the per-command dispatch timeout (#221). Mainly for tests that
    /// need to assert the timeout fires without waiting the production window;
    /// production callers can also shorten/lengthen it to taste.
    pub fn set_dispatch_timeout(&mut self, timeout: Duration) {
        self.dispatch_timeout = timeout;
    }

    /// Connect a UDS transport. Returns the client, the persistent signal
    /// stream, and a drop-notifier receiver that fires once per underlying
    /// socket close (#246) — the Connector uses the latter to drive reconnect.
    ///
    /// `system_id` / `host_label` (#248) ride the handshake frame so the daemon
    /// can compute exact co-location.
    ///
    /// `bearer_token` is `None` for the local peer-cred path (#407): the daemon
    /// authenticates the connection by its kernel `SO_PEERCRED`, so no JWT is
    /// sent and the handshake omits the `jwt` field entirely. A `Some(token)` is
    /// still forwarded (an explicitly-configured static token) and honored by the
    /// tolerant daemon, but local clients no longer mint one.
    pub async fn connect(
        socket_path: &Path,
        bearer_token: Option<&str>,
        system_id: Option<&str>,
        host_label: Option<&str>,
    ) -> Result<(
        Self,
        mpsc::UnboundedReceiver<SignalEvent>,
        mpsc::UnboundedReceiver<()>,
    )> {
        let pending = Arc::new(Mutex::new(PendingState {
            map: HashMap::new(),
            closed: None,
        }));
        let (signal_tx, signal_rx) = mpsc::unbounded_channel::<SignalEvent>();
        let (drop_tx, drop_rx) = mpsc::unbounded_channel::<()>();

        let outbound_tx = Self::spawn_connection(
            socket_path,
            bearer_token,
            system_id,
            host_label,
            Arc::clone(&pending),
            signal_tx.clone(),
            drop_tx.clone(),
        )
        .await?;

        Ok((
            Self {
                conn: Arc::new(Mutex::new(ConnState { outbound_tx })),
                pending,
                signal_tx,
                drop_tx,
                dispatch_timeout: DISPATCH_TIMEOUT,
            },
            signal_rx,
            drop_rx,
        ))
    }

    /// Connect a fresh socket, perform the JWT handshake, and spawn the
    /// reader/writer tasks wired to the **persistent** `pending` / `signal_tx` /
    /// `drop_tx`. Returns the new writer handle. Shared by the initial
    /// [`connect`](Self::connect) and [`reconnect`](Self::reconnect) (#246).
    async fn spawn_connection(
        socket_path: &Path,
        bearer_token: Option<&str>,
        system_id: Option<&str>,
        host_label: Option<&str>,
        pending: Arc<Mutex<PendingState>>,
        signal_tx: mpsc::UnboundedSender<SignalEvent>,
        drop_tx: mpsc::UnboundedSender<()>,
    ) -> Result<mpsc::UnboundedSender<Vec<u8>>> {
        let stream = UnixStream::connect(socket_path)
            .await
            .map_err(|e| anyhow!("failed to connect uds {}: {e}", socket_path.display()))?;
        let (mut read_half, mut write_half) = stream.into_split();

        // Handshake: the first frame optionally carries a JWT plus, optionally,
        // the client's per-machine system id + host label for co-location (#248).
        // Absent fields are skipped — the local peer-cred path (#407) sends no
        // `jwt` at all (the daemon authenticates by `SO_PEERCRED`). On success the
        // server sends nothing back and proceeds straight to the dispatcher; on
        // failure it writes an error frame and closes, which the reader below
        // turns into a connection-level failure.
        let handshake = serde_json::to_vec(&api::UdsHandshake {
            jwt: bearer_token.map(str::to_string),
            system_id: system_id.map(str::to_string),
            host_label: host_label.map(str::to_string),
        })?;
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

        let pending_for_reader = Arc::clone(&pending);
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

            // Teardown: fail any outstanding requests so callers don't hang.
            // We do NOT emit a `Disconnected` on the signal stream here — that
            // stream persists across reconnects (#246), so a close on it would
            // wrongly read as a permanent end. Instead we notify the reconnect
            // supervisor via `drop_tx`; it emits the terminal `Disconnected` to
            // subscribers and drives the reconnect.
            pending_for_reader
                .lock()
                .await
                .close("uds connection closed");
            let _ = drop_tx.send(());
        });

        Ok(outbound_tx)
    }

    /// Re-establish the underlying socket after a drop (#246): re-run the JWT
    /// handshake, spawn fresh reader/writer tasks bound to the persistent
    /// channels, and swap in the new writer. On success the same
    /// `&TransportClient` resumes working; on failure the error is returned so
    /// the supervisor can back off and retry.
    ///
    /// The system id + host label (#248) are re-sent on every reconnect — the
    /// caller (`TransportClient::reconnect`) re-reads them from the stored
    /// `ConnectionConfig`, so a handshake field added in #248 survives a daemon
    /// restart exactly like the bearer token does.
    pub(crate) async fn reconnect(
        &self,
        socket_path: &Path,
        bearer_token: Option<&str>,
        system_id: Option<&str>,
        host_label: Option<&str>,
    ) -> Result<()> {
        let outbound_tx = Self::spawn_connection(
            socket_path,
            bearer_token,
            system_id,
            host_label,
            Arc::clone(&self.pending),
            self.signal_tx.clone(),
            self.drop_tx.clone(),
        )
        .await?;
        // Accept commands again, then swap the writer.
        self.pending.lock().await.reopen();
        self.conn.lock().await.outbound_tx = outbound_tx;
        Ok(())
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

        if self.conn.lock().await.outbound_tx.send(body).is_err() {
            self.pending.lock().await.map.remove(&id);
            return Err(anyhow!("failed to send uds request: writer closed"));
        }

        // Bound the wait for the response frame (#221): a server that accepts
        // the connection but never replies must not hang the caller forever. On
        // expiry we drop the pending slot so it can't leak and return a clear
        // transport error.
        match tokio::time::timeout(self.dispatch_timeout, rx).await {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(error))) => Err(anyhow!(error)),
            Ok(Err(_closed)) => Err(anyhow!("uds response channel closed")),
            Err(_elapsed) => {
                self.pending.lock().await.map.remove(&id);
                Err(anyhow!(
                    "uds command timed out after {:?} with no response from the server",
                    self.dispatch_timeout
                ))
            }
        }
    }
}
