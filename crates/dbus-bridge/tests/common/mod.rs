//! Shared test fixtures: a stub daemon UDS server that does the handshake + a
//! configurable request/response/event script.
//!
//! Kept small on purpose — the daemon is the only external surface the bridge
//! talks to (since #407 the local UDS hop is peer-cred authenticated, so there's
//! no minter), so a small fake is enough to exercise every code path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::transport::{read_frame, write_frame};
use serde_json::Value;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, mpsc, oneshot};

/// What a stub daemon does after the handshake.
///
/// Beyond `EchoAck`, the variants are reusable failure/event-injection
/// scaffolding for the bridge's failure-path / soak tests (#317/#318) — the
/// handshake-rejection and per-frame paths are unit-tested in `client-common`,
/// so they're currently unused here.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum DaemonScript {
    /// Accept the handshake; for every inbound request reply with
    /// `WsFrame::Result { id, result: Ack }`.
    EchoAck,
    /// Accept the handshake then immediately close.
    AcceptThenClose,
    /// Reject the handshake with a wire error frame.
    RejectHandshake { error: String },
    /// Accept handshake, then on receipt of any request, push the
    /// provided events first, then reply with `Ack`.
    EchoAckWithEvents { events: Vec<api::Event> },
}

/// Spawn a stub daemon at `path`. Returns:
/// - the path,
/// - a `Vec` of received handshake tokens,
/// - a `Vec` of received `WsRequest` envelopes (parsed),
/// - a oneshot to shut it down.
pub struct StubDaemonHandle {
    #[allow(dead_code)] // recorded for handshake-assertion tests (now in client-common)
    pub handshakes: Arc<Mutex<Vec<String>>>,
    pub requests: Arc<Mutex<Vec<api::WsRequest>>>,
    #[allow(dead_code)] // surfaced for future tests that push events dynamically
    pub event_tx: mpsc::UnboundedSender<api::Event>,
    pub stop_tx: oneshot::Sender<()>,
}

pub async fn spawn_stub_daemon(path: &Path, script: DaemonScript) -> StubDaemonHandle {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path).expect("bind daemon socket");
    let handshakes = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests = Arc::new(Mutex::new(Vec::<api::WsRequest>::new()));
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<api::Event>();
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();

    let handshakes_clone = Arc::clone(&handshakes);
    let requests_clone = Arc::clone(&requests);
    tokio::spawn(async move {
        // Track live connection tasks so stopping the daemon closes their
        // sockets (not just the accept loop). That's what lets a reconnect test
        // simulate a real daemon restart: aborting the task drops the server end,
        // the client sees EOF, and the Connector's reconnect supervisor fires.
        let mut conns: Vec<tokio::task::JoinHandle<()>> = Vec::new();
        loop {
            tokio::select! {
                _ = &mut stop_rx => {
                    for c in &conns {
                        c.abort();
                    }
                    break;
                }
                accept = listener.accept() => {
                    let Ok((stream, _)) = accept else { continue };
                    let script = script.clone();
                    let handshakes_clone = Arc::clone(&handshakes_clone);
                    let requests_clone = Arc::clone(&requests_clone);
                    // We rebuild a receiver per connection by passing a sub-channel.
                    let (sub_tx, mut sub_rx) = mpsc::unbounded_channel::<api::Event>();
                    // Drain any queued events on the outer channel into this sub-channel.
                    // Best-effort: tests typically queue events before requests fly.
                    while let Ok(ev) = event_rx.try_recv() {
                        let _ = sub_tx.send(ev);
                    }
                    conns.retain(|c| !c.is_finished());
                    conns.push(tokio::spawn(async move {
                        handle_daemon_connection(
                            stream,
                            script,
                            handshakes_clone,
                            requests_clone,
                            &mut sub_rx,
                        ).await;
                    }));
                }
            }
        }
        // `listener` drops here, unbinding the socket so a replacement daemon
        // can rebind the same path.
    });

    StubDaemonHandle {
        handshakes,
        requests,
        event_tx,
        stop_tx,
    }
}

async fn handle_daemon_connection(
    stream: UnixStream,
    script: DaemonScript,
    handshakes: Arc<Mutex<Vec<String>>>,
    requests: Arc<Mutex<Vec<api::WsRequest>>>,
    events: &mut mpsc::UnboundedReceiver<api::Event>,
) {
    let (mut read_half, mut write_half) = stream.into_split();

    // Handshake.
    let Ok(handshake_bytes) = read_frame(&mut read_half).await else {
        return;
    };
    let handshake: Value = match serde_json::from_slice(&handshake_bytes) {
        Ok(v) => v,
        Err(_) => return,
    };
    let token = handshake
        .get("jwt")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();
    handshakes.lock().await.push(token);

    if let DaemonScript::RejectHandshake { ref error } = script {
        let frame = api::WsFrame::Error {
            id: String::new(),
            error: error.clone(),
        };
        let body = serde_json::to_vec(&frame).unwrap();
        let _ = write_frame(&mut write_half, &body).await;
        return;
    }
    if let DaemonScript::AcceptThenClose = script {
        return;
    }

    // Dispatch loop.
    loop {
        let frame = match read_frame(&mut read_half).await {
            Ok(b) => b,
            Err(_) => break,
        };
        if frame.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_slice::<api::WsRequest>(&frame) else {
            continue;
        };
        requests.lock().await.push(req.clone());

        if let DaemonScript::EchoAckWithEvents { events: ref queued } = script {
            for ev in queued {
                let f = api::WsFrame::Event { event: ev.clone() };
                let body = serde_json::to_vec(&f).unwrap();
                if write_frame(&mut write_half, &body).await.is_err() {
                    return;
                }
            }
        }
        // Drain any dynamically-injected events.
        while let Ok(ev) = events.try_recv() {
            let f = api::WsFrame::Event { event: ev };
            let body = serde_json::to_vec(&f).unwrap();
            if write_frame(&mut write_half, &body).await.is_err() {
                return;
            }
        }

        let reply = api::WsFrame::Result {
            id: req.id,
            result: api::CommandResult::Ack,
        };
        let body = serde_json::to_vec(&reply).unwrap();
        if write_frame(&mut write_half, &body).await.is_err() {
            return;
        }
    }
}

/// Return a unique tempdir-rooted path to use for one of the stub
/// sockets. Caller is responsible for keeping the tempdir alive.
pub fn unique_socket_path(dir: &Path, name: &str) -> PathBuf {
    let id = uuid::Uuid::new_v4();
    dir.join(format!("{name}-{id}.sock"))
}
