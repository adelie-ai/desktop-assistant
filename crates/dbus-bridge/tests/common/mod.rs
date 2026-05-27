//! Shared test fixtures: a stub minter that returns a canned JWT and
//! a stub daemon UDS server that does the handshake + a configurable
//! request/response/event script.
//!
//! Kept small on purpose — these are the only two external surfaces
//! the bridge talks to, so a small fake of each is enough to exercise
//! every code path.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::transport::{read_frame, write_frame};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, mpsc, oneshot};

/// Whether the stub minter should succeed and what token to return.
#[derive(Debug, Clone)]
pub enum MinterScript {
    /// Return a one-shot success token.
    Success { token: String },
    /// Return an error string in the response body.
    Error { message: String },
    /// Hang (never reply) so the caller observes a timeout.
    HangForever,
    /// Reply with malformed JSON.
    MalformedReply,
}

/// Spawn a stub minter on `path`. Records each received request body.
pub async fn spawn_stub_minter(
    path: &Path,
    script: MinterScript,
) -> (Arc<Mutex<Vec<String>>>, oneshot::Sender<()>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path).expect("bind minter socket");
    let received = Arc::new(Mutex::new(Vec::<String>::new()));
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();

    let received_clone = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accept = listener.accept() => {
                    let Ok((mut stream, _)) = accept else { continue };
                    let received_clone = Arc::clone(&received_clone);
                    let script = script.clone();
                    tokio::spawn(async move {
                        let (read_half, mut write_half) = stream.split();
                        let mut reader = BufReader::new(read_half);
                        let mut line = String::new();
                        let _ = reader.read_line(&mut line).await;
                        received_clone.lock().await.push(line.trim().to_string());

                        match script {
                            MinterScript::Success { token } => {
                                let reply = serde_json::json!({
                                    "token": token,
                                    "exp": 9_999_999_999_u64,
                                });
                                let _ = write_half.write_all(reply.to_string().as_bytes()).await;
                                let _ = write_half.write_all(b"\n").await;
                            }
                            MinterScript::Error { message } => {
                                let reply = serde_json::json!({
                                    "error": message,
                                });
                                let _ = write_half.write_all(reply.to_string().as_bytes()).await;
                                let _ = write_half.write_all(b"\n").await;
                            }
                            MinterScript::HangForever => {
                                // Hold the stream forever.
                                tokio::time::sleep(Duration::from_secs(60)).await;
                            }
                            MinterScript::MalformedReply => {
                                let _ = write_half.write_all(b"not json").await;
                                let _ = write_half.write_all(b"\n").await;
                            }
                        }
                        let _ = write_half.flush().await;
                    });
                }
            }
        }
    });
    (received, stop_tx)
}

/// What a stub daemon does after the handshake.
#[derive(Debug, Clone)]
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
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
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
                    tokio::spawn(async move {
                        handle_daemon_connection(
                            stream,
                            script,
                            handshakes_clone,
                            requests_clone,
                            &mut sub_rx,
                        ).await;
                    });
                }
            }
        }
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
