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
use tempfile::TempDir;
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

/// Longest an AF_UNIX socket path may be, in bytes.
///
/// `sockaddr_un::sun_path` is a fixed-size array — 104 bytes on macOS/BSD, 108
/// on Linux, NUL terminator included — and `bind` rejects anything longer with a
/// bare `InvalidInput`. We hold to the *smaller* of the two on every platform so
/// a path that binds on Linux also binds on macOS; that is also why this needs no
/// `cfg(target_os)`, which would let the two diverge silently.
const MAX_SOCKET_PATH: usize = 104;

/// A temp dir short enough to root AF_UNIX socket paths in.
///
/// `tempfile::tempdir()` honours `$TMPDIR`, and macOS sets that to a ~49-byte
/// per-user path (`/var/folders/<hash>/T/`). Adding tempfile's own segment and a
/// socket filename overran `sun_path` by a handful of bytes, so the bridge tests
/// could not bind at all (#672). Rooting at `/tmp` — short, and present on both
/// Linux and macOS — leaves ample headroom.
///
/// Falls back to the default location when `/tmp` is unusable (an unusual
/// sandbox); [`unique_socket_path`] still checks the budget, so the failure is a
/// clear assertion rather than a cryptic `bind` error.
pub fn socket_tempdir() -> TempDir {
    TempDir::new_in("/tmp")
        .or_else(|_| TempDir::new())
        .expect("create a temp dir for sockets")
}

/// Return a unique tempdir-rooted path to use for one of the stub
/// sockets. Caller is responsible for keeping the tempdir alive.
///
/// The suffix is deliberately short (8 hex chars, not a full 36-char UUID):
/// every byte counts against [`MAX_SOCKET_PATH`], and uniqueness only has to
/// hold among a handful of sockets within one temp dir.
pub fn unique_socket_path(dir: &Path, name: &str) -> PathBuf {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let path = dir.join(format!("{name}-{}.sock", &id[..8]));
    // Fail loudly and specifically here rather than letting `bind` return an
    // opaque InvalidInput several frames away from the actual cause.
    assert!(
        path.as_os_str().len() < MAX_SOCKET_PATH,
        "socket path is {} bytes, over the {MAX_SOCKET_PATH}-byte AF_UNIX limit: {}",
        path.as_os_str().len(),
        path.display(),
    );
    path
}

#[cfg(test)]
mod socket_path_tests {
    use super::*;

    /// The generated path must fit `sun_path` with room to spare — not merely
    /// squeak under it. Before #672 this was over by 4 bytes on macOS, which is
    /// the kind of margin that breaks again on the next slightly-deeper $TMPDIR.
    #[test]
    fn generated_socket_path_has_headroom() {
        let dir = socket_tempdir();
        let path = unique_socket_path(dir.path(), "daemon");
        let len = path.as_os_str().len();
        assert!(
            len < MAX_SOCKET_PATH,
            "path {len} bytes exceeds the {MAX_SOCKET_PATH}-byte limit: {}",
            path.display()
        );
        // A third of the budget free, so a deeper temp root or a longer stub
        // name does not silently put us back on the edge.
        assert!(
            len < MAX_SOCKET_PATH * 2 / 3,
            "path {len} bytes leaves too little headroom under {MAX_SOCKET_PATH}: {}",
            path.display()
        );
    }

    /// The end-to-end property that actually failed: a socket at the generated
    /// path can be bound. Asserting on length alone would not catch a wrong
    /// limit, so bind a real listener.
    #[test]
    fn generated_socket_path_can_be_bound() {
        let dir = socket_tempdir();
        let path = unique_socket_path(dir.path(), "daemon");
        let listener = std::os::unix::net::UnixListener::bind(&path)
            .unwrap_or_else(|e| panic!("bind {} failed: {e}", path.display()));
        drop(listener);
    }

    /// Two sockets in the same dir must not collide — the short suffix still has
    /// to do its job.
    #[test]
    fn socket_paths_are_unique_within_a_dir() {
        let dir = socket_tempdir();
        let a = unique_socket_path(dir.path(), "daemon");
        let b = unique_socket_path(dir.path(), "daemon");
        assert_ne!(a, b, "two generated socket paths must differ");
    }
}
