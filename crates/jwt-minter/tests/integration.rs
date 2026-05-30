//! Integration tests for the local JWT minter (issue #101).

use std::path::PathBuf;
use std::time::Duration;

use desktop_assistant_auth_jwt as auth_jwt;
use desktop_assistant_jwt_minter::config::{MAX_TTL_SECS, MIN_TTL_SECS, MintConfig};
use desktop_assistant_jwt_minter::group::{self, uid_in_groups};
use desktop_assistant_jwt_minter::peer::{self, PeerIdentity};
use desktop_assistant_jwt_minter::request::{MintResponse, handle_request};
use desktop_assistant_jwt_minter::server::{ServerOptions, serve};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

fn test_config(dir: &TempDir) -> (MintConfig, String) {
    let key_path = dir.path().join("key");
    let signing_key = auth_jwt::ensure_signing_key_at(&key_path).expect("ensure key");
    let mut cfg = MintConfig::with_default_paths();
    cfg.signing_key_path = key_path;
    (cfg, signing_key)
}

fn test_peer() -> PeerIdentity {
    PeerIdentity {
        uid: peer::current_uid(),
        username: peer::username_for_uid(peer::current_uid())
            .expect("uid lookup")
            .expect("user exists"),
    }
}

#[tokio::test]
async fn peer_cred_identity_extraction_returns_correct_uid_and_username() {
    let dir = TempDir::new().expect("tempdir");
    let socket_path = dir.path().join("peer.sock");
    let listener = UnixListener::bind(&socket_path).expect("bind");

    let accept_task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        peer::extract_peer_identity(&stream).expect("extract")
    });

    let _client = UnixStream::connect(&socket_path).await.expect("connect");
    let identity = accept_task.await.expect("accept task");

    assert_eq!(identity.uid, peer::current_uid());
    let expected = peer::username_for_uid(peer::current_uid())
        .expect("username syscall")
        .expect("username present");
    assert_eq!(identity.username, expected);
    assert!(!identity.username.is_empty());
}

#[tokio::test]
async fn minted_token_round_trips_through_daemon_validator() {
    let dir = TempDir::new().expect("tempdir");
    let (cfg, signing_key) = test_config(&dir);
    let peer = test_peer();

    let raw = handle_request("{}", &peer, &cfg, &signing_key);
    let resp: MintResponse = serde_json::from_str(&raw).expect("parse response");
    assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
    let token = resp.token.expect("token present");

    // Independently re-read the key file — same path the daemon uses.
    let key_b = auth_jwt::read_signing_key_at(&cfg.signing_key_path).expect("read key");
    let claims = auth_jwt::decode(&token, &key_b, &cfg.issuer, &cfg.default_audience)
        .expect("daemon-style decode");
    assert_eq!(claims.sub, peer.username);
    assert_eq!(claims.iss, cfg.issuer);
    assert_eq!(claims.aud, cfg.default_audience);
}

#[test]
fn ttl_clamping_enforces_max_and_min() {
    let cfg = MintConfig::with_default_paths();

    // Below the floor: clamps up to min.
    assert_eq!(cfg.clamp_ttl(Some(1)), Duration::from_secs(MIN_TTL_SECS));
    // Above the ceiling: clamps down to max.
    assert_eq!(
        cfg.clamp_ttl(Some(u64::MAX)),
        Duration::from_secs(MAX_TTL_SECS),
    );
    // Inside the window: passed through.
    assert_eq!(cfg.clamp_ttl(Some(600)), Duration::from_secs(600));
    // None: default.
    assert_eq!(cfg.clamp_ttl(None), cfg.default_ttl);
}

#[test]
fn malformed_input_returns_error_json_not_panic() {
    let dir = TempDir::new().expect("tempdir");
    let (cfg, signing_key) = test_config(&dir);
    let peer = test_peer();

    for bad in [
        "not json at all",
        "{ unterminated",
        "[1, 2, 3]",
        "{\"ttl_seconds\": \"not a number\"}",
    ] {
        let raw = handle_request(bad, &peer, &cfg, &signing_key);
        let resp: MintResponse = serde_json::from_str(&raw).expect("response is valid JSON");
        assert!(
            resp.error.is_some(),
            "expected error for input {bad:?}, got {raw}"
        );
        assert!(resp.token.is_none());
    }
}

#[test]
fn default_audience_matches_daemon_expectation() {
    let dir = TempDir::new().expect("tempdir");
    let (cfg, signing_key) = test_config(&dir);
    let peer = test_peer();

    let raw = handle_request("{}", &peer, &cfg, &signing_key);
    let resp: MintResponse = serde_json::from_str(&raw).expect("parse");
    let token = resp.token.expect("token");

    let claims = auth_jwt::decode(
        &token,
        &signing_key,
        "org.desktopAssistant.local",
        "desktop-assistant-ws",
    )
    .expect("decode with daemon defaults");
    assert_eq!(claims.aud, "desktop-assistant-ws");
}

#[tokio::test]
async fn socket_is_removed_on_shutdown() {
    let dir = TempDir::new().expect("tempdir");
    let socket_path = dir.path().join("shutdown.sock");
    let key_path = dir.path().join("key");
    auth_jwt::ensure_signing_key_at(&key_path).expect("key");

    let mut cfg = MintConfig::with_default_paths();
    cfg.signing_key_path = key_path;

    let opts = ServerOptions {
        socket_path: socket_path.clone(),
        group_gate: None,
    };

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let serve_task = tokio::spawn(async move {
        serve(opts, cfg, async move {
            let _ = rx.await;
        })
        .await
    });

    // Give the server a moment to bind.
    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "socket should exist after bind");

    tx.send(()).expect("trigger shutdown");
    let result = serve_task.await.expect("serve join");
    result.expect("serve ok");

    assert!(
        !socket_path.exists(),
        "socket file should be removed after shutdown"
    );
}

#[test]
fn group_gate_rejects_non_member() {
    // Synthetic GID that's guaranteed not to appear in any real grouplist.
    let synthetic_gid = u32::MAX;
    let user_groups: Vec<u32> = vec![1000, 1001, 27]; // arbitrary fixture
    assert!(!uid_in_groups(synthetic_gid, &user_groups));
}

#[test]
fn group_gate_allows_member() {
    let user_groups: Vec<u32> = vec![1000, 1001, 27];
    assert!(uid_in_groups(1001, &user_groups));
}

#[test]
fn group_gate_allows_member_via_real_lookup() {
    // Look up the current user's primary GID and confirm it's in their
    // grouplist. This exercises `grouplist_for` against the real system.
    let uid = peer::current_uid();
    let username = peer::username_for_uid(uid)
        .expect("username syscall")
        .expect("user");
    let primary = group::primary_gid_for_uid(uid)
        .expect("primary gid syscall")
        .expect("user");
    let groups = group::grouplist_for(&username, primary).expect("grouplist");
    assert!(uid_in_groups(primary, &groups));
}

#[test]
fn nonexistent_group_at_startup_is_a_clean_error() {
    // A name astronomically unlikely to exist on any system.
    let name = "adelie-nosuch-group-zzzzzzzzzzzzzzzz";
    let result = group::resolve_group(name).expect("syscall ok");
    assert!(
        result.is_none(),
        "expected no group named {name}, got {result:?}"
    );
}

#[test]
fn existing_group_resolves_with_gid() {
    // "root" exists on every Linux system. We don't care about membership
    // — just that lookup of an existing name returns Some.
    let result = group::resolve_group("root").expect("syscall ok");
    let entry = result.expect("root exists");
    assert_eq!(entry.name, "root");
    assert_eq!(entry.gid, 0);
}

#[tokio::test]
async fn end_to_end_server_mints_token_for_local_client() {
    let dir = TempDir::new().expect("tempdir");
    let socket_path = dir.path().join("e2e.sock");
    let key_path = dir.path().join("key");
    let signing_key = auth_jwt::ensure_signing_key_at(&key_path).expect("key");

    let mut cfg = MintConfig::with_default_paths();
    cfg.signing_key_path = key_path.clone();

    let opts = ServerOptions {
        socket_path: socket_path.clone(),
        group_gate: None,
    };

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let serve_task = tokio::spawn(async move {
        serve(opts, cfg, async move {
            let _ = rx.await;
        })
        .await
    });

    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let mut stream = UnixStream::connect(&socket_path).await.expect("connect");
    stream.write_all(b"{}\n").await.expect("write request");

    let mut reader = BufReader::new(&mut stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.expect("read line");
    let resp: MintResponse = serde_json::from_str(line.trim()).expect("parse");
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let token = resp.token.expect("token");

    let claims = auth_jwt::decode(
        &token,
        &signing_key,
        "org.desktopAssistant.local",
        "desktop-assistant-ws",
    )
    .expect("decode");
    let expected_user = peer::username_for_uid(peer::current_uid())
        .expect("uid")
        .expect("user");
    assert_eq!(claims.sub, expected_user);

    let _ = tx.send(());
    let _ = serve_task.await.expect("join");
    // Quiet unused warning on PathBuf
    let _: PathBuf = key_path;
}
