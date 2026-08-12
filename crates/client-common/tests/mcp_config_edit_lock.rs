//! Cross-process serialization of `client-mcp.toml` edits (#1291).
//!
//! `client-mcp.toml` is machine-wide: every Adele client on the box reads and
//! writes the same file. `ClientMcpConfig::save` is atomic against a torn read,
//! but the transaction a caller performs is load, mutate, save, and only the
//! last step was protected. Two clients that load the same bytes and then save
//! their own change lose one of the two edits.
//!
//! [`ClientMcpConfig::edit`] takes an exclusive lock on a sidecar file for the
//! whole read-mutate-write transaction. That property is only observable
//! **between processes** - a `flock` belongs to an open file description, so a
//! test that races threads inside one process passes whether or not the lock is
//! there. So the load-bearing test here spawns real child processes.
//!
//! The children are this same test binary, re-executed and selected by name.
//! The child entry points are `#[ignore]`d, so a normal run never executes
//! them, and each one returns without doing anything when the environment that
//! drives it is absent. A child that did nothing cannot make its parent pass:
//! the parent asserts on the state of the file, not on the child's word.

#![cfg(feature = "mcp-host")]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use desktop_assistant_client_common::mcp_host::config::{ClientMcpConfig, McpServerConfig};

/// Absolute path of the config the child edits.
const ENV_CONFIG: &str = "ADELE_TEST_EDIT_CONFIG";
/// The child's own index, which names the servers it adds.
const ENV_CHILD: &str = "ADELE_TEST_EDIT_CHILD";
/// How many separate `edit` transactions the child performs.
const ENV_EDITS: &str = "ADELE_TEST_EDIT_COUNT";
/// Wall-clock instant (nanoseconds since the Unix epoch) every child waits for
/// before its first edit, so they collide instead of queueing by start order.
const ENV_START: &str = "ADELE_TEST_EDIT_START_NANOS";
/// File the lock-holding child creates once it holds the lock.
const ENV_READY: &str = "ADELE_TEST_LOCK_READY";

/// Child processes in the race, and edits each performs. 4 x 5 = 20 separate
/// read-mutate-write transactions against one file, each one contending with
/// three others.
const CHILDREN: usize = 4;
const EDITS_PER_CHILD: usize = 5;

/// How long a child may take before the parent kills it and fails. Far beyond
/// the barrier plus twenty locked edits, so it only fires on a real hang.
const CHILD_DEADLINE: Duration = Duration::from_secs(60);

/// Build a minimal [`McpServerConfig`] by name through the parser, so the test
/// does not depend on the cross-crate struct's full field set.
fn server(name: &str) -> McpServerConfig {
    ClientMcpConfig::from_toml(&format!(
        "[[servers]]\nname = \"{name}\"\ncommand = \"cmd\"\n"
    ))
    .expect("valid single-server toml")
    .servers
    .into_iter()
    .next()
    .expect("one server")
}

fn nanos_since_epoch(at: SystemTime) -> u128 {
    at.duration_since(UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_nanos()
}

/// The name each child/edit pair writes. Every one of these must survive.
fn edit_name(child: usize, edit: usize) -> String {
    format!("child-{child}-edit-{edit}")
}

// ----- Acceptance criterion: serialized across processes -----

/// Four separate processes each perform five `edit` transactions against one
/// config file, all released at the same wall-clock instant. Every server any
/// child asked for must be present at the end: a lost update drops one.
#[test]
fn edit_serializes_concurrent_editors_across_processes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");

    // Seed a real file so every child's first transaction reads a parsed
    // config rather than taking the absent-file first-write path.
    let mut seed = ClientMcpConfig::default();
    seed.add_server(server("seed")).expect("seed add");
    seed.save(&path).expect("seed save");

    // Give the children enough runway to be spawned and to reach the barrier.
    let start = nanos_since_epoch(SystemTime::now() + Duration::from_millis(1500));

    let mut kids = Vec::new();
    for child in 0..CHILDREN {
        let mut cmd = Command::new(std::env::current_exe().expect("current_exe"));
        cmd.args([
            "--exact",
            "child_process_edit_worker",
            "--ignored",
            "--test-threads=1",
        ])
        .env(ENV_CONFIG, &path)
        .env(ENV_CHILD, child.to_string())
        .env(ENV_EDITS, EDITS_PER_CHILD.to_string())
        .env(ENV_START, start.to_string());
        kids.push(cmd.spawn().expect("spawn child editor"));
    }

    // Reap every child before asserting, and kill any that outstays the
    // deadline: a regression to a blocking `lock()` must fail this test rather
    // than hang the suite in `wait`.
    let mut failures = Vec::new();
    for (child, kid) in kids.into_iter().enumerate() {
        match wait_bounded(kid, CHILD_DEADLINE) {
            Some(status) if status.success() => {}
            Some(status) => failures.push(format!("child {child} exited with {status}")),
            None => failures.push(format!(
                "child {child} did not finish within {CHILD_DEADLINE:?} and was killed"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "children did not all apply their edits: {failures:?}"
    );

    // Read strictly: the file must still parse, and hold every requested name.
    let contents = std::fs::read_to_string(&path).expect("read final config");
    let final_config = ClientMcpConfig::from_toml(&contents)
        .expect("the raced file must still be valid TOML with no duplicate names");
    let present: Vec<&str> = final_config
        .list_defined_servers()
        .iter()
        .map(|s| s.name.as_str())
        .collect();

    let mut missing = Vec::new();
    for child in 0..CHILDREN {
        for edit in 0..EDITS_PER_CHILD {
            let name = edit_name(child, edit);
            if !present.iter().any(|p| *p == name) {
                missing.push(name);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {} edits were lost to the race: {missing:?}\nsurvivors: {present:?}",
        missing.len(),
        CHILDREN * EDITS_PER_CHILD
    );
    assert!(
        present.contains(&"seed"),
        "the seeded definition was overwritten: {present:?}"
    );
}

/// Wait for `kid`, killing and reaping it if it outstays `deadline`. `None`
/// means it had to be killed.
fn wait_bounded(mut kid: std::process::Child, deadline: Duration) -> Option<ExitStatus> {
    let until = Instant::now() + deadline;
    loop {
        match kid.try_wait().expect("poll child") {
            Some(status) => return Some(status),
            None if Instant::now() >= until => {
                let _ = kid.kill();
                let _ = kid.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

/// The child half of [`edit_serializes_concurrent_editors_across_processes`].
///
/// `#[ignore]` keeps it out of a normal run, and it returns without doing
/// anything when the environment that drives it is absent, so
/// `cargo test -- --include-ignored` stays green. Nothing is lost by that: a
/// child that silently did nothing still fails the parent, whose assertion is
/// that every requested name is in the file at the end.
#[test]
#[ignore = "re-executed as a child process by edit_serializes_concurrent_editors_across_processes"]
fn child_process_edit_worker() {
    let Ok(config) = std::env::var(ENV_CONFIG) else {
        return;
    };
    let path = PathBuf::from(config);
    let child: usize = std::env::var(ENV_CHILD)
        .expect("child worker selected without an index")
        .parse()
        .expect("child index is a number");
    let edits: usize = std::env::var(ENV_EDITS)
        .expect("child worker selected without an edit count")
        .parse()
        .expect("edit count is a number");
    let start: u128 = std::env::var(ENV_START)
        .expect("child worker selected without a start instant")
        .parse()
        .expect("start instant is a number");

    // Spin to the barrier so the children contend rather than queue.
    while nanos_since_epoch(SystemTime::now()) < start {
        std::hint::spin_loop();
    }

    for edit in 0..edits {
        let name = edit_name(child, edit);
        ClientMcpConfig::edit(&path, |config| config.add_server(server(&name)))
            .unwrap_or_else(|err| panic!("child {child} edit {edit} ({name}) failed: {err}"));
    }
}

// ----- Acceptance criterion: an Err closure writes nothing -----

/// A change closure that returns `Err` leaves the file exactly as it was, byte
/// for byte.
#[test]
fn edit_change_returning_err_leaves_file_byte_identical() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let mut seed = ClientMcpConfig::default();
    seed.add_server(server("keep")).expect("seed add");
    seed.save(&path).expect("seed save");
    let before = std::fs::read(&path).expect("read before");

    let err = ClientMcpConfig::edit(&path, |config| {
        // Mutate first, to prove the mutation is discarded and not written.
        config.add_server(server("must-not-land")).expect("add");
        Err::<(), String>("the caller declined".to_string())
    })
    .expect_err("an Err closure must fail the transaction");
    assert!(err.contains("declined"), "got: {err}");

    assert_eq!(
        before,
        std::fs::read(&path).expect("read after"),
        "a declined change must leave the file byte-identical"
    );
}

// ----- The lock is released on every exit path -----

/// How long a following `edit` may take when the lock is genuinely free. Well
/// under the bounded retry, so a still-held lock shows up as a slow success
/// rather than passing unnoticed.
const UNCONTENDED: Duration = Duration::from_millis(500);

/// A failed transaction must not strand the lock: the next editor gets it at
/// once, rather than waiting out the retry.
#[test]
fn edit_releases_the_lock_when_the_change_returns_err() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");

    let err = ClientMcpConfig::edit(&path, |_config| {
        Err::<(), String>("the caller declined".to_string())
    })
    .expect_err("an Err closure must fail the transaction");
    assert!(err.contains("declined"), "got: {err}");

    let started = Instant::now();
    ClientMcpConfig::edit(&path, |config| config.add_server(server("after-err")))
        .expect("the next edit must not be blocked by the failed one");
    assert!(
        started.elapsed() < UNCONTENDED,
        "the lock was still held after a declined change: waited {:?}",
        started.elapsed()
    );
}

/// The same on the way out of a panic. The lock is released by dropping the
/// owned `File`, so unwinding through `edit` frees it with no unwind handling
/// of its own.
#[test]
fn edit_releases_the_lock_when_the_change_panics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ClientMcpConfig::edit(&path, |_config| -> Result<(), String> {
            panic!("the change closure blew up")
        })
    }));
    assert!(panicked.is_err(), "the panic must reach the caller");

    let started = Instant::now();
    ClientMcpConfig::edit(&path, |config| config.add_server(server("after-panic")))
        .expect("the next edit must not be blocked by the panicking one");
    assert!(
        started.elapsed() < UNCONTENDED,
        "the lock was still held after a panic: waited {:?}",
        started.elapsed()
    );
}

// ----- Acceptance criterion: an unparseable config is refused -----

/// A config that cannot be parsed is refused before the change closure runs,
/// and left byte-identical. `edit` reads with `from_toml`, not the tolerant
/// `load`, so a damaged file is never overwritten with an empty one.
#[test]
fn edit_refuses_unparseable_config_and_leaves_it_byte_identical() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let damaged = b"this is : not valid toml [[[\n";
    std::fs::write(&path, damaged).expect("write damaged config");

    let mut ran = false;
    let err = ClientMcpConfig::edit(&path, |config| {
        ran = true;
        config.add_server(server("must-not-land"))
    })
    .expect_err("an unparseable config must be refused");
    assert!(!ran, "the change closure must not run on a damaged config");
    // Specifically the parse failure. Every error `edit` can return names the
    // path, so asserting on the path would also accept a lock failure and let
    // this test pass without the refusal path ever running.
    assert!(
        err.contains("parse error"),
        "the error must name the parse failure; got: {err}"
    );
    assert_eq!(
        damaged.as_slice(),
        std::fs::read(&path).expect("read after").as_slice(),
        "a refused config must be left byte-identical"
    );
}

/// A duplicate server name is the other way `from_toml` fails closed, and it
/// must be refused for the same reason: overwriting would silently drop one of
/// the two definitions.
#[test]
fn edit_refuses_config_with_duplicate_server_names() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let duplicated = b"[[servers]]\nname = \"dup\"\ncommand = \"a\"\n\
                       [[servers]]\nname = \"dup\"\ncommand = \"b\"\n";
    std::fs::write(&path, duplicated).expect("write duplicated config");

    let err = ClientMcpConfig::edit(&path, |config| config.add_server(server("new")))
        .expect_err("a duplicate-name config must be refused");
    assert!(err.contains("duplicate"), "got: {err}");
    assert_eq!(
        duplicated.as_slice(),
        std::fs::read(&path).expect("read after").as_slice(),
        "a refused config must be left byte-identical"
    );
}

// ----- Acceptance criterion: an absent config is a first write -----

/// No file yet is not an error: the change closure sees an empty config and the
/// result is written, parent directory and all. The closure's return value
/// comes back to the caller.
#[test]
fn edit_on_absent_config_is_a_first_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A nested path exercises parent-directory creation for both the config
    // and its sidecar lock.
    let path = dir.path().join("nested").join("client-mcp.toml");
    assert!(!path.exists());

    let count = ClientMcpConfig::edit(&path, |config| {
        config.add_server(server("first"))?;
        Ok(config.list_defined_servers().len())
    })
    .expect("first write");
    assert_eq!(count, 1, "the closure's value must be returned");

    let written = ClientMcpConfig::from_toml(&std::fs::read_to_string(&path).expect("read"))
        .expect("the first write must be valid");
    assert_eq!(written.list_defined_servers().len(), 1);
    assert_eq!(written.list_defined_servers()[0].name, "first");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "expected 0600, got {:o}", mode & 0o777);
    }
}

// ----- Acceptance criterion: a held lock fails, bounded, with a named cause -----

/// The sidecar path `edit` locks. Derived here independently of the
/// implementation so the test pins the contract rather than following it.
fn lock_path(config: &Path) -> PathBuf {
    let mut name = config
        .file_name()
        .expect("config has a file name")
        .to_owned();
    name.push(".lock");
    config.with_file_name(name)
}

/// Take the sidecar lock the way another process would, and hold it.
///
/// A `flock` belongs to an open file description, so a second `open` in this
/// same process contends exactly as a second process does - which is what makes
/// this test able to stand in for one.
fn hold_lock(config: &Path) -> std::fs::File {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(config))
        .expect("open sidecar lock");
    file.try_lock().expect("take sidecar lock");
    file
}

/// A lock held by another editor makes `edit` fail with a message naming the
/// cause, inside its bounded retry - it must not park the caller forever.
#[test]
fn edit_fails_with_named_cause_when_lock_held_within_bounded_retry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    ClientMcpConfig::default().save(&path).expect("seed save");
    let before = std::fs::read(&path).expect("read before");

    let held = hold_lock(&path);

    let started = Instant::now();
    let err = ClientMcpConfig::edit(&path, |config| config.add_server(server("blocked")))
        .expect_err("a held lock must fail the transaction");
    let waited = started.elapsed();

    assert!(
        err.contains("another Adele client is editing"),
        "the error must name the cause; got: {err}"
    );
    // Bounded at roughly two seconds. The bracket is wide enough not to flake
    // on a loaded machine and tight enough to fail if the wait is dropped, or
    // widened to something a person would experience as a hang.
    assert!(
        waited >= Duration::from_secs(1),
        "must retry for about two seconds, gave up after {waited:?}"
    );
    assert!(
        waited < Duration::from_secs(6),
        "must give up after about two seconds, waited {waited:?}"
    );
    assert_eq!(
        before,
        std::fs::read(&path).expect("read after"),
        "a refused edit must leave the file byte-identical"
    );

    drop(held);
    // Once the holder releases, the very next edit succeeds.
    ClientMcpConfig::edit(&path, |config| config.add_server(server("after")))
        .expect("edit after the lock is released");
}

// ----- Acceptance criterion: readers are never blocked -----

/// `load` takes no lock, so a reader is served while the edit lock is held.
///
/// Named for what it checks: the lock is held here directly, not by an edit in
/// flight. What a reader has to survive during a real edit is the config's
/// atomic replace, and that is `save`'s own property.
///
/// The read runs on its own thread and the result is collected with a timeout,
/// so a `load` that started waiting on the lock fails this test instead of
/// hanging it.
#[test]
fn load_takes_no_lock_while_the_edit_lock_is_held() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let mut seed = ClientMcpConfig::default();
    seed.add_server(server("visible")).expect("seed add");
    seed.save(&path).expect("seed save");

    let held = hold_lock(&path);

    let (tx, rx) = std::sync::mpsc::channel();
    let reader_path = path.clone();
    std::thread::spawn(move || {
        let read = ClientMcpConfig::load(&reader_path);
        let _ = tx.send(
            read.list_defined_servers()
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
        );
    });

    let names = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("a reader must not wait on the edit lock");
    assert_eq!(names, vec!["visible".to_string()]);
    drop(held);
}

// ----- Acceptance criterion: the lock file's mode, and leftovers -----

/// The sidecar is created 0600: it sits beside a 0600 config in a shared
/// config directory and must not widen what is visible there.
#[test]
#[cfg(unix)]
fn edit_creates_the_lock_file_0600() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    assert!(!lock_path(&path).exists());

    ClientMcpConfig::edit(&path, |config| config.add_server(server("a"))).expect("edit");

    let mode = std::fs::metadata(lock_path(&path))
        .expect("the sidecar lock must exist after an edit")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "expected 0600, got {:o}", mode & 0o777);
}

/// A lock file left behind by a dead editor is just a file - `flock` releases
/// when the process dies, so nothing about the leftover blocks a later edit.
#[test]
fn a_leftover_lock_file_does_not_block_a_later_edit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    ClientMcpConfig::default().save(&path).expect("seed save");
    // Left behind, unlocked, with stale content in it.
    std::fs::write(lock_path(&path), b"stale").expect("write leftover lock");

    ClientMcpConfig::edit(&path, |config| config.add_server(server("later")))
        .expect("a leftover lock file must not block an edit");

    let after = ClientMcpConfig::load(&path);
    assert_eq!(after.list_defined_servers().len(), 1);
    assert_eq!(after.list_defined_servers()[0].name, "later");
}

/// A lock held by a process that is then killed is released by the kernel, so
/// the next edit proceeds without waiting out the retry.
///
/// This is the case the doc comment claims and the unlocked-leftover test does
/// not reach: the file is left behind by a process that died mid-edit.
#[test]
fn a_lock_held_by_a_dead_process_does_not_block_a_later_edit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let ready = dir.path().join("holder-ready");

    let mut holder = Command::new(std::env::current_exe().expect("current_exe"))
        .args([
            "--exact",
            "child_process_lock_holder",
            "--ignored",
            "--test-threads=1",
        ])
        .env(ENV_CONFIG, &path)
        .env(ENV_READY, &ready)
        .spawn()
        .expect("spawn lock holder");

    let until = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        assert!(
            Instant::now() < until,
            "the lock holder never took the lock"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    // The holder really has it: this process cannot take it now.
    let probe = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(&path))
        .expect("open sidecar lock");
    assert!(
        probe.try_lock().is_err(),
        "the holder was expected to hold the lock"
    );
    drop(probe);

    holder.kill().expect("kill the lock holder");
    holder.wait().expect("reap the lock holder");

    let started = Instant::now();
    ClientMcpConfig::edit(&path, |config| config.add_server(server("after-death")))
        .expect("a lock held by a dead process must not block a later edit");
    assert!(
        started.elapsed() < UNCONTENDED,
        "the dead holder's lock was still in force: waited {:?}",
        started.elapsed()
    );
}

/// Takes the sidecar lock, reports that it has it, and waits to be killed.
///
/// `#[ignore]`d and inert without its environment, like the edit worker.
#[test]
#[ignore = "re-executed as a child process by a_lock_held_by_a_dead_process_does_not_block_a_later_edit"]
fn child_process_lock_holder() {
    let (Ok(config), Ok(ready)) = (std::env::var(ENV_CONFIG), std::env::var(ENV_READY)) else {
        return;
    };
    let held = hold_lock(Path::new(&config));
    std::fs::write(ready, b"held").expect("report that the lock is held");
    // The parent kills this process; the kernel releases the lock.
    std::thread::sleep(Duration::from_secs(120));
    drop(held);
}

/// The sidecar is a file of its own and is never renamed over: the config's own
/// atomic replace must leave it in place, or the next process would lock a
/// different inode and serialize against nobody.
#[test]
fn the_lock_file_survives_the_config_being_replaced() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");

    ClientMcpConfig::edit(&path, |config| config.add_server(server("one"))).expect("first edit");
    let first_inode = inode_of(&lock_path(&path));

    ClientMcpConfig::edit(&path, |config| config.add_server(server("two"))).expect("second edit");
    assert_eq!(
        first_inode,
        inode_of(&lock_path(&path)),
        "the sidecar must be the same inode across an atomic config replace"
    );

    let after = ClientMcpConfig::load(&path);
    assert_eq!(after.list_defined_servers().len(), 2);
}

#[cfg(unix)]
fn inode_of(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).expect("lock file metadata").ino()
}

#[cfg(not(unix))]
fn inode_of(path: &Path) -> u64 {
    // No inode concept to assert on; existence is the portable part.
    assert!(path.exists(), "the sidecar lock must exist");
    0
}
