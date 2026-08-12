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
//! The children are this same test binary, re-executed with
//! [`CHILD_TEST`](child) selected by name. `child_process_edit_worker` is
//! `#[ignore]`d so a normal run never executes it, and it panics when the
//! environment that drives it is absent, so a mis-selected child fails loudly
//! rather than exiting 0 with nothing done.

#![cfg(feature = "mcp-host")]

use std::path::{Path, PathBuf};
use std::process::Command;
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

/// Child processes in the race, and edits each performs. 4 x 5 = 20 separate
/// read-mutate-write transactions against one file, each one contending with
/// three others.
const CHILDREN: usize = 4;
const EDITS_PER_CHILD: usize = 5;

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
            "--nocapture",
        ])
        .env(ENV_CONFIG, &path)
        .env(ENV_CHILD, child.to_string())
        .env(ENV_EDITS, EDITS_PER_CHILD.to_string())
        .env(ENV_START, start.to_string());
        kids.push(cmd.spawn().expect("spawn child editor"));
    }

    for (child, mut kid) in kids.into_iter().enumerate() {
        let status = kid.wait().expect("wait for child editor");
        assert!(
            status.success(),
            "child {child} failed with {status}; its edits were not all applied"
        );
    }

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
        present.iter().any(|p| *p == "seed"),
        "the seeded definition was overwritten: {present:?}"
    );
}

/// The child half of [`edit_serializes_concurrent_editors_across_processes`].
///
/// `#[ignore]` keeps it out of a normal run; the parent selects it by name with
/// `--ignored --exact`. It panics when its environment is absent so a child
/// that was never really selected cannot exit 0 and read as a clean run.
#[test]
#[ignore = "re-executed as a child process by edit_serializes_concurrent_editors_across_processes"]
fn child_process_edit_worker() {
    let path = PathBuf::from(
        std::env::var(ENV_CONFIG)
            .expect("child worker selected without a config path; the parent must set it"),
    );
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

/// A change closure that returns `Err` releases the lock and leaves the file
/// exactly as it was, byte for byte.
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
    assert!(
        err.contains("parse error") || err.contains("client-mcp.toml"),
        "the error must name the parse failure or the file; got: {err}"
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
    let mut name = config.file_name().expect("config has a file name").to_owned();
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
    // Bounded: roughly two seconds of retry, generously bracketed so a loaded
    // machine does not turn this into a flake.
    assert!(
        waited >= Duration::from_millis(500),
        "must retry before giving up, gave up after {waited:?}"
    );
    assert!(
        waited < Duration::from_secs(15),
        "must give up rather than hang, waited {waited:?}"
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

/// `load` takes no lock, so a reader is served while an edit holds the sidecar.
#[test]
fn load_is_not_blocked_while_an_edit_holds_the_lock() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("client-mcp.toml");
    let mut seed = ClientMcpConfig::default();
    seed.add_server(server("visible")).expect("seed add");
    seed.save(&path).expect("seed save");

    let held = hold_lock(&path);

    let started = Instant::now();
    let read = ClientMcpConfig::load(&path);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a reader must not wait on the edit lock"
    );
    assert_eq!(read.list_defined_servers().len(), 1);
    assert_eq!(read.list_defined_servers()[0].name, "visible");
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
