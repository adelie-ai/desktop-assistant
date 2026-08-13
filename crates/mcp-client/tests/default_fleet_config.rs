//! The shipped default MCP fleet config (`deploy/mcp/mcp_servers.default.toml`)
//! is a contract: the composable base image (#492) bundles it and the daemon
//! seeds it on first boot (#491). A typo in a field name or path would silently
//! break the whole fleet (the seed logs a warning and the daemon starts with no
//! MCP servers), so these tests parse the REAL shipped file through the REAL
//! loader + seeder and pin its contents.

use std::fs;
use std::path::PathBuf;

use desktop_assistant_mcp_client::config::{ensure_mcp_config_exists, load_mcp_configs};

/// The shipped default, relative to this crate (`crates/mcp-client`).
fn shipped_default() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../deploy/mcp/mcp_servers.default.toml")
}

/// Copy the shipped default into `dir` so the loader's 0600-enforcement chmods a
/// throwaway copy, never the tracked repo file.
fn staged_source(dir: &std::path::Path) -> PathBuf {
    let src = dir.join("default.toml");
    fs::copy(shipped_default(), &src).expect("copy shipped default into temp source");
    src
}

#[test]
fn shipped_default_seeds_and_parses_to_the_expected_fleet() {
    assert!(
        shipped_default().exists(),
        "shipped default config missing at {}",
        shipped_default().display()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let src = staged_source(dir.path());
    let dest = dir.path().join("mcp_servers.toml");

    // Seeds when absent (the first-boot path #491 takes in the container).
    assert!(
        ensure_mcp_config_exists(&dest, Some(&src)).expect("seed"),
        "should seed the default when dest is absent"
    );
    assert!(dest.exists(), "seed must create the dest file");

    let servers = load_mcp_configs(&dest).expect("load seeded config");
    assert_eq!(servers.len(), 12, "expected the full 12-server fleet");

    let enabled: Vec<&str> = servers
        .iter()
        .filter(|s| s.enabled)
        .map(|s| s.name.as_str())
        .collect();
    let disabled: Vec<&str> = servers
        .iter()
        .filter(|s| !s.enabled)
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(
        enabled,
        [
            "weather-forecast",
            "geocode",
            "openstreetmap",
            "cve",
            "tasks",
            "timeclock",
            "skills",
            "web"
        ],
        "the safe, zero-config servers ship enabled; `web` joined once Chromium \
         was bundled into the base image (#508) with the SSRF guard on"
    );
    assert_eq!(
        disabled,
        ["terminal", "command", "fileio", "internet-radio"],
        "the dangerous / dependency-needing servers ship disabled; \
         `homeassistant` is not among them because the image cannot build it \
         (#1235, #1290)"
    );
}

#[test]
fn every_shipped_server_declares_an_absolute_stdio_command() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = staged_source(dir.path());
    let dest = dir.path().join("mcp_servers.toml");
    ensure_mcp_config_exists(&dest, Some(&src)).expect("seed");
    let servers = load_mcp_configs(&dest).expect("load");

    for s in &servers {
        assert!(
            s.command.starts_with("/opt/adele/mcp/"),
            "{}: command must be an absolute bundled path (the daemon spawns via Command::new with no PATH augmentation), got {:?}",
            s.name,
            s.command
        );
        assert_eq!(
            s.args.first().map(String::as_str),
            Some("serve"),
            "{}: fleet servers launch via `<bin> serve`",
            s.name
        );
        assert!(
            s.http.is_none(),
            "{}: the bundled fleet is stdio, not http",
            s.name
        );
    }

    // `web` carries the container Chrome flags on top of `serve` (Chromium is
    // bundled in the base image; #508). Pin them so the contract is explicit.
    let web = servers
        .iter()
        .find(|s| s.name == "web")
        .expect("web server present in the fleet");
    assert_eq!(
        web.args,
        vec![
            "serve".to_string(),
            "--chrome-arg=--no-sandbox".to_string(),
            "--chrome-arg=--disable-dev-shm-usage".to_string(),
        ],
        "web launches headless Chrome with the container-required flags"
    );
}

/// `tasks` and `internet-radio` are the only two servers scoped to receive a
/// session-bus/runtime-dir variable via `inherit_env` (#910 round 3) -
/// deliberately narrow rather than a global grant, since both variables are
/// also the standard D-Bus session-bus auto-discovery route to the
/// freedesktop Secret Service. Pin the grant to exactly these two servers so
/// a future edit to the shipped config cannot silently widen it.
#[test]
fn only_tasks_and_internet_radio_inherit_a_session_variable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = staged_source(dir.path());
    let dest = dir.path().join("mcp_servers.toml");
    ensure_mcp_config_exists(&dest, Some(&src)).expect("seed");
    let servers = load_mcp_configs(&dest).expect("load");

    // Sorted by name: `[[servers]]` block order in the TOML is not part of
    // the contract this test pins, so reordering the shipped config must not
    // fail it spuriously.
    let mut with_inherit_env: Vec<(&str, &[String])> = servers
        .iter()
        .filter(|s| !s.inherit_env.is_empty())
        .map(|s| (s.name.as_str(), s.inherit_env.as_slice()))
        .collect();
    with_inherit_env.sort_by_key(|(name, _)| *name);

    assert_eq!(
        with_inherit_env,
        [
            ("internet-radio", ["XDG_RUNTIME_DIR".to_string()].as_slice()),
            ("tasks", ["DBUS_SESSION_BUS_ADDRESS".to_string()].as_slice()),
        ],
        "only tasks (session-bus signal service) and internet-radio (mpv's \
         audio session) should inherit a session-scoped variable"
    );
}

#[test]
fn seeding_never_clobbers_an_existing_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = staged_source(dir.path());
    let dest = dir.path().join("mcp_servers.toml");
    fs::write(&dest, "servers = []\n").expect("write pre-existing empty config");

    // dest already exists -> the default must NOT overwrite the operator's config.
    assert!(
        !ensure_mcp_config_exists(&dest, Some(&src)).expect("seed"),
        "an existing config must never be clobbered by the shipped default"
    );
    assert!(
        load_mcp_configs(&dest).expect("load").is_empty(),
        "the pre-existing (empty) config must survive untouched"
    );
}

// --- The image actually carries what the config offers -----------------------
//
// `every_shipped_server_declares_an_absolute_stdio_command` above checks the
// SHAPE of each command path. It cannot check that the binary exists, because
// the binary is produced by `Dockerfile.fleet` in another stage entirely. That gap let the
// shipped config offer `homeassistant`, whose repository is private and so was
// removed from the build context (#1235), for as long as the server existed:
// enabling it from the settings UI spawned a path that was never in the image
// (#1290). These tests close the gap by reading the Dockerfile.

/// The repo root, from this crate (`crates/mcp-client`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_repo_file(relative: &str) -> String {
    let path = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `for d in <dirs>; do` list in `Dockerfile.fleet`, in file order.
///
/// The Dockerfile names the fleet twice - once to build and once to collect the
/// binaries into `/out/mcp/` - and the runtime stage copies that directory to
/// `/opt/adele/mcp/`. Both lists must agree, so both are returned.
fn dockerfile_fleet_loops(dockerfile: &str) -> Vec<Vec<String>> {
    // Undo shell line continuations so a list can be read as one span.
    let joined = dockerfile.replace("\\\n", " ");
    joined
        .match_indices("for d in ")
        .map(|(at, marker)| {
            let rest = &joined[at + marker.len()..];
            let end = rest
                .find("; do")
                .expect("a `for d in` list must be terminated by `; do`");
            rest[..end].split_whitespace().map(str::to_string).collect()
        })
        .collect()
}

/// The `*-mcp` source trees `Dockerfile.fleet` copies into the build stage.
fn dockerfile_fleet_copies(dockerfile: &str) -> Vec<String> {
    dockerfile
        .lines()
        .filter_map(|l| l.strip_prefix("COPY "))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(|src| src.strip_suffix('/'))
        .filter(|src| src.ends_with("-mcp"))
        .map(str::to_string)
        .collect()
}

/// The fleet the image is built from and ships: the single agreed list.
fn fleet_in_the_image() -> Vec<String> {
    let dockerfile = read_repo_file("Dockerfile.fleet");
    let loops = dockerfile_fleet_loops(&dockerfile);
    assert_eq!(
        loops.len(),
        2,
        "Dockerfile.fleet should name the fleet exactly twice (build, then collect)"
    );
    assert_eq!(
        loops[0], loops[1],
        "the build loop and the collect loop must name the same servers, or the \
         image builds a binary it never copies (or copies one it never built)"
    );
    assert_eq!(
        dockerfile_fleet_copies(&dockerfile),
        loops[0],
        "every fleet source COPYed into the build stage must be built, and every \
         server built must have had its source COPYed"
    );
    // Everything above maps a config command basename onto a Dockerfile
    // DIRECTORY name, which only holds while the collect step names the binary
    // after its directory. Pin that step rather than assume it.
    assert!(
        dockerfile.contains(r#"cp "$d/target/release/$d" "/out/mcp/$d""#),
        "the collect step must copy $d/target/release/$d to /out/mcp/$d. If it \
         stops naming the binary after its directory, a config command can point \
         at a path the image does not carry while these tests still pass"
    );
    assert!(
        dockerfile.contains("COPY --from=builder /out/mcp/ /opt/adele/mcp/"),
        "the runtime stage must place /out/mcp at /opt/adele/mcp, or the paths \
         the shipped config names are not the paths the image carries"
    );
    loops.into_iter().next().expect("checked non-empty above")
}

/// The property `every_shipped_server_declares_an_absolute_stdio_command` fell
/// short of: the command is not merely shaped like a bundled path, the image
/// really does put a binary there.
#[test]
fn every_command_in_the_shipped_config_is_a_binary_the_image_builds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = staged_source(dir.path());
    let dest = dir.path().join("mcp_servers.toml");
    ensure_mcp_config_exists(&dest, Some(&src)).expect("seed");
    let servers = load_mcp_configs(&dest).expect("load");

    let built = fleet_in_the_image();
    for s in &servers {
        let binary = s
            .command
            .strip_prefix("/opt/adele/mcp/")
            .unwrap_or_else(|| panic!("{}: command is not a bundled path", s.name));
        assert!(
            built.iter().any(|d| d == binary),
            "{}: the shipped config offers {:?}, but Dockerfile.fleet never builds \
             {binary}. Enabling it from the settings UI spawns a path that is not \
             in the image. Either build it, or drop the server from the shipped \
             default. Built: {built:?}",
            s.name,
            s.command,
        );
    }
}

/// The docs tell a reader which sources to stage into the build context. Stage
/// too few and the build fails loudly; stage one the Dockerfile ignores and it
/// succeeds while quietly producing an image without that server - which is how
/// #1290 survived two images built five weeks apart.
#[test]
fn the_documented_staging_lists_name_exactly_the_sources_the_image_needs() {
    let mut expected = vec!["desktop-assistant".to_string()];
    expected.extend(fleet_in_the_image());
    expected.sort();

    for doc in ["docs/k8s-deployment.md", "deploy/mcp/README.md"] {
        let text = read_repo_file(doc);
        let mut loops: Vec<(usize, &str)> = text
            .match_indices("for r in ")
            .chain(text.match_indices("for repo in "))
            .collect();
        loops.sort_by_key(|(at, _)| *at);
        // Exactly one, like the Dockerfile check. With `.next()` alone, a second
        // shell example added to the doc later would never be inspected, and
        // this test would report on whichever loop happened to come first.
        assert_eq!(
            loops.len(),
            1,
            "{doc}: expected exactly one staging loop, found {}. Checking only \
             the first would leave the others free to drift",
            loops.len()
        );
        let (at, marker) = loops[0];
        let rest = &text[at + marker.len()..];
        let end = rest
            .find("; do")
            .unwrap_or_else(|| panic!("{doc}: staging loop is not terminated by `; do`"));
        let mut listed: Vec<String> = rest[..end]
            .replace('\\', " ")
            .split_whitespace()
            .map(str::to_string)
            .collect();
        listed.sort();
        assert_eq!(
            listed, expected,
            "{doc}: the staging list and Dockerfile.fleet disagree. A source \
             staged but never built yields an image silently missing that server"
        );
    }
}

/// The fleet size also appears written out in prose, in four files across two
/// directories. The staging-list test above holds the LISTS to the Dockerfile,
/// but a number in a sentence is a separate claim and drifts on its own: when
/// the fleet lost a server, three of the four sentences went stale, and a
/// hand search for them still missed one.
///
/// This check is deliberately narrow - it matches two phrasings, not English -
/// so it asserts that it matched something. A doc that rephrases its way out of
/// the pattern fails here rather than passing vacuously.
#[test]
fn every_written_out_fleet_size_matches_the_fleet() {
    const DOCS: [&str; 4] = [
        "Dockerfile.fleet",
        "docs/k8s-deployment.md",
        "deploy/mcp/README.md",
        "deploy/k8s/README.md",
    ];
    let expected = fleet_in_the_image().len();
    let mut found = 0usize;

    for doc in DOCS {
        let text = read_repo_file(doc);
        for (idx, _) in text
            .match_indices("bundled MCP servers")
            .chain(text.match_indices("`*-mcp`"))
        {
            // The count is the last number before the phrase.
            let before = &text[..idx];
            let number: String = before
                .trim_end()
                .rsplit(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap_or_default()
                .to_string();
            let Ok(stated) = number.parse::<usize>() else {
                continue; // the phrase without a count in front of it
            };
            found += 1;
            assert_eq!(
                stated, expected,
                "{doc}: says {stated} servers, but Dockerfile.fleet builds {expected}"
            );
        }
    }

    assert!(
        found >= DOCS.len(),
        "expected a written-out fleet size in each of {} files, matched {found}. \
         A doc that rephrases past this check stops being checked, so widen the \
         match rather than leaving it silently vacuous",
        DOCS.len()
    );
}
