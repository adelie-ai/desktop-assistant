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
    assert_eq!(servers.len(), 13, "expected the full 13-server fleet");

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
        [
            "terminal",
            "command",
            "fileio",
            "homeassistant",
            "internet-radio"
        ],
        "the dangerous / dependency-needing servers ship disabled"
    );
}

#[test]
fn every_shipped_server_is_a_bundled_stdio_binary() {
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
