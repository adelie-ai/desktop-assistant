//! The tool-provenance table (`core::tool_provenance::CLASSIFIED_SOURCES`)
//! must name every tool this build ships (#741).
//!
//! `core` cannot see the built-in tool list or the shipped fleet config: it
//! sits below both. This crate sits above both, so the coverage check lives
//! here, next to the fleet-config contract test. It follows the pattern of
//! `builtin_provider_map_is_exhaustive`: the table is held against the real
//! list at test time, so a tool or a server added without a classification
//! fails a named test rather than falling silently into the gated default.

use std::fs;
use std::path::PathBuf;

use desktop_assistant_core::tool_provenance::{CLASSIFIED_SOURCES, ToolTier, classify_tool};
use desktop_assistant_mcp_client::config::load_mcp_configs;
use desktop_assistant_mcp_client::executor::BuiltinToolService;

/// The shipped default fleet config, relative to this crate.
fn shipped_default() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../deploy/mcp/mcp_servers.default.toml")
}

#[test]
fn every_builtin_tool_has_an_explicit_classification() {
    let unclassified: Vec<&str> = BuiltinToolService::ALL_TOOL_NAMES
        .iter()
        .copied()
        .filter(|name| classify_tool(name).tier == ToolTier::Unclassified)
        .collect();
    assert!(
        unclassified.is_empty(),
        "these built-in tools have no entry in CLASSIFIED_SOURCES, so they would be gated \
         after any external ingest: {unclassified:?}"
    );
}

#[test]
fn the_classification_table_names_no_builtin_that_does_not_exist() {
    // The other direction: a stale entry classifies a tool that was renamed
    // or removed, which reads as coverage while covering nothing.
    let builtin_entries: Vec<&str> = CLASSIFIED_SOURCES
        .iter()
        .filter(|s| s.source == "builtin")
        .flat_map(|s| s.tools.iter().map(|t| t.name))
        .collect();
    let stale: Vec<&&str> = builtin_entries
        .iter()
        .filter(|name| !BuiltinToolService::ALL_TOOL_NAMES.contains(name))
        .collect();
    assert!(
        stale.is_empty(),
        "CLASSIFIED_SOURCES classifies built-ins that no longer exist: {stale:?}"
    );
    assert_eq!(
        builtin_entries.len(),
        BuiltinToolService::ALL_TOOL_NAMES.len(),
        "the built-in classifications and the built-in tool list must match one for one"
    );
}

#[test]
fn every_shipped_mcp_server_has_an_explicit_classification() {
    let servers = load_mcp_configs(&shipped_default()).expect("load the shipped fleet config");
    assert!(!servers.is_empty(), "the shipped fleet must not be empty");

    let classified: Vec<&str> = CLASSIFIED_SOURCES.iter().map(|s| s.source).collect();
    let missing: Vec<&str> = servers
        .iter()
        .map(|s| s.name.as_str())
        .filter(|name| !classified.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "these shipped MCP servers have no entry in CLASSIFIED_SOURCES: {missing:?}. \
         Add one, even an empty one, so the decision is written down."
    );
}

#[test]
fn the_classification_table_names_no_mcp_server_that_is_not_shipped() {
    // Sources that are not MCP servers are named here so the check stays
    // honest about what it is skipping.
    const NON_SERVER_SOURCES: [&str; 3] = ["builtin", "subagent", "client"];

    let servers = load_mcp_configs(&shipped_default()).expect("load the shipped fleet config");
    let shipped: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
    let stale: Vec<&str> = CLASSIFIED_SOURCES
        .iter()
        .map(|s| s.source)
        .filter(|name| !NON_SERVER_SOURCES.contains(name) && !shipped.contains(name))
        .collect();
    assert!(
        stale.is_empty(),
        "CLASSIFIED_SOURCES names MCP servers the fleet no longer ships: {stale:?}"
    );
}

#[test]
fn the_shipped_fleet_config_is_readable_where_the_table_expects_it() {
    // The two tests above are vacuous if the path drifts, so pin it.
    let path = shipped_default();
    assert!(
        fs::metadata(&path).is_ok(),
        "the shipped fleet config must exist at {}",
        path.display()
    );
}
