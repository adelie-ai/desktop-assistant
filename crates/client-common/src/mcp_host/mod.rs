//! Client-side MCP host.
//!
//! Runs local MCP servers on the *edge* (the machine a client runs on) and
//! exposes their tools to the (possibly remote) daemon as **client-side tools**
//! — so a brain running elsewhere (e.g. in k8s) can invoke tools that act on
//! the user's own machine. Selection is driven by [`config::ClientMcpConfig`]
//! (`client-mcp.toml`) with per-surface enable lists.
//!
//! Phase 1 (this module) provides the config layer; the host orchestrator and
//! the `Connector` registration bridge land in follow-up phases.

pub mod config;
pub mod host;

pub use config::{ClientMcpConfig, McpServerConfig, SurfaceConfig, default_client_mcp_path};
pub use host::McpHost;
