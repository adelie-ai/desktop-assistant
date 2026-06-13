//! Pure classification of what a config reload implies (#222).
//!
//! Mirrors the voice daemon's `plan_reload` (voice config#52): a
//! side-effect-free diff of the old vs. new [`DaemonConfig`] into a
//! [`ReloadPlan`] so the apply decision is unit-tested without touching
//! disk, the registry, or any live connection.
//!
//! Three classes of knob:
//!
//! - **Hot-apply** — picked up by rebuilding the in-memory connection
//!   registry under its `RwLock`. New turns route through the new clients;
//!   in-flight turns keep the `Arc<dyn LlmClient>` they already cloned alive
//!   by refcount (see [`crate::api_surface::RegistryHandle::apply_reload`]).
//!   This covers `[connections]`, `[purposes]`, and the legacy `[llm]` block.
//!
//! - **Rebuild** — same mechanism as hot-apply today (a full registry
//!   rebuild), called out separately so the log explains what changed.
//!
//! - **Restart-required** — wired once at process start and not swappable
//!   live: the database pool/url, embeddings backend, persistence, WS auth,
//!   TLS, and profiling. A reload still applies every hot knob in the same
//!   edit; these are flagged in the plan so the daemon logs that a restart is
//!   needed for them to take effect, rather than silently ignoring them.

use super::DaemonConfig;

/// The work a reload implies, derived purely from the old/new
/// [`DaemonConfig`]. Pure and side-effect-free.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReloadPlan {
    /// Rebuild the connection registry: `[connections]`, `[purposes]`, or the
    /// legacy `[llm]` block changed. New turns route through the new clients.
    pub rebuild_registry: bool,
    /// Knobs that only take effect on a full process restart (database,
    /// embeddings, persistence, ws-auth, TLS, profiling). Human-readable
    /// labels for the log; applying a reload does not act on them.
    pub restart_required: Vec<String>,
}

impl ReloadPlan {
    /// True when nothing changed — the watcher/`Reload` can skip a no-op.
    pub fn is_empty(&self) -> bool {
        !self.rebuild_registry && self.restart_required.is_empty()
    }

    /// True when at least one knob requires a restart to take effect.
    pub fn needs_restart(&self) -> bool {
        !self.restart_required.is_empty()
    }
}

/// Diff two [`DaemonConfig`] snapshots into the concrete work a reload implies.
///
/// - `[connections]` / `[purposes]` / `[llm]` changes set `rebuild_registry`
///   (hot-applied by swapping the registry under its lock).
/// - `[database]` / `[embeddings]` / `[persistence]` / `[ws_auth]` / `[tls]` /
///   `[profiling]` changes are flagged in `restart_required` because those
///   subsystems are constructed once at daemon startup.
pub fn plan_reload(old: &DaemonConfig, new: &DaemonConfig) -> ReloadPlan {
    let mut plan = ReloadPlan::default();

    // Hot-applicable: anything the registry rebuild observes. `ConnectionConfig`
    // / `Purposes` / `LlmConfig` aren't `PartialEq`, so compare via their
    // serialized form — cheap, allocation-light at reload cadence, and exact.
    if !areas_eq(&old.connections, &new.connections)
        || !areas_eq(&old.purposes, &new.purposes)
        || !areas_eq(&old.llm, &new.llm)
        || !areas_eq(&old.backend_tasks, &new.backend_tasks)
    {
        plan.rebuild_registry = true;
    }

    // Restart-required: subsystems wired once at startup.
    if !areas_eq(&old.database, &new.database) {
        plan.restart_required.push("database".to_string());
    }
    if !areas_eq(&old.embeddings, &new.embeddings) {
        plan.restart_required.push("embeddings".to_string());
    }
    if !areas_eq(&old.persistence, &new.persistence) {
        plan.restart_required.push("persistence (git)".to_string());
    }
    if !areas_eq(&old.ws_auth, &new.ws_auth) {
        plan.restart_required.push("ws_auth".to_string());
    }
    if !areas_eq(&old.tls, &new.tls) {
        plan.restart_required.push("tls".to_string());
    }
    if !areas_eq(&old.profiling, &new.profiling) {
        plan.restart_required.push("profiling".to_string());
    }

    plan
}

/// Structural equality of two config sub-areas via their TOML form. The config
/// value types don't all implement `PartialEq`, and `serde` round-trips are the
/// project's existing equality proxy for these. Serialization failures (which
/// don't happen for valid in-memory config) conservatively report "changed".
fn areas_eq<T: serde::Serialize>(a: &T, b: &T) -> bool {
    match (toml::to_string(a), toml::to_string(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{ConnectionConfig, OllamaConnection};

    fn cfg_with_ollama(id: &str, base: &str) -> DaemonConfig {
        let mut cfg = DaemonConfig::default();
        cfg.connections.insert(
            id.to_string(),
            ConnectionConfig::Ollama(OllamaConnection {
                base_url: Some(base.to_string()),
                ..Default::default()
            }),
        );
        cfg
    }

    #[test]
    fn no_change_is_an_empty_plan() {
        let cfg = cfg_with_ollama("local", "http://localhost:11434");
        let plan = plan_reload(&cfg, &cfg);
        assert!(
            plan.is_empty(),
            "an unchanged config must be a no-op reload"
        );
        assert!(!plan.needs_restart());
    }

    #[test]
    fn connection_change_rebuilds_registry_without_restart() {
        let old = cfg_with_ollama("local", "http://localhost:11434");
        let new = cfg_with_ollama("local", "http://localhost:9999");
        let plan = plan_reload(&old, &new);
        assert!(plan.rebuild_registry, "a [connections] edit hot-applies");
        assert!(
            !plan.needs_restart(),
            "a connection change never forces a restart"
        );
        assert!(!plan.is_empty());
    }

    #[test]
    fn adding_a_connection_rebuilds_registry() {
        let old = cfg_with_ollama("a", "http://localhost:11434");
        let mut new = old.clone();
        new.connections.insert(
            "b".to_string(),
            ConnectionConfig::Ollama(OllamaConnection {
                base_url: Some("http://localhost:11435".to_string()),
                ..Default::default()
            }),
        );
        let plan = plan_reload(&old, &new);
        assert!(plan.rebuild_registry);
        assert!(!plan.needs_restart());
    }

    #[test]
    fn database_change_flags_restart_required() {
        let old = DaemonConfig::default();
        let mut new = old.clone();
        new.database.url = Some("postgres://localhost/da".to_string());
        let plan = plan_reload(&old, &new);
        assert!(plan.needs_restart());
        assert!(
            plan.restart_required.iter().any(|s| s == "database"),
            "database edits must be flagged restart-required: {:?}",
            plan.restart_required
        );
        // A pure database edit does not by itself rebuild the registry.
        assert!(!plan.rebuild_registry);
    }

    #[test]
    fn mixed_edit_hot_applies_connection_and_flags_database_restart() {
        let old = cfg_with_ollama("local", "http://localhost:11434");
        let mut new = cfg_with_ollama("local", "http://localhost:9999");
        new.database.url = Some("postgres://localhost/da".to_string());
        let plan = plan_reload(&old, &new);
        // The hot knob in the same edit still applies …
        assert!(plan.rebuild_registry);
        // … while the restart-only knob is flagged.
        assert!(plan.restart_required.iter().any(|s| s == "database"));
    }

    #[test]
    fn embeddings_and_tls_changes_are_restart_required() {
        let old = DaemonConfig::default();
        let mut new = old.clone();
        new.embeddings.connector = Some("ollama".to_string());
        new.tls.enabled = !old.tls.enabled;
        let plan = plan_reload(&old, &new);
        assert!(plan.restart_required.iter().any(|s| s == "embeddings"));
        assert!(plan.restart_required.iter().any(|s| s == "tls"));
        assert!(!plan.rebuild_registry);
    }
}
