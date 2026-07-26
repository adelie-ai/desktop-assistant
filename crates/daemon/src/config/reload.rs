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
    use crate::connections::{ConnectionConfig, ConnectionId, OllamaConnection};
    use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, PurposeKind};

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

    /// A fully-named purpose binding (no `primary` inheritance).
    fn named_purpose(connection: &str, model: &str) -> PurposeConfig {
        PurposeConfig {
            connection: ConnectionRef::Named(
                ConnectionId::new(connection).expect("test slug is valid"),
            ),
            model: ModelRef::Named(model.to_string()),
            effort: None,
            max_context_tokens: None,
        }
    }

    /// A purpose binding that inherits both halves from `interactive`.
    fn inheriting_purpose() -> PurposeConfig {
        PurposeConfig {
            connection: ConnectionRef::Primary,
            model: ModelRef::Primary,
            effort: None,
            max_context_tokens: None,
        }
    }

    /// A config with one ollama connection and a named `interactive` purpose,
    /// which is the anchor every other purpose may inherit from.
    fn cfg_with_interactive(model: &str) -> DaemonConfig {
        let mut cfg = cfg_with_ollama("local", "http://localhost:11434");
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(named_purpose("local", model)),
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
            plan.restart_required.contains(&RestartArea::Database),
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
        assert!(plan.restart_required.contains(&RestartArea::Database));
    }

    #[test]
    fn embeddings_and_tls_changes_are_restart_required() {
        let old = DaemonConfig::default();
        let mut new = old.clone();
        new.embeddings.connector = Some("ollama".to_string());
        new.tls.enabled = !old.tls.enabled;
        let plan = plan_reload(&old, &new);
        assert!(plan.restart_required.contains(&RestartArea::Embeddings));
        assert!(plan.restart_required.contains(&RestartArea::Tls));
        assert!(!plan.rebuild_registry);
    }

    #[test]
    fn ws_auth_change_is_restart_required() {
        // Changing the allowed authentication methods is exactly the edit an
        // operator makes under pressure; the method set is wired into the
        // listener once at startup, so it cannot take effect until a restart.
        let old = DaemonConfig::default();
        let mut new = old.clone();
        new.ws_auth.methods = vec!["oidc".to_string()];
        let plan = plan_reload(&old, &new);
        assert!(
            plan.restart_required.contains(&RestartArea::WsAuth),
            "a [ws_auth] edit must be flagged restart-required: {:?}",
            plan.restart_required
        );
        assert!(!plan.rebuild_registry);
    }

    // --- [purposes.embedding] classification (#686) -----------------------
    //
    // The embedding client is built once in `main` and is NOT in the
    // connection registry, so the registry rebuild every `[purposes]` edit
    // triggers does nothing for it. Until #685 hot-applies the embedding
    // backend, an embedding-purpose edit needs a restart to take effect.

    #[test]
    fn embedding_purpose_change_is_restart_required() {
        let old = {
            let mut cfg = cfg_with_interactive("llama3");
            cfg.purposes.set(
                PurposeKind::Embedding,
                Some(named_purpose("local", "nomic-embed-text")),
            );
            cfg
        };
        let new = {
            let mut cfg = cfg_with_interactive("llama3");
            cfg.purposes.set(
                PurposeKind::Embedding,
                Some(named_purpose("local", "amazon.titan-embed-text-v2:0")),
            );
            cfg
        };
        let plan = plan_reload(&old, &new);
        assert!(
            plan.restart_required.contains(&RestartArea::Embeddings),
            "a [purposes.embedding] model swap must be flagged restart-required: {:?}",
            plan.restart_required
        );
        // The registry rebuild still happens in the same edit; it just does
        // nothing for the embedding client, which is the whole bug.
        assert!(
            plan.rebuild_registry,
            "a [purposes] edit still rebuilds the registry"
        );
    }

    #[test]
    fn adding_an_embedding_purpose_is_restart_required() {
        let old = cfg_with_interactive("llama3");
        let mut new = old.clone();
        new.purposes.set(
            PurposeKind::Embedding,
            Some(named_purpose("local", "nomic-embed-text")),
        );
        let plan = plan_reload(&old, &new);
        assert!(
            plan.restart_required.contains(&RestartArea::Embeddings),
            "configuring the embedding purpose for the first time needs a restart: {:?}",
            plan.restart_required
        );
    }

    #[test]
    fn removing_an_embedding_purpose_is_restart_required() {
        let mut old = cfg_with_interactive("llama3");
        old.purposes.set(
            PurposeKind::Embedding,
            Some(named_purpose("local", "nomic-embed-text")),
        );
        let mut new = old.clone();
        new.purposes.set(PurposeKind::Embedding, None);
        let plan = plan_reload(&old, &new);
        assert!(
            plan.restart_required.contains(&RestartArea::Embeddings),
            "clearing the embedding purpose falls back to [embeddings], which also needs \
             a restart: {:?}",
            plan.restart_required
        );
    }

    #[test]
    fn inheriting_embedding_purpose_tracks_interactive_changes() {
        // `connection = "primary"` / `model = "primary"` resolves through the
        // interactive purpose, so an interactive edit silently re-points the
        // embedding backend too.
        let mut old = cfg_with_interactive("llama3");
        old.purposes
            .set(PurposeKind::Embedding, Some(inheriting_purpose()));
        let mut new = cfg_with_interactive("mistral");
        new.purposes
            .set(PurposeKind::Embedding, Some(inheriting_purpose()));
        let plan = plan_reload(&old, &new);
        assert!(
            plan.restart_required.contains(&RestartArea::Embeddings),
            "an inheriting embedding purpose must follow the interactive purpose: {:?}",
            plan.restart_required
        );
    }

    #[test]
    fn interactive_purpose_change_alone_needs_no_restart() {
        // The hot-applicable case: no embedding purpose is configured, so
        // re-pointing interactive is a pure registry rebuild.
        let old = cfg_with_interactive("llama3");
        let new = cfg_with_interactive("mistral");
        let plan = plan_reload(&old, &new);
        assert!(plan.rebuild_registry, "a [purposes] edit hot-applies");
        assert!(
            !plan.needs_restart(),
            "an interactive-only purpose edit must not claim a restart is needed: {:?}",
            plan.restart_required
        );
    }

    #[test]
    fn embedding_backend_edits_report_a_single_area() {
        // The legacy `[embeddings]` block and `[purposes.embedding]` are two
        // ways to configure one backend; changing both at once must not report
        // the same restart twice.
        let mut old = cfg_with_interactive("llama3");
        old.purposes.set(
            PurposeKind::Embedding,
            Some(named_purpose("local", "nomic-embed-text")),
        );
        let mut new = old.clone();
        new.embeddings.connector = Some("ollama".to_string());
        new.purposes.set(
            PurposeKind::Embedding,
            Some(named_purpose("local", "mxbai-embed-large")),
        );
        let plan = plan_reload(&old, &new);
        let embedding_entries = plan
            .restart_required
            .iter()
            .filter(|area| **area == RestartArea::Embeddings)
            .count();
        assert_eq!(
            embedding_entries, 1,
            "one backend, one restart entry: {:?}",
            plan.restart_required
        );
    }

    #[test]
    fn restart_area_keys_are_stable_wire_identifiers() {
        // These keys cross the wire to clients; renaming one silently breaks a
        // settings UI that matches on them.
        assert_eq!(RestartArea::Database.as_key(), "database");
        assert_eq!(RestartArea::Embeddings.as_key(), "embeddings");
        assert_eq!(RestartArea::Persistence.as_key(), "persistence");
        assert_eq!(RestartArea::WsAuth.as_key(), "ws_auth");
        assert_eq!(RestartArea::Tls.as_key(), "tls");
        assert_eq!(RestartArea::Profiling.as_key(), "profiling");
    }
}
