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
//!   TLS, the administrator allowlist, and pre-prompt recall. A
//!   reload still applies every hot knob in the same edit; these are flagged in
//!   the plan so the daemon logs that a restart is needed for them to take
//!   effect, rather than silently ignoring them.
//!
//! The embedding backend is a purpose-driven knob that lands in the *hot*
//! arm's diff (`[purposes]`) but is not hot-applicable: the embedding client is
//! built once in `main` and is not in the connection registry, so the rebuild
//! does nothing for it. It is classified restart-required here until #685
//! rebuilds it live (see [`embedding_backend_changed`]).
//!
//! [`RestartArea`] carries one member this diff never produces:
//! `ConfigLoadFailed`, reported by the registry handle when the file will not
//! load at all. It rides the same wire field because it answers the same client
//! question - what in the file is not in force - and there is nothing to diff
//! when the file did not parse.

use super::DaemonConfig;
use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, PurposeKind};

/// Something in the config file that the running process is not acting on and
/// cannot pick up without a restart - in almost every case a config area whose
/// value is wired once at process start.
///
/// A closed set rather than free strings because [`Self::as_key`] crosses the
/// wire to clients (`api-model`'s `Config::restart_required`): a typo in a
/// classifier would silently break a settings UI that matches on these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartArea {
    /// The whole file: it could not be loaded, so *nothing* in it is in force
    /// and the daemon is running built-in defaults.
    ///
    /// The odd one out - not an area, and never produced by [`plan_reload`],
    /// which diffs two configs that both parsed. It is reported by
    /// `RegistryHandle::restart_required` so a settings panel can tell a
    /// degraded daemon from a genuinely unconfigured one, and it clears when
    /// the file loads again.
    ConfigLoadFailed,
    /// `[database]`: the pool and URL are opened once at startup.
    Database,
    /// The embedding backend, whether configured via the legacy `[embeddings]`
    /// block or `[purposes.embedding]`. Removed by #685.
    Embeddings,
    /// `[persistence]`: the git-backed history mirror is wired once.
    Persistence,
    /// `[ws_auth]`: the allowed authentication methods, OIDC discovery, and
    /// permitted browser origins are read into the listener once.
    WsAuth,
    /// `[tls]`: the certificate resolver is built once, so rotation needs a
    /// restart.
    Tls,
    /// `[authz]`: the remote-administrator allowlist is read into the transport
    /// validators once at startup, so an edit does not reach a live connection.
    Authz,
    /// `[recall]`: whether the pre-prompt lookup is wired, and how many lines
    /// its block shows, are both read once when the conversation handler is
    /// built.
    Recall,
}

impl RestartArea {
    /// Stable identifier for this area, safe to put on the wire and to match on
    /// in a client. Never carries a configured *value*, only the area name.
    pub fn as_key(self) -> &'static str {
        match self {
            Self::ConfigLoadFailed => "config_load_failed",
            Self::Database => "database",
            Self::Embeddings => "embeddings",
            Self::Persistence => "persistence",
            Self::WsAuth => "ws_auth",
            Self::Tls => "tls",
            Self::Authz => "authz",
            Self::Recall => "recall",
        }
    }
}

/// The work a reload implies, derived purely from the old/new
/// [`DaemonConfig`]. Pure and side-effect-free.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReloadPlan {
    /// Rebuild the connection registry: `[connections]`, `[purposes]`, or the
    /// legacy `[llm]` block changed. New turns route through the new clients.
    pub rebuild_registry: bool,
    /// Areas that only take effect on a full process restart. Applying a reload
    /// does not act on them; they are reported so the daemon logs, and tells
    /// the client that made the change, that a restart is still needed.
    pub restart_required: Vec<RestartArea>,
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

    /// The restart-required areas as stable wire/log identifiers.
    pub fn restart_required_keys(&self) -> Vec<String> {
        self.restart_required
            .iter()
            .map(|area| area.as_key().to_string())
            .collect()
    }
}

/// Diff two [`DaemonConfig`] snapshots into the concrete work a reload implies.
///
/// - `[connections]` / `[purposes]` / `[llm]` changes set `rebuild_registry`
///   (hot-applied by swapping the registry under its lock).
/// - Edits to any [`RestartArea`] are reported in `restart_required` because
///   those subsystems are constructed once at daemon startup.
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
        plan.restart_required.push(RestartArea::Database);
    }
    // Until #685 rebuilds the embedding client live, every way of configuring
    // it is restart-bound. Reported as ONE area because it is one backend.
    if embedding_backend_changed(old, new) {
        plan.restart_required.push(RestartArea::Embeddings);
    }
    if !areas_eq(&old.persistence, &new.persistence) {
        plan.restart_required.push(RestartArea::Persistence);
    }
    if !areas_eq(&old.ws_auth, &new.ws_auth) {
        plan.restart_required.push(RestartArea::WsAuth);
    }
    if !areas_eq(&old.tls, &new.tls) {
        plan.restart_required.push(RestartArea::Tls);
    }
    if old.authz != new.authz {
        plan.restart_required.push(RestartArea::Authz);
    }
    if old.recall != new.recall {
        plan.restart_required.push(RestartArea::Recall);
    }

    plan
}

/// Whether the embedding backend the daemon would build differs between the two
/// configs. Removed by #685 along with its call site.
///
/// Three inputs feed `resolve_embeddings_config`, so all three are compared:
/// the legacy `[embeddings]` block, the `[purposes.embedding]` entry that
/// overrides it, and, when that entry inherits via the `primary` sentinel,
/// the `[purposes.interactive]` entry it inherits from.
///
/// Deliberately structural rather than calling the resolver: resolution reads
/// API keys out of the secret backend, and `plan_reload` must stay pure.
fn embedding_backend_changed(old: &DaemonConfig, new: &DaemonConfig) -> bool {
    if !areas_eq(&old.embeddings, &new.embeddings) {
        return true;
    }

    let old_embedding = old.purposes.get(PurposeKind::Embedding);
    let new_embedding = new.purposes.get(PurposeKind::Embedding);
    if old_embedding != new_embedding {
        return true;
    }

    if inherits_from_interactive(old_embedding) {
        return old.purposes.get(PurposeKind::Interactive)
            != new.purposes.get(PurposeKind::Interactive);
    }

    false
}

/// Whether a purpose entry resolves through the `interactive` purpose. A mixed
/// pair is refused on the write path, but a config already on disk can carry
/// one, so either half counts.
fn inherits_from_interactive(purpose: Option<&PurposeConfig>) -> bool {
    purpose.is_some_and(|cfg| {
        matches!(cfg.connection, ConnectionRef::Primary) || matches!(cfg.model, ModelRef::Primary)
    })
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
        assert_eq!(RestartArea::Authz.as_key(), "authz");
        assert_eq!(RestartArea::Recall.as_key(), "recall");
    }

    /// The admin allowlist is read into the transport validators once, so an
    /// edit is reported honestly as pending a restart rather than looking live.
    #[test]
    fn changing_admin_subjects_requires_a_restart() {
        let old = DaemonConfig::default();
        let mut new = DaemonConfig::default();
        new.authz.admin_subjects = vec!["operator".to_string()];
        let plan = plan_reload(&old, &new);
        assert!(plan.restart_required.contains(&RestartArea::Authz));
        assert!(!plan.rebuild_registry);
    }

    /// An unchanged allowlist is not reported, so a desktop reload stays quiet.
    #[test]
    fn an_unchanged_authz_section_reports_nothing() {
        let cfg = DaemonConfig::default();
        assert!(plan_reload(&cfg, &cfg).is_empty());
    }

    /// Pre-prompt recall is wired once, when the conversation handler is built,
    /// so an edit must read as pending a restart. Without this the reload says
    /// "nothing to apply" about a file that demonstrably changed.
    #[test]
    fn changing_the_recall_section_requires_a_restart() {
        let old = DaemonConfig::default();

        let mut narrower = DaemonConfig::default();
        narrower.recall.max_entries = Some(8);
        let plan = plan_reload(&old, &narrower);
        assert!(plan.restart_required.contains(&RestartArea::Recall));
        assert!(!plan.rebuild_registry);

        let mut off = DaemonConfig::default();
        off.recall.enabled = false;
        assert!(
            plan_reload(&old, &off)
                .restart_required
                .contains(&RestartArea::Recall),
            "the switch is wired once as well"
        );
    }

    /// An unchanged `[recall]` section is not reported, so a desktop reload
    /// stays quiet.
    #[test]
    fn an_unchanged_recall_section_reports_nothing() {
        let mut cfg = DaemonConfig::default();
        cfg.recall.max_entries = Some(12);
        assert!(plan_reload(&cfg, &cfg).is_empty());
    }
}
