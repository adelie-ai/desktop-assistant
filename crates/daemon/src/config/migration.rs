//! Legacy → new-format config migration.
//!
//! Extracted from `config.rs` (#41). Two migration passes run during
//! [`super::load_daemon_config`]:
//!
//! 1. **Legacy `[llm]` → `[connections.default]`**: when the file
//!    contains a top-level `[llm]` block but no `[connections]` table,
//!    synthesize a `default` connection from the legacy fields and
//!    rewrite the file in place (with a `.bak` of the original).
//!
//! 2. **Legacy `[llm]`/`[backend_tasks.llm]` → `[purposes.*]`**: when
//!    no `[purposes]` table is present *and* a legacy shape was
//!    detected, synthesize the `interactive` / `dreaming` / `titling` /
//!    `embedding` purposes — possibly creating a second `backend`
//!    connection if `[backend_tasks.llm]` targets a different
//!    connector — and rewrite. Cleared `backend_tasks.llm` is dropped
//!    from the serialized shape; in-memory it stays `None` so
//!    consumers fall back to the primary LLM.
//!
//! Helpers:
//! - [`file_has_top_level_table`] — cheap "is this section present?"
//!   scan over the raw file text. We use this rather than the parsed
//!   `DaemonConfig` because serde's defaults make every section
//!   *look* present after parse, which would defeat the migration
//!   gate.
//! - [`pick_free_connection_id`] — find an unused slug like
//!   `backend`, `backend_2`, etc., for synthesized connections.
//! - [`pick_backup_path`] — pick a never-overwriting `.bak` path so a
//!   mid-migration crash leaves the original recoverable.

use std::path::{Path, PathBuf};

use anyhow::Context;
use indexmap::IndexMap;

use crate::connections::{ConnectionConfig, ConnectionId, connection_from_legacy_llm};
use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, PurposeKind};

use super::{DaemonConfig, default_backend_llm_model, default_llm_model, save_daemon_config};

pub(super) fn maybe_migrate_legacy_connections(
    path: &Path,
    mut parsed: DaemonConfig,
    original_content: &str,
) -> anyhow::Result<DaemonConfig> {
    // Detect the legacy case: `[llm]` literally present in the file AND no
    // `[connections]` table. Using the raw file text for `[llm]` detection
    // avoids treating serde's default `LlmConfig` as "legacy present".
    let has_legacy_llm_section = file_has_top_level_table(original_content, "llm");
    let has_connections_section = file_has_top_level_table(original_content, "connections");

    if !has_legacy_llm_section || has_connections_section {
        return Ok(parsed);
    }

    tracing::warn!(
        "daemon config at {} uses the legacy `[llm]` block; \
         auto-migrating to `[connections.default]` \
         (one-time; the deprecated block will be removed in a future release)",
        path.display()
    );

    let default_id = ConnectionId::new("default").expect("literal slug is valid");
    let connection = connection_from_legacy_llm(&parsed.llm);
    parsed
        .connections
        .insert(default_id.into_string(), connection);

    // Back up the original file before we overwrite it, picking a fresh
    // `.bak.N` suffix if `.bak` already exists. We write the backup *before*
    // rewriting the config so a mid-migration crash leaves the user with the
    // original file recoverable from disk.
    let backup_path = pick_backup_path(path);
    std::fs::write(&backup_path, original_content).with_context(|| {
        format!(
            "failed to write daemon config backup at {}",
            backup_path.display()
        )
    })?;
    tracing::info!(
        "backed up legacy daemon config to {}",
        backup_path.display()
    );

    save_daemon_config(path, &parsed).with_context(|| {
        format!(
            "failed to rewrite migrated daemon config at {}",
            path.display()
        )
    })?;

    Ok(parsed)
}

/// Synthesize a `[purposes]` block from legacy `[llm]` / `[backend_tasks.llm]`
/// when migrating an older config.
///
/// Trigger conditions (all must hold):
/// - `parsed.purposes` is empty (`Purposes::default()`).
/// - The file does not already have an explicit `[purposes]` table (even an
///   empty one — treating an explicit empty table as "user authored, don't
///   touch" matches how `[connections]` is handled).
/// - At least one connection exists (either from prior migration or from an
///   author-written `[connections]` table). Without any connection we cannot
///   produce a valid interactive purpose.
///
/// Synthesis rules:
/// - `interactive`: reference the first connection in declaration order.
///   Model is taken from legacy `[llm].model` if set, otherwise connector
///   defaults at dispatch time — represented here as the legacy value or
///   `"primary"` (which we cannot use for interactive). We therefore fall
///   back to the connector-default model name when no explicit model was
///   configured, so the resolved purpose always has a concrete model.
/// - `dreaming`, `titling`, `embedding`: if `[backend_tasks.llm]` is set and
///   targets a different connector than `[llm]`, we synthesize an additional
///   connection (`backend`) using [`connection_from_legacy_llm`] and point
///   these purposes at it. Otherwise they inherit via `connection = "primary"`
///   and the backend-tasks model is used for dreaming/titling (or `"primary"`
///   when no backend-tasks model was set).
///
/// Post-migration, `backend_tasks.llm` is cleared in-memory (it will not
/// serialize). Other `[backend_tasks]` fields (dreaming_enabled, intervals,
/// archive_after_days) are preserved verbatim.
pub(super) fn maybe_migrate_legacy_purposes(
    path: &Path,
    mut parsed: DaemonConfig,
    explicit_purposes_table: bool,
    legacy_shape_present: bool,
) -> anyhow::Result<DaemonConfig> {
    if !parsed.purposes.is_empty() || explicit_purposes_table {
        return Ok(parsed);
    }
    if !legacy_shape_present {
        // New-format config with no legacy markers and no `[purposes]` yet.
        // Leave it untouched — first-run users configure purposes explicitly
        // (either through the settings API or by editing TOML directly).
        return Ok(parsed);
    }
    if parsed.connections.is_empty() {
        // Legacy shape but no connections resulted from migration. Cannot
        // produce a valid interactive purpose without at least one; skip.
        return Ok(parsed);
    }

    // Pick interactive's connection: prefer `default` (the name #8's migration
    // assigns), else the first declared connection.
    let interactive_conn_id = if parsed.connections.contains_key("default") {
        "default".to_string()
    } else {
        parsed
            .connections
            .keys()
            .next()
            .cloned()
            .expect("connections non-empty")
    };
    let interactive_conn = ConnectionId::new(interactive_conn_id.clone()).with_context(|| {
        format!("cannot migrate purposes: connection id {interactive_conn_id:?} is invalid")
    })?;

    // Model for interactive: take from [llm].model, else use the connector's
    // built-in default so the resolved purpose always has a concrete model.
    let interactive_model = parsed
        .llm
        .model
        .clone()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_llm_model(&parsed.llm.connector));

    tracing::warn!(
        "daemon config at {} has no `[purposes]` block; \
         synthesizing one from legacy `[llm]`/`[backend_tasks.llm]` \
         (one-time; future releases drop the compatibility shims)",
        path.display()
    );

    // Decide how to handle backend tasks (dreaming / titling / embedding).
    //
    // Case A: `[backend_tasks.llm]` is absent → everything inherits via
    //         `connection = "primary"`, `model = "primary"`.
    // Case B: `[backend_tasks.llm]` matches the primary connector → use the
    //         `primary` connection but pin dreaming/titling to the backend
    //         model if it was set.
    // Case C: `[backend_tasks.llm]` targets a different connector → synthesize
    //         a second connection (`backend`, with a suffix if taken) and
    //         point dreaming/titling/embedding at it.
    let bt_llm_ref = parsed.backend_tasks.llm.as_ref();
    let primary_connector = parsed.llm.connector.trim().to_ascii_lowercase();

    let (backend_conn_ref, backend_model_opt) = if let Some(bt_llm) = bt_llm_ref {
        let bt_connector = bt_llm.connector.trim().to_ascii_lowercase();
        let bt_model = bt_llm.model.clone().filter(|v| !v.trim().is_empty());

        if bt_connector.is_empty() || bt_connector == primary_connector {
            // Case B: same connector as primary — share the connection.
            (ConnectionRef::Primary, bt_model)
        } else {
            // Case C: different connector. Synthesize a new connection.
            let synthesized = connection_from_legacy_llm(bt_llm);
            let backend_id = pick_free_connection_id(&parsed.connections, "backend");
            parsed.connections.insert(backend_id.clone(), synthesized);
            let id = ConnectionId::new(backend_id).expect("pick_free returns a valid slug");
            (ConnectionRef::Named(id), bt_model)
        }
    } else {
        // Case A.
        (ConnectionRef::Primary, None)
    };

    // Build the purposes set.
    parsed.purposes.set(
        PurposeKind::Interactive,
        Some(PurposeConfig {
            connection: ConnectionRef::Named(interactive_conn),
            model: ModelRef::Named(interactive_model),
            effort: None,
            max_context_tokens: None,
        }),
    );

    let dreaming_model = match (&backend_conn_ref, &backend_model_opt) {
        (ConnectionRef::Primary, Some(m)) => ModelRef::Named(m.clone()),
        (ConnectionRef::Primary, None) => ModelRef::Primary,
        (ConnectionRef::Named(_), Some(m)) => ModelRef::Named(m.clone()),
        (ConnectionRef::Named(_), None) => {
            // Different connector but no explicit model — fall back to the
            // connector default so the resolved purpose is concrete.
            let bt_connector = bt_llm_ref
                .map(|l| l.connector.trim().to_ascii_lowercase())
                .unwrap_or_else(|| primary_connector.clone());
            ModelRef::Named(default_backend_llm_model(&bt_connector))
        }
    };

    parsed.purposes.set(
        PurposeKind::Dreaming,
        Some(PurposeConfig {
            connection: backend_conn_ref.clone(),
            model: dreaming_model.clone(),
            effort: None,
            max_context_tokens: None,
        }),
    );
    parsed.purposes.set(
        PurposeKind::Titling,
        Some(PurposeConfig {
            connection: backend_conn_ref,
            model: dreaming_model,
            effort: None,
            max_context_tokens: None,
        }),
    );
    // Embeddings always inherit from the primary connection: the embedding
    // model lives in `[embeddings]`, not in `backend_tasks.llm`, so there is
    // nothing connector-specific to carry over. Users with a dedicated
    // embeddings connector keep their `[embeddings]` config unchanged.
    parsed.purposes.set(
        PurposeKind::Embedding,
        Some(PurposeConfig {
            connection: ConnectionRef::Primary,
            model: ModelRef::Primary,
            effort: None,
            max_context_tokens: None,
        }),
    );

    // Drop `backend_tasks.llm` from the serialized shape. The field remains
    // in memory so existing consumers (main.rs, settings views) can still
    // read it as `None` and fall back to the primary LLM — that fallback is
    // already their documented behavior.
    parsed.backend_tasks.llm = None;

    save_daemon_config(path, &parsed).with_context(|| {
        format!(
            "failed to rewrite purpose-migrated daemon config at {}",
            path.display()
        )
    })?;

    Ok(parsed)
}

/// Find a `ConnectionId`-valid slug that is not already in use. Starts with
/// `base` (e.g. `backend`) and appends `_2`, `_3`, ... as needed.
fn pick_free_connection_id(existing: &IndexMap<String, ConnectionConfig>, base: &str) -> String {
    if !existing.contains_key(base) {
        return base.to_string();
    }
    for n in 2..=u32::MAX {
        let candidate = format!("{base}_{n}");
        if !existing.contains_key(&candidate) {
            return candidate;
        }
    }
    // Effectively unreachable.
    format!("{base}_{}", u32::MAX)
}

/// Pick a backup path: prefer `<path>.bak`, falling back to `<path>.bak.2`,
/// `<path>.bak.3`, ... when earlier slots are taken. Never overwrites.
///
/// `pub(super)` so the parent module's tests can exercise the
/// "primary slot exists, escalate to `.bak.N`" branch without having
/// to drive a full migration.
pub(super) fn pick_backup_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config");

    let primary = parent.join(format!("{file_name}.bak"));
    if !primary.exists() {
        return primary;
    }
    // `.bak.2`, `.bak.3`, ... keep trying until we find a free slot.
    // The cap is just a sanity bound; practical users will never hit it.
    for n in 2..=u32::MAX {
        let candidate = parent.join(format!("{file_name}.bak.{n}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    // Extremely unlikely: fall back to overwriting the highest-numbered slot.
    parent.join(format!("{file_name}.bak.{}", u32::MAX))
}

/// Cheap detector for a top-level `[<name>]` (or `[<name>.sub]`) TOML table in
/// the raw file text. Good enough for "is this section present?" gating during
/// migration; we do not try to handle all TOML edge cases (comments inside
/// headers, multiline strings that look like headers, etc.) because the config
/// file is a human-edited file we generated ourselves.
pub(super) fn file_has_top_level_table(content: &str, name: &str) -> bool {
    let prefix_eq = format!("[{name}]");
    let prefix_dot = format!("[{name}.");
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == prefix_eq || line.starts_with(&prefix_dot) {
            return true;
        }
    }
    false
}
