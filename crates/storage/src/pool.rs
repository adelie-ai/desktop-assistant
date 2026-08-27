use std::collections::HashSet;

use sqlx::postgres::PgPoolOptions;
use sqlx::{Acquire, Connection, PgConnection, PgPool};

/// Create a connection pool to PostgreSQL.
pub async fn create_pool(url: &str, max_connections: u32) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(url)
        .await
}

/// One registered migration.
struct Migration {
    /// The file's name, recorded verbatim in `schema_migrations`. Renaming a
    /// file therefore re-applies it, which is why migrations are append-only.
    name: &'static str,
    /// The SQL applied when it runs.
    sql: &'static str,
}

/// Register a migration by file name, so the name that identifies it in the
/// ledger and the file it loads cannot drift apart.
macro_rules! migration {
    ($file:literal) => {
        Migration {
            name: $file,
            sql: include_str!(concat!("../migrations/", $file)),
        }
    };
}

/// The ledger of applied migrations. Created before anything else runs, so it
/// cannot itself be a numbered migration.
///
/// Absent on databases created before the ledger existed; see
/// [`run_migrations`] for how those are brought forward.
const LEDGER_DDL: &str = "CREATE TABLE IF NOT EXISTS schema_migrations (\
     name       TEXT PRIMARY KEY, \
     applied_at TIMESTAMPTZ NOT NULL DEFAULT now())";

/// First half of the advisory-lock key — an arbitrary constant ("ADEL")
/// identifying this runner, so the lock cannot collide with an unrelated
/// application's advisory lock on the same database.
const MIGRATION_LOCK_NAMESPACE: i32 = 0x4144_454C;

/// Every migration, in the order it must be applied.
///
/// Append-only and ordinally numbered: an entry that has shipped is never
/// renamed, reordered, or edited in a way that changes the schema it produces,
/// because databases that already recorded it will not run it again.
const MIGRATIONS: &[Migration] = &[
    // Core tables — always required.
    migration!("001_initial_schema.sql"),
    // Vector tables.
    migration!("002_vector_tables.sql"),
    migration!("002b_tool_definitions.sql"),
    // Indexes (GIN for full-text, btree for flags).
    migration!("003_vector_indexes.sql"),
    // Track which embedding model produced each vector.
    migration!("004_embedding_model_tracking.sql"),
    // Convert messages.id from BIGSERIAL to TEXT (UUIDv7).
    migration!("005_uuidv7_ids.sql"),
    // Dreaming watermarks — tracks per-conversation extraction progress.
    migration!("006_dreaming_watermarks.sql"),
    // Chunked embeddings — knowledge_base.embedding becomes vector[].
    migration!("007_chunked_embeddings.sql"),
    // Collapsible message summaries — reversible range summaries.
    migration!("008_message_summaries.sql"),
    // Conversation archival — nullable archived_at timestamp.
    migration!("009_conversation_archived_at.sql"),
    // Repair damage from pre-idempotent runs of migration 007 on existing
    // databases. No-op on fresh installs.
    migration!("010_fix_damaged_embeddings.sql"),
    // Per-conversation model selection (issue #11) — nullable JSONB column
    // on `conversations`.
    migration!("011_conversation_last_model.sql"),
    // Active-task anchor (issue #57) — nullable text column capturing the
    // user's current goal so it can be re-injected after windowing/summary.
    migration!("012_conversation_active_task.sql"),
    // Conversation full-text search (issue #71) — generated tsvector
    // columns + GIN indexes on `messages` and `conversations`. Generated-
    // stored columns auto-backfill on `ALTER TABLE`; the rewrite takes a
    // write lock proportional to message count, so on a large history the
    // boot that applies this one takes a while.
    migration!("013_conversation_message_fts.sql"),
    // Tag registry (issue #108) — formal vocabulary for KB tags. Categorical
    // tags emitted by the extractor are constrained to the registry; new
    // tags are created via a tool call with description and examples.
    migration!("014_tag_registry.sql"),
    // Knowledge-base review columns (issue #108) — `reviewed_at` watermark
    // gates per-memory consolidation; `review_generation` caps mutation
    // re-review loops; `deleted_at` enables soft-delete with TTL.
    migration!("015_knowledge_base_review_columns.sql"),
    // Multi-tenant schema (issue #102) — every personal-data table gains
    // `user_id NOT NULL` plus a `(user_id, …)` composite index for the
    // hot query paths #105's scoping will use. Pre-existing rows are
    // backfilled to the sentinel `'default'` user so single-tenant
    // installs keep working without auth changes.
    migration!("016_multi_tenant_user_id.sql"),
    // Turn state machine (issue #107) — DB-persisted turn state for
    // client-side execution of client-local MCP tools. A `pending_client_tool`
    // row is the daemon's record of "the LLM asked for a client-local
    // tool; we're waiting for the client to post the result back".
    migration!("017_turn_state.sql"),
    // Background tasks (issue #115) — persistent mirror of the in-memory
    // `BackgroundTaskRegistry`. On daemon restart the cold-restart sweep
    // reads this table to surface tasks that were running when the
    // previous daemon died.
    migration!("018_background_tasks.sql"),
    // Conversation scratchpad (issue #184) — ephemeral per-conversation
    // keyed notes, cascade-deleted with the conversation, with an FTS column.
    migration!("019_scratchpads.sql"),
    // Scratchpad note kind/order/done (issue #188) — note_type / seq / done
    // columns so a scratchpad can hold an ordered, checkable plan of TODOs.
    migration!("020_scratchpad_type_sequence_done.sql"),
    // Message FTS INSERT guard (issue #177) — the migration-013 generated
    // `tsv` column ran `to_tsvector` over full message content, which on a
    // large/high-entropy message exceeds Postgres's 1 MB tsvector limit and
    // aborts the INSERT. Redefine it to skip `tool`-role rows and bound the
    // indexed input so a large message can always be stored.
    migration!("021_message_fts_guard.sql"),
    // Learned error-classification cache (issue #178, tier 2) — global
    // (no user_id) connector knowledge mapping opaque error signatures to a
    // normalized cause, populated by the cheap-LLM tier so repeats are
    // recognized locally.
    migration!("022_error_classifications.sql"),
    // SendMessage idempotency keys (#204): records a completed turn's reply
    // keyed by (user_id, conversation_id, idempotency_key) so a dropped-then-
    // retried turn replays instead of re-running.
    migration!("023_idempotency_keys.sql"),
    // #227: per-conversation personality override (JSONB column on
    // conversations), mirroring 011's last_model_selection.
    migration!("024_conversation_personality.sql"),
    // #343: learned effective context-window observations — the reactive
    // safety net that `min()`s an observed-overflow ceiling into budget
    // resolution (down-only), complementing #342's proactive provisioning.
    migration!("025_context_window_observations.sql"),
    // Dream-cycle overhaul foundation — `embeddings_updated_at` (embedding
    // generation decoupled from content writes; a background task regenerates
    // NULL/stale vectors) and a first-class `source` provenance column
    // ('extraction' | 'consolidation' | 'explicit') replacing the
    // `source:dreaming` tag convention.
    migration!("026_knowledge_base_source_and_embedding_freshness.sql"),
    // Per-conversation tags (`TEXT[]`) so callers can label conversations at
    // creation time (e.g. "voice") and the UI can filter on them.
    migration!("027_conversation_tags.sql"),
    // Success high-water mark for learned context windows (#425): the other
    // half of the #343 bracket, so a mis-parsed overflow can't pin the budget
    // below a proven-good size and the budget can recover.
    migration!("028_context_window_success_watermark.sql"),
    // #434: Row-Level Security backstop for the LLM-facing db_query read
    // path — enables RLS + a per-user isolation policy on every user-scoped
    // table, so Postgres itself enforces tenant scoping even if the AST
    // grafter (#141) ever misses a table. Owner-only, so it is safe as the
    // daemon's un-privileged role; the privileged role/grant half is a
    // one-time superuser bootstrap (`bootstrap/rls_role.sql`). The daemon's
    // owner role is exempt (non-FORCE RLS), so trusted paths are unaffected.
    migration!("029_rls_backstop.sql"),
    // Provider identity + index for provider-level tool surfacing (Phase 1):
    // real tools carry their MCP server / builtin-group provider, and the
    // daemon registers one synthetic `provider:<provider>` row per provider that
    // boosts its members' search scores when it matches a query.
    migration!("030_tool_definitions_provider.sql"),
    // #287: namespace the scratchpad by owner_todo (subagent-tree path) so
    // subagent writes are confined and reads snapshot by spawn marker.
    migration!("031_scratchpad_owner_todo.sql"),
    // #287: persist owner_todo + spawn_marker on background tasks so a
    // wait=false subagent's namespace/snapshot survive a daemon restart.
    migration!("032_subagent_task_columns.sql"),
    // Host-global skill index (#573): the disk-sourced skill/workflow catalog,
    // searchable by hybrid vector + full-text, mirroring `tool_definitions`.
    migration!("033_skill_index.sql"),
    // #570 Phase 1b: nullable `idempotency_key` on messages, carried on USER
    // rows only, so a transcript reload/reconnect surfaces the client's key and
    // clients dedup an echoed UserMessageAdded by exact match.
    migration!("034_message_idempotency_key.sql"),
    // #639: the skill catalog is cumulative -- a skill a scan no longer sees is
    // marked absent rather than deleted, so presence needs somewhere to live.
    migration!("035_skill_presence.sql"),
    // #597: a note can be pinned so its content is re-surfaced every turn,
    // rather than only its key appearing in the `[Scratchpad]` index.
    migration!("036_scratchpad_pinned.sql"),
    // #599: recover a message's creation time from its UUIDv7 id, so "when did
    // this happen" needs no timestamp column and works on existing rows.
    migration!("037_uuidv7_timestamp_fn.sql"),
    // Tombstones record why they were retired: merge (with the superseding id)
    // or prune (with the model's stated reason).
    migration!("038_kb_delete_provenance.sql"),
    // Per-conversation override for the tool-provenance gate (#1007).
    migration!("039_conversation_tool_gate.sql"),
    // Scratchpad embeddings (#717) — the pad joins the embedded tables, so the
    // agent finds its own note by meaning and not only by wording.
    migration!("040_scratchpad_embeddings.sql"),
    // #1097: a one-line summary per knowledge entry, so a reader can list many
    // entries without printing or truncating each whole body.
    migration!("041_knowledge_base_summary.sql"),
    // #1104: a scratchpad note can attach a knowledge entry, so pinning that
    // note keeps the entry's live content in view instead of a stale copy.
    migration!("042_scratchpad_knowledge_entry.sql"),
    // #1099: when a knowledge entry's summary was last written, so the dream
    // cycle can find the entries whose body changed after their summary did.
    migration!("043_knowledge_base_summary_freshness.sql"),
    // #698: the use log - what a knowledge entry was offered for, what was
    // opened, and what was marked, so a deployment can measure the weights
    // its retrieval path uses instead of carrying values fitted elsewhere.
    migration!("044_knowledge_use_log.sql"),
    // #1144: the scratchpad notes a completed step distilled a tool result
    // into, recorded on the result's own row. The decision, not the
    // replacement - `content` still holds every byte the tool returned - so a
    // later turn rebuilds the pointer instead of reading the payload again.
    migration!("045_message_distilled_into.sql"),
    // #1155: the approval axis -- whether a person has consented to a skill,
    // separate from `trust_tier`'s provenance. Existing rows backfill to
    // `indexed_at`, since every skill already in the catalog arrived by a
    // person putting a file in a skill root.
    migration!("046_skill_approval.sql"),
    // #1125: the situations an entry has been seen in -- where it was written
    // and where it has proved useful -- so a recurring situation can reach it
    // again. One row per (entry, field, value), bounded per entry by the writer.
    migration!("047_knowledge_situation.sql"),
    // #1154: the skill use log -- which skills the `[Recall]` block offered and
    // which of those the model opened. Its own tables rather than migration
    // 044's, whose foreign key to `knowledge_base(id)` a skill has no row for.
    migration!("048_skill_use_log.sql"),
    // #1126: negative memory -- the actions that went badly, and the facets
    // they went badly with, so the same act meets its own lesson before it is
    // taken again. Extinction is an overlay, so a corrected burn stays
    // readable beside the correction written over it.
    migration!("049_negative_memory.sql"),
    migration!("050_negative_memory_widening.sql"),
    // #1175: the situations a skill has been followed in, so the cue #1125
    // built can reach the procedural arm #1154 shipped without it. The same
    // shape as 047's, keyed on the catalog name rather than an entry id.
    migration!("051_skill_situation.sql"),
    // #1175: what the mis-filed-procedure sweep has already judged, so a store
    // is read once per entry per edit rather than once per entry per night.
    migration!("052_knowledge_procedure_sweep.sql"),
    // #1247: whether the turn that wrote a durable record had already read
    // content from outside the trust boundary. The record keeps its words; the
    // model-facing render is what withholds them.
    migration!("053_record_after_outside_read.sql"),
    // #588: one row per turn saying what filled its prompt, and which tier
    // resolved the budget it ran under. Both were measured on every turn and
    // then discarded.
    migration!("054_context_breakdown.sql"),
    // #1252: the full text of every turn - the request as sent, the reply, the
    // tool calls and their results. One row per turn, one per round inside it.
    migration!("055_turn_records.sql"),
    // #893: widen delete provenance into a disposition vocabulary, decoupled
    // from `deleted_at`, so consolidation can mark an entry wrong, stale or
    // redundant without erasing it.
    migration!("056_kb_disposition.sql"),
    // #694: revive merge-member tombstones where the disk still holds enough
    // to say what happened - live where the successor still exists, active
    // where it was hard-reaped. Prune tombstones are untouched.
    migration!("058_revive_merge_tombstones.sql"),
    // #1327: one row per turn recording the retrieval plan the [Recall]
    // lookup considered, each candidate's activation score broken down by
    // term, and which candidates were offered and later opened.
    migration!("059_context_plans.sql"),
];

/// Second half of the advisory-lock key: the schema the migrations write to.
///
/// Why: the ledger and the tables live in the first schema on the connection's
/// `search_path`, so two runs against *different* schemas share nothing and
/// must not block each other (the storage suites migrate dozens of private
/// schemas in parallel). Two runs against the same schema must serialize.
///
/// FNV-1a over the schema name, computed here rather than in SQL so it depends
/// on no Postgres internals. The value is a bit pattern, not a number: it is
/// reinterpreted as `i32` because that is what `pg_advisory_lock` takes.
fn schema_lock_key(schema: &str) -> i32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in schema.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    i32::from_ne_bytes(hash.to_ne_bytes())
}

/// Run embedded migrations against the database.
///
/// Each migration runs **at most once**: applying it and recording its name in
/// the `schema_migrations` ledger happen in one transaction, so a boot that
/// finds the name already there skips the file. This is what keeps startup
/// cost flat — several migrations (013 and 021 in particular) rewrite the
/// whole `messages` heap and rebuild its GIN index, work that must not repeat
/// on every restart.
///
/// The run is serialized by a `pg_advisory_lock` scoped to the target schema,
/// so two daemons booting against one database queue rather than race. The
/// lock is taken on a connection detached from the pool, so it is released
/// when this function returns — including on error, when the connection is
/// dropped and the server releases the session's locks.
///
/// # Databases created before the ledger existed
///
/// They have no ledger, so the first boot under this runner replays every
/// migration once — exactly the work the previous runner did on *every* boot —
/// and records the result; later boots do nothing. The alternative, marking
/// everything applied without running it, would silently skip migrations on an
/// install whose last daemon predated them. Every migration is idempotent, and
/// has to be: this transition re-applies all of them.
///
/// The `pgvector` extension is required and must be available in the
/// database — migrations will fail if it cannot be created. It is asserted on
/// every run rather than ledger-tracked, because an extension is database-wide
/// state that can be dropped independently of the tables that need it.
///
/// HNSW indexes on the `embedding` column are NOT created here because the
/// vector dimension depends on which embedding model the user configures.
/// GIN/btree indexes for full-text search and tags are created.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    // Detached from the pool on purpose: a session-level advisory lock outlives
    // the statement that took it, so a pooled connection handed back mid-run
    // (a panic, a dropped future) would strand the lock for the pool's
    // lifetime. A detached connection is closed instead, and the server
    // releases the lock with the session. The cost is that the pool may open a
    // replacement while this one is alive, so a migrating process can sit one
    // connection over `max_connections` until this returns.
    let mut conn = pool.acquire().await?.detach();

    // NULL when `search_path` names no existing schema; the migrations then
    // fail on their own terms, and the empty key still serializes everyone in
    // that state against each other.
    let schema: String = sqlx::query_scalar::<_, Option<String>>("SELECT current_schema()::text")
        .fetch_one(&mut conn)
        .await?
        .unwrap_or_default();
    let key = schema_lock_key(&schema);

    sqlx::query("SELECT pg_advisory_lock($1, $2)")
        .bind(MIGRATION_LOCK_NAMESPACE)
        .bind(key)
        .execute(&mut conn)
        .await?;

    let outcome = apply_pending(&mut conn, &schema).await;

    // Best-effort: closing the connection below releases the lock anyway, and
    // a failure here means the session is already gone. Reported, never
    // allowed to mask the migration's own outcome.
    if let Err(e) = sqlx::query("SELECT pg_advisory_unlock($1, $2)")
        .bind(MIGRATION_LOCK_NAMESPACE)
        .bind(key)
        .execute(&mut conn)
        .await
    {
        tracing::warn!(error = %e, "releasing the migration advisory lock failed");
    }
    if let Err(e) = conn.close().await {
        tracing::warn!(error = %e, "closing the migration connection failed");
    }

    outcome
}

/// Apply every migration the ledger does not already record, on a connection
/// that already holds the migration lock.
async fn apply_pending(conn: &mut PgConnection, schema: &str) -> Result<(), sqlx::Error> {
    sqlx::raw_sql(LEDGER_DDL).execute(&mut *conn).await?;

    // Deliberately not ledger-tracked: the extension is database-wide state
    // that can be dropped independently of the tables that need it, and
    // migration 002 onwards references the `vector` type. Re-asserting it is a
    // catalog lookup once it exists.
    sqlx::raw_sql("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&mut *conn)
        .await?;

    // Names this build does not carry (a downgrade, a hand-edited row) are
    // simply not in `MIGRATIONS` and are left alone.
    let recorded: HashSet<String> = sqlx::query_scalar("SELECT name FROM schema_migrations")
        .fetch_all(&mut *conn)
        .await?
        .into_iter()
        .collect();

    let mut applied = 0usize;
    for migration in MIGRATIONS {
        if recorded.contains(migration.name) {
            continue;
        }
        // One transaction per migration: the schema change and the ledger row
        // commit together, so a run interrupted here resumes at exactly the
        // migration that did not finish.
        let mut tx = Acquire::begin(&mut *conn).await?;
        if let Err(e) = sqlx::raw_sql(migration.sql).execute(&mut *tx).await {
            tracing::error!(migration = migration.name, error = %e, "migration failed");
            return Err(e);
        }
        sqlx::query("INSERT INTO schema_migrations (name) VALUES ($1)")
            .bind(migration.name)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        applied += 1;
    }

    tracing::info!(
        schema,
        applied,
        registered = MIGRATIONS.len(),
        "migrations up to date"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MIGRATIONS, schema_lock_key};

    /// Every `.sql` file in `migrations/` must be registered in [`MIGRATIONS`],
    /// and every registered name must be a file. Migrations are a
    /// hand-maintained list, NOT auto-discovered from the directory — so a new
    /// migration file that nobody registers compiles fine and silently never
    /// runs, surfacing only as a runtime "column does not exist" error against
    /// the live DB. This guard turns that into a build-time failure instead.
    ///
    /// (A registered file that doesn't exist is already caught at compile time,
    /// since `include_str!` fails to build — but a registered *name* that
    /// disagrees with the file it loads would not be, and is caught here: a
    /// wrong name would let the ledger record a migration that never ran.)
    #[test]
    fn every_migration_is_registered() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");

        let mut on_disk: Vec<String> = std::fs::read_dir(dir)
            .expect("read migrations/ dir")
            .map(|e| e.expect("dir entry").file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".sql"))
            .collect();
        on_disk.sort();

        let mut registered: Vec<String> = MIGRATIONS.iter().map(|m| m.name.to_string()).collect();
        registered.sort();

        assert_eq!(
            registered, on_disk,
            "the MIGRATIONS list in pool.rs and the migrations/ directory disagree — \
             add a `migration!(\"<name>.sql\")` entry for a new file (an unregistered \
             file never runs), or drop a stale entry"
        );
    }

    /// Application order is the ordinal order of the file names, and no
    /// migration is registered twice — a duplicate would apply once and then be
    /// skipped, quietly changing what a fresh database gets.
    #[test]
    fn registered_migrations_are_unique_and_in_ordinal_order() {
        let names: Vec<&str> = MIGRATIONS.iter().map(|m| m.name).collect();

        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "a migration is registered twice");

        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(
            names, sorted,
            "MIGRATIONS must be listed in ordinal order — it is the order they \
             are applied in on a fresh database"
        );
    }

    /// The boot lock is keyed on the schema being migrated: same schema, same
    /// key (two daemons serialize); different schema, different key (parallel
    /// migrations of private schemas do not block each other).
    #[test]
    fn schema_lock_key_is_stable_and_schema_specific() {
        assert_eq!(schema_lock_key("public"), schema_lock_key("public"));
        assert_ne!(schema_lock_key("public"), schema_lock_key("adele"));
        assert_ne!(schema_lock_key(""), schema_lock_key("public"));
    }
}
