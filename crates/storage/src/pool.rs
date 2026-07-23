use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

/// Create a connection pool to PostgreSQL.
pub async fn create_pool(url: &str, max_connections: u32) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(url)
        .await
}

/// Run embedded migrations against the database.
///
/// The `pgvector` extension is required and must be available in the
/// database — migrations will fail if it cannot be created.
///
/// HNSW indexes on the `embedding` column are NOT created here because the
/// vector dimension depends on which embedding model the user configures.
/// GIN/btree indexes for full-text search and tags are created.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    // Core tables — always required.
    sqlx::raw_sql(include_str!("../migrations/001_initial_schema.sql"))
        .execute(pool)
        .await?;

    // pgvector is required — fail fast if it cannot be enabled.
    sqlx::raw_sql("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;

    // Vector tables.
    sqlx::raw_sql(include_str!("../migrations/002_vector_tables.sql"))
        .execute(pool)
        .await?;
    sqlx::raw_sql(include_str!("../migrations/002b_tool_definitions.sql"))
        .execute(pool)
        .await?;

    // Indexes (GIN for full-text, btree for flags).
    sqlx::raw_sql(include_str!("../migrations/003_vector_indexes.sql"))
        .execute(pool)
        .await?;

    // Track which embedding model produced each vector.
    sqlx::raw_sql(include_str!(
        "../migrations/004_embedding_model_tracking.sql"
    ))
    .execute(pool)
    .await?;

    // Convert messages.id from BIGSERIAL to TEXT (UUIDv7).
    sqlx::raw_sql(include_str!("../migrations/005_uuidv7_ids.sql"))
        .execute(pool)
        .await?;

    // Dreaming watermarks — tracks per-conversation extraction progress.
    sqlx::raw_sql(include_str!("../migrations/006_dreaming_watermarks.sql"))
        .execute(pool)
        .await?;

    // Chunked embeddings — knowledge_base.embedding becomes vector[].
    sqlx::raw_sql(include_str!("../migrations/007_chunked_embeddings.sql"))
        .execute(pool)
        .await?;

    // Collapsible message summaries — reversible range summaries.
    sqlx::raw_sql(include_str!("../migrations/008_message_summaries.sql"))
        .execute(pool)
        .await?;

    // Conversation archival — nullable archived_at timestamp.
    sqlx::raw_sql(include_str!(
        "../migrations/009_conversation_archived_at.sql"
    ))
    .execute(pool)
    .await?;

    // Repair damage from pre-idempotent runs of migration 007 on existing
    // databases. No-op on fresh installs.
    sqlx::raw_sql(include_str!("../migrations/010_fix_damaged_embeddings.sql"))
        .execute(pool)
        .await?;

    // Per-conversation model selection (issue #11) — nullable JSONB column
    // on `conversations`.
    sqlx::raw_sql(include_str!(
        "../migrations/011_conversation_last_model.sql"
    ))
    .execute(pool)
    .await?;

    // Active-task anchor (issue #57) — nullable text column capturing the
    // user's current goal so it can be re-injected after windowing/summary.
    sqlx::raw_sql(include_str!(
        "../migrations/012_conversation_active_task.sql"
    ))
    .execute(pool)
    .await?;

    // Conversation full-text search (issue #71) — generated tsvector
    // columns + GIN indexes on `messages` and `conversations`. Generated-
    // stored columns auto-backfill on `ALTER TABLE`; the rewrite takes a
    // write lock proportional to message count, so first-run on large
    // histories may take a moment.
    sqlx::raw_sql(include_str!(
        "../migrations/013_conversation_message_fts.sql"
    ))
    .execute(pool)
    .await?;

    // Tag registry (issue #108) — formal vocabulary for KB tags. Categorical
    // tags emitted by the extractor are constrained to the registry; new
    // tags are created via a tool call with description and examples.
    sqlx::raw_sql(include_str!("../migrations/014_tag_registry.sql"))
        .execute(pool)
        .await?;

    // Knowledge-base review columns (issue #108) — `reviewed_at` watermark
    // gates per-memory consolidation; `review_generation` caps mutation
    // re-review loops; `deleted_at` enables soft-delete with TTL.
    sqlx::raw_sql(include_str!(
        "../migrations/015_knowledge_base_review_columns.sql"
    ))
    .execute(pool)
    .await?;

    // Multi-tenant schema (issue #102) — every personal-data table gains
    // `user_id NOT NULL` plus a `(user_id, …)` composite index for the
    // hot query paths #105's scoping will use. Pre-existing rows are
    // backfilled to the sentinel `'default'` user so single-tenant
    // installs keep working without auth changes.
    sqlx::raw_sql(include_str!("../migrations/016_multi_tenant_user_id.sql"))
        .execute(pool)
        .await?;

    // Turn state machine (issue #107) — DB-persisted turn state for
    // client-side execution of client-local MCP tools. A `pending_client_tool`
    // row is the daemon's record of "the LLM asked for a client-local
    // tool; we're waiting for the client to post the result back".
    sqlx::raw_sql(include_str!("../migrations/017_turn_state.sql"))
        .execute(pool)
        .await?;

    // Background tasks (issue #115) — persistent mirror of the in-memory
    // `BackgroundTaskRegistry`. On daemon restart the cold-restart sweep
    // reads this table to surface tasks that were running when the
    // previous daemon died.
    sqlx::raw_sql(include_str!("../migrations/018_background_tasks.sql"))
        .execute(pool)
        .await?;

    // Conversation scratchpad (issue #184) — ephemeral per-conversation
    // keyed notes, cascade-deleted with the conversation, with an FTS column.
    sqlx::raw_sql(include_str!("../migrations/019_scratchpads.sql"))
        .execute(pool)
        .await?;

    // Scratchpad note kind/order/done (issue #188) — note_type / seq / done
    // columns so a scratchpad can hold an ordered, checkable plan of TODOs.
    sqlx::raw_sql(include_str!(
        "../migrations/020_scratchpad_type_sequence_done.sql"
    ))
    .execute(pool)
    .await?;

    // Message FTS INSERT guard (issue #177) — the migration-013 generated
    // `tsv` column ran `to_tsvector` over full message content, which on a
    // large/high-entropy message exceeds Postgres's 1 MB tsvector limit and
    // aborts the INSERT. Redefine it to skip `tool`-role rows and bound the
    // indexed input so a large message can always be stored.
    sqlx::raw_sql(include_str!("../migrations/021_message_fts_guard.sql"))
        .execute(pool)
        .await?;

    // Learned error-classification cache (issue #178, tier 2) — global
    // (no user_id) connector knowledge mapping opaque error signatures to a
    // normalized cause, populated by the cheap-LLM tier so repeats are
    // recognized locally.
    sqlx::raw_sql(include_str!("../migrations/022_error_classifications.sql"))
        .execute(pool)
        .await?;

    // SendMessage idempotency keys (#204): records a completed turn's reply
    // keyed by (user_id, conversation_id, idempotency_key) so a dropped-then-
    // retried turn replays instead of re-running.
    sqlx::raw_sql(include_str!("../migrations/023_idempotency_keys.sql"))
        .execute(pool)
        .await?;

    // #227: per-conversation personality override (JSONB column on
    // conversations), mirroring 011's last_model_selection.
    sqlx::raw_sql(include_str!(
        "../migrations/024_conversation_personality.sql"
    ))
    .execute(pool)
    .await?;

    // #343: learned effective context-window observations — the reactive
    // safety net that `min()`s an observed-overflow ceiling into budget
    // resolution (down-only), complementing #342's proactive provisioning.
    sqlx::raw_sql(include_str!(
        "../migrations/025_context_window_observations.sql"
    ))
    .execute(pool)
    .await?;

    // Dream-cycle overhaul foundation — `embeddings_updated_at` (embedding
    // generation decoupled from content writes; a background task regenerates
    // NULL/stale vectors) and a first-class `source` provenance column
    // ('extraction' | 'consolidation' | 'explicit') replacing the
    // `source:dreaming` tag convention.
    sqlx::raw_sql(include_str!(
        "../migrations/026_knowledge_base_source_and_embedding_freshness.sql"
    ))
    .execute(pool)
    .await?;

    // Per-conversation tags (`TEXT[]`) so callers can label conversations at
    // creation time (e.g. "voice") and the UI can filter on them.
    sqlx::raw_sql(include_str!("../migrations/027_conversation_tags.sql"))
        .execute(pool)
        .await?;

    // Success high-water mark for learned context windows (#425): the other
    // half of the #343 bracket, so a mis-parsed overflow can't pin the budget
    // below a proven-good size and the budget can recover.
    sqlx::raw_sql(include_str!(
        "../migrations/028_context_window_success_watermark.sql"
    ))
    .execute(pool)
    .await?;

    // #434: Row-Level Security backstop for the LLM-facing db_query read
    // path — enables RLS + a per-user isolation policy on every user-scoped
    // table, so Postgres itself enforces tenant scoping even if the AST
    // grafter (#141) ever misses a table. Owner-only + idempotent (re-run
    // every startup) so it is safe as the daemon's un-privileged role; the
    // privileged role/grant half is a one-time superuser bootstrap
    // (`bootstrap/rls_role.sql`). The daemon's owner role is exempt
    // (non-FORCE RLS), so trusted paths are unaffected.
    sqlx::raw_sql(include_str!("../migrations/029_rls_backstop.sql"))
        .execute(pool)
        .await?;

    // Provider identity + index for provider-level tool surfacing (Phase 1):
    // real tools carry their MCP server / builtin-group provider, and the
    // daemon registers one synthetic `provider:<provider>` row per provider that
    // boosts its members' search scores when it matches a query.
    sqlx::raw_sql(include_str!(
        "../migrations/030_tool_definitions_provider.sql"
    ))
    .execute(pool)
    .await?;

    // #287: namespace the scratchpad by owner_todo (subagent-tree path) so
    // subagent writes are confined and reads snapshot by spawn marker.
    sqlx::raw_sql(include_str!("../migrations/031_scratchpad_owner_todo.sql"))
        .execute(pool)
        .await?;

    // #287: persist owner_todo + spawn_marker on background tasks so a
    // wait=false subagent's namespace/snapshot survive a daemon restart.
    sqlx::raw_sql(include_str!("../migrations/032_subagent_task_columns.sql"))
        .execute(pool)
        .await?;

    // Host-global skill index (#573): the disk-sourced skill/workflow catalog,
    // searchable by hybrid vector + full-text, mirroring `tool_definitions`.
    sqlx::raw_sql(include_str!("../migrations/033_skill_index.sql"))
        .execute(pool)
        .await?;

    // #570 Phase 1b: nullable `idempotency_key` on messages, carried on USER
    // rows only, so a transcript reload/reconnect surfaces the client's key and
    // clients dedup an echoed UserMessageAdded by exact match.
    sqlx::raw_sql(include_str!(
        "../migrations/034_message_idempotency_key.sql"
    ))
    .execute(pool)
    .await?;

    // #639: the skill catalog is cumulative -- a skill a scan no longer sees is
    // marked absent rather than deleted, so presence needs somewhere to live.
    sqlx::raw_sql(include_str!("../migrations/035_skill_presence.sql"))
        .execute(pool)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Every `.sql` file in `migrations/` must be wired into `run_migrations`
    /// above. Migrations are a hand-maintained `include_str!` list, NOT
    /// auto-discovered from the directory — so a new migration file that nobody
    /// registers compiles fine and silently never runs, surfacing only as a
    /// runtime "column does not exist" error against the live DB. This guard
    /// turns that into a build-time failure instead.
    ///
    /// (The reverse direction — a registered file that doesn't exist — is
    /// already caught at compile time, since `include_str!` fails to build.)
    #[test]
    fn every_migration_is_registered() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
        let source = include_str!("pool.rs");

        let mut unregistered: Vec<String> = std::fs::read_dir(dir)
            .expect("read migrations/ dir")
            .map(|e| e.expect("dir entry").file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".sql"))
            .filter(|name| !source.contains(name.as_str()))
            .collect();
        unregistered.sort();

        assert!(
            unregistered.is_empty(),
            "migration file(s) exist in migrations/ but are not referenced in \
             run_migrations() in pool.rs — add an \
             `sqlx::raw_sql(include_str!(\"../migrations/<name>\"))` call: {unregistered:?}"
        );
    }
}
