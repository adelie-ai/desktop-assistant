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

    Ok(())
}
