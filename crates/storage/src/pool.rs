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
/// Core tables (conversations, messages) are always created.
/// Vector-dependent tables (knowledge_base, tool_definitions) require the
/// `pgvector` extension; if it is unavailable those tables are skipped
/// with a warning — the daemon will still work for conversations.
///
/// HNSW indexes on the `embedding` column are NOT created here because the
/// vector dimension depends on which embedding model the user configures.
/// GIN/btree indexes for full-text search and tags are created.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    // Core tables — always required.
    sqlx::raw_sql(include_str!("../migrations/001_initial_schema.sql"))
        .execute(pool)
        .await?;

    // Best-effort: enable pgvector if the user has the privilege.
    if let Err(e) = sqlx::raw_sql("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await
    {
        tracing::warn!("could not create pgvector extension: {e}");
    }

    // Vector tables — each run independently so one failure doesn't block the other.
    for (table, sql) in [
        ("knowledge_base", include_str!("../migrations/002_vector_tables.sql")),
        ("tool_definitions", concat!(
            "CREATE TABLE IF NOT EXISTS tool_definitions (",
            "    name        TEXT PRIMARY KEY,",
            "    description TEXT NOT NULL,",
            "    parameters  JSONB NOT NULL,",
            "    source      TEXT NOT NULL,",
            "    is_core     BOOLEAN NOT NULL DEFAULT FALSE,",
            "    embedding   vector,",
            "    tsv         tsvector GENERATED ALWAYS AS (",
            "                    to_tsvector('english', name || ' ' || description)",
            "                ) STORED,",
            "    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            ");"
        )),
    ] {
        if let Err(e) = sqlx::raw_sql(sql).execute(pool).await {
            tracing::warn!("could not create {table} table (pgvector may be missing): {e}");
        }
    }

    // Non-vector indexes (GIN for full-text, btree for flags).
    if let Err(e) = sqlx::raw_sql(include_str!("../migrations/003_vector_indexes.sql"))
        .execute(pool)
        .await
    {
        tracing::warn!("could not create indexes: {e}");
    }

    // Track which embedding model produced each vector.
    if let Err(e) =
        sqlx::raw_sql(include_str!("../migrations/004_embedding_model_tracking.sql"))
            .execute(pool)
            .await
    {
        tracing::warn!("could not apply embedding model tracking migration: {e}");
    }

    // Convert messages.id from BIGSERIAL to TEXT (UUIDv7).
    sqlx::raw_sql(include_str!("../migrations/005_uuidv7_ids.sql"))
        .execute(pool)
        .await?;

    Ok(())
}
