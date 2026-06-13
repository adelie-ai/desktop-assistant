//! Integration test for the learned context-window store (issue #343), the
//! reactive safety net that complements #342. Gated on `TEST_DATABASE_URL`;
//! pass-skips without a DB. Runs in a private throwaway schema, so it never
//! touches live tables.

use std::sync::Arc;

use desktop_assistant_core::ports::store::LearnedWindowStore;
use desktop_assistant_storage::{PgLearnedWindowStore, run_migrations};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

struct SchemaFixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl SchemaFixture {
    async fn try_new() -> Option<Self> {
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let schema = format!("cwo_test_{}", Uuid::now_v7().simple());
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to TEST_DATABASE_URL");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA \"{schema}\"")))
            .execute(&admin)
            .await
            .expect("create test schema");
        admin.close().await;

        let schema_for_hook = Arc::new(schema.clone());
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .after_connect(move |conn, _meta| {
                let schema = Arc::clone(&schema_for_hook);
                Box::pin(async move {
                    let sql = format!("SET search_path TO \"{schema}\", public");
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await
            .expect("connect per-test pool");

        Some(Self {
            pool,
            schema,
            admin_url: url,
        })
    }

    async fn cleanup(self) {
        self.pool.close().await;
        if let Ok(admin) = PgPoolOptions::new()
            .max_connections(1)
            .connect(&self.admin_url)
            .await
        {
            let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                "DROP SCHEMA \"{}\" CASCADE",
                self.schema
            )))
            .execute(&admin)
            .await;
            admin.close().await;
        }
    }
}

#[tokio::test]
async fn learned_window_ratchets_down_invalidates_and_isolates() {
    let Some(fixture) = SchemaFixture::try_new().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; learned_window_ratchets_down_invalidates_and_isolates pass-skipped"
        );
        return;
    };
    run_migrations(&fixture.pool)
        .await
        .expect("migrations (incl. 025) apply");
    let store = PgLearnedWindowStore::new(fixture.pool.clone());

    // Turn 1: an overflow under an 8192 configured window observed a 4096
    // ceiling — record it.
    store
        .record("ollama", "qwen2.5", 4_096, 8_192)
        .await
        .expect("record");
    let got = store.lookup("ollama", "qwen2.5").await.expect("lookup");
    assert_eq!(
        got.map(|w| (w.observed_limit, w.configured_window)),
        Some((4_096, 8_192))
    );

    // Ratchet DOWN: a smaller observation under the same configured window
    // overwrites.
    store
        .record("ollama", "qwen2.5", 2_048, 8_192)
        .await
        .expect("record smaller");
    assert_eq!(
        store
            .lookup("ollama", "qwen2.5")
            .await
            .expect("lookup")
            .map(|w| w.observed_limit),
        Some(2_048),
        "a smaller observation must ratchet the stored ceiling down"
    );

    // NEVER UP: a larger observation under the same configured window is
    // ignored (down-only).
    store
        .record("ollama", "qwen2.5", 6_000, 8_192)
        .await
        .expect("record larger");
    assert_eq!(
        store
            .lookup("ollama", "qwen2.5")
            .await
            .expect("lookup")
            .map(|w| w.observed_limit),
        Some(2_048),
        "a larger observation under the same configured window must NOT raise the stored ceiling"
    );

    // INVALIDATION: a record under a DIFFERENT configured window replaces the
    // row wholesale — a deliberate window change starts fresh (even if the new
    // observed value is larger than the old one).
    store
        .record("ollama", "qwen2.5", 12_000, 16_384)
        .await
        .expect("record new configured window");
    assert_eq!(
        store
            .lookup("ollama", "qwen2.5")
            .await
            .expect("lookup")
            .map(|w| (w.observed_limit, w.configured_window)),
        Some((12_000, 16_384)),
        "a new configured window must replace the stale lower observation"
    );

    // CROSS-(connector, model) ISOLATION: a different model's cap doesn't leak.
    store
        .record("ollama", "llama3", 1_024, 8_192)
        .await
        .expect("record other model");
    assert_eq!(
        store
            .lookup("ollama", "qwen2.5")
            .await
            .expect("lookup")
            .map(|w| w.observed_limit),
        Some(12_000),
        "another model's learned cap must not leak into qwen2.5"
    );
    assert_eq!(
        store
            .lookup("ollama", "llama3")
            .await
            .expect("lookup")
            .map(|w| w.observed_limit),
        Some(1_024)
    );
    // Different connector, same model name is also isolated.
    assert!(
        store
            .lookup("bedrock", "qwen2.5")
            .await
            .expect("lookup")
            .is_none(),
        "another connector must not see this model's cap"
    );

    // SURVIVES RESTART: a fresh store over the same pool reads the persisted
    // value (the row outlives any in-process state).
    let reopened = PgLearnedWindowStore::new(fixture.pool.clone());
    assert_eq!(
        reopened
            .lookup("ollama", "qwen2.5")
            .await
            .expect("lookup")
            .map(|w| w.observed_limit),
        Some(12_000),
        "persisted observation must survive a store restart"
    );

    fixture.cleanup().await;
}
