//! Integration test for the learned error-classification store (epic #178,
//! tier 2). Gated on `TEST_DATABASE_URL`; pass-skips without a DB. Runs in a
//! private throwaway schema, so it never touches live tables.

mod support;

use std::sync::Arc;

use desktop_assistant_core::ports::store::ErrorClassificationStore;
use desktop_assistant_storage::{PgErrorClassificationStore, run_migrations};
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
        let url = support::test_database_url()?;
        let schema = format!("ec_test_{}", Uuid::now_v7().simple());
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
async fn learned_classification_record_and_lookup() {
    let Some(fixture) = SchemaFixture::try_new().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; learned_classification_record_and_lookup pass-skipped"
        );
        return;
    };
    run_migrations(&fixture.pool)
        .await
        .expect("migrations (incl. 022) apply");
    let store = PgErrorClassificationStore::new(fixture.pool.clone());

    // Record a learned mapping.
    store
        .record("bedrock", "exceeded your current quota", "billing_fatal")
        .await
        .expect("record");

    // A message containing the signature matches (case-insensitive).
    let hit = store
        .lookup(
            "bedrock",
            "Error: You EXCEEDED YOUR CURRENT QUOTA for this account.",
        )
        .await
        .expect("lookup");
    assert_eq!(
        hit.as_ref().map(|c| c.cause.as_str()),
        Some("billing_fatal")
    );

    // A non-matching message misses.
    let miss = store
        .lookup("bedrock", "some unrelated transient blip")
        .await
        .expect("lookup");
    assert!(miss.is_none());

    // Wrong connector misses even with the same text.
    let other = store
        .lookup("openai", "you exceeded your current quota")
        .await
        .expect("lookup");
    assert!(other.is_none());

    // Most specific (longest) signature wins when several match.
    store
        .record("bedrock", "quota", "rate_limited")
        .await
        .expect("record short");
    let specific = store
        .lookup("bedrock", "you exceeded your current quota now")
        .await
        .expect("lookup");
    assert_eq!(
        specific.as_ref().map(|c| c.cause.as_str()),
        Some("billing_fatal"),
        "longer signature must win over the shorter 'quota'"
    );

    // record is an idempotent upsert on (connector, signature).
    store
        .record("bedrock", "exceeded your current quota", "rate_limited")
        .await
        .expect("upsert");
    let updated = store
        .lookup("bedrock", "you exceeded your current quota")
        .await
        .expect("lookup");
    assert_eq!(
        updated.as_ref().map(|c| c.cause.as_str()),
        Some("rate_limited"),
        "upsert must overwrite the cause for an existing (connector, signature)"
    );

    fixture.cleanup().await;
}
