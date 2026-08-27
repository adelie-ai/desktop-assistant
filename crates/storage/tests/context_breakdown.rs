//! Integration tests for the per-turn context breakdown record (#588).
//!
//! Exercises `PgContextBreakdownStore` against a real Postgres with the
//! migrations applied. The record joins two measurements that are taken in
//! different places and were both discarded before this feature existed: the
//! assembler's per-part estimate, and the provider's own reported count. The
//! whole value of the row is that the two stay apart, so these tests read them
//! apart.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test context_breakdown
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log line so
//! the suite stays green without a DB.

mod support;

use std::sync::Arc;

use desktop_assistant_core::ports::context_breakdown::{
    ContextBreakdown, ContextBreakdownStore, PromptBreakdown, PromptPart,
};
use desktop_assistant_core::ports::llm::BudgetSource;
use desktop_assistant_storage::context_breakdown::PgContextBreakdownStore;
use desktop_assistant_storage::{UserId, run_migrations, with_user_id};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("issue588_{}", Uuid::now_v7().simple());
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect admin pool");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA \"{schema}\"")))
            .execute(&admin)
            .await
            .expect("create schema");
        admin.close().await;

        let pool = PgPoolOptions::new()
            .max_connections(2)
            .after_connect({
                let schema = Arc::new(schema.clone());
                move |conn, _| {
                    let schema = Arc::clone(&schema);
                    Box::pin(async move {
                        let sql = format!("SET search_path TO \"{schema}\", public");
                        sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                        Ok(())
                    })
                }
            })
            .connect(&url)
            .await
            .expect("connect scoped pool");
        run_migrations(&pool).await.expect("migrations");
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

/// The raw per-part figures a fixture writes, before they ever pass through
/// [`PromptBreakdown`]. A test that wants to check conservation across the
/// write/read round trip sums these directly, rather than reading the parts
/// back out through the same accessor the code under test also uses to sum
/// them - the latter would agree with a broken total by construction.
fn part_figures() -> Vec<(PromptPart, u64)> {
    PromptPart::ALL
        .iter()
        .enumerate()
        .map(|(i, part)| (*part, (i as u64 + 1) * 100))
        .collect()
}

/// A breakdown whose every part carries a different figure, so a slot written
/// to the wrong column is visible rather than hidden behind equal values.
fn parts() -> PromptBreakdown {
    PromptBreakdown::from_parts(part_figures(), 7)
}

/// One record for `conversation_id`, at `turn_ordinal`, keyed by `request_id`.
fn record(request_id: &str, conversation_id: &str, turn_ordinal: i32) -> ContextBreakdown {
    ContextBreakdown {
        request_id: request_id.to_string(),
        conversation_id: conversation_id.to_string(),
        turn_ordinal,
        model: "a-model".to_string(),
        provider_used_tokens: Some(41_000),
        budget_tokens: Some(200_000),
        budget_source: Some(BudgetSource::ConnectorTable),
        compaction_active: false,
        parts: parts(),
        projected_messages: 2,
        recorded_at: None,
    }
}

#[tokio::test]
async fn context_breakdown_rows_are_retrievable_for_the_whole_conversation() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        for (i, id) in ["r1", "r2", "r3"].iter().enumerate() {
            store
                .record(&record(id, "c1", i as i32 * 2))
                .await
                .expect("record");
        }
        let rows = store.list("c1", 50, 0).await.expect("list");
        assert_eq!(
            rows.len(),
            3,
            "every turn of the conversation is inspectable, not only the last: {rows:?}"
        );
        let ids: Vec<&str> = rows.iter().map(|r| r.request_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["r1", "r2", "r3"],
            "rows read back in conversation order"
        );

        let one = store.get("r2").await.expect("get");
        let one = one.expect("the row written under r2");
        assert_eq!(one.conversation_id, "c1");
        assert_eq!(one.turn_ordinal, 2);

        // The list and the get read the same row through two separate
        // statements. A column dropped from one of them would make a field
        // present in a get and absent in a list, which reads to a client as a
        // field the daemon sometimes forgets - so the listed row is compared
        // against the one the get returned, field by field.
        let listed = rows
            .iter()
            .find(|r| r.request_id == "r2")
            .expect("r2 in the listing");
        assert_eq!(listed, &one, "the list and the get must read one row alike");
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn context_breakdown_round_trips_every_recorded_field() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut written = record("r1", "c1", 4);
        written.compaction_active = true;
        store.record(&written).await.expect("record");

        let read = store.get("r1").await.expect("get").expect("the row");
        assert_eq!(read.model, written.model);
        assert_eq!(read.turn_ordinal, written.turn_ordinal);
        assert_eq!(read.provider_used_tokens, written.provider_used_tokens);
        assert_eq!(read.budget_tokens, written.budget_tokens);
        assert_eq!(read.budget_source, written.budget_source);
        assert!(read.compaction_active);
        assert_eq!(read.projected_messages, written.projected_messages);
        assert_eq!(
            read.parts.tool_count(),
            7,
            "the advertised tool count is a count, and survives as one"
        );
        for part in PromptPart::ALL {
            assert_eq!(
                read.parts.tokens(part),
                written.parts.tokens(part),
                "part `{}` came back as another part's figure",
                part.as_label()
            );
        }
        assert!(
            read.recorded_at.is_some(),
            "a stored row says when it was written"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn budget_source_survives_the_round_trip() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        // The case the tier exists for: a curated 200k and a silent
        // universal-fallback 200k are the same number and a different
        // situation. Only the tier tells them apart.
        let mut curated = record("r-curated", "c1", 0);
        curated.budget_source = Some(BudgetSource::ConnectorTable);
        let mut fallback = record("r-fallback", "c1", 2);
        fallback.budget_source = Some(BudgetSource::UniversalFallback);
        store.record(&curated).await.expect("record curated");
        store.record(&fallback).await.expect("record fallback");

        let read_curated = store.get("r-curated").await.expect("get").expect("row");
        let read_fallback = store.get("r-fallback").await.expect("get").expect("row");
        assert_eq!(
            read_curated.budget_tokens, read_fallback.budget_tokens,
            "precondition: the two turns resolved the same number"
        );
        assert_eq!(
            read_curated.budget_source,
            Some(BudgetSource::ConnectorTable)
        );
        assert_eq!(
            read_fallback.budget_source,
            Some(BudgetSource::UniversalFallback),
            "a curated limit and the universal fallback must not read alike"
        );

        // Every tier survives the round trip, not only the two above.
        for (i, source) in [
            BudgetSource::PurposeOverride,
            BudgetSource::ConnectorTable,
            BudgetSource::UniversalFallback,
            BudgetSource::LearnedCap,
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("r-tier-{i}");
            let mut row = record(&id, "c2", i as i32);
            row.budget_source = Some(source);
            store.record(&row).await.expect("record tier");
            assert_eq!(
                store
                    .get(&id)
                    .await
                    .expect("get")
                    .expect("row")
                    .budget_source,
                Some(source),
                "tier {source:?} did not survive the round trip"
            );
        }
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn context_breakdown_reports_provider_used_tokens_beside_the_estimate() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut row = record("r1", "c1", 0);
        // Deliberately unequal. The provider tokenises its own way, so the
        // estimate and the reported count are two measurements of one prompt
        // and never two halves of one figure.
        row.provider_used_tokens = Some(41_000);
        store.record(&row).await.expect("record");

        let read = store.get("r1").await.expect("get").expect("row");
        assert_eq!(
            read.provider_used_tokens,
            Some(41_000),
            "the provider's own count is stored as the provider gave it"
        );
        // The expected total is summed from the figures the fixture wrote,
        // on the write side, before any of them touched a `PromptBreakdown`
        // - never by reading the parts back out through `PromptBreakdown`'s
        // own accessor and re-summing, which would agree with a broken total
        // by construction (both walk the same backing array). Summing the
        // fixture's own inputs instead means this assertion exercises the
        // whole chain: fixture declares each part, the store writes it, the
        // store reads it back, and only then does production code sum it -
        // so a write that drops a part, a read that skips one, or a total
        // that double-counts one all show up as a mismatch here.
        let expected_total: u64 = part_figures().iter().map(|(_, tokens)| tokens).sum();
        assert_eq!(
            read.estimated_total_tokens(),
            expected_total,
            "the estimate is the sum of the measured parts and nothing else"
        );
        assert_ne!(
            read.provider_used_tokens,
            Some(read.estimated_total_tokens()),
            "the two figures are separate measurements; a record that made \
             them agree by construction would be reporting one of them twice"
        );

        // A provider that declines to report leaves the field absent. Zero
        // would invent a measurement, which is the one thing a reader must be
        // able to tell apart from an empty prompt.
        let mut silent = record("r2", "c1", 2);
        silent.provider_used_tokens = None;
        store.record(&silent).await.expect("record silent");
        let read_silent = store.get("r2").await.expect("get").expect("row");
        assert_eq!(read_silent.provider_used_tokens, None);
        assert!(
            read_silent.estimated_total_tokens() > 0,
            "the estimate stands on its own; it is not the provider's count"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn list_context_breakdowns_paginated() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        for i in 0..5 {
            store
                .record(&record(&format!("r{i}"), "c1", i * 2))
                .await
                .expect("record");
        }
        let first = store.list("c1", 2, 0).await.expect("page 1");
        let second = store.list("c1", 2, 2).await.expect("page 2");
        let third = store.list("c1", 2, 4).await.expect("page 3");
        let past_end = store.list("c1", 2, 6).await.expect("past the end");

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        assert_eq!(third.len(), 1, "the last page is short, not padded");
        assert!(past_end.is_empty(), "past the end is empty, not an error");

        let paged: Vec<String> = first
            .iter()
            .chain(second.iter())
            .chain(third.iter())
            .map(|r| r.request_id.clone())
            .collect();
        assert_eq!(
            paged,
            vec!["r0", "r1", "r2", "r3", "r4"],
            "the pages together are the whole conversation, in order and with \
             nothing repeated or skipped"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn context_breakdown_scoped_to_user_and_conversation() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("owner"), async {
        store.record(&record("r1", "c1", 0)).await.expect("record");
        store
            .record(&record("other-conv", "c2", 0))
            .await
            .expect("record");
    })
    .await;

    with_user_id(UserId::from("intruder"), async {
        assert!(
            store.list("c1", 50, 0).await.expect("list").is_empty(),
            "another user's conversation must read as empty"
        );
        assert_eq!(
            store.get("r1").await.expect("get"),
            None,
            "a request id is not a capability; the row belongs to its user"
        );
    })
    .await;

    with_user_id(UserId::from("owner"), async {
        let rows = store.list("c1", 50, 0).await.expect("list");
        assert_eq!(rows.len(), 1, "the owner still reads their own row");
        assert_eq!(
            rows[0].conversation_id, "c1",
            "the sibling conversation's row must not leak into this one"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn context_breakdown_record_is_idempotent() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let row = record("r1", "c1", 0);
        store.record(&row).await.expect("first write");
        store
            .record(&row)
            .await
            .expect("a repeat of the same write must not fail");

        let rows = store.list("c1", 50, 0).await.expect("list");
        assert_eq!(rows.len(), 1, "one turn, one row, however many writes");

        // A re-drive that carries a corrected figure replaces it rather than
        // adding a second row for the same turn.
        let mut corrected = record("r1", "c1", 0);
        corrected.provider_used_tokens = Some(9_999);
        store.record(&corrected).await.expect("re-drive");
        let rows = store.list("c1", 50, 0).await.expect("list");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].provider_used_tokens, Some(9_999));
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn a_reused_correlation_id_never_moves_a_record_between_conversations() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgContextBreakdownStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        // The key holds a value the CLIENT chose - a turn adopts the caller's
        // `turn_id` when it is a usable uuid - so a client that reuses one id
        // for two turns of two conversations writes the same key twice. Letting
        // the second write win would relocate the first conversation's record
        // into the second, leaving the first short one turn and reporting
        // nothing about it.
        store
            .record(&record("shared", "c1", 0))
            .await
            .expect("first");
        store
            .record(&record("shared", "c2", 0))
            .await
            .expect("a reused id is refused quietly, not as an error");

        let first = store.list("c1", 50, 0).await.expect("list c1");
        assert_eq!(
            first.len(),
            1,
            "the conversation that recorded the turn keeps its record"
        );
        assert_eq!(first[0].request_id, "shared");
        assert!(
            store.list("c2", 50, 0).await.expect("list c2").is_empty(),
            "the second conversation gains nothing, and takes nothing away"
        );
        assert_eq!(
            store
                .get("shared")
                .await
                .expect("get")
                .expect("row")
                .conversation_id,
            "c1",
            "the record still names the conversation whose turn it describes"
        );
    })
    .await;
    fx.cleanup().await;
}
