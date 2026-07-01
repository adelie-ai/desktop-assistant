//! DB-touching integration tests for the "dreaming" (memory-consolidation)
//! subsystem (issue #435).
//!
//! Before this suite, only the pure helpers (union-find clustering, JSON
//! extraction, op parsing) had coverage — the entire transactional DB path
//! (`apply_ops`, the holistic deletion cap, per-user consolidation scoping,
//! the extraction watermark, archival) was unverified, so a bad consolidation
//! run could silently gut or cross-leak a user's knowledge base with nothing
//! to catch it.
//!
//! These drive the real code through its public entry points
//! ([`run_consolidation_scan`], [`run_dreaming_scan`]) with fake LLM/embed
//! closures so the exact op plan is deterministic; the one path with no
//! reachable public entry point (the user-scoped watermark upsert guard) is
//! exercised via the surfaced [`update_watermark`].
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! # the `vector` extension must exist in the target database:
//! psql "$URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test dreaming_db_paths
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips (a loud banner
//! prints once) so the suite stays green without a database.

mod support;

use desktop_assistant_storage::dreaming::{
    BackfillEmbedFn, DreamingLlmFn, run_consolidation_scan, run_dreaming_scan, update_watermark,
};
use desktop_assistant_storage::{UserId, with_user_id};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/// A dreaming LLM that ignores its prompts and always returns `response`.
fn llm_returning(response: &str) -> DreamingLlmFn {
    let response = response.to_string();
    Box::new(move |_system, _user| {
        let response = response.clone();
        Box::pin(async move { Ok(response) })
    })
}

/// An embedder that must never be called (the extraction facts in these tests
/// carry no `new_tags`, which is the only thing that would invoke it).
fn unused_embed_fn() -> BackfillEmbedFn {
    Box::new(|_texts| {
        Box::pin(async move { Err("embed_fn must not be called in this test".to_string()) })
    })
}

// ---------------------------------------------------------------------------
// Seed helpers
// ---------------------------------------------------------------------------

async fn seed_kb(pool: &PgPool, user_id: &str, id: &str, content: &str) {
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content) VALUES ($1, $2, $3)")
        .bind(id)
        .bind(user_id)
        .bind(content)
        .execute(pool)
        .await
        .expect("seed knowledge_base row");
}

/// Seed a soft-deleted KB row whose `deleted_at` is `days_ago` days in the past.
async fn seed_kb_soft_deleted(
    pool: &PgPool,
    user_id: &str,
    id: &str,
    content: &str,
    days_ago: i32,
) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, deleted_at) \
         VALUES ($1, $2, $3, NOW() - make_interval(days => $4))",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(days_ago)
    .execute(pool)
    .await
    .expect("seed soft-deleted knowledge_base row");
}

async fn seed_conversation(pool: &PgPool, user_id: &str, id: &str) {
    sqlx::query("INSERT INTO conversations (id, title, user_id) VALUES ($1, 'test', $2)")
        .bind(id)
        .bind(user_id)
        .execute(pool)
        .await
        .expect("seed conversation");
}

async fn seed_message(
    pool: &PgPool,
    user_id: &str,
    conversation_id: &str,
    id: &str,
    ordinal: i32,
    role: &str,
    content: &str,
) {
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, user_id, ordinal, role, content) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(user_id)
    .bind(ordinal)
    .bind(role)
    .bind(content)
    .execute(pool)
    .await
    .expect("seed message");
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

async fn kb_content(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT content FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
        .expect("read kb content")
}

async fn kb_source(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT source FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read kb source")
}

async fn kb_is_deleted(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT deleted_at IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read kb deleted_at")
}

async fn kb_exists(pool: &PgPool, id: &str) -> bool {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("count kb row");
    count > 0
}

async fn kb_review_generation(pool: &PgPool, id: &str) -> i16 {
    sqlx::query_scalar("SELECT review_generation FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read review_generation")
}

async fn kb_count_for_user(pool: &PgPool, user_id: &str) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1")
        .bind(user_id)
        .fetch_one(pool)
        .await
        .expect("count kb rows for user")
}

async fn kb_count_deleted(pool: &PgPool, user_id: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND deleted_at IS NOT NULL",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await
    .expect("count soft-deleted kb rows")
}

async fn kb_count_active(pool: &PgPool, user_id: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND deleted_at IS NULL",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await
    .expect("count active kb rows")
}

async fn conversation_is_archived(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT archived_at IS NOT NULL FROM conversations WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read archived_at")
}

// ---------------------------------------------------------------------------
// apply_ops — canonical update + member soft-delete, and tenant isolation.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn apply_ops_soft_deletes_members_and_updates_canonical() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-aaa", "alpha fact").await;
    seed_kb(pool, "u1", "kb-bbb", "beta fact").await;

    // Merge both entries into the lexicographically-lowest canonical id.
    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["kb-aaa","kb-bbb"],"content":"UNIFIED","scope":null}]}"#,
    );
    let stats = run_consolidation_scan(pool, &llm, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(stats.merged_clusters, 1, "one merge cluster applied");
    assert_eq!(stats.soft_deleted, 1, "one cluster member soft-deleted");

    // The canonical row absorbs the synthesized content and is stamped
    // 'consolidation', and stays active.
    assert_eq!(kb_content(pool, "kb-aaa").await.as_deref(), Some("UNIFIED"));
    assert_eq!(
        kb_source(pool, "kb-aaa").await.as_deref(),
        Some("consolidation")
    );
    assert!(
        !kb_is_deleted(pool, "kb-aaa").await,
        "canonical row stays active"
    );

    // Central assertion: the non-canonical member is *soft*-deleted (row still
    // present, deleted_at set). Flipping the member soft-delete UPDATE breaks it.
    assert!(
        kb_is_deleted(pool, "kb-bbb").await,
        "cluster member is soft-deleted"
    );
    assert!(
        kb_exists(pool, "kb-bbb").await,
        "soft-delete is not a hard delete"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn apply_ops_never_touches_other_users_kb() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    // user2 owns exactly one row: soft-deleted 60 days ago (past the 30-day
    // TTL). It has NO active entries, so user2's own consolidation never runs —
    // the only thing that could reap this row is user1's apply_ops TTL sweep,
    // which must be scoped to user1.
    seed_kb_soft_deleted(pool, "u2", "u2-old", "user2 expired fact", 60).await;

    // user1 has two active entries; merging them drives apply_ops (and its
    // leading per-user TTL reap) under the user1 scope.
    seed_kb(pool, "u1", "u1-a", "alpha").await;
    seed_kb(pool, "u1", "u1-b", "beta").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["u1-a","u1-b"],"content":"MERGED","scope":null}]}"#,
    );
    run_consolidation_scan(pool, &llm, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    // Sanity: user1's merge landed.
    assert!(
        kb_is_deleted(pool, "u1-b").await,
        "user1's merge member was soft-deleted"
    );

    // Central assertion: user2's expired-soft-deleted row is untouched. Dropping
    // `user_id = $2` from the TTL-reap DELETE cross-deletes it during user1's run.
    assert!(
        kb_exists(pool, "u2-old").await,
        "user2's expired row must NOT be reaped by user1's consolidation cycle"
    );
    assert!(
        kb_is_deleted(pool, "u2-old").await,
        "user2's row is unchanged (still soft-deleted, not hard-deleted)"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// consolidation — deletion cap and per-user scoping.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn consolidation_respects_max_delete_fraction() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    for id in ["kb-1", "kb-2", "kb-3", "kb-4"] {
        seed_kb(pool, "u1", id, "trivial fact").await;
    }

    // A plan that deletes all 4. MAX_DELETE_FRACTION = 0.5 → cap = ceil(4*0.5) = 2,
    // so the excess is dropped this run and only 2 rows are soft-deleted.
    let llm = llm_returning(
        r#"{"operations":[{"op":"delete","ids":["kb-1","kb-2","kb-3","kb-4"],"reason":"trivial"}]}"#,
    );
    let stats = run_consolidation_scan(pool, &llm, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    // Central assertion: the delete plan is clamped to the cap. Removing the
    // `delete_ops.truncate(cap)` line lets all 4 through.
    assert_eq!(
        stats.soft_deleted, 2,
        "delete plan clamped to the 50% deletion cap"
    );
    assert_eq!(kb_count_deleted(pool, "u1").await, 2);
    assert_eq!(kb_count_active(pool, "u1").await, 2);

    fx.cleanup().await;
}

#[tokio::test]
async fn consolidate_user_is_tenant_isolated() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "u1-x", "user1 fact").await;
    seed_kb(pool, "u2", "u2-keep", "user2 fact").await;

    // The delete targets a user1 id. Under user2's scope this id is not a loaded
    // (valid) entry, so it is ignored — each `consolidate_user` pass runs inside
    // its own `with_user_id` scope and only sees/touches its own partition.
    let llm = llm_returning(r#"{"operations":[{"op":"delete","ids":["u1-x"],"reason":"x"}]}"#);
    run_consolidation_scan(pool, &llm, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    // Central assertion: user1's own pass loaded its entry (via the user-scoped
    // load) and applied the delete. Scoping the load to a wrong user makes this
    // fail (nothing loaded ⇒ nothing deleted).
    assert!(
        kb_is_deleted(pool, "u1-x").await,
        "user1's entry deleted under its own scope"
    );
    // user2 is untouched — neither user1's op nor the cross-user delete id reached
    // it.
    assert!(
        !kb_is_deleted(pool, "u2-keep").await,
        "user2's entry must not be deleted"
    );
    assert_eq!(
        kb_content(pool, "u2-keep").await.as_deref(),
        Some("user2 fact")
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn review_generation_saturates_at_max() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-x", "verbose original").await;
    // Pre-set the entry to the review cap (MAX_REVIEW_GENERATION = 2).
    sqlx::query("UPDATE knowledge_base SET review_generation = 2 WHERE id = 'kb-x'")
        .execute(pool)
        .await
        .expect("set review_generation to the cap");

    let llm = llm_returning(r#"{"operations":[{"op":"edit","id":"kb-x","content":"REWRITTEN"}]}"#);
    run_consolidation_scan(pool, &llm, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(
        kb_content(pool, "kb-x").await.as_deref(),
        Some("REWRITTEN"),
        "edit applied"
    );
    // Central assertion: LEAST(review_generation + 1, MAX) saturates at 2, not 3.
    // Dropping the LEAST clamp lets it climb to 3.
    assert_eq!(
        kb_review_generation(pool, "kb-x").await,
        2,
        "review_generation saturates at MAX_REVIEW_GENERATION"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// extraction — watermark idempotency and user-scoped watermark upsert.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_is_idempotent_via_watermark() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    seed_conversation(pool, "u1", "conv-1").await;
    seed_message(pool, "u1", "conv-1", "m1", 1, "user", "I always use vim").await;
    seed_message(pool, "u1", "conv-1", "m2", 2, "assistant", "Noted.").await;

    let llm =
        llm_returning(r#"{"facts":[{"content":"The user prefers vim.","tags":[],"scope":null}]}"#);
    let embed = unused_embed_fn();
    let token = CancellationToken::new();

    let first = run_dreaming_scan(pool, &llm, &embed, "test-model", 0, &token, None)
        .await
        .expect("first dreaming scan succeeds");
    assert_eq!(first, 1, "first run extracts exactly one fact");
    assert_eq!(kb_count_for_user(pool, "u1").await, 1);

    // Second run over the exact same messages: the watermark advanced to the max
    // ordinal, so the conversation is no longer selected. Central assertion: zero
    // new facts and no duplicate row. Not advancing the watermark re-extracts.
    let second = run_dreaming_scan(pool, &llm, &embed, "test-model", 0, &token, None)
        .await
        .expect("second dreaming scan succeeds");
    assert_eq!(
        second, 0,
        "second run over the same messages writes nothing"
    );
    assert_eq!(
        kb_count_for_user(pool, "u1").await,
        1,
        "no duplicate fact written"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn watermark_is_user_scoped() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    // A single conversation row (global PK) that user1 owns a watermark for.
    seed_conversation(pool, "u1", "conv-shared").await;

    with_user_id(UserId::new("u1"), async {
        update_watermark(pool, "conv-shared", 5).await
    })
    .await
    .expect("user1 watermark write");

    // user2 attempts to advance the SAME conversation-id watermark. The
    // `(conversation_id)` upsert is guarded by `WHERE user_id = $1`, so for a row
    // owned by user1 this is a silent no-op that returns Ok.
    with_user_id(UserId::new("u2"), async {
        update_watermark(pool, "conv-shared", 99).await
    })
    .await
    .expect("user2 watermark write returns Ok (no-op)");

    let (owner, ordinal): (String, i32) = sqlx::query_as(
        "SELECT user_id, last_processed_ordinal FROM dreaming_watermarks \
         WHERE conversation_id = 'conv-shared'",
    )
    .fetch_one(pool)
    .await
    .expect("read back watermark");

    // Central assertion: user1's watermark survives untouched. Dropping the
    // `WHERE user_id = $1` guard lets user2 clobber it to 99.
    assert_eq!(owner, "u1", "watermark still owned by user1");
    assert_eq!(ordinal, 5, "user2 must not clobber user1's watermark");

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// archival — idle conversations get flagged, fresh ones don't.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn archival_moves_expected_rows() {
    let Some(fx) = support::DbFixture::try_new("dream435").await else {
        return;
    };
    let pool = &fx.pool;

    // One conversation idle for 100 days, one fresh. Neither has messages, so the
    // extraction phase is a clean no-op and only archival acts.
    sqlx::query(
        "INSERT INTO conversations (id, title, user_id, updated_at) \
         VALUES ('conv-old', 'old', 'default', NOW() - make_interval(days => 100))",
    )
    .execute(pool)
    .await
    .expect("seed idle conversation");
    sqlx::query(
        "INSERT INTO conversations (id, title, user_id, updated_at) \
         VALUES ('conv-new', 'new', 'default', NOW())",
    )
    .execute(pool)
    .await
    .expect("seed fresh conversation");

    let llm = llm_returning("{}");
    let embed = unused_embed_fn();
    run_dreaming_scan(
        pool,
        &llm,
        &embed,
        "test-model",
        30,
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("dreaming scan succeeds");

    // Central assertion: only the idle conversation is archived. Inverting the
    // archival age predicate flips both rows.
    assert!(
        conversation_is_archived(pool, "conv-old").await,
        "conversation idle beyond the window is archived"
    );
    assert!(
        !conversation_is_archived(pool, "conv-new").await,
        "fresh conversation is not archived"
    );

    fx.cleanup().await;
}
