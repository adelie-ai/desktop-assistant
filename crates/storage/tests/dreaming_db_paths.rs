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
    BackfillEmbedFn, DreamingLlmFn, MAX_DELETE_REASON_CHARS, MAX_REVIEW_GENERATION,
    SOURCE_EXPLICIT, run_consolidation_scan, run_dreaming_scan, update_watermark,
};
use desktop_assistant_storage::knowledge_delete::{DEFAULT_PRUNE_FRACTION, KnowledgeDeletePolicy};
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

/// Seed an active KB row carrying an explicit `source` provenance value.
async fn seed_kb_sourced(pool: &PgPool, user_id: &str, id: &str, content: &str, source: &str) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, source) VALUES ($1, $2, $3, $4)",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(source)
    .execute(pool)
    .await
    .expect("seed sourced knowledge_base row");
}

/// Seed an active KB row already at `generation` review generations.
async fn seed_kb_at_generation(
    pool: &PgPool,
    user_id: &str,
    id: &str,
    content: &str,
    generation: i16,
) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, review_generation) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(generation)
    .execute(pool)
    .await
    .expect("seed knowledge_base row at a review generation");
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

async fn kb_deleted_kind(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT deleted_kind FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_kind")
}

async fn kb_deleted_reason(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT deleted_reason FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_reason")
}

async fn kb_superseded_by(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT superseded_by FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read superseded_by")
}

/// Tombstone counts per `deleted_kind`, as an auditor would ask for them:
/// `(kind, count)` sorted by kind, NULL kinds excluded.
async fn tombstones_by_kind(pool: &PgPool, user_id: &str) -> Vec<(String, i64)> {
    sqlx::query_as(
        "SELECT deleted_kind, COUNT(*) FROM knowledge_base \
         WHERE user_id = $1 AND deleted_at IS NOT NULL AND deleted_kind IS NOT NULL \
         GROUP BY deleted_kind ORDER BY deleted_kind",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .expect("group tombstones by deleted_kind")
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
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
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
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
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

/// The cap is a blast-radius bound on one run's unreviewed judgment, so the
/// shipped value is part of the contract, not an implementation detail. It is
/// configurable now, and the default is what an instance that sets nothing
/// gets.
#[test]
fn the_default_prune_fraction_is_the_reviewed_value() {
    assert!(
        (DEFAULT_PRUNE_FRACTION - 0.1).abs() < f64::EPSILON,
        "the default prune fraction is {DEFAULT_PRUNE_FRACTION}; changing the share of a \
         knowledge base one run may destroy is a reviewed decision, not a tweak"
    );
}

#[tokio::test]
async fn delete_cap_is_enforced_at_the_documented_fraction() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    let ids: Vec<String> = (1..=10).map(|i| format!("kb-{i}")).collect();
    for id in &ids {
        seed_kb(pool, "u1", id, "trivial fact").await;
    }

    // A plan that deletes all 10. cap = ceil(10 * 0.1) = 1, so nine are
    // deferred to a later run.
    let plan = format!(
        r#"{{"operations":[{{"op":"delete","ids":[{}],"reason":"trivial"}}]}}"#,
        ids.iter()
            .map(|id| format!("\"{id}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    let llm = llm_returning(&plan);
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        stats.soft_deleted, 1,
        "delete plan clamped to ceil(10 * the configured prune fraction)"
    );
    assert_eq!(kb_count_deleted(pool, "u1").await, 1);
    assert_eq!(kb_count_active(pool, "u1").await, 9);

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// Deliberately-entered entries are not the model's to remove.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn explicit_entry_proposed_for_deletion_is_not_pruned_and_is_counted() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_sourced(
        pool,
        "u1",
        "kb-a",
        "a fact the user entered",
        SOURCE_EXPLICIT,
    )
    .await;
    seed_kb_sourced(
        pool,
        "u1",
        "kb-b",
        "a fact dreaming extracted",
        "extraction",
    )
    .await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"delete","ids":["kb-a","kb-b"],"reason":"trivial"}]}"#,
    );
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert!(
        !kb_is_deleted(pool, "kb-a").await,
        "a source = 'explicit' entry must survive a proposed prune"
    );
    assert_eq!(
        kb_deleted_kind(pool, "kb-a").await,
        None,
        "the protected entry carries no tombstone provenance"
    );
    assert_eq!(
        stats.protected_from_delete, 1,
        "the refusal is reported in the run's stats"
    );

    // The protected id does not consume the delete cap, so the unprotected
    // entry is still pruned in the same run.
    assert!(
        kb_is_deleted(pool, "kb-b").await,
        "the unprotected entry is still pruned"
    );
    assert_eq!(stats.soft_deleted, 1);

    fx.cleanup().await;
}

#[tokio::test]
async fn edit_of_an_explicit_entry_does_not_launder_its_source() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_sourced(pool, "u1", "kb-a", "verbose original", SOURCE_EXPLICIT).await;

    let llm = llm_returning(r#"{"operations":[{"op":"edit","id":"kb-a","content":"TIGHTER"}]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        kb_content(pool, "kb-a").await.as_deref(),
        Some("TIGHTER"),
        "editing an explicit entry is allowed"
    );
    // Rewriting `source` to 'consolidation' would strip the protection and make
    // the entry prunable on the next night's run.
    assert_eq!(
        kb_source(pool, "kb-a").await.as_deref(),
        Some(SOURCE_EXPLICIT),
        "an edit must not launder explicit provenance into 'consolidation'"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn merge_whose_canonical_is_explicit_keeps_the_explicit_source() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    // kb-a sorts lowest, so it is the canonical (surviving) row.
    seed_kb_sourced(pool, "u1", "kb-a", "user-entered fact", SOURCE_EXPLICIT).await;
    seed_kb_sourced(pool, "u1", "kb-b", "extracted restatement", "extraction").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(kb_content(pool, "kb-a").await.as_deref(), Some("UNIFIED"));
    assert_eq!(
        kb_source(pool, "kb-a").await.as_deref(),
        Some(SOURCE_EXPLICIT),
        "merging into an explicit row must not downgrade it to 'consolidation'"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn merge_with_an_explicit_member_keeps_explicit_source_on_the_canonical() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    // The explicit row is the *member* here, not the canonical: its content is
    // carried into kb-a, so kb-a inherits the protection with it.
    seed_kb_sourced(pool, "u1", "kb-a", "extracted restatement", "extraction").await;
    seed_kb_sourced(pool, "u1", "kb-b", "user-entered fact", SOURCE_EXPLICIT).await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        kb_source(pool, "kb-a").await.as_deref(),
        Some(SOURCE_EXPLICIT),
        "explicit provenance survives a merge that absorbs an explicit member"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// Delete provenance: merge and prune are distinguishable on disk.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn merge_records_superseding_id_on_every_soft_deleted_member() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    for id in ["kb-a", "kb-b", "kb-c"] {
        seed_kb(pool, "u1", id, "a near-duplicate fact").await;
    }

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["kb-a","kb-b","kb-c"],"content":"UNIFIED","scope":null}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    for member in ["kb-b", "kb-c"] {
        assert!(kb_is_deleted(pool, member).await, "{member} is retired");
        assert_eq!(
            kb_deleted_kind(pool, member).await.as_deref(),
            Some("merge"),
            "{member} is recorded as a merge, not a prune"
        );
        assert_eq!(
            kb_superseded_by(pool, member).await.as_deref(),
            Some("kb-a"),
            "{member} names the canonical row that absorbed it"
        );
        assert_eq!(
            kb_deleted_reason(pool, member).await,
            None,
            "a merge member has no stated reason; superseded_by is the reason"
        );
    }

    fx.cleanup().await;
}

#[tokio::test]
async fn standalone_prune_records_prune_kind_and_the_models_reason() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "circumstantial detail").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"delete","ids":["kb-b"],"reason":"mattered only in the moment"}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert!(kb_is_deleted(pool, "kb-b").await);
    assert_eq!(
        kb_deleted_kind(pool, "kb-b").await.as_deref(),
        Some("prune")
    );
    assert_eq!(
        kb_deleted_reason(pool, "kb-b").await.as_deref(),
        Some("mattered only in the moment"),
        "the model's stated reason is persisted, not discarded"
    );
    assert_eq!(
        kb_superseded_by(pool, "kb-b").await,
        None,
        "nothing supersedes a prune"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn merge_and_prune_tombstones_are_distinguishable_by_sql() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    for id in ["kb-a", "kb-b", "kb-c"] {
        seed_kb(pool, "u1", id, "a fact").await;
    }

    let llm = llm_returning(
        r#"{"operations":[
            {"op":"merge","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null},
            {"op":"delete","ids":["kb-c"],"reason":"trivial"}
        ]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    // The audit the epic could not run: split tombstones into relocated vs
    // destroyed, from SQL alone.
    assert_eq!(
        tombstones_by_kind(pool, "u1").await,
        vec![("merge".to_string(), 1), ("prune".to_string(), 1)],
        "merge and prune tombstones are separable without reading logs"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn prune_with_no_stated_reason_records_null_not_an_empty_string() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "trivia").await;

    // `reason` omitted entirely, then whitespace-only: both mean "unstated".
    let llm = llm_returning(r#"{"operations":[{"op":"delete","ids":["kb-b"]}]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        kb_deleted_kind(pool, "kb-b").await.as_deref(),
        Some("prune"),
        "an unstated reason still records that this was a prune"
    );
    assert_eq!(
        kb_deleted_reason(pool, "kb-b").await,
        None,
        "an absent reason is NULL, not an empty string"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn delete_reason_from_the_model_is_bounded_before_storage() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "trivia").await;

    // Multi-byte so a byte-slicing truncation would panic on a char boundary.
    let over_long: String = "é".repeat(MAX_DELETE_REASON_CHARS * 2);
    let llm = llm_returning(&format!(
        r#"{{"operations":[{{"op":"delete","ids":["kb-b"],"reason":"{over_long}"}}]}}"#
    ));
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    let stored = kb_deleted_reason(pool, "kb-b")
        .await
        .expect("a reason was stored");
    assert_eq!(
        stored.chars().count(),
        MAX_DELETE_REASON_CHARS,
        "an unbounded model reason is clamped to the storage bound"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn delete_provenance_is_never_written_to_another_tenants_rows() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-u1", "user1 trivia").await;
    seed_kb(pool, "u2", "kb-u2", "user2 fact").await;

    // The plan names a user1 id. user2's pass sees the same response but the id
    // is not in its partition, so nothing of user2's may be stamped.
    let llm =
        llm_returning(r#"{"operations":[{"op":"delete","ids":["kb-u1"],"reason":"trivial"}]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        kb_deleted_kind(pool, "kb-u1").await.as_deref(),
        Some("prune")
    );
    assert!(!kb_is_deleted(pool, "kb-u2").await, "user2's row is active");
    assert_eq!(
        kb_deleted_kind(pool, "kb-u2").await,
        None,
        "user2's row carries no provenance from user1's run"
    );
    assert_eq!(kb_superseded_by(pool, "kb-u2").await, None);
    assert_eq!(kb_deleted_reason(pool, "kb-u2").await, None);

    fx.cleanup().await;
}

#[tokio::test]
async fn empty_knowledge_base_consolidates_to_a_clean_no_op() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    // The user's only row is already a tombstone, so there is nothing active to
    // recompute and the tombstone's provenance must not be rewritten.
    seed_kb_soft_deleted(pool, "u1", "kb-gone", "already retired", 1).await;

    let llm = llm_returning(r#"{"operations":[{"op":"delete","ids":["kb-gone"],"reason":"x"}]}"#);
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation over an empty active KB succeeds");

    assert_eq!(stats.reviewed, 0);
    assert_eq!(stats.soft_deleted, 0);
    assert_eq!(stats.protected_from_delete, 0);
    assert_eq!(stats.settled_unchanged, 0);
    assert_eq!(
        kb_deleted_kind(pool, "kb-gone").await,
        None,
        "an existing tombstone is not restamped by a no-op run"
    );

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
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
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

// ---------------------------------------------------------------------------
// review_generation: an entry's prose settles after MAX_REVIEW_GENERATION
// rewrites, so consolidation stops paraphrasing its own paraphrases.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn review_generation_increments_on_each_mutation() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-x", "verbose original", 0).await;

    let llm = llm_returning(r#"{"operations":[{"op":"edit","id":"kb-x","content":"REWRITTEN"}]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(kb_content(pool, "kb-x").await.as_deref(), Some("REWRITTEN"));
    assert_eq!(
        kb_review_generation(pool, "kb-x").await,
        1,
        "a rewrite counts one review generation"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn entry_at_review_generation_cap_is_excluded_from_further_mutation() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-x", "settled prose", MAX_REVIEW_GENERATION).await;

    let llm = llm_returning(r#"{"operations":[{"op":"edit","id":"kb-x","content":"REWRITTEN"}]}"#);
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    assert_eq!(
        kb_content(pool, "kb-x").await.as_deref(),
        Some("settled prose"),
        "an entry at the review cap is not rewritten again"
    );
    assert_eq!(
        kb_review_generation(pool, "kb-x").await,
        MAX_REVIEW_GENERATION,
        "the counter does not advance past the cap"
    );
    assert_eq!(stats.updated, 0, "no edit was applied");
    assert_eq!(
        stats.settled_unchanged, 1,
        "the refusal is reported in the run's stats"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn merge_touching_a_settled_entry_is_skipped_entirely() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-a", "settled prose", MAX_REVIEW_GENERATION).await;
    seed_kb(pool, "u1", "kb-b", "a near-duplicate").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
    );
    let stats = run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    // Merging "around" the settled member would apply content synthesized to
    // stand for both rows while one of them stays live, duplicating it.
    assert_eq!(stats.merged_clusters, 0, "the whole merge is dropped");
    assert_eq!(
        kb_content(pool, "kb-a").await.as_deref(),
        Some("settled prose")
    );
    assert_eq!(
        kb_content(pool, "kb-b").await.as_deref(),
        Some("a near-duplicate")
    );
    assert!(!kb_is_deleted(pool, "kb-b").await, "no member is retired");
    assert_eq!(stats.settled_unchanged, 1);

    fx.cleanup().await;
}

#[tokio::test]
async fn settled_entry_can_still_be_pruned() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-a", "settled prose", MAX_REVIEW_GENERATION).await;
    seed_kb(pool, "u1", "kb-b", "keeper").await;

    let llm = llm_returning(r#"{"operations":[{"op":"delete","ids":["kb-a"],"reason":"wrong"}]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    // The cap settles an entry's *prose*, not the store: consolidation's own
    // output must stay removable, or the interpretation layer ossifies.
    assert!(
        kb_is_deleted(pool, "kb-a").await,
        "a settled entry is still prunable"
    );
    assert_eq!(
        kb_deleted_kind(pool, "kb-a").await.as_deref(),
        Some("prune")
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
