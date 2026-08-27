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
    BackfillEmbedFn, Disposition, DreamingLlmFn, MAX_DELETE_REASON_CHARS, MAX_REVIEW_GENERATION,
    OpBuffer, ProposedOp, SOURCE_EXPLICIT, SynthesizedMerge, apply_ops, run_consolidation_scan,
    run_dreaming_scan, update_watermark,
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

/// A dreaming LLM that always returns `response` and keeps every user prompt it
/// was given, so a test can assert what the model was actually shown.
fn llm_capturing_prompts(
    response: &str,
) -> (DreamingLlmFn, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let response = response.to_string();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = std::sync::Arc::clone(&seen);
    let llm: DreamingLlmFn = Box::new(move |_system, user| {
        captured
            .lock()
            .expect("the capture buffer is not poisoned")
            .push(user);
        let response = response.clone();
        Box::pin(async move { Ok(response) })
    });
    (llm, seen)
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

/// Seed an active KB row carrying tags, which is what the daily pass groups by.
async fn seed_kb_tagged(pool: &PgPool, user_id: &str, id: &str, content: &str, tags: &[&str]) {
    let tags: Vec<String> = tags.iter().map(|t| (*t).to_string()).collect();
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content, tags) VALUES ($1, $2, $3, $4)")
        .bind(id)
        .bind(user_id)
        .bind(content)
        .bind(&tags)
        .execute(pool)
        .await
        .expect("seed tagged knowledge_base row");
}

/// Record `opens` recent opens against an entry, as the use log stores them:
/// counters plus the exact timestamps of the recent window.
async fn seed_recent_opens(pool: &PgPool, user_id: &str, entry_id: &str, opens: i64) {
    sqlx::query(
        "INSERT INTO knowledge_use_stats \
             (user_id, entry_id, offered_count, opened_count, first_seen_at, last_offered_at, \
              recent_uses) \
         SELECT $1, $2, $3, $3, NOW() - make_interval(hours => 6), NOW() - make_interval(mins => 5), \
                array_agg(NOW() - make_interval(mins => 5 * g)) \
         FROM generate_series(1, $3::int) AS g",
    )
    .bind(user_id)
    .bind(entry_id)
    .bind(opens)
    .execute(pool)
    .await
    .expect("seed knowledge_use_stats row");
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

/// Seed an active KB row carrying a single scope dimension.
async fn seed_kb_scoped(
    pool: &PgPool,
    user_id: &str,
    id: &str,
    content: &str,
    dim: &str,
    value: &str,
) {
    let metadata = serde_json::json!({"scope": {dim: value}});
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, metadata) VALUES ($1, $2, $3, $4)",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(metadata)
    .execute(pool)
    .await
    .expect("seed scoped knowledge_base row");
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

async fn kb_disposition(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT disposition FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read disposition")
}

async fn kb_disposition_reason(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT disposition_reason FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read disposition_reason")
}

async fn kb_superseded_by(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT superseded_by FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read superseded_by")
}

/// Dispositioned-row counts per `disposition`, as an auditor would ask for
/// them: `(disposition, count)` sorted by disposition. Every one of these
/// rows is normally still live (`deleted_at IS NULL`) - disposition is
/// decoupled from deletion.
async fn dispositioned_rows(pool: &PgPool, user_id: &str) -> Vec<(String, i64)> {
    sqlx::query_as(
        "SELECT disposition, COUNT(*) FROM knowledge_base \
         WHERE user_id = $1 AND disposition <> 'active' \
         GROUP BY disposition ORDER BY disposition",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .expect("group dispositioned rows")
}

/// The id of the one row carrying exactly `content`, for a caller that does
/// not know a `merge_new` row's deterministic id in advance.
async fn kb_id_with_content(pool: &PgPool, user_id: &str, content: &str) -> Option<String> {
    sqlx::query_scalar(
        "SELECT id FROM knowledge_base WHERE user_id = $1 AND content = $2 AND deleted_at IS NULL",
    )
    .bind(user_id)
    .bind(content)
    .fetch_optional(pool)
    .await
    .expect("look up kb row by content")
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
// apply_ops — merge_new writes a new row and preserves its members, and
// tenant isolation.
// ---------------------------------------------------------------------------

/// Acceptance (#893): a merge writes a NEW row for the unified content and
/// dispositions every member `redundant` with a link back. No member is
/// rewritten or removed - this is the shape that dissolves the settled-merge
/// deadlock and stops a merge from dropping a member's own provenance.
#[tokio::test]
async fn a_merge_writes_a_new_row_and_dispositions_members_redundant_with_links() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-aaa", "alpha fact").await;
    seed_kb(pool, "u1", "kb-bbb", "beta fact").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge_new","ids":["kb-aaa","kb-bbb"],"content":"UNIFIED","scope":null}]}"#,
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
    assert_eq!(
        stats.soft_deleted, 2,
        "both members dispositioned redundant"
    );

    // Central assertion: a NEW row carries the synthesized content. Neither
    // original member's own row was rewritten to hold it.
    let new_id = kb_id_with_content(pool, "u1", "UNIFIED")
        .await
        .expect("the merge wrote a new row for the unified content");
    assert_ne!(new_id, "kb-aaa");
    assert_ne!(new_id, "kb-bbb");
    assert_eq!(
        kb_source(pool, &new_id).await.as_deref(),
        Some("consolidation")
    );
    assert!(!kb_is_deleted(pool, &new_id).await, "the new row is live");
    assert_eq!(
        kb_disposition(pool, &new_id).await.as_deref(),
        Some("active")
    );

    // Both members stay live rows, dispositioned redundant, pointing at the
    // new row. Neither is soft-deleted: disposition and deletion are
    // orthogonal.
    for member in ["kb-aaa", "kb-bbb"] {
        assert!(
            !kb_is_deleted(pool, member).await,
            "{member} stays live - a merge member is never deleted"
        );
        assert_eq!(
            kb_disposition(pool, member).await.as_deref(),
            Some("redundant"),
            "{member} is dispositioned redundant, not superseded (it duplicates, it is not \
             replaced)"
        );
        assert_eq!(
            kb_superseded_by(pool, member).await.as_deref(),
            Some(new_id.as_str()),
            "{member} names the new row that absorbed it"
        );
        assert_eq!(
            kb_disposition_reason(pool, member).await,
            None,
            "a merge member has no stated reason; superseded_by is the reason"
        );
        assert_eq!(
            kb_content(pool, member).await.as_deref().unwrap(),
            if member == "kb-aaa" {
                "alpha fact"
            } else {
                "beta fact"
            },
            "{member}'s own content is never rewritten - the synthesis lives on the new row"
        );
    }

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
        r#"{"operations":[{"op":"merge_new","ids":["u1-a","u1-b"],"content":"MERGED","scope":null}]}"#,
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
    assert_eq!(
        kb_disposition(pool, "u1-b").await.as_deref(),
        Some("redundant"),
        "user1's merge member was dispositioned"
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
async fn disposition_cap_is_enforced_at_the_documented_fraction() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    let ids: Vec<String> = (1..=10).map(|i| format!("kb-{i}")).collect();
    for id in &ids {
        seed_kb(pool, "u1", id, "trivial fact").await;
    }

    // A plan that dispositions all 10. cap = ceil(10 * 0.1) = 1, so nine are
    // deferred to a later run. Each id needs its own op: `disposition` names
    // exactly one entry.
    let ops: Vec<String> = ids
        .iter()
        .map(|id| {
            format!(r#"{{"op":"disposition","id":"{id}","as":"trivial","reason":"trivial"}}"#)
        })
        .collect();
    let plan = format!(r#"{{"operations":[{}]}}"#, ops.join(","));
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
        "disposition plan clamped to ceil(10 * the configured prune fraction)"
    );
    assert_eq!(kb_count_active(pool, "u1").await, 10, "no row is deleted");
    let trivial_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND disposition = 'trivial'",
    )
    .bind("u1")
    .fetch_one(pool)
    .await
    .expect("count trivial rows");
    assert_eq!(trivial_count, 1);

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// Deliberately-entered entries are not the model's to disposition freely.
// ---------------------------------------------------------------------------

/// Acceptance (#893): an explicit entry refuses `trivial` and `redundant` -
/// the two dispositions the never-prune rule translates into - but accepts
/// `refuted`, because a user-entered fact can still be corrected by the
/// user's own later statement.
#[tokio::test]
async fn an_explicit_entry_refuses_trivial_and_redundant_but_accepts_refuted() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
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
    // The redundant disposition needs a target to name.
    seed_kb(pool, "u1", "kb-canonical", "the entry kb-a would duplicate").await;

    let llm = llm_returning(
        r#"{"operations":[
            {"op":"disposition","id":"kb-a","as":"trivial","reason":"trivial"},
            {"op":"disposition","id":"kb-b","as":"trivial","reason":"trivial"}
        ]}"#,
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

    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("active"),
        "a source = 'explicit' entry must survive a proposed trivial disposition"
    );
    assert_eq!(
        stats.protected_from_delete, 1,
        "the refusal is reported in the run's stats"
    );
    // The protected id does not consume the disposition budget, so the
    // unprotected entry is still dispositioned in the same run.
    assert_eq!(
        kb_disposition(pool, "kb-b").await.as_deref(),
        Some("trivial"),
        "the unprotected entry is still dispositioned"
    );

    // redundant is refused too.
    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-a","as":"redundant","reason":"dup","superseded_by":"kb-canonical"}]}"#,
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
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("active"),
        "an explicit entry must survive a proposed redundant disposition too"
    );

    // refuted is accepted: a user-entered fact can be corrected by the
    // user's own later statement.
    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-a","as":"refuted","reason":"the user later said otherwise"}]}"#,
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
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("refuted"),
        "an explicit entry may be refuted"
    );
    assert!(
        !kb_is_deleted(pool, "kb-a").await,
        "a disposition is never a deletion"
    );

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
async fn merge_new_with_an_explicit_member_stamps_the_new_row_explicit() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_sourced(pool, "u1", "kb-a", "user-entered fact", SOURCE_EXPLICIT).await;
    seed_kb_sourced(pool, "u1", "kb-b", "extracted restatement", "extraction").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
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

    let new_id = kb_id_with_content(pool, "u1", "UNIFIED")
        .await
        .expect("the merge wrote a new row");
    assert_eq!(
        kb_source(pool, &new_id).await.as_deref(),
        Some(SOURCE_EXPLICIT),
        "a merge that absorbs an explicit member stamps the new row explicit, so the \
         protection is not laundered away by writing a fresh row"
    );
    // Neither original member's own row is rewritten to hold the protection.
    assert_eq!(
        kb_source(pool, "kb-a").await.as_deref(),
        Some(SOURCE_EXPLICIT)
    );
    assert_eq!(kb_source(pool, "kb-b").await.as_deref(), Some("extraction"));

    fx.cleanup().await;
}

#[tokio::test]
async fn merge_new_with_no_explicit_member_stamps_the_new_row_consolidation() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_sourced(pool, "u1", "kb-a", "extracted restatement", "extraction").await;
    seed_kb_sourced(pool, "u1", "kb-b", "another restatement", "extraction").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
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

    let new_id = kb_id_with_content(pool, "u1", "UNIFIED")
        .await
        .expect("the merge wrote a new row");
    assert_eq!(
        kb_source(pool, &new_id).await.as_deref(),
        Some("consolidation")
    );

    fx.cleanup().await;
}

/// Acceptance (#893): consolidation can disposition an entry without
/// deleting it. `deleted_at` never moves; the row stays live and findable,
/// marked with what it is and why.
#[tokio::test]
async fn consolidation_can_disposition_an_entry_without_deleting_it() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "circumstantial detail").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-b","as":"trivial","reason":"mattered only in the moment"}]}"#,
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

    assert!(
        !kb_is_deleted(pool, "kb-b").await,
        "a dispositioned entry is never soft-deleted - disposition and deletion are orthogonal"
    );
    assert!(kb_exists(pool, "kb-b").await);
    assert_eq!(
        kb_disposition(pool, "kb-b").await.as_deref(),
        Some("trivial")
    );
    assert_eq!(
        kb_disposition_reason(pool, "kb-b").await.as_deref(),
        Some("mattered only in the moment"),
        "the model's stated reason is persisted, not discarded"
    );
    assert_eq!(
        kb_superseded_by(pool, "kb-b").await,
        None,
        "nothing supersedes a standalone disposition"
    );

    fx.cleanup().await;
}

/// A merge's and a standalone disposition's rows are separable by SQL alone,
/// and neither one is a tombstone.
#[tokio::test]
async fn merge_and_disposition_outcomes_are_distinguishable_by_sql_and_stay_live() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    for id in ["kb-a", "kb-b", "kb-c"] {
        seed_kb(pool, "u1", id, "a fact").await;
    }

    let llm = llm_returning(
        r#"{"operations":[
            {"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null},
            {"op":"disposition","id":"kb-c","as":"trivial","reason":"trivial"}
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

    assert_eq!(
        dispositioned_rows(pool, "u1").await,
        vec![("redundant".to_string(), 2), ("trivial".to_string(), 1)],
        "a merge's members and a standalone disposition are separable without reading logs"
    );
    for id in ["kb-a", "kb-b", "kb-c"] {
        assert!(!kb_is_deleted(pool, id).await, "{id} stays live");
    }

    fx.cleanup().await;
}

#[tokio::test]
async fn disposition_with_no_stated_reason_records_null_not_an_empty_string() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "trivia").await;

    // `reason` omitted entirely: unstated.
    let llm = llm_returning(r#"{"operations":[{"op":"disposition","id":"kb-b","as":"trivial"}]}"#);
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
        kb_disposition(pool, "kb-b").await.as_deref(),
        Some("trivial"),
        "an unstated reason still records the disposition"
    );
    assert_eq!(
        kb_disposition_reason(pool, "kb-b").await,
        None,
        "an absent reason is NULL, not an empty string"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn disposition_reason_from_the_model_is_bounded_before_storage() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "keeper").await;
    seed_kb(pool, "u1", "kb-b", "trivia").await;

    // Multi-byte so a byte-slicing truncation would panic on a char boundary.
    let over_long: String = "é".repeat(MAX_DELETE_REASON_CHARS * 2);
    let llm = llm_returning(&format!(
        r#"{{"operations":[{{"op":"disposition","id":"kb-b","as":"trivial","reason":"{over_long}"}}]}}"#
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

    let stored = kb_disposition_reason(pool, "kb-b")
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
async fn disposition_provenance_is_never_written_to_another_tenants_rows() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-u1", "user1 trivia").await;
    seed_kb(pool, "u2", "kb-u2", "user2 fact").await;

    // The plan names a user1 id. user2's pass sees the same response but the id
    // is not in its partition, so nothing of user2's may be stamped.
    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-u1","as":"trivial","reason":"trivial"}]}"#,
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
        kb_disposition(pool, "kb-u1").await.as_deref(),
        Some("trivial")
    );
    assert!(!kb_is_deleted(pool, "kb-u2").await, "user2's row is active");
    assert_eq!(
        kb_disposition(pool, "kb-u2").await.as_deref(),
        Some("active"),
        "user2's row carries no disposition from user1's run"
    );
    assert_eq!(kb_superseded_by(pool, "kb-u2").await, None);
    assert_eq!(kb_disposition_reason(pool, "kb-u2").await, None);

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

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-gone","as":"trivial","reason":"x"}]}"#,
    );
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
        kb_disposition(pool, "kb-gone").await.as_deref(),
        Some("active"),
        "an existing tombstone's disposition is not restamped by a no-op run"
    );

    fx.cleanup().await;
}

/// Acceptance (#1127): the daily pass reads the use log and examines a
/// recently retrieved entry before one that was only recently written.
///
/// Driven through the real entry point, so it covers the whole path the unit
/// tests cannot: the `knowledge_use_stats` read, its `user_id` predicate, the
/// stitch into a domain record, the slice ordering, and the prompt the model is
/// actually shown.
///
/// **Each entry is given a body big enough to fill a slice**, because ordering
/// moves slices and not entries. A store that fits in one prompt is shown to the
/// model in one call and has no order to get wrong, so a two-entry fixture would
/// assert nothing. The never-retrieved entry is seeded first and sorts first by
/// tag, so a pass that did not read the log would prompt it first.
#[tokio::test]
async fn the_daily_pass_prompts_a_recently_retrieved_entry_before_a_recently_written_one() {
    let Some(fx) = support::DbFixture::try_new("dream1127").await else {
        return;
    };
    let pool = &fx.pool;

    // Comfortably over half the holistic prompt budget, so no two of these
    // share a slice.
    let filler = "the same sentence again and again. ".repeat(700);

    seed_kb_tagged(
        pool,
        "u1",
        "kb-written",
        &format!("a fact nobody has reached for. {filler}"),
        &["a-topic"],
    )
    .await;
    seed_kb_tagged(
        pool,
        "u1",
        "kb-retrieved",
        &format!("a fact the work keeps needing. {filler}"),
        &["b-topic"],
    )
    .await;
    seed_recent_opens(pool, "u1", "kb-retrieved", 3).await;
    // Another tenant's history, on an entry of its own. It must not reach u1's
    // pass, and u1's must not reach this one.
    seed_kb_tagged(
        pool,
        "u2",
        "kb-other",
        "another tenant's fact",
        &["a-topic"],
    )
    .await;
    seed_recent_opens(pool, "u2", "kb-other", 9).await;

    let (llm, prompts) = llm_capturing_prompts(r#"{"operations":[]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    // Taken out of the guard before the `await` below, so no lock is held
    // across one.
    let seen: Vec<String> = {
        let captured = prompts.lock().expect("the capture buffer is not poisoned");
        captured.clone()
    };

    let retrieved_at = seen
        .iter()
        .position(|p| p.contains("## kb-retrieved"))
        .expect("user1's retrieved entry was prompted");
    let written_at = seen
        .iter()
        .position(|p| p.contains("## kb-written"))
        .expect("user1's written entry was prompted");
    assert_ne!(
        retrieved_at, written_at,
        "precondition: the two entries are big enough to be sliced apart"
    );
    assert!(
        retrieved_at < written_at,
        "the slice holding the entry the log says was opened must be examined first; the \
         prompts arrived in the order {:?}",
        seen.iter()
            .map(|p| p.lines().find(|l| l.starts_with("## ")).unwrap_or(""))
            .collect::<Vec<_>>()
    );

    let for_u1 = &seen[retrieved_at];
    assert!(
        !for_u1.contains("kb-other"),
        "another tenant's entry must never reach this prompt"
    );
    let for_u2 = seen
        .iter()
        .find(|p| p.contains("kb-other"))
        .expect("user2's slice was prompted");
    assert!(
        !for_u2.contains("kb-retrieved") && !for_u2.contains("kb-written"),
        "and user1's entries must never reach user2's"
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

    // The disposition targets a user1 id. Under user2's scope this id is not a
    // loaded (valid) entry, so it is ignored — each `consolidate_user` pass
    // runs inside its own `with_user_id` scope and only sees/touches its own
    // partition.
    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"u1-x","as":"trivial","reason":"x"}]}"#,
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

    // Central assertion: user1's own pass loaded its entry (via the user-scoped
    // load) and applied the disposition. Scoping the load to a wrong user makes
    // this fail (nothing loaded ⇒ nothing dispositioned).
    assert_eq!(
        kb_disposition(pool, "u1-x").await.as_deref(),
        Some("trivial"),
        "user1's entry dispositioned under its own scope"
    );
    // user2 is untouched — neither user1's op nor the cross-user disposition id
    // reached it.
    assert_eq!(
        kb_disposition(pool, "u2-keep").await.as_deref(),
        Some("active"),
        "user2's entry must not be dispositioned"
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

/// Acceptance (#893): a merge touching a settled member is no longer refused.
/// `merge_new` writes a new row and leaves every member's own prose
/// untouched, so the reason the old in-place merge had to refuse a settled
/// member - it would have rewritten that member's content - no longer
/// applies. Only `edit` still refuses a settled entry.
#[tokio::test]
async fn a_merge_touching_a_settled_member_is_no_longer_refused() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-a", "settled prose", MAX_REVIEW_GENERATION).await;
    seed_kb(pool, "u1", "kb-b", "a near-duplicate").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null}]}"#,
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

    assert_eq!(
        stats.merged_clusters, 1,
        "the merge is applied, not dropped"
    );
    assert_eq!(
        stats.settled_unchanged, 0,
        "nothing about this merge was refused"
    );

    let new_id = kb_id_with_content(pool, "u1", "UNIFIED")
        .await
        .expect("the merge wrote a new row");
    assert_ne!(new_id, "kb-a", "the settled member's own row is not reused");

    // Neither member's own prose was rewritten - the settled member's least
    // of all.
    assert_eq!(
        kb_content(pool, "kb-a").await.as_deref(),
        Some("settled prose"),
        "the settled member's own content is untouched"
    );
    assert_eq!(
        kb_content(pool, "kb-b").await.as_deref(),
        Some("a near-duplicate")
    );
    for member in ["kb-a", "kb-b"] {
        assert!(!kb_is_deleted(pool, member).await, "{member} stays live");
        assert_eq!(
            kb_disposition(pool, member).await.as_deref(),
            Some("redundant")
        );
    }

    fx.cleanup().await;
}

#[tokio::test]
async fn settled_entry_can_still_be_dispositioned() {
    let Some(fx) = support::DbFixture::try_new("dream695").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-a", "settled prose", MAX_REVIEW_GENERATION).await;
    seed_kb(pool, "u1", "kb-b", "keeper").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-a","as":"trivial","reason":"wrong"}]}"#,
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

    // The cap settles an entry's *prose*, not the store: consolidation's own
    // output must stay dispositionable, or the interpretation layer ossifies.
    assert!(
        !kb_is_deleted(pool, "kb-a").await,
        "a disposition is never a deletion"
    );
    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("trivial")
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// The scope guard, the disposition budget's clustering order, the applier's
// SQL backstop, and replay idempotency (#893).
// ---------------------------------------------------------------------------

/// Acceptance (#893): a refuted/superseded/redundant disposition naming a
/// target is refused when the two entries' scopes are both set and share
/// nothing - two facts about different scopes cannot contradict each other -
/// and the refusal is counted.
#[tokio::test]
async fn disjoint_scopes_refuse_a_contradiction_disposition_and_count_it() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_scoped(pool, "u1", "kb-a", "true for alpha", "project", "alpha").await;
    seed_kb_scoped(pool, "u1", "kb-b", "true for beta", "project", "beta").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-a","as":"superseded","reason":"newer","superseded_by":"kb-b"}]}"#,
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

    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("active"),
        "disjoint-scoped entries cannot contradict each other, so the disposition is refused"
    );
    assert_eq!(
        stats.scope_guard_refusals, 1,
        "the refusal is reported in the run's stats"
    );

    fx.cleanup().await;
}

/// A disposition naming a target whose scope agrees on a shared dimension is
/// not refused - only a genuinely disjoint pair is.
#[tokio::test]
async fn a_shared_scope_dimension_does_not_trip_the_scope_guard() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_scoped(pool, "u1", "kb-a", "old wording", "project", "adelie-ai").await;
    seed_kb_scoped(pool, "u1", "kb-b", "new wording", "project", "adelie-ai").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-a","as":"superseded","reason":"newer","superseded_by":"kb-b"}]}"#,
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

    assert_eq!(stats.scope_guard_refusals, 0);
    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("superseded")
    );

    fx.cleanup().await;
}

/// Acceptance (#893): dispositioned entries are excluded from the
/// consolidation prompt. Once judged, an entry is not shown again - re-
/// judging it nightly is spend with no product.
#[tokio::test]
async fn dispositioned_entries_are_excluded_from_the_consolidation_prompt() {
    let Some(fx) = support::DbFixture::try_new("dream893").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-active", "still under review").await;
    seed_kb(pool, "u1", "kb-judged", "already decided").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"disposition","id":"kb-judged","as":"trivial","reason":"x"}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("first scan dispositions kb-judged");

    let (llm, prompts) = llm_capturing_prompts(r#"{"operations":[]}"#);
    run_consolidation_scan(
        pool,
        &llm,
        KnowledgeDeletePolicy::default(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("second scan succeeds");

    let seen: Vec<String> = {
        let captured = prompts.lock().expect("the capture buffer is not poisoned");
        captured.clone()
    };
    assert!(
        seen.iter().any(|p| p.contains("## kb-active")),
        "the still-active entry must still be prompted: {seen:?}"
    );
    assert!(
        !seen.iter().any(|p| p.contains("## kb-judged")),
        "a dispositioned entry must never reach a later prompt: {seen:?}"
    );

    fx.cleanup().await;
}

/// Acceptance (#712 item 3): the disposition budget is computed AFTER
/// clustering. An id a merge already absorbs is excluded from the count
/// before the cap is applied, so it does not spend the budget a genuinely
/// standalone disposition needs.
#[tokio::test]
async fn the_disposition_budget_is_computed_after_clustering() {
    let Some(fx) = support::DbFixture::try_new("dream712").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "will be merged").await;
    seed_kb(pool, "u1", "kb-b", "will be merged too").await;
    seed_kb(pool, "u1", "kb-c", "a genuine standalone disposition").await;
    for i in 0..7 {
        seed_kb(pool, "u1", &format!("kb-filler-{i}"), "filler").await;
    }
    // 10 active entries; cap = ceil(10 * 0.1) = 1.

    let llm = llm_returning(
        r#"{"operations":[
            {"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null},
            {"op":"disposition","id":"kb-a","as":"trivial","reason":"redundant with the merge"},
            {"op":"disposition","id":"kb-c","as":"trivial","reason":"trivial"}
        ]}"#,
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

    assert_eq!(
        stats.prunes_over_cap, 0,
        "kb-a's disposition is subsumed by the merge and must not count against the cap"
    );
    assert_eq!(
        kb_disposition(pool, "kb-c").await.as_deref(),
        Some("trivial"),
        "the one budget slot went to the genuinely standalone disposition"
    );

    fx.cleanup().await;
}

/// Acceptance (#712 item 3): a plan whose dispositions are ALL subsumed by
/// merges leaves the full budget for anything else in the same run.
#[tokio::test]
async fn a_plan_whose_dispositions_are_all_subsumed_by_merges_leaves_the_full_budget() {
    let Some(fx) = support::DbFixture::try_new("dream712").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "will be merged").await;
    seed_kb(pool, "u1", "kb-b", "will be merged too").await;
    seed_kb(pool, "u1", "kb-d", "a genuine standalone disposition").await;
    for i in 0..7 {
        seed_kb(pool, "u1", &format!("kb-filler-{i}"), "filler").await;
    }
    // 10 active entries; cap = ceil(10 * 0.1) = 1.

    let llm = llm_returning(
        r#"{"operations":[
            {"op":"merge_new","ids":["kb-a","kb-b"],"content":"UNIFIED","scope":null},
            {"op":"disposition","id":"kb-a","as":"trivial","reason":"redundant with the merge"},
            {"op":"disposition","id":"kb-b","as":"trivial","reason":"redundant with the merge"},
            {"op":"disposition","id":"kb-d","as":"trivial","reason":"trivial"}
        ]}"#,
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

    assert_eq!(
        stats.prunes_over_cap, 0,
        "both subsumed dispositions are excluded before the cap is computed, so the cap of 1 \
         is spent only by kb-d and nothing overflows"
    );
    assert_eq!(
        kb_disposition(pool, "kb-d").await.as_deref(),
        Some("trivial"),
        "the full budget was still available for the one standalone disposition, even though \
         two other dispositions were proposed in the same run"
    );

    fx.cleanup().await;
}

/// Acceptance (#712 item 2): a SQL backstop predicate firing increments a
/// counter and warns naming the row. Driven directly through the exported
/// `apply_ops`/`OpBuffer`, bypassing `consolidation.rs`'s own pre-filter -
/// through the public entry point the two always agree, so this is the only
/// way to prove the applier's own guard actually holds rather than merely
/// riding along behind one that already agrees with it.
#[tokio::test]
async fn a_sql_backstop_firing_increments_a_counter_and_warns_naming_the_row() {
    let Some(fx) = support::DbFixture::try_new("dream712").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_sourced(pool, "u1", "kb-a", "user-entered fact", SOURCE_EXPLICIT).await;

    let mut buffer = OpBuffer::new();
    buffer.absorb(ProposedOp::Disposition {
        id: "kb-a".to_string(),
        disposition: Disposition::Trivial,
        reason: Some("the guard above should have refused this".to_string()),
        superseded_by: None,
    });

    let policy = KnowledgeDeletePolicy::default();
    let (stats, logged) = support::capture_tracing_at(tracing::Level::WARN, || {
        with_user_id(UserId::new("u1".to_string()), async {
            apply_ops(pool, &buffer, &[], policy)
                .await
                .expect("apply_ops succeeds even when it refuses a write")
        })
    })
    .await;

    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("active"),
        "the explicit entry must survive even when the op reached the applier directly"
    );
    assert_eq!(
        stats.backstop_firings, 1,
        "the SQL predicate refused the write and the refusal is counted"
    );
    assert!(
        logged.contains("kb-a"),
        "the warning must name the row the backstop refused: {logged}"
    );
    // Captured at `Level::WARN`: the subscriber's own max-level filter drops
    // anything below it, so a message reaching this buffer at all is proof
    // it was logged at WARN or above - the level name itself is stripped
    // from the formatted line (`with_level(false)`), so it cannot be found
    // by substring.
    assert!(
        !logged.is_empty(),
        "a guard hole needs an operator-visible level: nothing reached the WARN-filtered buffer"
    );

    fx.cleanup().await;
}

/// The settled-entry guard has the same two layers as the explicit-entry
/// guard: consolidation.rs's pre-filter excludes a settled entry's edit
/// before it is ever proposed, and this SQL predicate backs it up. Through
/// the public entry point the two always agree, so - as with the explicit
/// backstop above - the only way to prove this SQL predicate holds on its
/// own is to call `apply_ops` directly with an edit the guard above would
/// have refused.
#[tokio::test]
async fn a_settled_entrys_edit_is_refused_by_the_sql_backstop_when_called_directly() {
    let Some(fx) = support::DbFixture::try_new("dream712").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_at_generation(pool, "u1", "kb-a", "settled prose", MAX_REVIEW_GENERATION).await;

    let mut buffer = OpBuffer::new();
    buffer.absorb(ProposedOp::Update {
        id: "kb-a".to_string(),
        new_content: "the guard above should have refused this".to_string(),
    });

    let policy = KnowledgeDeletePolicy::default();
    let (stats, logged) = support::capture_tracing_at(tracing::Level::WARN, || {
        with_user_id(UserId::new("u1".to_string()), async {
            apply_ops(pool, &buffer, &[], policy)
                .await
                .expect("apply_ops succeeds even when it refuses a write")
        })
    })
    .await;

    assert_eq!(
        kb_content(pool, "kb-a").await.as_deref(),
        Some("settled prose"),
        "a settled entry's prose must survive even when the op reached the applier directly"
    );
    assert_eq!(
        stats.backstop_firings, 1,
        "the SQL predicate refused the write and the refusal is counted"
    );
    assert!(
        logged.contains("kb-a"),
        "the warning must name the row the backstop refused: {logged}"
    );

    fx.cleanup().await;
}

/// Acceptance (#893, design 5.6/8.4): replaying an applied batch is a no-op.
/// `merge_new`'s deterministic id makes a retried apply upsert rather than
/// duplicate.
#[tokio::test]
async fn replaying_an_applied_batch_is_a_no_op() {
    let Some(fx) = support::DbFixture::try_new("dream712").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "alpha fact").await;
    seed_kb(pool, "u1", "kb-b", "beta fact").await;

    let merge = SynthesizedMerge {
        member_ids: vec!["kb-a".to_string(), "kb-b".to_string()],
        new_content: "UNIFIED".to_string(),
        new_scope: None,
    };
    let policy = KnowledgeDeletePolicy::default();

    with_user_id(UserId::new("u1".to_string()), async {
        apply_ops(pool, &OpBuffer::new(), std::slice::from_ref(&merge), policy)
            .await
            .expect("first apply succeeds")
    })
    .await;

    let new_id = kb_id_with_content(pool, "u1", "UNIFIED")
        .await
        .expect("the first apply wrote a new row");
    let count_after_first = kb_count_for_user(pool, "u1").await;

    // Replay the identical batch, as a retried apply after a crash would.
    with_user_id(UserId::new("u1".to_string()), async {
        apply_ops(pool, &OpBuffer::new(), std::slice::from_ref(&merge), policy)
            .await
            .expect("replayed apply succeeds")
    })
    .await;

    assert_eq!(
        kb_count_for_user(pool, "u1").await,
        count_after_first,
        "no row was added by the replay"
    );
    assert_eq!(
        kb_id_with_content(pool, "u1", "UNIFIED").await.as_deref(),
        Some(new_id.as_str()),
        "the replay upserted the same deterministic id rather than writing a second row"
    );
    assert_eq!(
        kb_disposition(pool, "kb-a").await.as_deref(),
        Some("redundant")
    );
    assert_eq!(
        kb_superseded_by(pool, "kb-a").await.as_deref(),
        Some(new_id.as_str())
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

// ---------------------------------------------------------------------------
// The level contract for the dreaming worker.
//
// > INFO carries ids, counts, durations, model names and token counts. Never
// > content.
// > DEBUG carries prompts, the full assembled context, and tool arguments.
//
// An extracted fact is, by construction, personal: the extraction prompt asks
// for preferences and personal context. The worker used to write the head of
// each one on an INFO line, which every shipped deployment turns on, so a
// central log store accumulated one tenant's personal facts where a reader
// with pod-log access could see them - a much wider audience than the
// per-user row-level boundary the rest of the module maintains.
// ---------------------------------------------------------------------------

/// A fact body no other test emits.
const FACT_SENTINEL: &str = "SENTINEL-THE-USER-HAS-A-PEANUT-ALLERGY";

#[tokio::test]
async fn no_extracted_fact_content_at_info() {
    let Some(fx) = support::DbFixture::try_new("dream1005").await else {
        return;
    };
    let pool = &fx.pool;

    seed_conversation(pool, "u1", "conv-1").await;
    seed_message(pool, "u1", "conv-1", "m1", 1, "user", "remember this").await;

    let llm = llm_returning(&format!(
        r#"{{"facts":[{{"content":"{FACT_SENTINEL}","tags":[],"scope":null}}]}}"#
    ));
    let embed = unused_embed_fn();
    let token = CancellationToken::new();

    let (written, logs) = support::capture_tracing_at(tracing::Level::INFO, || async {
        run_dreaming_scan(pool, &llm, &embed, "test-model", 0, &token, None)
            .await
            .expect("the dreaming scan succeeds")
    })
    .await;

    assert_eq!(written, 1, "the scan wrote the fact under test");
    assert!(
        !logs.contains(FACT_SENTINEL),
        "an extracted fact is personal content and must not reach an INFO line\n\
         --- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains("dreaming: wrote fact"),
        "the INFO line itself must survive - the fix is to drop the content, not the line\n\
         --- captured at INFO ---\n{logs}"
    );

    fx.cleanup().await;
}
