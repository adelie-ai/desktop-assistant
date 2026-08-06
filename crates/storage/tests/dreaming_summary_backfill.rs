//! The dream cycle's summary-backfill pass (issue #1099).
//!
//! `knowledge_base.summary` is nullable and unenforced at the write boundary,
//! so two populations of rows carry none: every entry written before the column
//! existed, and any later write that named no summary. The `[Recall]` block
//! renders one line per candidate entry and that line is the summary, so an
//! entry without one is offered back as a cut-down prefix of its body.
//!
//! This suite pins what the pass owes the store:
//!
//! 1. It fills a missing summary, and leaves a current one alone.
//! 2. It rewrites a summary whose content changed after it was written. A
//!    summary that describes content that has since changed is worse than none,
//!    because it is confidently wrong.
//! 3. It never touches `content`. #694 is the standing concern that the store
//!    becomes model-rewritten prose rather than accumulated evidence, and a
//!    summarising pass that edits the body is exactly that failure.
//! 4. It is bounded: it takes at most a capped number of rows per cycle and
//!    leaves the rest for the next one.
//! 5. A row it cannot summarise keeps no summary, does not stop the other rows,
//!    and is picked up again on the next cycle.
//! 6. It writes only inside the owning user's partition. Row-level security is
//!    a non-FORCE backstop that the table owner bypasses, so the scoping has to
//!    be in the query.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test dreaming_summary_backfill
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::{Arc, Mutex};

use desktop_assistant_core::domain::SUMMARY_MAX_CHARS;
use desktop_assistant_storage::dreaming::{
    BackfillEmbedFn, DreamingLlmFn, MAX_SUMMARIES_PER_CYCLE, run_dreaming_scan,
};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/// The ids a summary prompt names, in the order it names them. The prompt lists
/// one `## <id>` heading per entry, the same shape consolidation uses.
fn ids_in_prompt(prompt: &str) -> Vec<String> {
    prompt
        .lines()
        .filter_map(|l| l.strip_prefix("## "))
        .map(|id| id.trim().to_string())
        .collect()
}

/// A dreaming LLM that answers a summary prompt by applying `line` to each id it
/// was shown. `line` returning `None` models the model omitting that entry.
///
/// Every prompt it is asked is recorded, so a test can assert how many calls a
/// batch of entries cost and what each one contained.
fn summarising_llm(
    prompts: Arc<Mutex<Vec<String>>>,
    line: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
) -> DreamingLlmFn {
    Box::new(move |_system, user| {
        let ids = ids_in_prompt(&user);
        prompts
            .lock()
            .expect("prompt log is not poisoned")
            .push(user);
        let entries: Vec<String> = ids
            .iter()
            .filter_map(|id| {
                line(id).map(|summary| {
                    format!(
                        r#"{{"id":{},"summary":{}}}"#,
                        serde_json::to_string(id).expect("an id serializes"),
                        serde_json::to_string(&summary).expect("a summary serializes"),
                    )
                })
            })
            .collect();
        let response = format!(r#"{{"summaries":[{}]}}"#, entries.join(","));
        Box::pin(async move { Ok(response) })
    })
}

/// The prompts a recording LLM was asked, copied out of the log so no guard is
/// held across the fixture teardown that follows.
fn recorded(prompts: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    prompts.lock().expect("prompt log is not poisoned").clone()
}

/// A dreaming LLM that answers every entry with `"summary of <id>"`.
fn llm_summarising_everything(prompts: Arc<Mutex<Vec<String>>>) -> DreamingLlmFn {
    summarising_llm(prompts, |id| Some(format!("summary of {id}")))
}

/// A dreaming LLM whose every call fails, modelling an unreachable or
/// unauthorized backend.
fn failing_llm() -> DreamingLlmFn {
    Box::new(move |_system, _user| Box::pin(async move { Err("backend refused".to_string()) }))
}

/// An embedder that must never be called: none of these tests seed a
/// conversation, so extraction never proposes a tag.
fn unused_embed_fn() -> BackfillEmbedFn {
    Box::new(|_texts| {
        Box::pin(async move { Err("embed_fn must not be called in this test".to_string()) })
    })
}

/// Run one dream cycle with archival off, which leaves extraction (no
/// conversations are seeded, so it is a no-op) and the summary pass.
async fn run_cycle(pool: &PgPool, llm: &DreamingLlmFn) {
    let embed = unused_embed_fn();
    run_dreaming_scan(
        pool,
        llm,
        &embed,
        "test-model",
        0,
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("the dream cycle succeeds");
}

// ---------------------------------------------------------------------------
// Seed and read helpers
// ---------------------------------------------------------------------------

async fn seed_kb(pool: &PgPool, user_id: &str, id: &str, content: &str) {
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content, tags) VALUES ($1, $2, $3, $4)")
        .bind(id)
        .bind(user_id)
        .bind(content)
        .bind(Vec::<String>::new())
        .execute(pool)
        .await
        .expect("seed knowledge_base row");
}

async fn seed_kb_tagged(pool: &PgPool, user_id: &str, id: &str, content: &str, tags: &[&str]) {
    let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content, tags) VALUES ($1, $2, $3, $4)")
        .bind(id)
        .bind(user_id)
        .bind(content)
        .bind(&tags)
        .execute(pool)
        .await
        .expect("seed tagged knowledge_base row");
}

/// Seed a row that already carries a summary written at the same moment as its
/// content, which is what a fresh, current summary looks like on disk.
async fn seed_kb_summarised(pool: &PgPool, user_id: &str, id: &str, content: &str, summary: &str) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, tags, summary, summary_updated_at) \
         VALUES ($1, $2, $3, $4, $5, NOW())",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(Vec::<String>::new())
    .bind(summary)
    .execute(pool)
    .await
    .expect("seed summarised knowledge_base row");
}

async fn seed_kb_soft_deleted(pool: &PgPool, user_id: &str, id: &str, content: &str) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, tags, deleted_at) \
         VALUES ($1, $2, $3, $4, NOW())",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(Vec::<String>::new())
    .execute(pool)
    .await
    .expect("seed soft-deleted knowledge_base row");
}

/// Seed `count` unsummarised rows in one statement, so the cap test can build a
/// backlog larger than one cycle may take without paying a round trip per row.
async fn seed_many(pool: &PgPool, user_id: &str, count: i32) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, tags) \
         SELECT 'kb-bulk-' || g, $1, 'A durable fact numbered ' || g, '{}'::text[] \
         FROM generate_series(1, $2) g",
    )
    .bind(user_id)
    .bind(count)
    .execute(pool)
    .await
    .expect("seed a backlog of unsummarised rows");
}

/// Rewrite a row's content the way a live write does: the body changes and
/// `updated_at` moves, while the stored summary is deliberately preserved.
async fn rewrite_content(pool: &PgPool, id: &str, content: &str) {
    sqlx::query("UPDATE knowledge_base SET content = $2, updated_at = NOW() WHERE id = $1")
        .bind(id)
        .bind(content)
        .execute(pool)
        .await
        .expect("rewrite content");
}

async fn kb_summary(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT summary FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read kb summary")
}

async fn kb_content(pool: &PgPool, id: &str) -> String {
    sqlx::query_scalar("SELECT content FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read kb content")
}

async fn kb_summarised_count(pool: &PgPool, user_id: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND summary IS NOT NULL",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await
    .expect("count summarised rows")
}

// ---------------------------------------------------------------------------
// The pass fills what is missing, and leaves what is current alone.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dream_summary_pass_fills_a_null_summary() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(
        pool,
        "u1",
        "kb-a",
        "The user keeps facet colons in tag names.",
    )
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts)).await;

    assert_eq!(
        kb_summary(pool, "kb-a").await.as_deref(),
        Some("summary of kb-a"),
        "an entry with no summary gets the line the model wrote"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_leaves_an_existing_summary_alone() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_summarised(
        pool,
        "u1",
        "kb-a",
        "The user keeps facet colons in tag names.",
        "Keeps facet colons in tag names",
    )
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts.clone())).await;

    assert_eq!(
        kb_summary(pool, "kb-a").await.as_deref(),
        Some("Keeps facet colons in tag names"),
        "a summary that still describes the content is not rewritten"
    );
    assert!(
        prompts
            .lock()
            .expect("prompt log is not poisoned")
            .is_empty(),
        "a store with nothing to summarise must not spend a model call"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_rewrites_a_summary_older_than_its_content() {
    // The write path preserves a stored summary when an update names none
    // (#1098), which is exactly how a summary comes to describe content that no
    // longer says that. A confidently wrong line is worse than no line.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_summarised(
        pool,
        "u1",
        "kb-a",
        "The user prefers a dark theme.",
        "Prefers dark themes",
    )
    .await;
    rewrite_content(pool, "kb-a", "The user prefers a light theme after all.").await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts)).await;

    assert_eq!(
        kb_summary(pool, "kb-a").await.as_deref(),
        Some("summary of kb-a"),
        "a summary older than the content it describes is written again"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// The body is evidence, not the pass's to edit.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dream_summary_pass_never_changes_content() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    // Whitespace, a newline and a multi-byte character, so a pass that
    // normalized or re-encoded the body would show up as a difference.
    let body = "  The user prefers a dark theme.\n  Café notes: keep the accent.  ";
    seed_kb(pool, "u1", "kb-a", body).await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts)).await;

    assert!(
        kb_summary(pool, "kb-a").await.is_some(),
        "premise: the pass did write a summary for this row"
    );
    assert_eq!(
        kb_content(pool, "kb-a").await.as_bytes(),
        body.as_bytes(),
        "the pass writes the summary and leaves the body byte-identical"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// Bounded per cycle.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dream_summary_pass_caps_the_rows_it_takes_per_cycle() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    let backlog = MAX_SUMMARIES_PER_CYCLE as i32 + 5;
    seed_many(pool, "u1", backlog).await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let llm = llm_summarising_everything(prompts);
    run_cycle(pool, &llm).await;

    assert_eq!(
        kb_summarised_count(pool, "u1").await,
        MAX_SUMMARIES_PER_CYCLE as i64,
        "one cycle takes at most the per-cycle cap, whatever the backlog"
    );

    // It is a backfill, not a deadline: what is left over is picked up next time.
    run_cycle(pool, &llm).await;
    assert_eq!(
        kb_summarised_count(pool, "u1").await,
        backlog as i64,
        "the rows left behind are summarised on the following cycle"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_asks_for_a_batch_of_entries_in_one_call() {
    // One call per row is the expensive way to spend a backfill of hundreds of
    // rows, so the pass must group them.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    for i in 0..8 {
        seed_kb(
            pool,
            "u1",
            &format!("kb-{i}"),
            "A durable fact worth keeping.",
        )
        .await;
    }

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts.clone())).await;

    let calls = recorded(&prompts);
    assert!(
        calls.len() < 8,
        "8 entries must not cost 8 model calls, made {}",
        calls.len()
    );
    assert!(
        calls.iter().any(|p| ids_in_prompt(p).len() > 1),
        "at least one call must carry several entries"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// A row that cannot be summarised does not take the cycle down with it.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dream_summary_pass_survives_one_row_failing() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-good-1", "A fact the model can summarise.").await;
    seed_kb(
        pool,
        "u1",
        "kb-bad",
        "A fact the model refuses to summarise.",
    )
    .await;
    seed_kb(
        pool,
        "u1",
        "kb-good-2",
        "Another fact the model can summarise.",
    )
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let llm = summarising_llm(prompts, |id| {
        (id != "kb-bad").then(|| format!("summary of {id}"))
    });
    run_cycle(pool, &llm).await;

    assert_eq!(
        kb_summary(pool, "kb-bad").await,
        None,
        "the row the model skipped keeps no summary"
    );
    assert_eq!(
        kb_summary(pool, "kb-good-1").await.as_deref(),
        Some("summary of kb-good-1"),
        "one failed row does not cost the rows around it"
    );
    assert_eq!(
        kb_summary(pool, "kb-good-2").await.as_deref(),
        Some("summary of kb-good-2")
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_retries_next_cycle_a_row_it_could_not_summarise() {
    // A row that stays NULL must stay in the worklist. Stamping it attempted
    // would put it permanently beyond the pass, and the entry would never get a
    // recall line.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "A fact worth a line.").await;

    run_cycle(pool, &failing_llm()).await;
    assert_eq!(
        kb_summary(pool, "kb-a").await,
        None,
        "premise: the failed cycle wrote nothing"
    );

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts)).await;
    assert_eq!(
        kb_summary(pool, "kb-a").await.as_deref(),
        Some("summary of kb-a"),
        "the row is offered to the model again on the next cycle"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_failure_does_not_fail_the_cycle() {
    // The pass runs beside extraction and archival. A summarising model that is
    // down must not take the whole dream cycle with it.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "A fact worth a line.").await;

    let embed = unused_embed_fn();
    let llm = failing_llm();
    run_dreaming_scan(
        pool,
        &llm,
        &embed,
        "test-model",
        0,
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("a failed summary pass is reported, not raised");

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// Scope, shape and bounds of what gets written.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dream_summary_pass_writes_only_to_the_owning_users_rows() {
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-u1", "A fact owned by the first user.").await;
    seed_kb_summarised(
        pool,
        "u2",
        "kb-u2",
        "A fact owned by the second user.",
        "The second user's own line",
    )
    .await;

    // The model answers for whatever id it is shown, including one it saw under
    // another user's pass. A write that forgot its `user_id` predicate would
    // stamp the other partition.
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let llm = summarising_llm(prompts, |_| Some("cross-tenant line".to_string()));
    run_cycle(pool, &llm).await;

    assert_eq!(
        kb_summary(pool, "kb-u1").await.as_deref(),
        Some("cross-tenant line"),
        "the first user's own row is summarised under its own scope"
    );
    assert_eq!(
        kb_summary(pool, "kb-u2").await.as_deref(),
        Some("The second user's own line"),
        "the second user's current summary is untouched by the first user's pass"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_skips_a_soft_deleted_entry() {
    // A tombstone is invisible to every read path, so summarising one is spend
    // with no reader.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_soft_deleted(pool, "u1", "kb-gone", "A retired fact.").await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts.clone())).await;

    assert_eq!(
        kb_summary(pool, "kb-gone").await,
        None,
        "a retired entry is not summarised"
    );
    assert!(
        prompts
            .lock()
            .expect("prompt log is not poisoned")
            .is_empty(),
        "a store whose only row is a tombstone spends no model call"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_cuts_an_over_long_summary_to_the_display_limit() {
    // Nothing bounds what the model returns, and the line is rendered into a
    // list row and a context block with a fixed budget.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "A fact worth a line.").await;

    // Multi-byte, so a byte-indexed cut would panic rather than truncate.
    let over_long = "é".repeat(SUMMARY_MAX_CHARS * 3);
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let llm = summarising_llm(prompts, move |_| Some(over_long.clone()));
    run_cycle(pool, &llm).await;

    let stored = kb_summary(pool, "kb-a")
        .await
        .expect("a summary was stored");
    assert_eq!(
        stored.chars().count(),
        SUMMARY_MAX_CHARS,
        "an unbounded model line is cut to the shared display limit"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_gives_the_model_the_entry_tags() {
    // A summary says what the entry states, and the tags say what kind of fact
    // it is. Without them the model has to guess the register from prose alone.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb_tagged(
        pool,
        "u1",
        "kb-a",
        "The user keeps facet colons in tag names.",
        &["preference", "project:adelie-ai"],
    )
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_cycle(pool, &llm_summarising_everything(prompts.clone())).await;

    let calls = recorded(&prompts);
    let asked = calls.first().expect("the pass asked the model once");
    assert!(
        asked.contains("preference") && asked.contains("project:adelie-ai"),
        "the prompt carries the entry's tags as context: {asked}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_ignores_a_blank_line_from_the_model() {
    // A blank answer is not a summary. Storing it would leave an empty string,
    // which reads as a blank row and takes the entry permanently out of a
    // `WHERE summary IS NULL` worklist.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "A fact worth a line.").await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let llm = summarising_llm(prompts, |_| Some("   \n  ".to_string()));
    run_cycle(pool, &llm).await;

    assert_eq!(
        kb_summary(pool, "kb-a").await,
        None,
        "a blank line leaves the row unsummarised rather than storing an empty string"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn dream_summary_pass_ignores_an_id_it_did_not_ask_about() {
    // The model answers with ids it was shown. One it was not shown is either a
    // hallucination or another partition's row.
    let Some(fx) = support::DbFixture::try_new("dream1099").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, "u1", "kb-a", "A fact worth a line.").await;
    seed_kb_summarised(
        pool,
        "u1",
        "kb-untouched",
        "A fact that already reads well.",
        "Already has a line",
    )
    .await;

    // Answers for the row it was shown, and for one it was not.
    let llm: DreamingLlmFn = Box::new(move |_system, _user| {
        Box::pin(async move {
            Ok(r#"{"summaries":[
                {"id":"kb-a","summary":"a real line"},
                {"id":"kb-untouched","summary":"an unasked-for line"}
            ]}"#
            .to_string())
        })
    });
    run_cycle(pool, &llm).await;

    assert_eq!(
        kb_summary(pool, "kb-a").await.as_deref(),
        Some("a real line")
    );
    assert_eq!(
        kb_summary(pool, "kb-untouched").await.as_deref(),
        Some("Already has a line"),
        "an answer for an entry the call did not name is ignored"
    );

    fx.cleanup().await;
}
