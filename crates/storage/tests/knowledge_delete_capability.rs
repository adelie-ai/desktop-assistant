//! Deletion of knowledge rows is a configured capability, not a compile-time
//! constant (issue #1122).
//!
//! Consolidation decides what to prune from prose alone, once per cycle, with
//! no signal about whether an entry was ever read. A deployment must be able to
//! run consolidation for its merges and decline its deletes, and must be able
//! to say that only a person may destroy a row. This suite pins both controls
//! and the one thing they must never do: block a person who asks to be
//! forgotten.
//!
//! Each test is named for the acceptance criterion it holds, so a failing run
//! names the unmet requirement.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_delete_capability
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::knowledge_delete::{DeleteInitiator, with_delete_initiator};
use desktop_assistant_storage::dreaming::{run_consolidation_scan, sweep_expired_trash};
use desktop_assistant_storage::knowledge_delete::{
    HardDeleteTarget, KnowledgeDeletePolicy, MAX_REFUSAL_IDS, hard_delete_knowledge,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, with_user_id};
use sqlx::PgPool;
use std::io;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::fmt::MakeWriter;

const ALICE: &str = "kb1122-alice";

// ---------------------------------------------------------------------------
// Fakes and helpers
// ---------------------------------------------------------------------------

/// A dreaming LLM that ignores its prompts and always returns `response`.
fn llm_returning(response: &str) -> desktop_assistant_storage::dreaming::DreamingLlmFn {
    let response = response.to_string();
    Box::new(move |_system, _user| {
        let response = response.clone();
        Box::pin(async move { Ok(response) })
    })
}

/// The shipped defaults, which is what an instance that sets nothing gets.
fn default_policy() -> KnowledgeDeletePolicy {
    KnowledgeDeletePolicy::default()
}

/// Defaults with the safety flag set: only a person may destroy a row.
fn person_only_policy() -> KnowledgeDeletePolicy {
    KnowledgeDeletePolicy {
        require_person_for_hard_delete: true,
        ..KnowledgeDeletePolicy::default()
    }
}

async fn seed_kb(pool: &PgPool, user_id: &str, id: &str, content: &str) {
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content) VALUES ($1, $2, $3)")
        .bind(id)
        .bind(user_id)
        .bind(content)
        .execute(pool)
        .await
        .expect("seed knowledge_base row");
}

/// Seed a soft-deleted row whose `deleted_at` is `days_ago` days in the past.
async fn seed_tombstone(pool: &PgPool, user_id: &str, id: &str, days_ago: i32) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, deleted_at) \
         VALUES ($1, $2, 'retired', NOW() - make_interval(days => $3))",
    )
    .bind(id)
    .bind(user_id)
    .bind(days_ago)
    .execute(pool)
    .await
    .expect("seed tombstone");
}

async fn kb_exists(pool: &PgPool, id: &str) -> bool {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("count kb row");
    count > 0
}

async fn kb_is_deleted(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT deleted_at IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_at")
}

async fn kb_content(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT content FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
        .expect("read content")
}

/// How many of `user_id`'s rows carry `disposition`. Consolidation never sets
/// `deleted_at` any more, so this is what "how many did the prune budget
/// retire" means now: a dispositioned row stays live.
async fn kb_count_with_disposition(pool: &PgPool, user_id: &str, disposition: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND disposition = $2",
    )
    .bind(user_id)
    .bind(disposition)
    .fetch_one(pool)
    .await
    .expect("count rows by disposition")
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

/// Collects everything the code under test writes to `tracing`, so a test can
/// assert that a refusal reached the log and not only the returned value.
#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log lock")).into_owned()
    }
}

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for CapturedLog {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A runtime for a test that captures log output.
///
/// It is single-threaded on purpose. A captured subscriber is installed per
/// thread, so a future that a multi-threaded runtime is free to poll on another
/// worker would write its records where the test cannot see them - and the test
/// would then pass or fail on how the scheduler felt.
fn capture_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a current-thread runtime")
}

/// Drive `fut` to completion with every `tracing` record captured, and return
/// what was written beside the future's own result.
fn run_capturing_logs<F: std::future::Future>(
    rt: &tokio::runtime::Runtime,
    fut: F,
) -> (F::Output, String) {
    let captured = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_ansi(false)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, || rt.block_on(fut));
    (out, captured.text())
}

/// A disposition plan naming every id `trivial`. `disposition` names exactly
/// one entry per op, unlike the old `delete`'s array of ids, so this builds
/// one op per id.
fn disposition_plan(ids: &[String]) -> String {
    let ops = ids
        .iter()
        .map(|id| {
            format!(r#"{{"op":"disposition","id":"{id}","as":"trivial","reason":"trivial"}}"#)
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"operations":[{ops}]}}"#)
}

/// An edit plan rewriting every id.
fn edit_plan(ids: &[String]) -> String {
    let ops = ids
        .iter()
        .map(|id| format!(r#"{{"op":"edit","id":"{id}","content":"REWRITTEN {id}"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"operations":[{ops}]}}"#)
}

// ---------------------------------------------------------------------------
// The prune fraction comes from configuration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prune_fraction_is_read_from_configuration() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    let ids: Vec<String> = (1..=10).map(|i| format!("prune-{i}")).collect();
    for id in &ids {
        seed_kb(pool, ALICE, id, "trivial fact").await;
    }

    // Half, not the shipped tenth. A run that still read the old constant
    // would disposition one row.
    let policy = KnowledgeDeletePolicy {
        prune_fraction: 0.5,
        ..KnowledgeDeletePolicy::default()
    };
    let llm = llm_returning(&disposition_plan(&ids));
    let stats = run_consolidation_scan(pool, &llm, policy, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(
        stats.dispositioned, 5,
        "the disposition cap must come from the configured fraction, not a constant"
    );
    assert_eq!(
        kb_count_with_disposition(pool, ALICE, "trivial").await,
        5,
        "exactly the capped share was dispositioned"
    );
    assert_eq!(
        kb_count_with_disposition(pool, ALICE, "active").await,
        5,
        "the rest stays active, not deleted - disposition never destroys a row"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn zero_prune_fraction_merges_and_edits_but_deletes_nothing() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    seed_kb(pool, ALICE, "zero-merge-a", "alpha").await;
    seed_kb(pool, ALICE, "zero-merge-b", "beta").await;
    seed_kb(pool, ALICE, "zero-edit", "verbose text").await;
    seed_kb(pool, ALICE, "zero-prune", "trivial").await;

    let plan = r#"{"operations":[
        {"op":"merge_new","ids":["zero-merge-a","zero-merge-b"],"content":"UNIFIED","scope":null},
        {"op":"edit","id":"zero-edit","content":"TIGHTENED"},
        {"op":"disposition","id":"zero-prune","as":"trivial","reason":"trivial"}
    ]}"#;
    // Only the prune (now disposition) share is under test here, so the
    // rewrite share is opened up to keep its own cap out of the result.
    let policy = KnowledgeDeletePolicy {
        prune_fraction: 0.0,
        rewrite_fraction: 1.0,
        ..KnowledgeDeletePolicy::default()
    };
    let llm = llm_returning(plan);
    let stats = run_consolidation_scan(pool, &llm, policy, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(stats.merged_clusters, 1, "merges still apply");
    assert_eq!(stats.updated, 1, "edits still apply");
    // The merge writes a NEW row for the unified content; neither original
    // member is rewritten in place any more.
    let merged_id = kb_id_with_content(pool, ALICE, "UNIFIED")
        .await
        .expect("the merge wrote a new row for the unified content");
    assert_ne!(merged_id, "zero-merge-a");
    assert_ne!(merged_id, "zero-merge-b");
    assert_eq!(
        kb_content(pool, "zero-edit").await.as_deref(),
        Some("TIGHTENED")
    );
    assert_eq!(
        kb_count_with_disposition(pool, ALICE, "trivial").await,
        0,
        "a zero disposition fraction must retire nothing on its own"
    );
    assert_eq!(
        stats.dispositions_over_cap, 1,
        "the one proposed disposition was deferred, not silently dropped"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// The safety flag: who asked for the delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_initiated_hard_delete_is_refused_while_the_safety_flag_is_set() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), person_only_policy());

    seed_kb(pool, ALICE, "model-kill-1", "a fact").await;
    seed_kb(pool, ALICE, "model-kill-2", "another fact").await;

    // `builtin_knowledge_base_delete` reaches this method with no person
    // scope installed, which is what the unset task-local stands for.
    let err = with_user_id(UserId::new(ALICE), async {
        store
            .delete_many(&["model-kill-1".to_string(), "model-kill-2".to_string()])
            .await
    })
    .await
    .expect_err("a machine-initiated hard delete must be refused");

    let text = err.to_string();
    assert!(
        text.contains("model-kill-1"),
        "the refusal must name what it declined to destroy: {text}"
    );
    assert!(kb_exists(pool, "model-kill-1").await, "row must survive");
    assert!(kb_exists(pool, "model-kill-2").await, "row must survive");
    assert!(
        !kb_is_deleted(pool, "model-kill-1").await,
        "a refusal must not answer with a tombstone either"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn user_initiated_delete_still_erases_while_the_safety_flag_is_set() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), person_only_policy());

    with_user_id(UserId::new(ALICE), async {
        store
            .write(KnowledgeEntry::new(
                "forget-me",
                "a fact the person wants gone",
                vec!["notes".into()],
            ))
            .await
            .expect("write");
    })
    .await;
    seed_tombstone(pool, ALICE, "forget-me-too", 1).await;

    // "Forget that" arrives as a command from a client control, so the person
    // scope is installed at the handler.
    with_user_id(UserId::new(ALICE), async {
        with_delete_initiator(DeleteInitiator::Person, async {
            store.delete("forget-me").await.expect("delete entry");
            let emptied = store.empty_trash().await.expect("empty trash");
            assert_eq!(emptied, 1, "empty trash frees the tombstone");
        })
        .await;
    })
    .await;

    assert!(
        !kb_exists(pool, "forget-me").await,
        "a right-to-be-forgotten request must erase the row, not tombstone it"
    );
    assert!(
        !kb_exists(pool, "forget-me-too").await,
        "emptying the trash must still free tombstones"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn refusals_are_logged_with_entry_id_and_calling_path() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    seed_tombstone(pool, ALICE, "refused-1", 90).await;
    seed_tombstone(pool, ALICE, "refused-2", 90).await;

    let outcome = hard_delete_knowledge(
        pool,
        ALICE,
        HardDeleteTarget::ExpiredTombstones,
        person_only_policy(),
        "test::refusal_path",
    )
    .await
    .expect("a refusal is a normal outcome, not an error");

    assert_eq!(outcome.removed, 0, "nothing may be destroyed");
    let refusal = outcome.refusal.expect("the outcome carries the refusal");
    assert_eq!(refusal.call_path, "test::refusal_path");
    assert_eq!(refusal.total, 2, "the volume must be measurable");
    assert!(refusal.entry_ids.contains(&"refused-1".to_string()));
    assert!(refusal.entry_ids.contains(&"refused-2".to_string()));

    let line = refusal.log_line();
    assert!(
        line.contains("refused-1"),
        "log line names the entry: {line}"
    );
    assert!(
        line.contains("refused-2"),
        "log line names the entry: {line}"
    );
    assert!(
        line.contains("test::refusal_path"),
        "log line names the calling path: {line}"
    );

    assert!(kb_exists(pool, "refused-1").await);
    assert!(kb_exists(pool, "refused-2").await);

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// The periodic sweep and the consolidation reap inherit the same refusal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_periodic_trash_sweep_is_refused_while_the_safety_flag_is_set() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    seed_tombstone(pool, ALICE, "sweep-refused", 90).await;

    let reaped = sweep_expired_trash(pool, person_only_policy())
        .await
        .expect("a refused sweep is a normal outcome");

    assert_eq!(reaped, 0, "the sweep must free nothing");
    assert!(kb_exists(pool, "sweep-refused").await);

    fx.cleanup().await;
}

#[tokio::test]
async fn the_periodic_trash_sweep_reaps_while_the_safety_flag_is_clear() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    seed_tombstone(pool, ALICE, "sweep-allowed", 90).await;

    let reaped = sweep_expired_trash(pool, default_policy())
        .await
        .expect("sweep");

    assert_eq!(reaped, 1, "the shipped default keeps reaping");
    assert!(!kb_exists(pool, "sweep-allowed").await);

    fx.cleanup().await;
}

#[tokio::test]
async fn the_consolidation_reap_is_refused_while_the_safety_flag_is_set() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    // One expired tombstone, plus two live rows so consolidation has work and
    // opens its apply transaction.
    seed_tombstone(pool, ALICE, "consol-reap", 90).await;
    seed_kb(pool, ALICE, "consol-a", "alpha").await;
    seed_kb(pool, ALICE, "consol-b", "beta").await;

    let llm = llm_returning(
        r#"{"operations":[{"op":"merge_new","ids":["consol-a","consol-b"],"content":"UNIFIED","scope":null}]}"#,
    );
    run_consolidation_scan(
        pool,
        &llm,
        person_only_policy(),
        &CancellationToken::new(),
        None,
    )
    .await
    .expect("consolidation scan succeeds");

    // The merge still writes its new row (and dispositions its members) even
    // though the reap in the same transaction is refused - only the person-only
    // safety flag on the reap is under test here.
    assert!(
        kb_id_with_content(pool, ALICE, "UNIFIED").await.is_some(),
        "the merge still applies; only the reap is refused"
    );
    assert_eq!(
        sqlx::query_scalar::<_, String>(
            "SELECT disposition FROM knowledge_base WHERE id = 'consol-a'"
        )
        .fetch_one(pool)
        .await
        .expect("read consol-a's disposition"),
        "redundant",
        "the merge member is dispositioned, not left untouched"
    );
    assert!(
        kb_exists(pool, "consol-reap").await,
        "the in-transaction reap must free nothing"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// One run may rewrite only a bounded share of the store
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rewrite_share_of_one_run_is_bounded() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    let ids: Vec<String> = (1..=10).map(|i| format!("rewrite-{i}")).collect();
    for id in &ids {
        seed_kb(pool, ALICE, id, "original text").await;
    }

    // A degraded answer that rewrites everything it was shown.
    let policy = KnowledgeDeletePolicy {
        rewrite_fraction: 0.25,
        ..KnowledgeDeletePolicy::default()
    };
    let llm = llm_returning(&edit_plan(&ids));
    let stats = run_consolidation_scan(pool, &llm, policy, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(
        stats.updated, 3,
        "one run may rewrite at most ceil(10 * 0.25) entries"
    );
    let mut rewritten = 0usize;
    for id in &ids {
        if kb_content(pool, id)
            .await
            .is_some_and(|c| c.starts_with("REWRITTEN"))
        {
            rewritten += 1;
        }
    }
    assert_eq!(rewritten, 3, "only the capped share reached the rows");

    fx.cleanup().await;
}

#[tokio::test]
async fn a_merge_cluster_costs_one_rewrite_whatever_its_size() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    for id in ["cl-a", "cl-b", "cl-c", "cl-d"] {
        seed_kb(pool, ALICE, id, "cluster member").await;
    }
    for id in ["ed-e", "ed-f", "ed-g", "ed-h"] {
        seed_kb(pool, ALICE, id, "editable").await;
    }

    // cap = ceil(8 * 0.25) = 2. Merging is what consolidation is for, so the
    // cluster is taken first and one edit fits behind it.
    //
    // The invariant under test survives `merge_new` unchanged: the rewrite
    // budget still charges a cluster ONE slot regardless of its member count
    // (`take_within_rewrite_cap` counts merge ops, not members - untouched by
    // this unit). What changed is only how a merge is applied - a new row for
    // the synthesized content, every member dispositioned `redundant` rather
    // than one member rewritten in place - so the content assertion below
    // points at the new row instead of at `cl-a`.
    let plan = r#"{"operations":[
        {"op":"merge_new","ids":["cl-a","cl-b","cl-c","cl-d"],"content":"UNIFIED","scope":null},
        {"op":"edit","id":"ed-e","content":"REWRITTEN e"},
        {"op":"edit","id":"ed-f","content":"REWRITTEN f"},
        {"op":"edit","id":"ed-g","content":"REWRITTEN g"},
        {"op":"edit","id":"ed-h","content":"REWRITTEN h"}
    ]}"#;
    let policy = KnowledgeDeletePolicy {
        rewrite_fraction: 0.25,
        ..KnowledgeDeletePolicy::default()
    };
    let llm = llm_returning(plan);
    let stats = run_consolidation_scan(pool, &llm, policy, &CancellationToken::new(), None)
        .await
        .expect("consolidation scan succeeds");

    assert_eq!(stats.merged_clusters, 1, "the four-member cluster applies");
    assert_eq!(
        stats.updated, 1,
        "the cluster leaves room for exactly one edit - it cost one rewrite \
         slot, not four"
    );
    let merged_id = kb_id_with_content(pool, ALICE, "UNIFIED").await;
    assert!(
        merged_id.is_some(),
        "a cluster of four still costs one rewrite, not four - the budget still \
         let it through alongside one edit"
    );
    for member in ["cl-a", "cl-b", "cl-c", "cl-d"] {
        assert_eq!(
            kb_content(pool, member).await.as_deref(),
            Some("cluster member"),
            "{member}'s own content is never rewritten under merge_new"
        );
    }

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// A refusal reaches the log on both disposals, not only the returned value
// ---------------------------------------------------------------------------

#[test]
fn a_refused_background_reap_writes_the_entry_id_and_path_to_the_log() {
    let rt = capture_runtime();
    let Some(fx) = rt.block_on(support::DbFixture::try_new("kb1122")) else {
        return;
    };

    rt.block_on(seed_tombstone(&fx.pool, ALICE, "logged-sweep", 90));

    let (reaped, log) =
        run_capturing_logs(&rt, sweep_expired_trash(&fx.pool, person_only_policy()));

    assert_eq!(reaped.expect("sweep"), 0);
    assert!(log.contains("logged-sweep"), "log names the entry: {log}");
    assert!(
        log.contains("dreaming::trash::reap_expired_for_user"),
        "log names the calling path: {log}"
    );

    rt.block_on(fx.cleanup());
}

#[test]
fn a_refused_caller_delete_writes_the_entry_id_and_path_to_the_log() {
    let rt = capture_runtime();
    let Some(fx) = rt.block_on(support::DbFixture::try_new("kb1122")) else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), person_only_policy());

    rt.block_on(seed_kb(&fx.pool, ALICE, "logged-tool-delete", "a fact"));

    // A caller receives the refusal as a value. It must be recorded even so: a
    // caller that swallows the error must not erase the evidence.
    let (result, log) = run_capturing_logs(
        &rt,
        with_user_id(UserId::new(ALICE), async {
            store.delete_many(&["logged-tool-delete".to_string()]).await
        }),
    );

    assert!(result.is_err(), "the delete is refused");
    assert!(log.contains("logged-tool-delete"), "log names it: {log}");
    assert!(
        log.contains("knowledge::PgKnowledgeBaseStore::delete_many"),
        "log names the calling path: {log}"
    );

    rt.block_on(fx.cleanup());
}

#[tokio::test]
async fn a_large_refusal_reports_the_true_count_and_lists_a_bounded_sample() {
    let Some(fx) = support::DbFixture::try_new("kb1122").await else {
        return;
    };
    let pool = &fx.pool;

    // More tombstones than one refusal will list. The count must still be
    // exact, because the count is what makes the volume measurable, and the
    // list must stay bounded, because the flag lets tombstones accumulate.
    let total = MAX_REFUSAL_IDS + 7;
    for i in 0..total {
        seed_tombstone(pool, ALICE, &format!("bulk-{i:04}"), 90).await;
    }

    let outcome = hard_delete_knowledge(
        pool,
        ALICE,
        HardDeleteTarget::ExpiredTombstones,
        person_only_policy(),
        "test::bulk_refusal",
    )
    .await
    .expect("a refusal is a normal outcome");

    let refusal = outcome.refusal.expect("the outcome carries the refusal");
    assert_eq!(refusal.total, total, "the count covers every spared row");
    assert_eq!(
        refusal.entry_ids.len(),
        MAX_REFUSAL_IDS,
        "the listed sample stays bounded"
    );
    assert!(refusal.log_line().contains(&format!("of {total}")));

    fx.cleanup().await;
}

#[test]
fn a_refused_sweep_that_spared_nothing_does_not_log_a_warning() {
    let rt = capture_runtime();
    let Some(fx) = rt.block_on(support::DbFixture::try_new("kb1122")) else {
        return;
    };

    // One tombstone, still inside the retention window. The user holds trash,
    // so the sweep runs and the flag refuses it - but nothing is aged past
    // retention, so the refusal spares nothing. This is what an instance with
    // the flag set does on most hourly ticks, and a warning each time would
    // bury the refusals that carry entries.
    rt.block_on(seed_tombstone(&fx.pool, ALICE, "not-yet-expired", 1));

    let (reaped, log) =
        run_capturing_logs(&rt, sweep_expired_trash(&fx.pool, person_only_policy()));

    assert_eq!(reaped.expect("sweep"), 0, "the refusal still frees nothing");
    assert!(
        !log.contains("WARN"),
        "a refusal that spared nothing must not warn: {log}"
    );
    assert!(
        rt.block_on(kb_exists(&fx.pool, "not-yet-expired")),
        "the tombstone is kept"
    );

    rt.block_on(fx.cleanup());
}
