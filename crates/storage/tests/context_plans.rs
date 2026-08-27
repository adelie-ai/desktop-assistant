//! Integration tests for the per-turn context plan store (#1327).
//!
//! Exercises `PgContextPlanStore` against a real Postgres with the
//! migrations applied. The whole value of this table is the per-candidate
//! score broken down by term, so
//! `a_plan_row_round_trips_with_every_field` writes a plan whose candidates
//! cover every drop reason the enum declares and reads every field back
//! rather than sampling a few.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db-up
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test context_plans
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log line so
//! the suite stays green without a DB.

mod support;

use chrono::{Duration, Utc};
use desktop_assistant_core::domain::activation::{ActivationTerms, ActivationWeights};
use desktop_assistant_core::domain::knowledge_use::UseScoreWeights;
use desktop_assistant_core::ports::context_plan::{
    ArmSummaries, ArmSummary, ContextPlan, PlannedCandidate, PlannedDropReason, PlannedUseCounts,
    RecallArm,
};
use desktop_assistant_core::ports::recall::{RecallDispersion, RecallRelevance};
use desktop_assistant_storage::context_plans::{PgContextPlanStore, sweep_expired_context_plans};
use desktop_assistant_storage::{UserId, with_user_id};

async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("plan1327").await
}

fn arm(
    median: f64,
    deviation: f64,
    measured: bool,
    cue: bool,
    limit: usize,
    rows: usize,
    capped: bool,
) -> ArmSummary {
    ArmSummary {
        dispersion: RecallDispersion::assumed(median, deviation),
        dispersion_measured: measured,
        situation_cue_present: cue,
        scan_limit: limit,
        rows_returned: rows,
        capped,
    }
}

/// A candidate for every one of the six `PlannedDropReason` variants that
/// exist in the code today, plus a bar-refused candidate and an offered one -
/// eight candidates, spanning all three arms and both `RecallRelevance`
/// kinds, so the round trip has something of every shape to lose.
fn rich_candidates() -> Vec<PlannedCandidate> {
    let terms = |semantic: Option<f64>, lex, rf, sit, sal, disp, total| ActivationTerms {
        semantic,
        lexical: lex,
        reinforcement: rf,
        situation: sit,
        salience: sal,
        disposition: disp,
        total,
    };
    vec![
        // Offered - the one line that actually rendered.
        PlannedCandidate {
            arm: RecallArm::Entry,
            id: "kb-offered".to_string(),
            relevance: RecallRelevance::Distance(0.42),
            terms: terms(Some(9.1), 0.0, 1.2, 0.4, 0.1, 0.0, 10.8),
            use_counts: Some(PlannedUseCounts {
                offered: 4,
                opened: 2,
                marked: 1,
            }),
            cleared_bar: true,
            rank: Some(0),
            offered: true,
            drop_reason: None,
        },
        PlannedCandidate {
            arm: RecallArm::Entry,
            id: "kb-width".to_string(),
            relevance: RecallRelevance::Distance(0.51),
            terms: terms(Some(8.0), 0.0, 0.0, 0.0, 0.0, 0.0, 8.0),
            use_counts: None,
            cleared_bar: true,
            rank: Some(1),
            offered: false,
            drop_reason: Some(PlannedDropReason::WidthCap),
        },
        PlannedCandidate {
            arm: RecallArm::Entry,
            id: "kb-pinned".to_string(),
            relevance: RecallRelevance::Distance(0.60),
            terms: terms(Some(7.0), 0.0, 0.0, 0.0, 0.0, 0.0, 7.0),
            use_counts: Some(PlannedUseCounts {
                offered: 1,
                opened: 0,
                marked: 0,
            }),
            cleared_bar: true,
            rank: Some(2),
            offered: false,
            drop_reason: Some(PlannedDropReason::Pinned),
        },
        PlannedCandidate {
            arm: RecallArm::Note,
            id: "note-in-view".to_string(),
            relevance: RecallRelevance::Distance(0.30),
            terms: terms(Some(6.0), 0.0, 0.0, 0.0, 0.0, 0.0, 6.0),
            use_counts: None,
            cleared_bar: true,
            // Notes are never reordered by activation, so a note candidate
            // carries no rank even when it cleared the bar.
            rank: None,
            offered: false,
            drop_reason: Some(PlannedDropReason::InView),
        },
        PlannedCandidate {
            arm: RecallArm::Note,
            id: "note-empty".to_string(),
            relevance: RecallRelevance::Distance(0.35),
            terms: terms(Some(5.5), 0.0, 0.0, 0.0, 0.0, 0.0, 5.5),
            use_counts: None,
            cleared_bar: true,
            rank: None,
            offered: false,
            drop_reason: Some(PlannedDropReason::EmptyContent),
        },
        PlannedCandidate {
            arm: RecallArm::Note,
            id: "note-external".to_string(),
            relevance: RecallRelevance::Distance(0.38),
            terms: terms(Some(5.0), 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
            use_counts: None,
            cleared_bar: true,
            rank: None,
            offered: false,
            drop_reason: Some(PlannedDropReason::ExternalContent),
        },
        PlannedCandidate {
            arm: RecallArm::Skill,
            id: "skill-unrenderable".to_string(),
            relevance: RecallRelevance::Distance(0.20),
            terms: terms(Some(4.0), 0.0, 0.3, 0.0, 0.0, 0.0, 4.3),
            use_counts: Some(PlannedUseCounts {
                offered: 2,
                opened: 2,
                marked: 0,
            }),
            cleared_bar: true,
            rank: Some(0),
            offered: false,
            drop_reason: Some(PlannedDropReason::IdUnrenderable),
        },
        // Bar-refused, lexical relevance - no semantic term, no rank, no
        // drop reason (the bar itself is the reason).
        PlannedCandidate {
            arm: RecallArm::Skill,
            id: "skill-refused".to_string(),
            relevance: RecallRelevance::LexicalMatch,
            terms: terms(None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            use_counts: None,
            cleared_bar: false,
            rank: None,
            offered: false,
            drop_reason: None,
        },
    ]
}

fn rich_plan(request_id: &str, conversation_id: &str) -> ContextPlan {
    ContextPlan {
        request_id: request_id.to_string(),
        conversation_id: conversation_id.to_string(),
        recall_ran: true,
        query_text: Some("PLAN1327-what-does-the-runbook-say".to_string()),
        query_text_truncated: false,
        bar: 6.8,
        weights: ActivationWeights {
            use_lift: 0.83,
            use_score: UseScoreWeights {
                decay: 0.61,
                model_mark: 2.5,
                person_mark: 9.25,
            },
            trivial_penalty: 1.4,
        },
        scorer_version: "1327-v1".to_string(),
        arms: ArmSummaries {
            entries: arm(12.0, 3.0, true, true, 50, 9, false),
            notes: arm(6.0, 1.5, false, false, 25, 3, true),
            skills: arm(4.0, 2.0, true, false, 25, 2, false),
        },
        candidates: rich_candidates(),
        // Deliberately larger than `candidates.len()` (8), so the round trip
        // proves the two fields travel independently rather than one being
        // derived from the other on read.
        considered_count: 11,
        truncated: true,
        opened: vec!["kb-offered".to_string(), "skill-unrenderable".to_string()],
        recorded_at: None,
    }
}

#[tokio::test]
async fn a_plan_row_round_trips_with_every_field() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; a_plan_row_round_trips_with_every_field");
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let written = rich_plan("r1", "c1");
        store.record(&written).await.expect("record");

        let mut read = store
            .get_by_request_id("r1")
            .await
            .expect("get")
            .expect("the row written under r1");
        assert!(
            read.recorded_at.is_some(),
            "a stored plan says when it was written"
        );
        read.recorded_at = None;
        assert_eq!(
            read, written,
            "every field must survive the round trip, including each \
             candidate's per-term score, its drop reason, and the arm \
             summaries' dispersion"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn a_reused_request_id_cannot_move_a_plan_between_conversations() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             a_reused_request_id_cannot_move_a_plan_between_conversations"
        );
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        store
            .record(&rich_plan("shared", "c1"))
            .await
            .expect("first write");
        // A second conversation reuses the same client-chosen id. The
        // refusal is silent (a warning, not an error), matching
        // context_breakdown's rule for the same situation.
        store
            .record(&rich_plan("shared", "c2"))
            .await
            .expect("a reused id is refused quietly, not as an error");

        let kept = store
            .get_by_request_id("shared")
            .await
            .expect("get")
            .expect("the row");
        assert_eq!(
            kept.conversation_id, "c1",
            "the record still names the conversation whose turn it describes"
        );

        let c1_rows = store
            .list_for_conversation("c1", 50, 0)
            .await
            .expect("list c1");
        assert_eq!(c1_rows.len(), 1, "c1 keeps the row it recorded first");
        let c2_rows = store
            .list_for_conversation("c2", 50, 0)
            .await
            .expect("list c2");
        assert!(
            c2_rows.is_empty(),
            "c2 gains nothing from a reused id, and takes nothing away from c1"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn append_opened_is_idempotent_for_one_entry_id() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; append_opened_is_idempotent_for_one_entry_id");
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut written = rich_plan("r1", "c1");
        written.opened.clear();
        store.record(&written).await.expect("record");

        store
            .append_opened("r1", "kb-newly-opened")
            .await
            .expect("first append");
        store
            .append_opened("r1", "kb-newly-opened")
            .await
            .expect("a repeat append must not fail");
        store
            .append_opened("r1", "kb-newly-opened")
            .await
            .expect("a third repeat still must not fail");

        let read = store
            .get_by_request_id("r1")
            .await
            .expect("get")
            .expect("the row");
        assert_eq!(
            read.opened,
            vec!["kb-newly-opened".to_string()],
            "three appends of the same id leave exactly one entry, not three"
        );

        store
            .append_opened("r1", "kb-second-open")
            .await
            .expect("a different id still appends");
        let read = store
            .get_by_request_id("r1")
            .await
            .expect("get")
            .expect("the row");
        assert_eq!(
            read.opened,
            vec!["kb-newly-opened".to_string(), "kb-second-open".to_string()],
            "a genuinely new id still grows the array, in the order opened"
        );
    })
    .await;
    fx.cleanup().await;
}

/// The store's own defensive ceiling on `opened` (512, matching
/// `MAX_PLANNED_CANDIDATES` in order of magnitude - see
/// `crates/storage/src/context_plans/mod.rs`'s `MAX_OPENED_ENTRIES`). Nothing
/// calls the opened-recorder in production today, so this path is dormant,
/// but the array must not grow without bound once something does.
#[tokio::test]
async fn append_opened_does_not_grow_the_array_past_its_bound() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             append_opened_does_not_grow_the_array_past_its_bound"
        );
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut written = rich_plan("r1", "c1");
        written.opened.clear();
        store.record(&written).await.expect("record");

        for i in 0..520 {
            store
                .append_opened("r1", &format!("kb-opened-{i}"))
                .await
                .expect("append must not fail even once the array is at its cap");
        }

        let read = store
            .get_by_request_id("r1")
            .await
            .expect("get")
            .expect("the row");
        assert_eq!(
            read.opened.len(),
            512,
            "520 distinct ids were opened, but the stored array stopped at \
             the cap rather than growing to match"
        );
        assert_eq!(
            read.opened[0], "kb-opened-0",
            "the ids that fit are the first ones opened, not the last"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn plan_reads_are_scoped_to_the_calling_user() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; plan_reads_are_scoped_to_the_calling_user");
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("owner"), async {
        store
            .record(&rich_plan("r1", "c1"))
            .await
            .expect("record as owner");
    })
    .await;

    with_user_id(UserId::from("intruder"), async {
        assert_eq!(
            store.get_by_request_id("r1").await.expect("get"),
            None,
            "a request id is not a capability; the row belongs to its user"
        );
        assert!(
            store
                .list_for_conversation("c1", 50, 0)
                .await
                .expect("list")
                .is_empty(),
            "another user's conversation must read as empty"
        );
    })
    .await;

    with_user_id(UserId::from("owner"), async {
        assert!(
            store.get_by_request_id("r1").await.expect("get").is_some(),
            "the owner still reads their own row"
        );
    })
    .await;
    fx.cleanup().await;
}

async fn age_plan(pool: &sqlx::PgPool, request_id: &str, days: i64) {
    let when = Utc::now() - Duration::days(days);
    sqlx::query("UPDATE context_plans SET recorded_at = $2 WHERE request_id = $1")
        .bind(request_id)
        .bind(when)
        .execute(pool)
        .await
        .expect("backdate the plan");
}

#[tokio::test]
async fn the_sweep_deletes_plans_past_the_retention_window_and_no_younger_row() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             the_sweep_deletes_plans_past_the_retention_window_and_no_younger_row"
        );
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        store
            .record(&rich_plan("old", "c1"))
            .await
            .expect("record old");
        store
            .record(&rich_plan("fresh", "c1"))
            .await
            .expect("record fresh");
    })
    .await;
    age_plan(&fx.pool, "old", 9).await;

    let removed = sweep_expired_context_plans(&fx.pool, 7)
        .await
        .expect("the sweep runs");
    assert_eq!(removed, 1, "exactly one row was past the window");

    with_user_id(UserId::from("u1"), async {
        assert!(
            store.get_by_request_id("old").await.expect("get").is_none(),
            "the row past the window is gone"
        );
        assert!(
            store
                .get_by_request_id("fresh")
                .await
                .expect("get")
                .is_some(),
            "the row inside the window is untouched by the sweep"
        );
    })
    .await;
    fx.cleanup().await;
}

/// `n` candidates, cycled from `rich_candidates`'s eight shapes with a
/// unique id suffix each, so a wider row is realistic rather than one
/// candidate repeated byte-for-byte (which would let JSONB's own
/// dictionary-style compression hide a size regression this pin exists to
/// catch).
fn n_candidates(n: usize) -> Vec<PlannedCandidate> {
    let base = rich_candidates();
    let mut candidates = Vec::with_capacity(n);
    while candidates.len() < n {
        for c in &base {
            if candidates.len() == n {
                break;
            }
            let mut c = c.clone();
            c.id = format!("{}-{}", c.id, candidates.len());
            candidates.push(c);
        }
    }
    candidates
}

/// Today's arm scan limits (entry 50, note 25, skill 25 - see
/// `crates/core/src/recall.rs`) sum to 100, which is why a realistic turn's
/// plan looks like this test's fixture rather than
/// `MAX_PLANNED_CANDIDATES`'s 512. This pin exists to catch that shape
/// drifting wider without anyone measuring it, not to describe the cap - see
/// `a_five_hundred_twelve_candidate_plan_serializes_within_its_measured_cost`
/// below for the cap itself.
#[tokio::test]
async fn a_hundred_candidate_plan_serializes_under_64_kib() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; a_hundred_candidate_plan_serializes_under_64_kib"
        );
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut plan = rich_plan("big", "c1");
        plan.candidates = n_candidates(100);
        plan.considered_count = 100;
        plan.truncated = false;
        store
            .record(&plan)
            .await
            .expect("record a hundred candidates");

        // The JSON text length, not the on-disk (TOAST-compressed) column
        // size: compression on a real column can hide a regression in the
        // shape this pin exists to catch, since a wider object still
        // compresses well when its keys repeat. `octet_length(...::text)` is
        // what "serializes" means here - the bytes the candidate array
        // actually renders to.
        let size: i32 = sqlx::query_scalar(
            "SELECT octet_length(candidates::text) FROM context_plans WHERE request_id = $1",
        )
        .bind("big")
        .fetch_one(&fx.pool)
        .await
        .expect("measure the serialized candidates column");
        assert!(
            size < 64 * 1024,
            "a hundred candidates serialized to {size} bytes, past the 64 \
             KiB pin the design's per-candidate estimate (150-250 bytes) \
             implies"
        );
    })
    .await;
    fx.cleanup().await;
}

/// `MAX_PLANNED_CANDIDATES` (512) is the real ceiling `candidates` can reach
/// - `crates/core/src/recall.rs`'s `build_context_plan` cuts to it before
/// this store ever sees a row - and the 64 KiB pin above is deliberately
/// **not** a claim about that ceiling: at today's ~233 bytes/candidate, 512
/// candidates costs roughly 512x that, not 100x, which lands well past 64
/// KiB. A reviewer reading only the test above could mistake "under 64 KiB
/// at 100" for "under 64 KiB at the cap"; this test measures the cap
/// directly instead of leaving that read as an inference. 130 KiB is the
/// pin: comfortably above the design's own ~128 KB estimate for the full
/// cap, so this catches a real regression in the per-candidate shape without
/// chasing single-digit-percent noise between runs.
#[tokio::test]
async fn a_five_hundred_twelve_candidate_plan_serializes_within_its_measured_cost() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             a_five_hundred_twelve_candidate_plan_serializes_within_its_measured_cost"
        );
        return;
    };
    let store = PgContextPlanStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        let mut plan = rich_plan("full-cap", "c1");
        plan.candidates = n_candidates(512);
        plan.considered_count = 512;
        plan.truncated = false;
        store.record(&plan).await.expect("record 512 candidates");

        let size: i32 = sqlx::query_scalar(
            "SELECT octet_length(candidates::text) FROM context_plans WHERE request_id = $1",
        )
        .bind("full-cap")
        .fetch_one(&fx.pool)
        .await
        .expect("measure the serialized candidates column");
        assert!(
            size < 130 * 1024,
            "512 candidates - MAX_PLANNED_CANDIDATES, the row's real ceiling \
             - serialized to {size} bytes, past the 130 KiB pin. The design \
             estimated ~128 KB for a full-cap row; a size meaningfully past \
             that is a per-candidate shape regression, not noise."
        );
    })
    .await;
    fx.cleanup().await;
}
