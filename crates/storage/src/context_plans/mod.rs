//! Postgres adapter for the per-turn context plan (#1327).
//!
//! One row per turn, keyed on `(user_id, request_id)`, mirroring
//! `crates/storage/src/context_breakdown.rs`. The key is the turn's own
//! correlation id, so the write is idempotent by construction: a retried or
//! re-driven turn replaces its row instead of adding a second record of the
//! same turn.
//!
//! ## The candidate shape comes from the domain type, never from here
//!
//! [`PlannedCandidate`] has one definition, in
//! `crates/core/src/ports/context_plan.rs`, and every JSON key this file
//! writes or reads is a field of that type or of [`ActivationTerms`]. Nothing
//! in this file spells a second column list: `candidates`, `arms`, `weights`
//! and `opened` are JSONB precisely so the Rust type stays the only
//! definition of the shape it stores.
//!
//! ## Both reads spell their SQL out in full, on purpose
//!
//! Like `context_breakdown.rs`, `get_by_request_id` and `list_for_conversation`
//! each write the column list as a plain string literal rather than sharing
//! one built with `format!`. The static `user_id` audit
//! (`crates/storage/tests/audit_user_id_scoping.rs`) extracts the string
//! literal passed to a `sqlx::query...(` call and skips a call whose first
//! argument is anything else - so a query assembled from a shared constant is
//! a query the audit silently does not scan. The cost is one column list
//! written twice.
//!
//! ## Reading `RecallDispersion` back
//!
//! `RecallDispersion`'s two numbers are private to `core` (#1244's rule: a
//! ranking site outside the crate must not read behind the type it is
//! handed). This file recovers them through the type's own public API
//! instead of reaching around it: [`RecallDispersion::distance_at`] is
//! linear in its argument, so `distance_at(0.0)` is the median exactly and
//! `distance_at(0.0) - distance_at(1.0)` is the deviation. On read
//! [`RecallDispersion::assumed`] rebuilds the pair losslessly from the two
//! stored floats.
//!
//! ## A failing write must never fail a turn
//!
//! Mirrors `crates/storage/src/context_breakdown.rs`: the write is reached
//! only through `record_in_background` (`crates/core/src/service.rs`), which
//! logs and swallows a failure. Nothing in this file changes that contract -
//! every `Result` here is returned to the caller to decide, never panicked
//! on.

pub mod retention;

pub use retention::sweep_expired_context_plans;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::activation::{ActivationTerms, ActivationWeights};
use desktop_assistant_core::domain::knowledge_use::UseScoreWeights;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::context_plan::{
    ArmSummaries, ArmSummary, ContextPlan, PlannedCandidate, PlannedDropReason, PlannedUseCounts,
    RecallArm,
};
use desktop_assistant_core::ports::recall::{RecallDispersion, RecallRelevance};
use serde_json::{Value, json};
use sqlx::PgPool;
use sqlx::Row;
use sqlx::postgres::PgRow;

/// The bound one `list_for_conversation` call actually gives the database.
///
/// A function rather than an inline `min` so the ceiling is exercised by a
/// test rather than asserted about a constant - see
/// `crates/storage/src/context_breakdown.rs`'s `page_limit` for the same
/// shape.
const MAX_PLANS_PER_PAGE: u32 = 500;

fn page_limit(requested: u32) -> u32 {
    requested.min(MAX_PLANS_PER_PAGE)
}

/// The most ids the `opened` array keeps, mirroring
/// [`desktop_assistant_core::ports::context_plan::MAX_PLANNED_CANDIDATES`] in
/// order of magnitude.
///
/// `ContextPlan::opened` (`crates/core/src/ports/context_plan.rs`) carries no
/// cap of its own - nothing in `core` writes past a handful of entries today,
/// because nothing yet calls the opened-recorder hook this store wires (see
/// the PR that added this file). That is a dormant path, not a proven one,
/// and a dormant field is exactly the one nobody is watching when a future
/// writer starts growing it. `candidates` already defends itself with a cap
/// and a `truncated` flag; `opened` cannot carry the same flag without a
/// field `core` does not define, so this store enforces a ceiling on the
/// array itself instead, at both write paths (`record` and `append_opened`).
/// A model cannot open more distinct candidates than a turn ever offered, and
/// `candidates` is itself bounded at 512, so 512 is the natural ceiling here
/// too, not a number picked independently.
const MAX_OPENED_ENTRIES: usize = 512;

pub struct PgContextPlanStore {
    pool: PgPool,
}

impl PgContextPlanStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Record this turn's plan, upserting on `(user_id, request_id)`.
    ///
    /// Refuses to move a row between conversations: the key holds a value
    /// the client chose, so a client that reuses one id for two turns of two
    /// conversations would otherwise relocate the first conversation's
    /// record into the second. The `WHERE` on the `DO UPDATE` makes that
    /// write land nowhere instead, and a `rows_affected() == 0` logs why.
    pub async fn record(&self, plan: &ContextPlan) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let written = sqlx::query(
            "INSERT INTO context_plans \
                 (user_id, request_id, conversation_id, recall_ran, query_text, \
                  query_text_truncated, bar, weights, scorer_version, arms, \
                  candidates, considered_count, truncated, opened) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) \
             ON CONFLICT (user_id, request_id) DO UPDATE SET \
                 conversation_id = EXCLUDED.conversation_id, \
                 recall_ran = EXCLUDED.recall_ran, \
                 query_text = EXCLUDED.query_text, \
                 query_text_truncated = EXCLUDED.query_text_truncated, \
                 bar = EXCLUDED.bar, \
                 weights = EXCLUDED.weights, \
                 scorer_version = EXCLUDED.scorer_version, \
                 arms = EXCLUDED.arms, \
                 candidates = EXCLUDED.candidates, \
                 considered_count = EXCLUDED.considered_count, \
                 truncated = EXCLUDED.truncated, \
                 opened = EXCLUDED.opened \
             WHERE context_plans.conversation_id = EXCLUDED.conversation_id",
        )
        .bind(user_id.as_str())
        .bind(&plan.request_id)
        .bind(&plan.conversation_id)
        .bind(plan.recall_ran)
        .bind(&plan.query_text)
        .bind(plan.query_text_truncated)
        .bind(plan.bar)
        .bind(weights_to_json(&plan.weights))
        .bind(&plan.scorer_version)
        .bind(arms_to_json(&plan.arms))
        .bind(candidates_to_json(&plan.candidates))
        .bind(i32::try_from(plan.considered_count).unwrap_or(i32::MAX))
        .bind(plan.truncated)
        .bind(opened_to_json(bounded_opened(&plan.opened)))
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if written.rows_affected() == 0 {
            tracing::warn!(
                request_id = %plan.request_id,
                conversation_id = %plan.conversation_id,
                "a context plan for this correlation id is already recorded \
                 against another conversation, so this turn's plan was not \
                 written; the id is the client's own turn id and has been \
                 reused"
            );
        }
        Ok(())
    }

    /// Append one opened id to the plan's `opened` array, once per id the
    /// model fetches during the turn.
    ///
    /// Idempotent: the array is checked for the id before it is appended, so
    /// a repeat call for the same id (a retried tool call, a second read of
    /// the same entry) leaves the array unchanged rather than growing it. A
    /// request id with no matching row - a fetch outside any turn's scope -
    /// updates zero rows and is not an error.
    pub async fn append_opened(
        &self,
        request_id: &str,
        opened_id: &str,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // Three-way CASE: already present (idempotent no-op), at the cap
        // (refuse to grow further - also a no-op), otherwise append. The cap
        // check reads the array's own length rather than a stored counter, so
        // it stays correct however the array got to that length.
        sqlx::query(
            "UPDATE context_plans SET opened = CASE \
                 WHEN opened @> to_jsonb($3::text) THEN opened \
                 WHEN jsonb_array_length(opened) >= $4 THEN opened \
                 ELSE opened || to_jsonb($3::text) \
             END \
             WHERE user_id = $1 AND request_id = $2",
        )
        .bind(user_id.as_str())
        .bind(request_id)
        .bind(opened_id)
        .bind(i32::try_from(MAX_OPENED_ENTRIES).unwrap_or(i32::MAX))
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Read one turn's plan by its correlation id, scoped to the calling
    /// user - a request id is not a capability.
    pub async fn get_by_request_id(
        &self,
        request_id: &str,
    ) -> Result<Option<ContextPlan>, CoreError> {
        let user_id = current_user_id();
        let row = sqlx::query(
            "SELECT request_id, conversation_id, recall_ran, query_text, \
                    query_text_truncated, bar, weights, scorer_version, arms, \
                    candidates, considered_count, truncated, opened, recorded_at \
             FROM context_plans \
             WHERE user_id = $1 AND request_id = $2",
        )
        .bind(user_id.as_str())
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        row.as_ref().map(row_to_plan).transpose()
    }

    /// List every plan for one conversation, newest first, scoped to the
    /// calling user.
    pub async fn list_for_conversation(
        &self,
        conversation_id: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<ContextPlan>, CoreError> {
        let user_id = current_user_id();
        let limit = page_limit(limit);
        let rows = sqlx::query(
            "SELECT request_id, conversation_id, recall_ran, query_text, \
                    query_text_truncated, bar, weights, scorer_version, arms, \
                    candidates, considered_count, truncated, opened, recorded_at \
             FROM context_plans \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY recorded_at DESC, request_id ASC \
             LIMIT $3 OFFSET $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(i64::from(limit))
        .bind(i64::from(offset))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        rows.iter().map(row_to_plan).collect()
    }
}

// ---------- JSON mapping: RecallArm and PlannedDropReason labels ----------

fn arm_label(arm: RecallArm) -> &'static str {
    match arm {
        RecallArm::Entry => "entry",
        RecallArm::Note => "note",
        RecallArm::Skill => "skill",
    }
}

fn arm_from_label(label: &str) -> Option<RecallArm> {
    match label {
        "entry" => Some(RecallArm::Entry),
        "note" => Some(RecallArm::Note),
        "skill" => Some(RecallArm::Skill),
        _ => None,
    }
}

/// The stable label each of the six [`PlannedDropReason`] variants is stored
/// under. Every variant the type declares is covered here; a variant added to
/// the enum and not to this match fails to compile, which is what a `match`
/// with no wildcard arm buys.
fn drop_reason_label(reason: PlannedDropReason) -> &'static str {
    match reason {
        PlannedDropReason::WidthCap => "width_cap",
        PlannedDropReason::Pinned => "pinned",
        PlannedDropReason::InView => "in_view",
        PlannedDropReason::IdUnrenderable => "id_unrenderable",
        PlannedDropReason::EmptyContent => "empty_content",
        PlannedDropReason::ExternalContent => "external_content",
    }
}

fn drop_reason_from_label(label: &str) -> Option<PlannedDropReason> {
    match label {
        "width_cap" => Some(PlannedDropReason::WidthCap),
        "pinned" => Some(PlannedDropReason::Pinned),
        "in_view" => Some(PlannedDropReason::InView),
        "id_unrenderable" => Some(PlannedDropReason::IdUnrenderable),
        "empty_content" => Some(PlannedDropReason::EmptyContent),
        "external_content" => Some(PlannedDropReason::ExternalContent),
        _ => None,
    }
}

// ---------- JSON mapping: weights ----------

fn weights_to_json(weights: &ActivationWeights) -> Value {
    json!({
        "use_lift": weights.use_lift,
        "use_score": {
            "decay": weights.use_score.decay,
            "model_mark": weights.use_score.model_mark,
            "person_mark": weights.use_score.person_mark,
        },
    })
}

/// A field this build cannot read falls back to the scorer's own default -
/// the same rule `context_breakdown.rs` follows for a part its build cannot
/// name: reporting a wrong number is worse than reporting the value the
/// scorer would use unconfigured.
fn weights_from_json(stored: &Value) -> ActivationWeights {
    let defaults = ActivationWeights::default();
    let use_score = stored.get("use_score");
    ActivationWeights {
        use_lift: stored
            .get("use_lift")
            .and_then(Value::as_f64)
            .unwrap_or(defaults.use_lift),
        use_score: UseScoreWeights {
            decay: use_score
                .and_then(|u| u.get("decay"))
                .and_then(Value::as_f64)
                .unwrap_or(defaults.use_score.decay),
            model_mark: use_score
                .and_then(|u| u.get("model_mark"))
                .and_then(Value::as_f64)
                .unwrap_or(defaults.use_score.model_mark),
            person_mark: use_score
                .and_then(|u| u.get("person_mark"))
                .and_then(Value::as_f64)
                .unwrap_or(defaults.use_score.person_mark),
        },
    }
}

// ---------- JSON mapping: arm summaries ----------

fn dispersion_to_json(dispersion: RecallDispersion) -> Value {
    let median = dispersion.distance_at(0.0);
    let deviation = median - dispersion.distance_at(1.0);
    json!({ "median": median, "deviation": deviation })
}

fn dispersion_from_json(stored: &Value) -> RecallDispersion {
    let median = stored.get("median").and_then(Value::as_f64).unwrap_or(0.0);
    let deviation = stored
        .get("deviation")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    RecallDispersion::assumed(median, deviation)
}

fn arm_summary_to_json(arm: &ArmSummary) -> Value {
    json!({
        "dispersion": dispersion_to_json(arm.dispersion),
        "dispersion_measured": arm.dispersion_measured,
        "situation_cue_present": arm.situation_cue_present,
        "scan_limit": arm.scan_limit,
        "rows_returned": arm.rows_returned,
        "capped": arm.capped,
    })
}

fn arm_summary_from_json(stored: &Value) -> ArmSummary {
    ArmSummary {
        dispersion: stored
            .get("dispersion")
            .map(dispersion_from_json)
            .unwrap_or(RecallDispersion::assumed(0.0, 1.0)),
        dispersion_measured: stored
            .get("dispersion_measured")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        situation_cue_present: stored
            .get("situation_cue_present")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        scan_limit: stored
            .get("scan_limit")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        rows_returned: stored
            .get("rows_returned")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        capped: stored.get("capped").and_then(Value::as_bool).unwrap_or(false),
    }
}

fn arms_to_json(arms: &ArmSummaries) -> Value {
    json!({
        "entries": arm_summary_to_json(&arms.entries),
        "notes": arm_summary_to_json(&arms.notes),
        "skills": arm_summary_to_json(&arms.skills),
    })
}

fn arms_from_json(stored: &Value) -> ArmSummaries {
    ArmSummaries {
        entries: stored
            .get("entries")
            .map(arm_summary_from_json)
            .unwrap_or(ArmSummary::empty(0)),
        notes: stored
            .get("notes")
            .map(arm_summary_from_json)
            .unwrap_or(ArmSummary::empty(0)),
        skills: stored
            .get("skills")
            .map(arm_summary_from_json)
            .unwrap_or(ArmSummary::empty(0)),
    }
}

// ---------- JSON mapping: candidates ----------

fn candidate_to_json(candidate: &PlannedCandidate) -> Value {
    let (rel_kind, rel) = match candidate.relevance {
        RecallRelevance::Distance(d) => ("distance", Some(d)),
        RecallRelevance::LexicalMatch => ("lexical", None),
    };
    json!({
        "arm": arm_label(candidate.arm),
        "id": candidate.id,
        "rel_kind": rel_kind,
        "rel": rel,
        "sem": candidate.terms.semantic,
        "lex": candidate.terms.lexical,
        "rf": candidate.terms.reinforcement,
        "sit": candidate.terms.situation,
        "sal": candidate.terms.salience,
        "a": candidate.terms.total,
        "use": candidate.use_counts.map(|u| {
            json!({ "off": u.offered, "op": u.opened, "mk": u.marked })
        }),
        "bar_ok": candidate.cleared_bar,
        "rank": candidate.rank,
        "off": candidate.offered,
        "drop": candidate.drop_reason.map(drop_reason_label),
    })
}

/// Rebuild one candidate from its stored object, or `None` when the object
/// is missing a field this build cannot do without. A candidate this build
/// cannot read is skipped by the caller rather than surfaced as a malformed
/// one among otherwise-good candidates - the same "skip, do not attribute"
/// rule `context_breakdown.rs` applies to an unknown prompt part.
fn candidate_from_json(stored: &Value) -> Option<PlannedCandidate> {
    let arm = arm_from_label(stored.get("arm")?.as_str()?)?;
    let id = stored.get("id")?.as_str()?.to_string();
    let relevance = match stored.get("rel_kind")?.as_str()? {
        "distance" => RecallRelevance::Distance(stored.get("rel")?.as_f64()?),
        "lexical" => RecallRelevance::LexicalMatch,
        _ => return None,
    };
    let terms = ActivationTerms {
        semantic: stored.get("sem").and_then(Value::as_f64),
        lexical: stored.get("lex")?.as_f64()?,
        reinforcement: stored.get("rf")?.as_f64()?,
        situation: stored.get("sit")?.as_f64()?,
        salience: stored.get("sal")?.as_f64()?,
        total: stored.get("a")?.as_f64()?,
    };
    let use_counts = stored.get("use").and_then(|u| {
        if u.is_null() {
            return None;
        }
        Some(PlannedUseCounts {
            offered: u.get("off")?.as_u64()?,
            opened: u.get("op")?.as_u64()?,
            marked: u.get("mk")?.as_u64()?,
        })
    });
    let cleared_bar = stored.get("bar_ok")?.as_bool()?;
    let rank = stored
        .get("rank")
        .and_then(Value::as_u64)
        .map(|r| r as usize);
    let offered = stored.get("off")?.as_bool()?;
    let drop_reason = stored
        .get("drop")
        .and_then(Value::as_str)
        .and_then(drop_reason_from_label);
    Some(PlannedCandidate {
        arm,
        id,
        relevance,
        terms,
        use_counts,
        cleared_bar,
        rank,
        offered,
        drop_reason,
    })
}

fn candidates_to_json(candidates: &[PlannedCandidate]) -> Value {
    Value::Array(candidates.iter().map(candidate_to_json).collect())
}

fn candidates_from_json(stored: &Value) -> Vec<PlannedCandidate> {
    stored
        .as_array()
        .map(|array| array.iter().filter_map(candidate_from_json).collect())
        .unwrap_or_default()
}

// ---------- JSON mapping: opened ----------

/// Cut `opened` to [`MAX_OPENED_ENTRIES`] if it is somehow already past the
/// cap by the time `record` writes it. Logged, not silent: unlike
/// `candidates`, the stored row carries no flag that says this array was
/// truncated, so a warning here is the only trace of it.
fn bounded_opened(opened: &[String]) -> &[String] {
    if opened.len() > MAX_OPENED_ENTRIES {
        tracing::warn!(
            len = opened.len(),
            cap = MAX_OPENED_ENTRIES,
            "a plan's opened array arrived past its cap and was cut on write; \
             every writer today appends at most one id per call through \
             append_opened, which enforces the same cap, so this path is not \
             expected to fire"
        );
        &opened[..MAX_OPENED_ENTRIES]
    } else {
        opened
    }
}

fn opened_to_json(opened: &[String]) -> Value {
    Value::Array(opened.iter().map(|id| Value::String(id.clone())).collect())
}

fn opened_from_json(stored: &Value) -> Vec<String> {
    stored
        .as_array()
        .map(|array| {
            array
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

// ---------- row mapping ----------

fn row_to_plan(row: &PgRow) -> Result<ContextPlan, CoreError> {
    let read = |column: &str, cause: sqlx::Error| -> CoreError {
        CoreError::Storage(format!("malformed context_plans row: column {column}: {cause}"))
    };
    let weights_json: Value = row.try_get("weights").map_err(|e| read("weights", e))?;
    let arms_json: Value = row.try_get("arms").map_err(|e| read("arms", e))?;
    let candidates_json: Value = row
        .try_get("candidates")
        .map_err(|e| read("candidates", e))?;
    let opened_json: Value = row.try_get("opened").map_err(|e| read("opened", e))?;
    let considered_count: i32 = row
        .try_get("considered_count")
        .map_err(|e| read("considered_count", e))?;
    let recorded_at: chrono::DateTime<chrono::Utc> = row
        .try_get("recorded_at")
        .map_err(|e| read("recorded_at", e))?;
    Ok(ContextPlan {
        request_id: row.try_get("request_id").map_err(|e| read("request_id", e))?,
        conversation_id: row
            .try_get("conversation_id")
            .map_err(|e| read("conversation_id", e))?,
        recall_ran: row.try_get("recall_ran").map_err(|e| read("recall_ran", e))?,
        query_text: row.try_get("query_text").map_err(|e| read("query_text", e))?,
        query_text_truncated: row
            .try_get("query_text_truncated")
            .map_err(|e| read("query_text_truncated", e))?,
        bar: row.try_get("bar").map_err(|e| read("bar", e))?,
        weights: weights_from_json(&weights_json),
        scorer_version: row
            .try_get("scorer_version")
            .map_err(|e| read("scorer_version", e))?,
        arms: arms_from_json(&arms_json),
        candidates: candidates_from_json(&candidates_json),
        considered_count: considered_count.max(0) as usize,
        truncated: row.try_get("truncated").map_err(|e| read("truncated", e))?,
        opened: opened_from_json(&opened_json),
        recorded_at: Some(recorded_at.to_rfc3339()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::knowledge_use::UseScoreWeights as UseScoreWeightsDomain;

    #[test]
    fn every_drop_reason_variant_survives_its_own_label_round_trip() {
        // The six variants that exist in the enum today, not the four the
        // design's own table lists - see `PlannedDropReason` in
        // `crates/core/src/ports/context_plan.rs`. A variant this test does
        // not name is a variant this file's `match` in `drop_reason_label`
        // would still have to cover (a `match` with no wildcard fails to
        // compile otherwise), but the round trip is what proves the label
        // and the read agree.
        for reason in [
            PlannedDropReason::WidthCap,
            PlannedDropReason::Pinned,
            PlannedDropReason::InView,
            PlannedDropReason::IdUnrenderable,
            PlannedDropReason::EmptyContent,
            PlannedDropReason::ExternalContent,
        ] {
            let label = drop_reason_label(reason);
            assert_eq!(
                drop_reason_from_label(label),
                Some(reason),
                "`{label}` did not read back as {reason:?}"
            );
        }
    }

    #[test]
    fn every_recall_arm_survives_its_own_label_round_trip() {
        for arm in [RecallArm::Entry, RecallArm::Note, RecallArm::Skill] {
            assert_eq!(arm_from_label(arm_label(arm)), Some(arm));
        }
    }

    #[test]
    fn weights_round_trip_through_json_when_they_differ_from_the_default() {
        let written = ActivationWeights {
            use_lift: 0.83,
            use_score: UseScoreWeightsDomain {
                decay: 0.61,
                model_mark: 2.5,
                person_mark: 9.25,
            },
        };
        let json = weights_to_json(&written);
        let read = weights_from_json(&json);
        assert_eq!(read, written);
    }

    #[test]
    fn a_dispersion_round_trips_exactly_for_clean_values() {
        // `distance_at` recovers median and deviation through arithmetic
        // rather than a private field read (see the module header), so this
        // pins that the recovery is exact for the values a real turn's
        // dispersion is built from - not merely close.
        let written = RecallDispersion::assumed(12.0, 3.0);
        let json = dispersion_to_json(written);
        let read = dispersion_from_json(&json);
        assert_eq!(read, written);
    }

    #[test]
    fn a_candidate_round_trips_every_field_including_a_lexical_relevance_with_no_semantic_term() {
        let written = PlannedCandidate {
            arm: RecallArm::Skill,
            id: "skill-1".to_string(),
            relevance: RecallRelevance::LexicalMatch,
            terms: ActivationTerms {
                semantic: None,
                lexical: 0.0,
                reinforcement: 1.5,
                situation: 0.25,
                salience: 0.0,
                total: 1.75,
            },
            use_counts: None,
            cleared_bar: true,
            rank: None,
            offered: false,
            drop_reason: Some(PlannedDropReason::ExternalContent),
        };
        let json = candidate_to_json(&written);
        let read = candidate_from_json(&json).expect("a well-formed candidate parses");
        assert_eq!(read, written);
    }

    #[test]
    fn a_candidate_missing_a_required_field_is_skipped_rather_than_read_as_a_default() {
        let mut json = candidate_to_json(&PlannedCandidate {
            arm: RecallArm::Entry,
            id: "kb-1".to_string(),
            relevance: RecallRelevance::Distance(0.4),
            terms: ActivationTerms {
                semantic: Some(9.0),
                lexical: 0.0,
                reinforcement: 0.0,
                situation: 0.0,
                salience: 0.0,
                total: 9.0,
            },
            use_counts: None,
            cleared_bar: true,
            rank: Some(0),
            offered: true,
            drop_reason: None,
        });
        json.as_object_mut().expect("object").remove("a");
        assert!(
            candidate_from_json(&json).is_none(),
            "a candidate with no total activation must not silently read as 0.0"
        );
    }

    #[test]
    fn a_page_is_capped_however_many_plans_a_caller_asks_for() {
        assert_eq!(page_limit(u32::MAX), MAX_PLANS_PER_PAGE);
        assert_eq!(page_limit(10), 10);
        assert_eq!(page_limit(0), 0);
    }
}
