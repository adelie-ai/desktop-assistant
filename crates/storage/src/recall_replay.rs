//! The recall snapshot, the labelled set, and the replay that runs one
//! against the other (#1328).
//!
//! **Specification stub.** Every item below has its final signature so
//! `crates/storage/tests/recall_replay.rs` compiles and names the behaviour
//! this module must have; the bodies are placeholders that do not yet do
//! that work, so the suite is red for that reason and not for a missing
//! symbol. The real implementation follows in the next commit.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::activation::ActivationTerms;
use sqlx::PgPool;

/// Below this many active cases, a replay report states plainly that the set
/// is too small to generalise from (#1328's own discipline).
pub const SMALL_SET_CASE_THRESHOLD: usize = 16;

/// The sentence a replay report prints below [`SMALL_SET_CASE_THRESHOLD`]
/// cases.
pub const SMALL_SET_NOTICE: &str = "regression suite, not a fitting corpus: this set can show a \
     change made things worse; it cannot justify a coefficient";

/// How many of the top-ranked candidates a case's report carries for display.
pub const REPLAY_TOP_CANDIDATES_SHOWN: usize = 5;

/// The manifest row for one frozen corpus (#1328).
#[derive(Debug, Clone)]
pub struct SnapshotManifest {
    pub id: String,
    pub name: String,
    pub taken_at: DateTime<Utc>,
    pub embedding_model: String,
    pub entry_count: i32,
    pub use_count: i32,
    pub excluded_count: i32,
}

/// One row of a labelled case as read back from storage.
#[derive(Debug, Clone)]
pub struct CaseRecord {
    pub id: String,
    pub query_text: String,
    pub expected_entry_id: String,
    pub source_request_id: Option<String>,
    pub note: Option<String>,
    pub added_at: DateTime<Utc>,
    pub active: bool,
    pub baseline_snapshot_id: Option<String>,
}

/// What `add_case` needs to write one labelled case.
pub struct CaseInput<'a> {
    pub query_text: &'a str,
    pub expected_entry_id: &'a str,
    pub source_request_id: Option<&'a str>,
    pub note: Option<&'a str>,
    pub baseline_snapshot_id: Option<&'a str>,
}

/// One entry a snapshot's ranked scan measured, nearest first.
#[derive(Debug, Clone)]
pub struct RankedSnapshotEntry {
    pub entry_id: String,
    pub content: String,
    pub distance: f64,
    pub cleared_bar: bool,
    pub terms: ActivationTerms,
}

/// What replay found for one case.
#[derive(Debug, Clone)]
pub enum CaseOutcome {
    Ranked {
        rank: usize,
        total_candidates: usize,
        cleared_bar: bool,
        top: Vec<RankedSnapshotEntry>,
    },
    ExpectedEntryMissing {
        total_candidates: usize,
    },
}

/// One case's replay result.
#[derive(Debug, Clone)]
pub struct CaseReplayResult {
    pub case_id: String,
    pub query_text: String,
    pub expected_entry_id: String,
    pub outcome: CaseOutcome,
}

/// What one replay run answers (#1328).
#[derive(Debug, Clone)]
pub struct ReplayReport {
    pub snapshot_id: String,
    pub case_count: usize,
    pub too_small_to_generalize: bool,
    pub results: Vec<CaseReplayResult>,
}

/// Freeze this user's knowledge base and use history into a new snapshot.
///
/// Stub: builds a manifest with none of the real fields (majority-model
/// selection, exclusion counting) and writes nothing to any table.
pub async fn take_snapshot(
    _pool: &PgPool,
    _user_id: &str,
    name: &str,
) -> Result<SnapshotManifest, CoreError> {
    Ok(SnapshotManifest {
        id: "stub-snapshot".to_string(),
        name: name.to_string(),
        taken_at: Utc::now(),
        embedding_model: "stub-model".to_string(),
        entry_count: 0,
        use_count: 0,
        excluded_count: 0,
    })
}

/// Read one snapshot's manifest, or `None` where it does not exist.
///
/// Stub: never reads storage, so it always answers `None`.
pub async fn get_snapshot(
    _pool: &PgPool,
    _user_id: &str,
    _snapshot_id: &str,
) -> Result<Option<SnapshotManifest>, CoreError> {
    Ok(None)
}

/// Drop a snapshot, refusing when a labelled case's baseline still names it.
///
/// Stub: never refuses, because it never checks.
pub async fn drop_snapshot(
    _pool: &PgPool,
    _user_id: &str,
    _snapshot_id: &str,
) -> Result<(), CoreError> {
    Ok(())
}

/// Add one labelled case, refusing one with neither a source turn nor a note.
///
/// Stub: accepts a fixed id and never checks traceability, never writes to
/// storage.
pub async fn add_case(
    _pool: &PgPool,
    _user_id: &str,
    _input: CaseInput<'_>,
) -> Result<String, CoreError> {
    Ok("stub-case".to_string())
}

/// Seed a case from a stored turn's context plan.
///
/// Stub: always reports no stored plan.
pub async fn case_from_turn(
    _pool: &PgPool,
    _user_id: &str,
    request_id: &str,
    _expected_entry_id: &str,
) -> Result<String, CoreError> {
    Err(CoreError::InvalidInput {
        code: "recall_case_no_stored_plan",
        description: format!("stub: no context plan lookup implemented for {request_id}"),
        message: "case_from_turn is not yet implemented".to_string(),
    })
}

/// Every active case for `user_id`, oldest first.
///
/// Stub: never reads storage, so it always answers empty.
pub async fn list_active_cases(_pool: &PgPool, _user_id: &str) -> Result<Vec<CaseRecord>, CoreError> {
    Ok(Vec::new())
}

/// Anchor a case's baseline to a snapshot.
///
/// Stub: never writes to storage.
pub async fn set_case_baseline(
    _pool: &PgPool,
    _user_id: &str,
    _case_id: &str,
    _snapshot_id: &str,
) -> Result<(), CoreError> {
    Ok(())
}

/// The cached embedding of a case's query text under `embedding_model`.
///
/// Stub: never reads storage, so it always answers `None`.
pub async fn get_cached_case_embedding(
    _pool: &PgPool,
    _user_id: &str,
    _case_id: &str,
    _embedding_model: &str,
) -> Result<Option<Vec<f32>>, CoreError> {
    Ok(None)
}

/// Cache a case's query embedding under `embedding_model`.
///
/// Stub: never writes to storage.
pub async fn cache_case_embedding(
    _pool: &PgPool,
    _user_id: &str,
    _case_id: &str,
    _embedding_model: &str,
    _embedding: Vec<f32>,
) -> Result<(), CoreError> {
    Ok(())
}

/// Rank every entry in `snapshot_id` against `query_embedding`, nearest
/// first.
///
/// Stub: never reads storage, so it always answers an empty ranking.
pub async fn ranked_snapshot_scan(
    _pool: &PgPool,
    _user_id: &str,
    _snapshot_id: &str,
    _query_embedding: Vec<f32>,
    _now: DateTime<Utc>,
) -> Result<Vec<RankedSnapshotEntry>, CoreError> {
    Ok(Vec::new())
}

/// Replay every active case for `user_id` against `snapshot`.
///
/// Stub: never refuses a model mismatch and always answers an empty report,
/// because [`list_active_cases`] never has anything to give it.
pub async fn run_replay(
    pool: &PgPool,
    user_id: &str,
    snapshot: &SnapshotManifest,
) -> Result<ReplayReport, CoreError> {
    let cases = list_active_cases(pool, user_id).await?;
    Ok(ReplayReport {
        snapshot_id: snapshot.id.clone(),
        case_count: cases.len(),
        too_small_to_generalize: cases.len() < SMALL_SET_CASE_THRESHOLD,
        results: Vec::new(),
    })
}
