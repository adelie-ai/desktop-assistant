//! The recall snapshot, the labelled set, and the replay that runs one
//! against the other (#1328).
//!
//! Retrieval ranks knowledge entries by an ACT-R-style activation score, and
//! it is in daily use, but nothing before this module could say whether a
//! change to that score helped. Two things stood in the way. Background
//! writers (consolidation, extraction, embedding backfill) run continuously,
//! so the store measured today is not the store measured tomorrow, and any
//! difference between two rankings could be the corpus moving rather than
//! the change under test. Nothing recorded that a query had a known-correct
//! answer, so "recall improved" could be felt but never stated.
//!
//! A snapshot answers the first problem: a frozen copy of the knowledge
//! store and its use history, in tables nothing ever updates. A labelled
//! case answers the second: a query, the entry that should win, and where
//! that expectation came from. Replay runs the same public ranking function
//! the live `[Recall]` block uses
//! ([`rank_by_activation_traced`](desktop_assistant_core::ports::recall::rank_by_activation_traced))
//! against the frozen tables, with `now` fixed to the snapshot's own
//! `taken_at` rather than the wall clock — the reinforcement term decays with
//! time, so an unfrozen replay would give a different rank on every run and
//! the determinism this module exists to hold would fail for a true reason.
//!
//! **This is a regression suite, not a fitting corpus.** The labelled set
//! starts small and self-selected — seeded from real failures, one case per
//! diagnosed problem. That is enough to show a change made a known case
//! worse. It is not enough to justify a coefficient: a handful of cases from
//! one failure mode overfits on the first attempt. [`ReplayReport`] states
//! this on every run below [`SMALL_SET_CASE_THRESHOLD`] cases, in the report
//! itself rather than in a document a later reader may not open.
//!
//! ## Why an entry never disappears quietly
//!
//! A case's expected entry can stop existing in a snapshot — deleted before
//! the snapshot was taken, or excluded because it was embedded under a
//! different model. [`CaseOutcome::ExpectedEntryMissing`] reports that case
//! rather than dropping it from [`ReplayReport::results`]: a vanished ground
//! truth is a finding about the corpus, not a gap to paper over. The same
//! rule holds for the report as a whole —
//! [`ReplayReport::results`] always holds exactly one entry per active case,
//! never fewer, so a reader can never mistake a partial run for a complete
//! one.
//!
//! ## Distance parity
//!
//! The ranked scan in [`ranked_snapshot_scan`] reads `chunk <=> $1` over
//! `unnest(embedding)`, the same pgvector cosine-distance expression and the
//! same per-entry `MIN` and store-wide median/median-absolute-deviation CTEs
//! `NEAREST_BY_EMBEDDING_SQL` in `crate::knowledge` uses for the live
//! `[Recall]` block's vector arm — copied, not re-derived, because a replay
//! that measured distance its own way would diverge from the thing it is
//! meant to measure. The one intentional difference: the live query also
//! tolerates a source mid-migration between two embedding models
//! (`knowledge.rs`'s `split_part` fallback); a snapshot has already resolved
//! that at take-time (§ [`take_snapshot`]), so every row in
//! `recall_snapshot_entries` carries one model and the fallback has nothing
//! to do. `replay_ranks_a_seeded_corpus_exactly_as_the_live_scan_ranks_it`
//! (in `crates/storage/tests/recall_replay.rs`) pins the claim behaviourally:
//! it ranks one seeded corpus through both queries and asserts identical
//! order, which is the enforcement that matters — two SQL strings that read
//! alike can still diverge if pgvector's own semantics ever change under one
//! of them and not the other.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::activation::{ActivationTerms, LexicalMatch, NO_SITUATION};
use desktop_assistant_core::domain::knowledge_use::{
    KnowledgeMark, KnowledgeUseRecord, MarkPolarity, MarkSource,
};
use desktop_assistant_core::domain::salience::{SalienceReading, SalienceSource};
use desktop_assistant_core::domain::situation::SituationCue;
use desktop_assistant_core::domain::{Disposition, KnowledgeEntry};
use desktop_assistant_core::ports::recall::{
    Activatable, MixedSet, RecallDispersion, RecallRelevance, rank_by_activation_traced,
};
use desktop_assistant_core::recall::{RECALL_ASSUMED_DISPERSION, RECALL_BAR};
use pgvector::Vector;
use sqlx::PgPool;
use sqlx::types::Json;

/// How long a snapshot-take or a replay scan may run before the database
/// gives up on it.
///
/// Both are operator-invoked, offline of any turn, so the bound is generous
/// rather than the sub-second ceiling the per-turn recall paths carry
/// ([`crate::RECALL_SCAN_STATEMENT_TIMEOUT`]) — but it is still bounded,
/// because an unbounded scan against a large store would otherwise hold a
/// connection and the server's CPU for as long as an operator was willing to
/// wait, with no way to tell a slow corpus from a wedged one.
const RECALL_REPLAY_STATEMENT_TIMEOUT: Duration = Duration::from_secs(30);

/// Below this many active cases, [`ReplayReport`] states plainly that the set
/// is too small to generalise from (#1328's own discipline). Not a cutoff —
/// a replay still runs and reports every case — a caption that travels with
/// the numbers so nobody quotes an average over a handful of cases as a
/// benchmark.
pub const SMALL_SET_CASE_THRESHOLD: usize = 16;

/// The sentence [`ReplayReport`] prints below [`SMALL_SET_CASE_THRESHOLD`]
/// cases, verbatim, so it reads the same in every report rather than being
/// reworded at each call site.
pub const SMALL_SET_NOTICE: &str = "regression suite, not a fitting corpus: this set can show a \
     change made things worse; it cannot justify a coefficient";

// ---------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------

/// The manifest row for one frozen corpus (#1328).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SnapshotManifest {
    pub id: String,
    pub name: String,
    pub taken_at: DateTime<Utc>,
    /// The embedding model every row in `recall_snapshot_entries` carries.
    /// Part of the snapshot's identity — [`run_replay`] refuses a case with
    /// no cached embedding under this exact model rather than comparing
    /// vectors from two different spaces.
    pub embedding_model: String,
    pub entry_count: i32,
    pub use_count: i32,
    /// Knowledge rows read at take-time whose own `embedding_model` did not
    /// match the majority, and so were left out. Never silently absorbed
    /// into `entry_count`.
    pub excluded_count: i32,
}

#[derive(sqlx::FromRow)]
struct SourceEntryRow {
    id: String,
    content: String,
    tags: Vec<String>,
    embedding: Option<Vec<Vector>>,
    embedding_model: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    source: Option<String>,
    summary: Option<String>,
    disposition: String,
}

#[derive(sqlx::FromRow)]
struct SourceUseRow {
    entry_id: String,
    offered_count: i64,
    opened_count: i64,
    marked_count: i64,
    first_seen_at: DateTime<Utc>,
    last_offered_at: Option<DateTime<Utc>>,
    recent_uses: Vec<DateTime<Utc>>,
}

#[derive(sqlx::FromRow)]
struct SourceMarkRow {
    entry_id: String,
    marked_by: String,
    polarity: String,
    reason: Option<String>,
    marked_at: DateTime<Utc>,
}

/// One frozen mark, the shape `recall_snapshot_uses.marks` stores.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StoredMark {
    source: String,
    polarity: String,
    reason: Option<String>,
    marked_at: DateTime<Utc>,
}

/// Freeze this user's knowledge base and use history into a new snapshot
/// (#1328).
///
/// Reads every live (`deleted_at IS NULL`) `knowledge_base` row for `user_id`,
/// picks the embedding model the most rows carry, and copies only the rows
/// that carry exactly that model into `recall_snapshot_entries` — a row
/// embedded under another model, or not embedded at all, cannot be compared
/// by the ranked scan and is excluded rather than included with a
/// meaningless distance. Ties in the majority count break on the
/// lexicographically smallest model name, so the choice is deterministic
/// rather than an artifact of row order.
///
/// Refuses when not one row carries any embedding model at all: a snapshot
/// with nothing to compare answers no question a replay could ask.
///
/// The use log (`knowledge_use_stats` + `knowledge_use_marks`) is copied for
/// exactly the included entries, into `recall_snapshot_uses`.
pub async fn take_snapshot(
    pool: &PgPool,
    user_id: &str,
    name: &str,
) -> Result<SnapshotManifest, CoreError> {
    let mut scan = crate::scan_bound::begin_bounded(pool, RECALL_REPLAY_STATEMENT_TIMEOUT).await?;

    let rows: Vec<SourceEntryRow> = sqlx::query_as(
        "SELECT id, content, tags, embedding, embedding_model, created_at, updated_at, \
                source, summary, disposition \
         FROM knowledge_base \
         WHERE user_id = $1 AND deleted_at IS NULL",
    )
    .bind(user_id)
    .fetch_all(&mut *scan)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for row in &rows {
        if let Some(model) = row.embedding_model.as_deref() {
            *counts.entry(model).or_default() += 1;
        }
    }
    let majority = counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(model, _)| model.to_string());

    let Some(majority) = majority else {
        return Err(CoreError::InvalidInput {
            code: "recall_snapshot_no_embedded_rows",
            description: "no knowledge_base row carries an embedding model to snapshot".to_string(),
            message: "there is nothing embedded to snapshot yet; run the embedding backfill \
                      first"
                .to_string(),
        });
    };

    let included: Vec<&SourceEntryRow> = rows
        .iter()
        .filter(|r| r.embedding_model.as_deref() == Some(majority.as_str()))
        .collect();
    let excluded_count = rows.len() - included.len();

    let snapshot_id = uuid::Uuid::now_v7().to_string();

    sqlx::query(
        "INSERT INTO recall_snapshots \
             (id, user_id, name, embedding_model, entry_count, use_count, excluded_count) \
         VALUES ($1, $2, $3, $4, $5, 0, $6)",
    )
    .bind(&snapshot_id)
    .bind(user_id)
    .bind(name)
    .bind(&majority)
    .bind(included.len() as i32)
    .bind(excluded_count as i32)
    .execute(&mut *scan)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    for row in &included {
        sqlx::query(
            "INSERT INTO recall_snapshot_entries \
                 (snapshot_id, user_id, entry_id, content, tags, embedding, embedding_model, \
                  created_at, updated_at, source, summary, disposition) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
        )
        .bind(&snapshot_id)
        .bind(user_id)
        .bind(&row.id)
        .bind(&row.content)
        .bind(&row.tags)
        .bind(&row.embedding)
        .bind(&row.embedding_model)
        .bind(row.created_at)
        .bind(row.updated_at)
        .bind(&row.source)
        .bind(&row.summary)
        .bind(&row.disposition)
        .execute(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    }

    let included_ids: Vec<String> = included.iter().map(|r| r.id.clone()).collect();
    let mut use_count = 0i32;
    if !included_ids.is_empty() {
        let stats: Vec<SourceUseRow> = sqlx::query_as(
            "SELECT entry_id, offered_count, opened_count, marked_count, first_seen_at, \
                    last_offered_at, recent_uses \
             FROM knowledge_use_stats \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id)
        .bind(&included_ids)
        .fetch_all(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let marks: Vec<SourceMarkRow> = sqlx::query_as(
            "SELECT entry_id, marked_by, polarity, reason, marked_at \
             FROM knowledge_use_marks \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id)
        .bind(&included_ids)
        .fetch_all(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut marks_by_entry: HashMap<&str, Vec<StoredMark>> = HashMap::new();
        for m in &marks {
            marks_by_entry
                .entry(m.entry_id.as_str())
                .or_default()
                .push(StoredMark {
                    source: m.marked_by.clone(),
                    polarity: m.polarity.clone(),
                    reason: m.reason.clone(),
                    marked_at: m.marked_at,
                });
        }

        for stat in &stats {
            let marks_json = marks_by_entry
                .get(stat.entry_id.as_str())
                .cloned()
                .unwrap_or_default();
            sqlx::query(
                "INSERT INTO recall_snapshot_uses \
                     (snapshot_id, user_id, entry_id, offered_count, opened_count, \
                      marked_count, first_seen_at, last_offered_at, recent_uses, marks) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            )
            .bind(&snapshot_id)
            .bind(user_id)
            .bind(&stat.entry_id)
            .bind(stat.offered_count)
            .bind(stat.opened_count)
            .bind(stat.marked_count)
            .bind(stat.first_seen_at)
            .bind(stat.last_offered_at)
            .bind(&stat.recent_uses)
            .bind(Json(marks_json))
            .execute(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
            use_count += 1;
        }
    }

    sqlx::query("UPDATE recall_snapshots SET use_count = $1 WHERE id = $2 AND user_id = $3")
        .bind(use_count)
        .bind(&snapshot_id)
        .bind(user_id)
        .execute(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    scan.commit()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(SnapshotManifest {
        id: snapshot_id,
        name: name.to_string(),
        taken_at: Utc::now(),
        embedding_model: majority,
        entry_count: included.len() as i32,
        use_count,
        excluded_count: excluded_count as i32,
    })
}

/// Read one snapshot's manifest, or `None` where it does not exist for this
/// user.
pub async fn get_snapshot(
    pool: &PgPool,
    user_id: &str,
    snapshot_id: &str,
) -> Result<Option<SnapshotManifest>, CoreError> {
    sqlx::query_as(
        "SELECT id, name, taken_at, embedding_model, entry_count, use_count, excluded_count \
         FROM recall_snapshots WHERE user_id = $1 AND id = $2",
    )
    .bind(user_id)
    .bind(snapshot_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))
}

/// Drop a snapshot, refusing when any labelled case's `baseline_snapshot_id`
/// still names it (#1328).
///
/// Restore is deliberately absent from this module: replay reads snapshot
/// tables directly and nothing ever loads a snapshot back over the live
/// store, so there is no destructive restore path to guard — the refusal
/// here is the only safety property a drop needs. The refusal itself is the
/// database's own: `recall_cases.baseline_snapshot_id` references
/// `recall_snapshots(id)` with no `ON DELETE` clause, so Postgres declines
/// the delete on its own and this function only turns that into a message
/// naming the reason.
pub async fn drop_snapshot(
    pool: &PgPool,
    user_id: &str,
    snapshot_id: &str,
) -> Result<(), CoreError> {
    let result = sqlx::query("DELETE FROM recall_snapshots WHERE user_id = $1 AND id = $2")
        .bind(user_id)
        .bind(snapshot_id)
        .execute(pool)
        .await;

    match result {
        Ok(outcome) if outcome.rows_affected() == 1 => Ok(()),
        Ok(_) => Err(CoreError::InvalidInput {
            code: "recall_snapshot_not_found",
            description: format!("no snapshot {snapshot_id} for this user"),
            message: format!("snapshot {snapshot_id} does not exist"),
        }),
        Err(sqlx::Error::Database(db_err)) if db_err.is_foreign_key_violation() => {
            Err(CoreError::InvalidInput {
                code: "recall_snapshot_has_baseline_cases",
                description: format!(
                    "snapshot {snapshot_id} is still named by a case's baseline_snapshot_id"
                ),
                message: format!(
                    "snapshot {snapshot_id} cannot be dropped: at least one labelled case's \
                     baseline still names it"
                ),
            })
        }
        Err(e) => Err(CoreError::Storage(e.to_string())),
    }
}

// ---------------------------------------------------------------------
// Labelled cases
// ---------------------------------------------------------------------

/// One row of a labelled case as read back from storage.
#[derive(Debug, Clone, sqlx::FromRow)]
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
    /// The snapshot this case's "currently gets" rank is anchored to, where
    /// one is already known. `None` for a case added before any replay has
    /// run against it.
    pub baseline_snapshot_id: Option<&'a str>,
}

/// Add one labelled case (#1328).
///
/// **Refuses a case with neither `source_request_id` nor `note`.** A case
/// nobody can trace back to a real failure is an invented one, and an
/// invented case is how a regression suite becomes a mirror that agrees with
/// whatever change produced it. The same rule is also the database's own
/// (`recall_cases_traceable`, migration 060) — this check exists so the
/// refusal reads as a stated reason rather than a raw constraint violation.
pub async fn add_case(
    pool: &PgPool,
    user_id: &str,
    input: CaseInput<'_>,
) -> Result<String, CoreError> {
    if input.source_request_id.is_none() && input.note.is_none() {
        return Err(CoreError::InvalidInput {
            code: "recall_case_not_traceable",
            description: "a case with neither source_request_id nor note was refused".to_string(),
            message: "a case must carry the turn it came from (--from-turn) or a note stating \
                      why there is none — an untraceable case is how a regression suite \
                      becomes a mirror"
                .to_string(),
        });
    }

    let id = uuid::Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO recall_cases \
             (id, user_id, query_text, expected_entry_id, source_request_id, note, \
              baseline_snapshot_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(&id)
    .bind(user_id)
    .bind(input.query_text)
    .bind(input.expected_entry_id)
    .bind(input.source_request_id)
    .bind(input.note)
    .bind(input.baseline_snapshot_id)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(id)
}

/// Seed a case from a stored turn's context plan (#1328's `--from-turn`
/// seam), reading `context_plans.query_text` for `(user_id, request_id)`.
///
/// `context_plans` is written by a sibling unit of this same epic (#1327's
/// persistence). Where it has not been deployed yet, Postgres answers
/// "relation does not exist" and this function turns that into a plain
/// message rather than a raw SQL error, so the missing dependency is
/// diagnosable rather than confusing.
pub async fn case_from_turn(
    pool: &PgPool,
    user_id: &str,
    request_id: &str,
    expected_entry_id: &str,
) -> Result<String, CoreError> {
    let query_text: Option<(Option<String>,)> = sqlx::query_as(
        "SELECT query_text FROM context_plans WHERE user_id = $1 AND request_id = $2",
    )
    .bind(user_id)
    .bind(request_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        let text = e.to_string();
        if text.contains("does not exist") {
            CoreError::Storage(format!(
                "--from-turn requires the context_plans table (#1327's turn-plan \
                 persistence), which is not deployed on this database: {text}"
            ))
        } else {
            CoreError::Storage(text)
        }
    })?;

    let query_text = match query_text {
        Some((Some(text),)) => text,
        _ => {
            return Err(CoreError::InvalidInput {
                code: "recall_case_no_stored_plan",
                description: format!("no context plan with a query for request {request_id}"),
                message: format!(
                    "turn {request_id} has no stored plan with a query to seed a case from"
                ),
            });
        }
    };

    add_case(
        pool,
        user_id,
        CaseInput {
            query_text: &query_text,
            expected_entry_id,
            source_request_id: Some(request_id),
            note: None,
            baseline_snapshot_id: None,
        },
    )
    .await
}

/// Every active case for `user_id`, oldest first.
pub async fn list_active_cases(pool: &PgPool, user_id: &str) -> Result<Vec<CaseRecord>, CoreError> {
    sqlx::query_as(
        "SELECT id, query_text, expected_entry_id, source_request_id, note, added_at, active, \
                baseline_snapshot_id \
         FROM recall_cases WHERE user_id = $1 AND active ORDER BY added_at",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))
}

/// Anchor a case's baseline to a snapshot, recording the substrate its
/// "currently gets" rank is measured against.
pub async fn set_case_baseline(
    pool: &PgPool,
    user_id: &str,
    case_id: &str,
    snapshot_id: &str,
) -> Result<(), CoreError> {
    sqlx::query("UPDATE recall_cases SET baseline_snapshot_id = $1 WHERE user_id = $2 AND id = $3")
        .bind(snapshot_id)
        .bind(user_id)
        .bind(case_id)
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------
// Case query embeddings, cached per model
// ---------------------------------------------------------------------

/// The cached embedding of a case's query text under `embedding_model`, or
/// `None` where none has been cached yet.
///
/// Caching is what makes
/// `replaying_a_set_against_a_snapshot_twice_gives_identical_ranks` true by
/// construction: the second replay reads the same vector the first one
/// wrote, rather than trusting the embedder to answer the same text
/// bit-for-bit twice.
pub async fn get_cached_case_embedding(
    pool: &PgPool,
    user_id: &str,
    case_id: &str,
    embedding_model: &str,
) -> Result<Option<Vec<f32>>, CoreError> {
    let row: Option<(Vector,)> = sqlx::query_as(
        "SELECT embedding FROM recall_case_embeddings \
         WHERE user_id = $1 AND case_id = $2 AND embedding_model = $3",
    )
    .bind(user_id)
    .bind(case_id)
    .bind(embedding_model)
    .fetch_optional(pool)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(row.map(|(v,)| v.to_vec()))
}

/// Cache a case's query embedding under `embedding_model`. Idempotent: a
/// second embed of the same case under the same model overwrites with
/// (bit-for-bit or not) the same meaning rather than adding a second row.
pub async fn cache_case_embedding(
    pool: &PgPool,
    user_id: &str,
    case_id: &str,
    embedding_model: &str,
    embedding: Vec<f32>,
) -> Result<(), CoreError> {
    sqlx::query(
        "INSERT INTO recall_case_embeddings (case_id, user_id, embedding_model, embedding) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (case_id, embedding_model) \
         DO UPDATE SET embedding = EXCLUDED.embedding, cached_at = NOW()",
    )
    .bind(case_id)
    .bind(user_id)
    .bind(embedding_model)
    .bind(Vector::from(embedding))
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------
// The ranked scan over a snapshot
// ---------------------------------------------------------------------

/// The distance CTE over `recall_snapshot_entries`, copied from
/// `NEAREST_BY_EMBEDDING_SQL` in `crate::knowledge` — same `chunk <=> $1`
/// cosine-distance operator, same per-entry `MIN`, same store-wide
/// median/median-absolute-deviation shape. See the module doc's "Distance
/// parity" section for what is deliberately different and why.
const SNAPSHOT_NEAREST_SQL: &str = "\
    WITH d AS (
         SELECT entry_id AS id, MIN(chunk <=> $1) AS distance
         FROM recall_snapshot_entries, unnest(embedding) AS chunk
         WHERE user_id = $2
           AND snapshot_id = $3
           AND embedding IS NOT NULL
         GROUP BY entry_id
     ),
     m AS (
         SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS median,
                count(*) AS rows_read
         FROM d
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - m.median))
                    AS deviation
         FROM d CROSS JOIN m
         GROUP BY m.median, m.rows_read
     )
     SELECT e.entry_id AS id, e.content, e.tags, e.created_at, e.updated_at, e.source, \
            e.summary, e.disposition,
            d.distance, s.median, s.rows_read, s.deviation
     FROM d
     JOIN recall_snapshot_entries e ON e.entry_id = d.id AND e.user_id = $2 AND e.snapshot_id = $3
     CROSS JOIN s
     ORDER BY d.distance";

#[derive(sqlx::FromRow)]
struct SnapshotNearestRow {
    id: String,
    content: String,
    tags: Vec<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    source: Option<String>,
    summary: Option<String>,
    disposition: String,
    distance: f64,
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
}

/// One entry a snapshot's ranked scan measured, nearest first, and the
/// activation terms [`rank_by_activation_traced`] computed for it.
#[derive(Debug, Clone)]
pub struct RankedSnapshotEntry {
    pub entry_id: String,
    pub content: String,
    pub distance: f64,
    pub cleared_bar: bool,
    pub terms: ActivationTerms,
}

/// One row read from `recall_snapshot_entries`, ready to rank.
struct SnapshotCandidate {
    entry_id: String,
    content: String,
    distance: f64,
    entry: KnowledgeEntry,
    use_record: Option<KnowledgeUseRecord>,
}

impl Activatable for SnapshotCandidate {
    fn relevance(&self) -> RecallRelevance {
        RecallRelevance::Distance(self.distance)
    }

    fn use_record(&self) -> Option<&KnowledgeUseRecord> {
        self.use_record.as_ref()
    }

    /// Replay measures no present situation — it is retrospective, not a
    /// live turn — so every candidate answers [`NO_SITUATION`], which ranks
    /// exactly as it ranked before the situation term existed.
    fn situation_coverage(&self, _cue: Option<&SituationCue>) -> f64 {
        NO_SITUATION
    }

    fn salience_share(&self) -> f64 {
        SalienceReading::read(&SalienceSource::of(&self.entry)).share()
    }

    /// A snapshot's ranked scan carries no full-text arm, on the same terms
    /// the `[Recall]` block's own lookup does.
    fn lexical(&self) -> LexicalMatch {
        LexicalMatch::NONE
    }

    /// The entry's own frozen disposition (#893), read off the snapshot copy
    /// exactly as the live path reads it off the live row — a `trivial` entry
    /// costs the same penalty here as it does in production, so a replay
    /// cannot show a rank the live block would not have produced.
    fn disposition(&self) -> Disposition {
        self.entry.disposition
    }
}

/// Rank every entry in `snapshot_id` against `query_embedding`, nearest
/// first, using the store's own [`rank_by_activation_traced`] with `now`
/// fixed to `now` (the snapshot's own `taken_at` in every caller but the
/// distance-parity test, which needs to compare against a live scan taken at
/// the same instant).
///
/// No `LIMIT`: a replay needs the exact rank of one named entry, which a
/// capped top-K scan could silently misreport as "not found" when it is
/// merely outside the cap. The scan reads the whole snapshot instead — sized
/// for a personal knowledge base, not a hot per-turn path.
pub async fn ranked_snapshot_scan(
    pool: &PgPool,
    user_id: &str,
    snapshot_id: &str,
    query_embedding: Vec<f32>,
    now: DateTime<Utc>,
) -> Result<Vec<RankedSnapshotEntry>, CoreError> {
    if query_embedding.is_empty() {
        return Ok(Vec::new());
    }
    let mut scan = crate::scan_bound::begin_bounded(pool, RECALL_REPLAY_STATEMENT_TIMEOUT).await?;

    let rows: Vec<SnapshotNearestRow> = sqlx::query_as(SNAPSHOT_NEAREST_SQL)
        .bind(Vector::from(query_embedding))
        .bind(user_id)
        .bind(snapshot_id)
        .fetch_all(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let dispersion = rows
        .first()
        .and_then(|r| {
            RecallDispersion::measured(r.median?, r.deviation?, r.rows_read.max(0) as usize)
        })
        .unwrap_or(RECALL_ASSUMED_DISPERSION);

    let entry_ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
    let use_records = if entry_ids.is_empty() {
        HashMap::new()
    } else {
        load_use_records(&mut scan, user_id, snapshot_id, &entry_ids).await?
    };

    scan.commit()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let candidates: Vec<SnapshotCandidate> = rows
        .into_iter()
        .map(|row| SnapshotCandidate {
            entry_id: row.id.clone(),
            content: row.content.clone(),
            distance: row.distance,
            use_record: use_records.get(&row.id).cloned(),
            entry: KnowledgeEntry {
                id: row.id,
                content: row.content,
                tags: row.tags,
                metadata: serde_json::Value::Null,
                created_at: row.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                updated_at: row.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                source: row.source,
                disposition: Disposition::parse(&row.disposition).unwrap_or(Disposition::Obsolete),
                summary: row.summary,
            },
        })
        .collect();

    let ranked =
        rank_by_activation_traced(candidates, |c| c, dispersion, None, now, MixedSet::Refuse);

    Ok(ranked
        .into_iter()
        .map(|(candidate, terms)| {
            let cleared_bar =
                RecallRelevance::Distance(candidate.distance).clears_bar(dispersion, RECALL_BAR);
            RankedSnapshotEntry {
                entry_id: candidate.entry_id,
                content: candidate.content,
                distance: candidate.distance,
                cleared_bar,
                terms,
            }
        })
        .collect())
}

async fn load_use_records(
    scan: &mut sqlx::Transaction<'static, sqlx::Postgres>,
    user_id: &str,
    snapshot_id: &str,
    entry_ids: &[String],
) -> Result<HashMap<String, KnowledgeUseRecord>, CoreError> {
    #[derive(sqlx::FromRow)]
    struct UseRow {
        entry_id: String,
        offered_count: i64,
        opened_count: i64,
        marked_count: i64,
        first_seen_at: DateTime<Utc>,
        last_offered_at: Option<DateTime<Utc>>,
        recent_uses: Vec<DateTime<Utc>>,
        marks: Json<Vec<StoredMark>>,
    }

    let rows: Vec<UseRow> = sqlx::query_as(
        "SELECT entry_id, offered_count, opened_count, marked_count, first_seen_at, \
                last_offered_at, recent_uses, marks \
         FROM recall_snapshot_uses \
         WHERE user_id = $1 AND snapshot_id = $2 AND entry_id = ANY($3)",
    )
    .bind(user_id)
    .bind(snapshot_id)
    .bind(entry_ids)
    .fetch_all(&mut **scan)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(rows
        .into_iter()
        .map(|row| {
            let marks = row
                .marks
                .0
                .into_iter()
                .filter_map(|m| {
                    Some(KnowledgeMark {
                        source: MarkSource::from_stored(&m.source)?,
                        polarity: MarkPolarity::from_stored(&m.polarity)?,
                        reason: m.reason,
                        marked_at: m.marked_at,
                    })
                })
                .collect();
            (
                row.entry_id.clone(),
                KnowledgeUseRecord {
                    entry_id: row.entry_id,
                    offered_count: row.offered_count.max(0) as u64,
                    opened_count: row.opened_count.max(0) as u64,
                    marked_count: row.marked_count.max(0) as u64,
                    first_seen_at: row.first_seen_at,
                    last_offered_at: row.last_offered_at,
                    recent_uses: row.recent_uses,
                    marks,
                },
            )
        })
        .collect())
}

// ---------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------

/// How many of the top-ranked candidates [`CaseOutcome::Ranked::top`] carries
/// for display. The rank number itself is exact and unbounded by this — this
/// only bounds how many rows a reader is shown alongside it, the same split
/// the `[Recall]` block keeps between what it measures and what it renders.
pub const REPLAY_TOP_CANDIDATES_SHOWN: usize = 5;

/// What replay found for one case.
#[derive(Debug, Clone)]
pub enum CaseOutcome {
    /// The expected entry was found in the snapshot and ranked.
    Ranked {
        /// 1-based position among every entry the scan ranked.
        rank: usize,
        total_candidates: usize,
        cleared_bar: bool,
        /// The nearest [`REPLAY_TOP_CANDIDATES_SHOWN`] candidates, for
        /// display — not a cap on `rank` or `total_candidates`, both of
        /// which are exact.
        top: Vec<RankedSnapshotEntry>,
    },
    /// The case's `expected_entry_id` is not among the snapshot's entries —
    /// deleted before the snapshot was taken, or excluded for carrying a
    /// different embedding model. Reported, not skipped.
    ExpectedEntryMissing { total_candidates: usize },
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
///
/// **`case_count` is read first, before any aggregate** — it is the first
/// field for exactly that reason: a caller building a report renders it
/// before anything else, so nobody quotes a number computed over cases
/// without first being told how many there were.
#[derive(Debug, Clone)]
pub struct ReplayReport {
    pub snapshot_id: String,
    pub case_count: usize,
    /// [`desktop_assistant_core::domain::activation::ACTIVATION_SCORER_VERSION`]
    /// at the moment this run computed its ranks (#893, #1327) — the same
    /// role the snapshot's `embedding_model` plays for the corpus, applied
    /// to the ranking function instead of the data.
    ///
    /// **Recorded on the result, not the manifest, and never refused.** The
    /// embedding model is part of what is *stored*: a vector from a
    /// different model is a different geometry, and comparing across models
    /// is not merely invalid, it is meaningless, so [`run_replay`] refuses
    /// outright. The scorer is part of what *reads* the store: it is the
    /// code running this call, not a property of `snapshot`, so pinning it
    /// to the manifest would conflate "which corpus" with "which build
    /// ranked it" and would make a snapshot stale the moment the scorer
    /// gained a term, which is not a data problem, it is business as usual.
    /// A single run under any scorer version is a fully honest computation
    /// of that version's ranks — refusing it would block a legitimate use
    /// (checking today's ranking against a frozen corpus). What *is* invalid
    /// is reading two reports as the same experiment when their scorer
    /// versions differ, so this field exists to make that visible rather
    /// than to make it impossible: the daemon's rendered report prints it
    /// beside `case_count`, on the same first line, so nobody compares two
    /// runs without seeing that the scorer moved.
    pub scorer_version: String,
    /// Set below [`SMALL_SET_CASE_THRESHOLD`] cases — see [`SMALL_SET_NOTICE`]
    /// for the sentence a caller should print alongside it.
    pub too_small_to_generalize: bool,
    /// Exactly one entry per active case that was replayed. Never fewer: a
    /// case this run could not embed under the snapshot's model is a refusal
    /// of the whole run ([`run_replay`]'s `Err`), not a silently shortened
    /// `results`.
    pub results: Vec<CaseReplayResult>,
}

/// Replay every active case for `user_id` against `snapshot`, with `now`
/// fixed to `snapshot.taken_at` (#1328).
///
/// Reads each case's query embedding only from `recall_case_embeddings`,
/// under `snapshot.embedding_model` exactly — this function calls no
/// embedder itself. **Refuses when any active case has no cached embedding
/// under the snapshot's model**: the vector needed to compare against the
/// frozen entries is one only that model can produce, and if the live
/// embedder has since moved on, the only way to compare is a vector cached
/// while that model was still current. The caller (the daemon's CLI wiring)
/// is what decides whether to embed a case fresh and cache it before calling
/// this — see `crates/daemon/src/recall_replay.rs`.
///
/// Deterministic by construction: the same cached vectors, the same
/// immutable snapshot tables, and the same frozen `now` mean two calls over
/// the same inputs read the same rows in the same order and rank them the
/// same way.
pub async fn run_replay(
    pool: &PgPool,
    user_id: &str,
    snapshot: &SnapshotManifest,
) -> Result<ReplayReport, CoreError> {
    let cases = list_active_cases(pool, user_id).await?;

    let mut missing_embeddings = Vec::new();
    let mut embeddings = HashMap::new();
    for case in &cases {
        match get_cached_case_embedding(pool, user_id, &case.id, &snapshot.embedding_model).await? {
            Some(vector) => {
                embeddings.insert(case.id.clone(), vector);
            }
            None => missing_embeddings.push(case.id.clone()),
        }
    }

    if !missing_embeddings.is_empty() {
        return Err(CoreError::InvalidInput {
            code: "recall_replay_embedding_model_mismatch",
            description: format!(
                "{} of {} active cases have no cached embedding under snapshot model {}",
                missing_embeddings.len(),
                cases.len(),
                snapshot.embedding_model
            ),
            message: format!(
                "this snapshot was taken under embedding model \"{}\"; {} case(s) have no \
                 query embedding cached under that model, and this run cannot compare a query \
                 embedded under a different model against it — embed the missing case(s) under \
                 \"{}\" first (the daemon does this automatically when its live embedder still \
                 matches the snapshot's model)",
                snapshot.embedding_model,
                missing_embeddings.len(),
                snapshot.embedding_model
            ),
        });
    }

    let mut results = Vec::with_capacity(cases.len());
    for case in &cases {
        let embedding = embeddings.remove(&case.id).unwrap_or_default();
        let ranked =
            ranked_snapshot_scan(pool, user_id, &snapshot.id, embedding, snapshot.taken_at).await?;
        let total_candidates = ranked.len();
        let outcome = match ranked
            .iter()
            .position(|r| r.entry_id == case.expected_entry_id)
        {
            Some(index) => CaseOutcome::Ranked {
                rank: index + 1,
                total_candidates,
                cleared_bar: ranked[index].cleared_bar,
                top: ranked
                    .into_iter()
                    .take(REPLAY_TOP_CANDIDATES_SHOWN)
                    .collect(),
            },
            None => CaseOutcome::ExpectedEntryMissing { total_candidates },
        };
        results.push(CaseReplayResult {
            case_id: case.id.clone(),
            query_text: case.query_text.clone(),
            expected_entry_id: case.expected_entry_id.clone(),
            outcome,
        });
    }

    Ok(ReplayReport {
        snapshot_id: snapshot.id.clone(),
        case_count: results.len(),
        scorer_version: desktop_assistant_core::domain::activation::ACTIVATION_SCORER_VERSION
            .to_string(),
        too_small_to_generalize: results.len() < SMALL_SET_CASE_THRESHOLD,
        results,
    })
}
