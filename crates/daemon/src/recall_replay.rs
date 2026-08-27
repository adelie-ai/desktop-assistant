//! Operator commands for the recall snapshot, the labelled set, and replay
//! (#1328): `--recall-snapshot`, `--recall-snapshot-drop`,
//! `--recall-case-add`, `--recall-replay`.
//!
//! The measurement itself — taking a snapshot, writing a case, ranking a
//! snapshot's entries — lives in `desktop_assistant_storage::recall_replay`
//! and knows nothing about an embedder or a CLI. This module is the thin
//! part: it reads arguments, resolves this daemon's live embedding backend
//! where one step needs it, and renders the report a person reads.
//!
//! Module name is `recall_replay` throughout, distinct from `replay_eval`
//! (#1209), which measures the context-window ladder and is a different
//! instrument.

use std::sync::Arc;

use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_storage::{
    CaseInput, CaseOutcome, PgPool, ReplayReport, SMALL_SET_NOTICE, SnapshotManifest,
};

/// `desktop-assistant --recall-snapshot <user-id> [name]`: freeze this
/// user's knowledge base and use history, print the manifest, and exit
/// without starting the daemon.
pub async fn run_snapshot(pool: &PgPool, user: &str, name: &str) -> anyhow::Result<String> {
    let manifest = desktop_assistant_storage::take_snapshot(pool, user, name).await?;
    Ok(render_manifest(&manifest))
}

/// `desktop-assistant --recall-snapshot-drop <user-id> <snapshot-id>`:
/// delete a snapshot, refusing while a labelled case's baseline still names
/// it.
pub async fn run_snapshot_drop(
    pool: &PgPool,
    user: &str,
    snapshot_id: &str,
) -> anyhow::Result<String> {
    desktop_assistant_storage::drop_snapshot(pool, user, snapshot_id).await?;
    Ok(format!("snapshot {snapshot_id} dropped"))
}

/// What `--recall-case-add` parsed from the command line.
pub enum CaseSource<'a> {
    /// `--query "..." [--note "..."]`.
    Query {
        query_text: &'a str,
        note: Option<&'a str>,
    },
    /// `--from-turn <request-id>`: the query comes from that turn's stored
    /// context plan.
    FromTurn { request_id: &'a str },
}

/// `desktop-assistant --recall-case-add <user-id> <expected-entry-id> ...`:
/// add one labelled case, either from an explicit query and note, or from a
/// stored turn's context plan (`--from-turn`) — the #1327/#1328 seam that
/// turns a diagnosed failure into a case with one command.
pub async fn run_case_add(
    pool: &PgPool,
    user: &str,
    expected_entry_id: &str,
    source: CaseSource<'_>,
    baseline_snapshot_id: Option<&str>,
) -> anyhow::Result<String> {
    let case_id = match source {
        CaseSource::Query { query_text, note } => {
            desktop_assistant_storage::add_case(
                pool,
                user,
                CaseInput {
                    query_text,
                    expected_entry_id,
                    source_request_id: None,
                    note,
                    baseline_snapshot_id,
                },
            )
            .await?
        }
        CaseSource::FromTurn { request_id } => {
            desktop_assistant_storage::case_from_turn(pool, user, request_id, expected_entry_id)
                .await?
        }
    };
    Ok(format!("case {case_id} added"))
}

/// `desktop-assistant --recall-replay <user-id> <snapshot-id>`: replay every
/// active case against the named snapshot and print the report.
///
/// Resolves each active case's query embedding before calling
/// `desktop_assistant_storage::run_replay`, which reads only from the cache
/// (`recall_case_embeddings`) and calls no embedder itself: a case already
/// cached under the snapshot's own embedding model is left alone (this is
/// what makes two replays agree exactly); a case with none is embedded fresh
/// and cached only when this daemon's live embedder still reports the
/// snapshot's own model — otherwise it is left uncached, and
/// `run_replay` refuses the run and names the reason.
pub async fn run_replay(
    pool: &PgPool,
    user: &str,
    snapshot_id: &str,
    embedding_client: Option<&Arc<dyn EmbeddingClient>>,
) -> anyhow::Result<String> {
    let snapshot = desktop_assistant_storage::get_snapshot(pool, user, snapshot_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no snapshot {snapshot_id} for this user"))?;

    let live_model = match embedding_client {
        Some(client) => Some(client.model_identifier().await.map_err(|e| {
            anyhow::anyhow!("could not resolve the live embedder's model identifier: {e}")
        })?),
        None => None,
    };

    let cases = desktop_assistant_storage::list_active_cases(pool, user).await?;
    for case in &cases {
        let already_cached = desktop_assistant_storage::get_cached_case_embedding(
            pool,
            user,
            &case.id,
            &snapshot.embedding_model,
        )
        .await?
        .is_some();
        if already_cached {
            continue;
        }
        let (Some(model), Some(client)) = (&live_model, embedding_client) else {
            // No live embedder, or none configured at all — leave uncached.
            // `run_replay` below refuses the run and names which case and
            // which model, rather than this function guessing at a vector.
            continue;
        };
        if model != &snapshot.embedding_model {
            // The live embedder has moved on since this snapshot was taken.
            // A fresh embedding here would compare a vector from the wrong
            // space against the frozen entries, which is a silent version of
            // exactly the mismatch `run_replay` is meant to refuse.
            continue;
        }
        let mut vectors = client
            .embed(vec![case.query_text.clone()])
            .await
            .map_err(|e| anyhow::anyhow!("embedding case {} failed: {e}", case.id))?;
        let vector = vectors
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vector for case {}", case.id))?;
        desktop_assistant_storage::cache_case_embedding(
            pool,
            user,
            &case.id,
            &snapshot.embedding_model,
            vector,
        )
        .await?;
    }

    let report = desktop_assistant_storage::run_replay(pool, user, &snapshot).await?;
    Ok(render_report(&snapshot, &report))
}

fn render_manifest(manifest: &SnapshotManifest) -> String {
    format!(
        "snapshot {} (\"{}\") taken {}\n\
         embedding model: {}\n\
         entries: {}\n\
         use records: {}\n\
         excluded (embedded under another model or unembedded): {}",
        manifest.id,
        manifest.name,
        manifest.taken_at.to_rfc3339(),
        manifest.embedding_model,
        manifest.entry_count,
        manifest.use_count,
        manifest.excluded_count,
    )
}

/// Render a replay report for a person to read.
///
/// **The case count is the first line, before any aggregate** — so nobody
/// reading only the top of the output can quote a number computed over
/// cases without first being told how many there were. Below the small-set
/// threshold, [`SMALL_SET_NOTICE`] prints on every run, in the output
/// itself rather than in a document a reader may not have open. Every
/// active case gets one block: a case whose expected entry is missing from
/// the snapshot is named as such, never left out.
fn render_report(snapshot: &SnapshotManifest, report: &ReplayReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} case(s) replayed against snapshot {} (\"{}\", model {}, taken {}), scorer {}\n",
        report.case_count,
        snapshot.id,
        snapshot.name,
        snapshot.embedding_model,
        snapshot.taken_at.to_rfc3339(),
        report.scorer_version,
    ));
    if report.too_small_to_generalize {
        out.push_str(SMALL_SET_NOTICE);
        out.push('\n');
    }

    let mut hits_at_1 = 0usize;
    let mut hits_at_5 = 0usize;
    let mut missing = 0usize;
    let mut ranked_count = 0usize;
    let mut rank_sum = 0usize;

    for result in &report.results {
        out.push('\n');
        out.push_str(&format!(
            "case {} — query: {:?} — expected entry: {}\n",
            result.case_id, result.query_text, result.expected_entry_id
        ));
        match &result.outcome {
            CaseOutcome::Ranked {
                rank,
                total_candidates,
                cleared_bar,
                top,
            } => {
                ranked_count += 1;
                rank_sum += rank;
                if *rank == 1 {
                    hits_at_1 += 1;
                }
                if *rank <= 5 {
                    hits_at_5 += 1;
                }
                out.push_str(&format!(
                    "  rank {rank} of {total_candidates}, cleared the recall bar: {cleared_bar}\n"
                ));
                out.push_str("  nearest candidates:\n");
                for (position, entry) in top.iter().enumerate() {
                    out.push_str(&format!(
                        "    {}. {} — distance {:.4}, total {:.3} \
                         (semantic {:.3?}, lexical {:.3}, reinforcement {:.3}, situation {:.3}, \
                         salience {:.3}, disposition {:.3})\n",
                        position + 1,
                        entry.entry_id,
                        entry.distance,
                        entry.terms.total,
                        entry.terms.semantic,
                        entry.terms.lexical,
                        entry.terms.reinforcement,
                        entry.terms.situation,
                        entry.terms.salience,
                        entry.terms.disposition,
                    ));
                }
            }
            CaseOutcome::ExpectedEntryMissing { total_candidates } => {
                missing += 1;
                out.push_str(&format!(
                    "  expected entry not found in this snapshot ({total_candidates} \
                     candidate(s) scanned) — a vanished ground truth, reported rather than \
                     skipped\n"
                ));
            }
        }
    }

    out.push('\n');
    out.push_str(&format!(
        "aggregate: {ranked_count} case(s) ranked, {missing} with an expected entry missing \
         from the snapshot, hits-at-1={hits_at_1}, hits-at-5={hits_at_5}"
    ));
    if ranked_count > 0 {
        out.push_str(&format!(
            ", mean rank={:.2}",
            rank_sum as f64 / ranked_count as f64
        ));
    }
    out
}
