//! The mis-filed-procedure sweep (#1175): the routines already written as facts.
//!
//! `METHOD_IS_NOT_A_FACT` now reaches every path that writes a knowledge entry,
//! which stops the store gaining more of them. It does nothing about the ones
//! already there, and a prompt is not an enforcer in any case - so this pass is
//! the half that catches what the rule misses, before and after it existed.
//!
//! ## It proposes, and never rewrites
//!
//! The entry is the person's own writing. A background pass that decided a
//! sentence was really a procedure and rewrote it would be editing somebody's
//! notes on a guess, unattended and overnight. So the sweep writes a **new,
//! unapproved skill** naming the entry it came from, and leaves the entry
//! exactly as it stands. Approving the skill is the person's act (`#1175`'s
//! approve command); retiring the entry afterwards is theirs too.
//!
//! That is not a mechanism invented here. An unapproved skill is already how
//! this system proposes a procedure a person has not agreed to - it is what
//! `promote_plan_to_skill` and the extraction pass both write - so the sweep
//! reuses the proposal that exists rather than inventing a review queue beside
//! it.
//!
//! ## What stops it re-judging the store every night
//!
//! `knowledge_procedure_sweep` records every entry the pass has read and the
//! entry's own `updated_at` at that moment. An entry is judged once per edit,
//! and both answers are recorded - "we looked and it was a fact" is exactly the
//! answer the ledger exists to avoid paying for twice.
//!
//! ## The link is validated, never trusted
//!
//! One call shows the model several entries and asks which of them are methods,
//! so each proposal has to say which entry it came from. A `from_entry` naming
//! something this call did not show is dropped rather than followed: a
//! mis-linked proposal would tell a person that one of their entries is a
//! procedure when the sentence the model read was a different one.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::IndexedSkill;
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use desktop_assistant_core::skill_promotion::{METHOD_IS_NOT_A_FACT, MISFILED_SOURCE};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use super::skills::{ExtractedSkill, parse_one_skill, to_indexed_skill, write_proposed_skills};
use super::types::DreamingLlmFn;

/// How many entries one cycle may read, across every user with work.
///
/// The same shape of figure as [`MAX_SUMMARIES_PER_CYCLE`], and smaller: a
/// summary is owed to every entry that lacks one, where this question is asked
/// of each entry exactly once and then never again unless its text changes. A
/// store converges on nothing to do, so the cap decides how fast the first
/// sweep of an existing store drains rather than what the pass costs from then
/// on.
///
/// [`MAX_SUMMARIES_PER_CYCLE`]: super::types::MAX_SUMMARIES_PER_CYCLE
pub const MAX_SWEPT_ENTRIES_PER_CYCLE: usize = 60;

/// Entries described in one prompt.
///
/// Smaller than the summary batch, because the answer is longer: a summary is
/// one line per entry and a proposal is a whole method.
pub const MAX_SWEEP_BATCH_ROWS: usize = 8;

/// What one sweep did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MisfiledStats {
    /// Entries shown to the model and recorded in the ledger.
    pub judged: usize,
    /// Entries the model read as methods, and which became unapproved skill
    /// proposals.
    pub proposed: usize,
    /// Entries still unjudged when the pass finished. Reported rather than
    /// inferred, so the per-cycle cap is never a silent truncation.
    pub remaining: usize,
}

/// One entry as the sweep reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SweepTarget {
    id: String,
    content: String,
    tags: Vec<String>,
    /// The entry's `updated_at` when it was read. Recorded with the judgement,
    /// so an entry edited afterwards is judged again.
    content_at: DateTime<Utc>,
}

/// One proposal, with the entry it claims to have come from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Proposal {
    from_entry: String,
    skill: ExtractedSkill,
}

/// Sweep every user's knowledge base for procedures written as facts.
///
/// Cross-user iteration is a background-worker entry point; every per-user pass
/// installs a `with_user_id` scope so all sub-queries land in the right
/// partition.
///
/// A user whose pass fails is logged and skipped rather than failing the cycle:
/// the entries it did not judge stay in the worklist, and the next cycle reads
/// them.
pub async fn run_misfiled_sweep_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    cancellation: &CancellationToken,
) -> Result<MisfiledStats, CoreError> {
    let user_ids = load_user_ids_with_unjudged_entries(pool).await?;
    if user_ids.is_empty() {
        return Ok(MisfiledStats::default());
    }

    // Share the cycle's budget across the users that have work, so one large
    // store cannot starve every user behind it - the rule the summary pass
    // already keeps, and for the same reason.
    let user_count = user_ids.len();
    let per_user = MAX_SWEPT_ENTRIES_PER_CYCLE.div_ceil(user_count).max(1);
    let mut budget = MAX_SWEPT_ENTRIES_PER_CYCLE;
    let mut total = MisfiledStats::default();

    for user_id in user_ids {
        if cancellation.is_cancelled() {
            tracing::info!("dreaming: mis-filed sweep cancelled; stopping scan");
            break;
        }
        if budget == 0 {
            break;
        }
        let take = per_user.min(budget);
        // Charged before the call, for the reason the summary pass charges
        // before its own: a user whose batches all failed still read its rows
        // and spent its calls.
        budget = budget.saturating_sub(take);

        let result = with_user_id(UserId::new(user_id.clone()), async {
            sweep_one_user(pool, llm_fn, take, cancellation).await
        })
        .await;

        match result {
            Ok(stats) => {
                budget += take.saturating_sub(stats.judged);
                total.judged += stats.judged;
                total.proposed += stats.proposed;
                total.remaining += stats.remaining;
            }
            Err(e) => {
                tracing::warn!("dreaming: mis-filed sweep failed for user {user_id}: {e}");
            }
        }
    }

    Ok(total)
}

/// Judge up to `take` of the current user's unjudged entries.
async fn sweep_one_user(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    take: usize,
    cancellation: &CancellationToken,
) -> Result<MisfiledStats, CoreError> {
    let outstanding = count_unjudged_entries(pool).await?;
    let targets = load_unjudged_entries(pool, take).await?;
    if targets.is_empty() {
        return Ok(MisfiledStats::default());
    }

    let mut stats = MisfiledStats::default();
    for batch in targets.chunks(MAX_SWEEP_BATCH_ROWS) {
        if cancellation.is_cancelled() {
            break;
        }
        let proposals = match proposals_for_batch(llm_fn, batch).await {
            Ok(proposals) => proposals,
            Err(e) => {
                // The batch is left unjudged, so the next cycle reads it again.
                tracing::warn!("dreaming: mis-filed sweep batch failed: {e}");
                continue;
            }
        };

        let owner = current_user_id().as_str().to_string();
        let mut skills = Vec::with_capacity(proposals.len());
        let mut proposed_for: Vec<(String, String)> = Vec::new();
        for proposal in &proposals {
            let Some(skill) = as_proposed_skill(&proposal.skill, &proposal.from_entry, &owner)
            else {
                tracing::debug!(
                    entry = %proposal.from_entry,
                    "dreaming: a swept entry's method did not clear the skill bar"
                );
                continue;
            };
            proposed_for.push((proposal.from_entry.clone(), skill.name.clone()));
            skills.push(skill);
        }
        stats.proposed += write_proposed_skills(pool, &skills).await?;

        // Every entry the batch showed is recorded, method or not: an entry
        // that read as an ordinary fact is the answer this ledger exists to
        // avoid paying for a second time.
        record_judgements(pool, batch, &proposed_for).await?;
        stats.judged += batch.len();
    }

    stats.remaining = outstanding.saturating_sub(stats.judged);
    Ok(stats)
}

/// One call: show a batch and read back the methods among it.
async fn proposals_for_batch(
    llm_fn: &DreamingLlmFn,
    batch: &[SweepTarget],
) -> Result<Vec<Proposal>, String> {
    let response = llm_fn(build_system_prompt(), build_user_prompt(batch)).await?;
    let payload = super::common::extract_json_payload(&response);
    let parsed: serde_json::Value = serde_json::from_str(&payload)
        .map_err(|e| format!("the sweep answer was not JSON: {e}"))?;
    Ok(parse_proposals(&parsed, batch))
}

/// Read the proposals out of one answer, dropping any that names an entry this
/// call did not show.
///
/// **Validated rather than trusted**, which is the whole reason a batch may
/// carry more than one entry. A `from_entry` the batch did not contain would
/// tell a person that one of their entries is a procedure when the sentence the
/// model actually read was a different one - and there is nothing on the row to
/// show them that it was mis-linked.
fn parse_proposals(root: &serde_json::Value, batch: &[SweepTarget]) -> Vec<Proposal> {
    let Some(items) = root.get("skills").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let from_entry = item.get("from_entry")?.as_str()?.trim().to_string();
            if !batch.iter().any(|t| t.id == from_entry) {
                tracing::debug!(
                    entry = %from_entry,
                    "dreaming: dropping a sweep proposal for an entry this call did not show"
                );
                return None;
            }
            Some(Proposal {
                from_entry,
                skill: parse_one_skill(item)?,
            })
        })
        .collect()
}

/// One proposal as a catalog row: an unapproved skill that names the entry it
/// came from.
///
/// The bar is [`to_indexed_skill`]'s, unchanged - a proposal that does not
/// clear it is not a method worth splitting out. What this adds is the
/// provenance a person needs to act on both halves of the split: `source` says
/// this came from an entry rather than from a transcript or a plan, and
/// `metadata.from_entry` says which one.
fn as_proposed_skill(
    proposed: &ExtractedSkill,
    from_entry: &str,
    owner: &str,
) -> Option<IndexedSkill> {
    let mut skill = to_indexed_skill(proposed)?;
    skill.owner_user_id = Some(owner.to_string());
    skill.source = Some(MISFILED_SOURCE.to_string());
    skill.metadata = serde_json::json!({
        "authored_from": MISFILED_SOURCE,
        "from_entry": from_entry,
    });
    Some(skill)
}

fn build_system_prompt() -> String {
    let mut prompt = String::from(
        "You are auditing a personal knowledge base for entries that are really \
         procedures.\n\
         \n",
    );
    prompt.push_str(METHOD_IS_NOT_A_FACT);
    prompt.push_str(
        "\n\
         \n\
         You are shown numbered entries, each with its id. For every entry that is really a \
         method, return one skill. Return nothing at all for an entry that is an ordinary \
         fact, a preference, or a piece of project context - most entries are, and a wrong \
         proposal costs a person's attention.\n\
         \n\
         Return a JSON object with one `skills` array. Each skill has:\n\
         - `from_entry` (string): the id of the entry it came from, copied exactly from the \
         entry shown. An id you were not shown is discarded.\n\
         - `name` (string): a short kebab-case name, e.g. `weekly-status-report`.\n\
         - `description` (string): one or two sentences saying WHEN to use it.\n\
         - `steps` (array of objects): the method, in order, at least 3 of them. Each step has \
         `goal` (what the step does) and `outcome` (what it produces, or how you know it \
         worked). Take the steps from what the entry says; do not invent a method the entry \
         does not describe. An entry that names a method in one clause without saying what its \
         steps are is not enough to propose one - return nothing for it.\n\
         - `tags` (array of strings, optional).\n\
         \n\
         Return `{\"skills\": []}` when none of the entries is a method. Nothing you return \
         changes the entries: a skill written this way is UNAPPROVED, and a person decides \
         whether to keep it. Output ONLY the JSON object.",
    );
    prompt
}

fn build_user_prompt(batch: &[SweepTarget]) -> String {
    let mut prompt = String::with_capacity(batch.len() * 256);
    prompt.push_str("# Entries to audit\n\n");
    for target in batch {
        prompt.push_str("## ");
        prompt.push_str(&target.id);
        prompt.push('\n');
        prompt.push_str("tags: ");
        if target.tags.is_empty() {
            prompt.push_str("(none)");
        } else {
            prompt.push_str(&target.tags.join(", "));
        }
        prompt.push('\n');
        prompt.push_str(&target.content);
        prompt.push_str("\n\n");
    }
    prompt
}

/// Every user holding at least one entry the sweep has not judged at its
/// current text.
async fn load_user_ids_with_unjudged_entries(pool: &PgPool) -> Result<Vec<String>, CoreError> {
    // Cross-user by design: this is the background worker's own entry point,
    // and every per-user pass below installs its own scope.
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT kb.user_id \
         FROM knowledge_base kb \
         LEFT JOIN knowledge_procedure_sweep s \
           ON s.user_id = kb.user_id AND s.entry_id = kb.id \
         WHERE kb.deleted_at IS NULL \
           AND (s.entry_id IS NULL OR s.judged_content_at < kb.updated_at)",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load sweep users failed: {e}")))?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}

async fn count_unjudged_entries(pool: &PgPool) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let (count,): (i64,) = sqlx::query_as(
        "SELECT count(*) \
         FROM knowledge_base kb \
         LEFT JOIN knowledge_procedure_sweep s \
           ON s.user_id = kb.user_id AND s.entry_id = kb.id \
         WHERE kb.user_id = $1 AND kb.deleted_at IS NULL \
           AND (s.entry_id IS NULL OR s.judged_content_at < kb.updated_at)",
    )
    .bind(user_id.as_str())
    .fetch_one(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: count unjudged entries failed: {e}")))?;
    Ok(count.max(0) as usize)
}

async fn load_unjudged_entries(pool: &PgPool, limit: usize) -> Result<Vec<SweepTarget>, CoreError> {
    let user_id = current_user_id();
    type Row = (String, String, Vec<String>, DateTime<Utc>);
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT kb.id, kb.content, kb.tags, kb.updated_at \
         FROM knowledge_base kb \
         LEFT JOIN knowledge_procedure_sweep s \
           ON s.user_id = kb.user_id AND s.entry_id = kb.id \
         WHERE kb.user_id = $1 AND kb.deleted_at IS NULL \
           AND (s.entry_id IS NULL OR s.judged_content_at < kb.updated_at) \
         ORDER BY kb.created_at ASC, kb.id ASC \
         LIMIT $2",
    )
    .bind(user_id.as_str())
    .bind(limit as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load unjudged entries failed: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|(id, content, tags, content_at)| SweepTarget {
            id,
            content,
            tags,
            content_at,
        })
        .collect())
}

/// Record what the sweep decided about every entry in `batch`.
///
/// Guarded by ownership rather than only scoped by it, on the terms the use log
/// states: `knowledge_base.id` is a global primary key, so the insert selects
/// the entry out of `knowledge_base` under the caller's `user_id` and inserts
/// from that select.
async fn record_judgements(
    pool: &PgPool,
    batch: &[SweepTarget],
    proposed_for: &[(String, String)],
) -> Result<(), CoreError> {
    if batch.is_empty() {
        return Ok(());
    }
    let user_id = current_user_id();
    let ids: Vec<String> = batch.iter().map(|t| t.id.clone()).collect();
    let content_at: Vec<DateTime<Utc>> = batch.iter().map(|t| t.content_at).collect();
    let proposed: Vec<Option<String>> = batch
        .iter()
        .map(|t| {
            proposed_for
                .iter()
                .find(|(entry, _)| *entry == t.id)
                .map(|(_, name)| name.clone())
        })
        .collect();

    sqlx::query(
        "INSERT INTO knowledge_procedure_sweep \
             (user_id, entry_id, judged_at, judged_content_at, proposed_skill) \
         SELECT kb.user_id, kb.id, NOW(), judged.content_at, judged.proposed \
         FROM knowledge_base kb \
         JOIN UNNEST($2::text[], $3::timestamptz[], $4::text[]) \
              AS judged(entry_id, content_at, proposed) ON judged.entry_id = kb.id \
         WHERE kb.user_id = $1 \
         ON CONFLICT (user_id, entry_id) DO UPDATE SET \
             judged_at = NOW(), \
             judged_content_at = EXCLUDED.judged_content_at, \
             proposed_skill = EXCLUDED.proposed_skill",
    )
    .bind(user_id.as_str())
    .bind(&ids)
    .bind(&content_at)
    .bind(&proposed)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: record sweep judgement failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(id: &str) -> SweepTarget {
        SweepTarget {
            id: id.to_string(),
            content: "To publish: bump, tag, push.".to_string(),
            tags: vec!["instruction".to_string()],
            content_at: Utc::now(),
        }
    }

    fn a_method(from_entry: &str) -> serde_json::Value {
        serde_json::json!({
            "from_entry": from_entry,
            "name": "publish-a-crate",
            "description": "Cut a release and push it to the registry.",
            "steps": [
                {"goal": "Bump the version", "outcome": "the manifest names the new version"},
                {"goal": "Tag the commit", "outcome": "the tag points at the bump"},
                {"goal": "Push the tag", "outcome": "the registry has the release"},
            ],
        })
    }

    /// Acceptance (#1175): a knowledge entry that reads as a method is proposed
    /// as an unapproved skill that names the entry it came from.
    ///
    /// Naming the entry is what makes it a proposed *split* rather than a
    /// second copy: a person reading the skill can see which of their entries
    /// it says should not have been one.
    #[test]
    fn a_misfiled_entry_is_proposed_as_an_unapproved_skill_naming_the_entry_it_came_from() {
        let batch = vec![target("kb-1")];
        let answer = serde_json::json!({"skills": [a_method("kb-1")]});

        let proposals = parse_proposals(&answer, &batch);
        assert_eq!(proposals.len(), 1);

        let skill = as_proposed_skill(&proposals[0].skill, &proposals[0].from_entry, "alice")
            .expect("a three-step method clears the skill bar");

        assert!(
            !skill.is_approved(),
            "a proposal is not a decision: {skill:?}"
        );
        assert_eq!(skill.source.as_deref(), Some(MISFILED_SOURCE));
        assert_eq!(skill.metadata["from_entry"], "kb-1");
        assert_eq!(skill.owner_user_id.as_deref(), Some("alice"));
        assert!(
            !skill.present_on_disk,
            "nothing was written to a skill root"
        );
    }

    /// A proposal naming an entry the call did not show is dropped.
    ///
    /// One call shows several entries, so the link is the model's answer rather
    /// than the call's structure. A mis-linked proposal would tell a person
    /// that one of their entries is a procedure when the sentence the model
    /// read was a different one, and nothing on the row would show that.
    #[test]
    fn a_proposal_for_an_entry_the_call_did_not_show_is_dropped() {
        let batch = vec![target("kb-1"), target("kb-2")];
        let answer = serde_json::json!({
            "skills": [a_method("kb-1"), a_method("kb-99"), a_method("")],
        });

        let proposals = parse_proposals(&answer, &batch);

        assert_eq!(
            proposals
                .iter()
                .map(|p| p.from_entry.as_str())
                .collect::<Vec<_>>(),
            vec!["kb-1"],
            "only a proposal linked to an entry this call actually showed survives"
        );
    }

    /// An answer with no methods in it is the ordinary answer, not a failure.
    #[test]
    fn an_answer_with_no_methods_proposes_nothing() {
        let batch = vec![target("kb-1")];
        assert!(parse_proposals(&serde_json::json!({"skills": []}), &batch).is_empty());
        assert!(parse_proposals(&serde_json::json!({}), &batch).is_empty());
    }

    /// A proposal that does not clear the skill bar is not written.
    ///
    /// The bar is the promotion path's own: fewer than three steps is a single
    /// act rather than a method, and the value of a skill is the ordering.
    #[test]
    fn a_proposal_under_the_step_bar_is_not_written() {
        let batch = vec![target("kb-1")];
        let answer = serde_json::json!({"skills": [{
            "from_entry": "kb-1",
            "name": "one-step",
            "description": "Not really a method.",
            "steps": [{"goal": "do it", "outcome": "done"}],
        }]});

        let proposals = parse_proposals(&answer, &batch);
        assert_eq!(proposals.len(), 1, "it parses");
        assert!(
            as_proposed_skill(&proposals[0].skill, "kb-1", "alice").is_none(),
            "and then fails the bar rather than becoming a one-step skill"
        );
    }

    /// The sweep's prompt states the rule that decides which store a piece of
    /// learning belongs in, rather than restating it in its own words.
    #[test]
    fn the_sweep_prompt_states_the_method_is_not_a_fact_rule() {
        assert!(build_system_prompt().contains(METHOD_IS_NOT_A_FACT));
    }

    /// The prompt tells the model that nothing it returns changes the entries.
    ///
    /// A model told to audit a store will otherwise reach for the edit it
    /// cannot make, and the answer that matters here is a proposal.
    #[test]
    fn the_sweep_prompt_says_a_proposal_changes_no_entry() {
        let prompt = build_system_prompt();
        assert!(
            prompt.contains("Nothing you return changes the entries"),
            "{prompt}"
        );
        assert!(prompt.contains("UNAPPROVED"), "{prompt}");
    }
}
