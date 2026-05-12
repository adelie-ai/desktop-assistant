//! Phase 2: per-memory consolidation review (issue #108).
//!
//! For each KB entry that needs review (gated by `reviewed_at IS NULL`):
//!
//! 1. Retrieve candidate related entries by tag overlap + embedding
//!    similarity (high-precision filter: candidates likely interact with
//!    the focal entry).
//! 2. Ask the LLM to review the focal entry in context of its candidates
//!    and propose actions: keep, update, add scope, merge, delete, or
//!    request the source transcript for disambiguation.
//! 3. Buffer all proposed ops; do not apply during the review loop.
//! 4. After the loop: compute merge clusters via union-find. Clusters of
//!    size > 2 get an n-ary confirmation call. Each confirmed cluster gets
//!    a synthesis call to produce unified content.
//! 5. Apply everything in a single transaction (see `reconcile::apply_ops`).
//!
//! Prompt is biased toward "keep both". Merging requires same scope plus
//! evidence of contradiction or true duplication; different scopes never
//! merge.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use pgvector::Vector;
use sqlx::PgPool;

use super::common::{extract_json_payload, load_full_transcript};
use super::reconcile::{OpBuffer, ProposedOp, SynthesizedMerge, apply_ops};
use super::types::{
    BackfillEmbedFn, ConsolidationStats, DreamingLlmFn, MAX_REVIEWS_PER_CYCLE,
    MAX_REVIEW_CANDIDATES, MAX_REVIEW_GENERATION, SOFT_DELETE_TTL_DAYS,
};
use crate::kb_metadata::{KbMetadata, KbScope};

#[derive(Debug, Clone)]
struct KbRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: KbMetadata,
    /// For candidates: min cosine distance to the focal embedding. NaN for
    /// the focal itself.
    distance: f64,
}

pub async fn run_consolidation_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
) -> Result<ConsolidationStats, String> {
    let focals = load_entries_needing_review(pool).await?;
    if focals.is_empty() {
        tracing::debug!("dreaming: no entries needing review");
        return Ok(ConsolidationStats::default());
    }
    tracing::info!(
        "dreaming: consolidation reviewing {} focal entr{}",
        focals.len(),
        if focals.len() == 1 { "y" } else { "ies" }
    );

    let mut buffer = OpBuffer::new();

    for focal in &focals {
        let candidates = match retrieve_candidates(pool, focal).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "dreaming: candidate retrieval failed for {}: {e}",
                    focal.id
                );
                buffer.mark_reviewed(&focal.id);
                continue;
            }
        };

        // Always mark the focal as reviewed even if the LLM call fails —
        // we don't want failed calls to wedge a row above the watermark
        // forever.
        buffer.mark_reviewed(&focal.id);

        let actions = match review_focal(llm_fn, pool, focal, &candidates, false).await
        {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("dreaming: review failed for {}: {e}", focal.id);
                continue;
            }
        };

        for action in actions {
            if let Some(op) = action.into_op(&focal.id) {
                buffer.absorb(op);
            }
        }
    }

    // Merge-cluster confirmation + synthesis (outside the apply transaction).
    let clusters = buffer.merge_clusters();
    let id_to_row: HashMap<String, &KbRow> =
        focals.iter().map(|r| (r.id.clone(), r)).collect();

    let mut synthesized: Vec<SynthesizedMerge> = Vec::new();

    for cluster in clusters {
        let refined = if cluster.len() > 2 {
            confirm_cluster(pool, llm_fn, &cluster, &id_to_row).await
        } else {
            cluster.clone()
        };
        if refined.len() < 2 {
            continue;
        }

        match synthesize_cluster(pool, llm_fn, &refined, &id_to_row).await {
            Ok(merge) => synthesized.push(merge),
            Err(e) => {
                tracing::warn!(
                    "dreaming: synthesis failed for cluster of {}: {e}",
                    refined.len()
                );
            }
        }
    }

    let stats = apply_ops(
        pool,
        embed_fn,
        embedding_model,
        &buffer,
        &synthesized,
        SOFT_DELETE_TTL_DAYS,
    )
    .await?;

    Ok(stats)
}

async fn load_entries_needing_review(pool: &PgPool) -> Result<Vec<KbRow>, String> {
    let rows: Vec<(String, String, Vec<String>, serde_json::Value)> = sqlx::query_as(
        "SELECT id, content, tags, metadata
         FROM knowledge_base
         WHERE reviewed_at IS NULL
           AND deleted_at IS NULL
           AND review_generation < $1
         ORDER BY created_at ASC
         LIMIT $2",
    )
    .bind(MAX_REVIEW_GENERATION)
    .bind(MAX_REVIEWS_PER_CYCLE)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: load focals failed: {e}"))?;

    Ok(rows
        .into_iter()
        .map(|(id, content, tags, metadata_json)| KbRow {
            id,
            content,
            tags,
            metadata: KbMetadata::from_json(&metadata_json),
            distance: f64::NAN,
        })
        .collect())
}

async fn retrieve_candidates(pool: &PgPool, focal: &KbRow) -> Result<Vec<KbRow>, String> {
    // Pull the focal's first embedding chunk to use as the similarity probe.
    let focal_chunk: Option<(Vec<Vector>,)> =
        sqlx::query_as("SELECT embedding FROM knowledge_base WHERE id = $1")
            .bind(&focal.id)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("dreaming: load focal embedding failed: {e}"))?;

    let probe = match focal_chunk.and_then(|(v,)| v.into_iter().next()) {
        Some(v) => v,
        None => {
            // No embedding for focal — fall back to tag-overlap only.
            return retrieve_by_tags_only(pool, focal).await;
        }
    };

    let mut by_id: BTreeMap<String, KbRow> = BTreeMap::new();

    // Tag-overlap search.
    if !focal.tags.is_empty() {
        let tag_rows: Vec<(String, String, Vec<String>, serde_json::Value, f64)> =
            sqlx::query_as(
                "SELECT kb.id, kb.content, kb.tags, kb.metadata,
                        COALESCE(MIN(u.chunk <=> $1), 2.0) AS distance
                 FROM knowledge_base kb
                 LEFT JOIN LATERAL unnest(kb.embedding) AS u(chunk) ON true
                 WHERE kb.id != $2
                   AND kb.deleted_at IS NULL
                   AND kb.tags && $3
                 GROUP BY kb.id, kb.content, kb.tags, kb.metadata
                 ORDER BY distance ASC
                 LIMIT $4",
            )
            .bind(&probe)
            .bind(&focal.id)
            .bind(&focal.tags)
            .bind(MAX_REVIEW_CANDIDATES)
            .fetch_all(pool)
            .await
            .map_err(|e| format!("dreaming: tag-overlap retrieve failed: {e}"))?;

        for (id, content, tags, metadata_json, distance) in tag_rows {
            by_id.insert(
                id.clone(),
                KbRow {
                    id,
                    content,
                    tags,
                    metadata: KbMetadata::from_json(&metadata_json),
                    distance,
                },
            );
        }
    }

    // Embedding-similarity search (catches related-by-content entries that
    // don't share tags).
    let emb_rows: Vec<(String, String, Vec<String>, serde_json::Value, f64)> =
        sqlx::query_as(
            "SELECT kb.id, kb.content, kb.tags, kb.metadata,
                    MIN(u.chunk <=> $1) AS distance
             FROM knowledge_base kb,
                  LATERAL unnest(kb.embedding) AS u(chunk)
             WHERE kb.id != $2
               AND kb.deleted_at IS NULL
               AND kb.embedding IS NOT NULL
             GROUP BY kb.id, kb.content, kb.tags, kb.metadata
             ORDER BY distance ASC
             LIMIT $3",
        )
        .bind(&probe)
        .bind(&focal.id)
        .bind(MAX_REVIEW_CANDIDATES)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("dreaming: embedding retrieve failed: {e}"))?;

    for (id, content, tags, metadata_json, distance) in emb_rows {
        let entry = KbRow {
            id: id.clone(),
            content,
            tags,
            metadata: KbMetadata::from_json(&metadata_json),
            distance,
        };
        by_id
            .entry(id)
            .and_modify(|existing| {
                if distance < existing.distance {
                    existing.distance = distance;
                }
            })
            .or_insert(entry);
    }

    let mut combined: Vec<KbRow> = by_id.into_values().collect();
    combined.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(MAX_REVIEW_CANDIDATES as usize);
    Ok(combined)
}

async fn retrieve_by_tags_only(pool: &PgPool, focal: &KbRow) -> Result<Vec<KbRow>, String> {
    if focal.tags.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<(String, String, Vec<String>, serde_json::Value)> = sqlx::query_as(
        "SELECT id, content, tags, metadata
         FROM knowledge_base
         WHERE id != $1 AND deleted_at IS NULL AND tags && $2
         ORDER BY updated_at DESC
         LIMIT $3",
    )
    .bind(&focal.id)
    .bind(&focal.tags)
    .bind(MAX_REVIEW_CANDIDATES)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: tag-only retrieve failed: {e}"))?;

    Ok(rows
        .into_iter()
        .map(|(id, content, tags, metadata_json)| KbRow {
            id,
            content,
            tags,
            metadata: KbMetadata::from_json(&metadata_json),
            distance: f64::NAN,
        })
        .collect())
}

// ── Review action types ──────────────────────────────────────────────

#[derive(Debug, Clone)]
enum ReviewAction {
    Keep,
    Update(String),
    AddScope(KbScope),
    MergeWith(String),
    Delete(String),
    FetchSource,
}

impl ReviewAction {
    fn into_op(self, focal_id: &str) -> Option<ProposedOp> {
        match self {
            ReviewAction::Keep | ReviewAction::FetchSource => None,
            ReviewAction::Update(content) => Some(ProposedOp::Update {
                id: focal_id.to_string(),
                new_content: content,
            }),
            ReviewAction::AddScope(scope) => Some(ProposedOp::AddScope {
                id: focal_id.to_string(),
                scope,
            }),
            ReviewAction::MergeWith(target) => Some(ProposedOp::Merge {
                a: focal_id.to_string(),
                b: target,
            }),
            ReviewAction::Delete(reason) => Some(ProposedOp::Delete {
                id: focal_id.to_string(),
                reason,
            }),
        }
    }
}

async fn review_focal(
    llm_fn: &DreamingLlmFn,
    pool: &PgPool,
    focal: &KbRow,
    candidates: &[KbRow],
    transcript_included: bool,
) -> Result<Vec<ReviewAction>, String> {
    let system_prompt = build_review_system_prompt();
    let user_prompt = build_review_user_prompt(focal, candidates, None);

    let response = llm_fn(system_prompt.clone(), user_prompt).await?;
    let actions = parse_review_response(&response);

    // If the LLM asked for the source and we haven't already provided it,
    // re-call once with the transcript appended.
    let wants_source = actions
        .iter()
        .any(|a| matches!(a, ReviewAction::FetchSource));
    if wants_source && !transcript_included {
        let transcript = if let Some(conv_id) = &focal.metadata.source_conversation_id {
            load_full_transcript(pool, conv_id).await.unwrap_or_default()
        } else {
            String::new()
        };

        if transcript.is_empty() {
            tracing::debug!(
                "dreaming: focal {} requested source but it's gone — proceeding on KB alone",
                focal.id
            );
            // Strip FetchSource from the action list; everything else stands.
            return Ok(actions
                .into_iter()
                .filter(|a| !matches!(a, ReviewAction::FetchSource))
                .collect());
        }

        let user_prompt_with_source =
            build_review_user_prompt(focal, candidates, Some(&transcript));
        let response = llm_fn(system_prompt, user_prompt_with_source).await?;
        return Ok(parse_review_response(&response)
            .into_iter()
            .filter(|a| !matches!(a, ReviewAction::FetchSource))
            .collect());
    }

    Ok(actions)
}

fn build_review_system_prompt() -> String {
    String::from(
        "You are a memory consolidation reviewer. You see one knowledge-base entry \
        (the FOCAL) plus a small set of CANDIDATES that may interact with it. \
        Your job is to decide what to do with the focal entry.\n\
        \n\
        ## Bias toward KEEP\n\
        \n\
        The default action is `keep`. Only propose other actions when you have \
        clear evidence. Specifically:\n\
        \n\
        - **Different scopes never merge.** Two facts with different `scope` \
        objects are about different contexts and must both be preserved, \
        even if their content looks similar (e.g. \"project dir is /a\" with \
        `scope: {project: \"adelie-ai\"}` vs \"project dir is /b\" with \
        `scope: {project: \"other\"}` — KEEP BOTH).\n\
        - **Merge only on same scope + duplication or contradiction.** If \
        two facts share scope AND say the same thing in different words, \
        propose `merge_with`. If they share scope AND contradict each \
        other, the newer one supersedes — propose `merge_with` and let \
        synthesis pick the right text.\n\
        - **Missing scope is a signal.** If the focal has `scope: null` but \
        all its candidates share the same non-null scope, the focal was \
        probably mis-extracted — propose `add_scope` with the inferred \
        scope, NOT a merge.\n\
        - **Use `fetch_source` when unsure.** If you can't decide because \
        the focal's wording is ambiguous, request the source transcript. \
        Don't guess.\n\
        \n\
        ## Output\n\
        \n\
        Return a JSON object with an `actions` array. Available actions:\n\
        \n\
        - `{\"op\": \"keep\"}` — no change (default).\n\
        - `{\"op\": \"update\", \"new_content\": \"...\"}` — rewrite the \
        focal for clarity. Use sparingly; the existing content is usually \
        fine.\n\
        - `{\"op\": \"add_scope\", \"scope\": {\"project\": \"adelie-ai\"}}` \
        — annotate a scope-naked focal with the scope its peers all share.\n\
        - `{\"op\": \"merge_with\", \"target_id\": \"<candidate id>\"}` — \
        propose merging focal with that candidate. Multiple `merge_with` \
        actions are allowed.\n\
        - `{\"op\": \"delete\", \"reason\": \"...\"}` — soft-delete the \
        focal (e.g. it's clearly noise, not a real fact). Recoverable for \
        a TTL window. Avoid unless confident.\n\
        - `{\"op\": \"fetch_source\"}` — ask for the source transcript.\n\
        \n\
        You may emit multiple actions in one response (e.g. `add_scope` + \
        `merge_with`). If unsure, just `keep`.",
    )
}

fn format_scope(scope: Option<&KbScope>) -> String {
    match scope {
        None => "null".to_string(),
        Some(s) => serde_json::to_string(&s.0).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn build_review_user_prompt(
    focal: &KbRow,
    candidates: &[KbRow],
    transcript: Option<&str>,
) -> String {
    let mut prompt = String::from("## FOCAL entry\n\n");
    prompt.push_str(&format!("- id: `{}`\n", focal.id));
    prompt.push_str(&format!("- content: {}\n", focal.content));
    prompt.push_str(&format!("- tags: [{}]\n", focal.tags.join(", ")));
    prompt.push_str(&format!(
        "- scope: {}\n",
        format_scope(focal.metadata.effective_scope())
    ));

    if candidates.is_empty() {
        prompt.push_str("\n## CANDIDATES\n\n(no related entries found)\n");
    } else {
        prompt.push_str("\n## CANDIDATES (potentially related)\n\n");
        for c in candidates {
            prompt.push_str(&format!("- id: `{}`\n", c.id));
            prompt.push_str(&format!("  content: {}\n", c.content));
            prompt.push_str(&format!("  tags: [{}]\n", c.tags.join(", ")));
            prompt.push_str(&format!(
                "  scope: {}\n",
                format_scope(c.metadata.effective_scope())
            ));
            if c.distance.is_finite() {
                prompt.push_str(&format!("  similarity-distance: {:.3}\n", c.distance));
            }
        }
    }

    if let Some(t) = transcript {
        prompt.push_str("\n## SOURCE TRANSCRIPT (you requested this)\n\n");
        prompt.push_str(t);
        prompt.push('\n');
    }

    prompt.push_str(
        "\nDecide what to do with the focal. Return a JSON object with an `actions` array. \
        Default to `keep` unless you have clear evidence.",
    );
    prompt
}

fn parse_review_response(response: &str) -> Vec<ReviewAction> {
    let payload = extract_json_payload(response.trim());
    let root: serde_json::Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("dreaming: review response not JSON: {e}");
            return vec![ReviewAction::Keep];
        }
    };

    let actions_array = match root.get("actions").and_then(|v| v.as_array()) {
        Some(a) => a.clone(),
        None => match root.as_array() {
            Some(a) => a.clone(),
            None => return vec![ReviewAction::Keep],
        },
    };

    let parsed: Vec<ReviewAction> =
        actions_array.iter().filter_map(parse_one_action).collect();
    if parsed.is_empty() {
        vec![ReviewAction::Keep]
    } else {
        parsed
    }
}

fn parse_one_action(value: &serde_json::Value) -> Option<ReviewAction> {
    let obj = value.as_object()?;
    let op = obj.get("op")?.as_str()?;
    match op {
        "keep" => Some(ReviewAction::Keep),
        "update" => {
            let new_content = obj.get("new_content")?.as_str()?.trim().to_string();
            if new_content.is_empty() {
                None
            } else {
                Some(ReviewAction::Update(new_content))
            }
        }
        "add_scope" => {
            let scope_val = obj.get("scope")?.as_object()?;
            let mut scope = KbScope::new();
            for (k, v) in scope_val {
                if let Some(s) = v.as_str() {
                    scope = scope.with(k.clone(), s.to_string());
                }
            }
            if scope.is_empty() {
                None
            } else {
                Some(ReviewAction::AddScope(scope))
            }
        }
        "merge_with" => {
            let target = obj.get("target_id")?.as_str()?.trim().to_string();
            if target.is_empty() {
                None
            } else {
                Some(ReviewAction::MergeWith(target))
            }
        }
        "delete" => {
            let reason = obj
                .get("reason")
                .and_then(|r| r.as_str())
                .unwrap_or("no reason given")
                .to_string();
            Some(ReviewAction::Delete(reason))
        }
        "fetch_source" => Some(ReviewAction::FetchSource),
        other => {
            tracing::warn!("dreaming: unknown review op '{other}'");
            None
        }
    }
}

// ── Cluster confirmation + synthesis ─────────────────────────────────

/// For clusters of size > 2, ask the LLM whether all members truly belong.
/// Returns the subset confirmed as belonging; non-confirmed ids drop back to
/// standalone (no merge).
async fn confirm_cluster(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    cluster: &BTreeSet<String>,
    id_to_focal: &HashMap<String, &KbRow>,
) -> BTreeSet<String> {
    let members = load_cluster_members(pool, cluster, id_to_focal).await;
    if members.len() < 2 {
        return BTreeSet::new();
    }

    let mut user = String::from(
        "Several entries have been transitively grouped as potentially the \
        same fact. Confirm which ones actually belong together (same scope, \
        same factual claim). It's fine to confirm only a subset; non-\
        confirmed entries stay standalone.\n\n## Candidates\n\n",
    );
    for m in &members {
        user.push_str(&format!("- id: `{}`\n", m.id));
        user.push_str(&format!("  content: {}\n", m.content));
        user.push_str(&format!(
            "  scope: {}\n",
            format_scope(m.metadata.effective_scope())
        ));
    }
    user.push_str(
        "\nReturn a JSON object: `{\"confirmed\": [\"id1\", \"id2\", ...]}` \
        listing the ids that truly belong to a single merged entry. If fewer \
        than 2 belong, return `{\"confirmed\": []}`.",
    );

    let system = String::from(
        "You confirm whether a transitively-built merge cluster is genuine. \
        Be conservative: it's better to split a real cluster than to merge \
        unrelated entries.",
    );

    let response = match llm_fn(system, user).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("dreaming: confirm cluster LLM call failed: {e}");
            return BTreeSet::new();
        }
    };

    parse_confirmed_ids(&response, cluster)
}

fn parse_confirmed_ids(response: &str, cluster: &BTreeSet<String>) -> BTreeSet<String> {
    let payload = extract_json_payload(response.trim());
    let parsed: serde_json::Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(_) => return BTreeSet::new(),
    };
    let arr = match parsed.get("confirmed").and_then(|v| v.as_array()) {
        Some(a) => a.clone(),
        None => return BTreeSet::new(),
    };
    arr.into_iter()
        .filter_map(|v| v.as_str().map(String::from))
        .filter(|id| cluster.contains(id))
        .collect()
}

async fn load_cluster_members(
    pool: &PgPool,
    cluster: &BTreeSet<String>,
    id_to_focal: &HashMap<String, &KbRow>,
) -> Vec<KbRow> {
    // Use focal rows when available (already in memory); otherwise hit DB.
    let mut have: Vec<KbRow> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for id in cluster {
        if let Some(row) = id_to_focal.get(id) {
            have.push((*row).clone());
        } else {
            missing.push(id.clone());
        }
    }
    if !missing.is_empty() {
        type KbRowTuple = (String, String, Vec<String>, serde_json::Value);
        let rows: Result<Vec<KbRowTuple>, _> = sqlx::query_as(
            "SELECT id, content, tags, metadata FROM knowledge_base
             WHERE id = ANY($1) AND deleted_at IS NULL",
        )
        .bind(&missing)
        .fetch_all(pool)
        .await;
        if let Ok(rows) = rows {
            for (id, content, tags, metadata_json) in rows {
                have.push(KbRow {
                    id,
                    content,
                    tags,
                    metadata: KbMetadata::from_json(&metadata_json),
                    distance: f64::NAN,
                });
            }
        }
    }
    have
}

async fn synthesize_cluster(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    cluster: &BTreeSet<String>,
    id_to_focal: &HashMap<String, &KbRow>,
) -> Result<SynthesizedMerge, String> {
    let members = load_cluster_members(pool, cluster, id_to_focal).await;
    if members.len() < 2 {
        return Err("synthesize: fewer than 2 members".to_string());
    }

    let canonical_id = OpBuffer::canonical_of(cluster)
        .cloned()
        .ok_or_else(|| "synthesize: empty cluster".to_string())?;

    let mut user = String::from(
        "Merge the following entries into a single unified knowledge-base \
        entry. Preserve all factual content; remove only redundancy. The \
        merged entry must keep the same scope as the inputs (they should \
        already share scope at this stage).\n\n## Entries to merge\n\n",
    );
    for m in &members {
        user.push_str(&format!("- id: `{}`\n", m.id));
        user.push_str(&format!("  content: {}\n", m.content));
        user.push_str(&format!(
            "  scope: {}\n",
            format_scope(m.metadata.effective_scope())
        ));
    }
    user.push_str(
        "\nReturn JSON: `{\"content\": \"<unified prose sentence(s)>\", \
        \"scope\": null | {<dimensions>}}`. The scope should reflect the \
        agreed-upon scope of the inputs.",
    );

    let system = String::from(
        "You synthesize a set of related knowledge-base entries into a single \
        unified entry. Preserve all factual claims; deduplicate redundant \
        phrasing; keep the entry self-contained and concise.",
    );

    let response = llm_fn(system, user).await?;
    let payload = extract_json_payload(response.trim());
    let parsed: serde_json::Value = serde_json::from_str(&payload)
        .map_err(|e| format!("synthesize: response not JSON: {e}"))?;

    let new_content = parsed
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "synthesize: missing content".to_string())?;
    if new_content.is_empty() {
        return Err("synthesize: empty content".to_string());
    }

    let new_scope = match parsed.get("scope") {
        Some(serde_json::Value::Object(map)) => {
            let mut scope = KbScope::new();
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    scope = scope.with(k.clone(), s.to_string());
                }
            }
            if scope.is_empty() { None } else { Some(scope) }
        }
        _ => None,
    };

    Ok(SynthesizedMerge {
        canonical_id,
        member_ids: cluster.iter().cloned().collect(),
        new_content,
        new_scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_keep_action() {
        let r = parse_review_response(r#"{"actions": [{"op": "keep"}]}"#);
        assert_eq!(r.len(), 1);
        assert!(matches!(r[0], ReviewAction::Keep));
    }

    #[test]
    fn parses_multiple_actions() {
        let r = parse_review_response(
            r#"{"actions": [
              {"op": "add_scope", "scope": {"project": "adelie-ai"}},
              {"op": "merge_with", "target_id": "xyz"}
            ]}"#,
        );
        assert_eq!(r.len(), 2);
        assert!(matches!(r[0], ReviewAction::AddScope(_)));
        assert!(matches!(r[1], ReviewAction::MergeWith(_)));
    }

    #[test]
    fn empty_actions_defaults_to_keep() {
        let r = parse_review_response(r#"{"actions": []}"#);
        assert!(matches!(r[0], ReviewAction::Keep));
    }

    #[test]
    fn invalid_json_defaults_to_keep() {
        let r = parse_review_response("not json");
        assert!(matches!(r[0], ReviewAction::Keep));
    }

    #[test]
    fn delete_without_reason_uses_placeholder() {
        let r = parse_review_response(r#"{"actions":[{"op":"delete"}]}"#);
        assert_eq!(r.len(), 1);
        if let ReviewAction::Delete(reason) = &r[0] {
            assert_eq!(reason, "no reason given");
        } else {
            panic!("expected delete");
        }
    }

    #[test]
    fn unknown_op_skipped_but_others_kept() {
        let r = parse_review_response(
            r#"{"actions":[{"op":"weird"},{"op":"keep"}]}"#,
        );
        assert_eq!(r.len(), 1);
        assert!(matches!(r[0], ReviewAction::Keep));
    }

    #[test]
    fn confirm_ids_filters_to_cluster_membership() {
        let cluster: BTreeSet<String> =
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let response = r#"{"confirmed": ["a", "b", "d"]}"#;
        let confirmed = parse_confirmed_ids(response, &cluster);
        assert_eq!(confirmed.len(), 2);
        assert!(confirmed.contains("a"));
        assert!(confirmed.contains("b"));
        assert!(!confirmed.contains("d"));
    }
}
