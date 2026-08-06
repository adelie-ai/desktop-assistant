//! Adapter behind the pre-prompt recall port (#1100).
//!
//! One user prompt, one embedding, two indexes. The knowledge base answers with
//! the entries nearest the prompt and how near each is; the tag registry
//! answers with the names of the tags nearest it, read from vectors the
//! near-duplicate check already built. The core decides what clears its
//! relevance floor and how the `[Recall]` block reads.
//!
//! ## Recall never fails a turn
//!
//! The embedding call is bounded by [`EMBED_TIMEOUT`], the same ceiling the
//! knowledge-base search tool already applies. On timeout, or on an embedding
//! error, the knowledge arm degrades to full-text search (the precedent is
//! #195) and the tag arm goes quiet, because the registry carries no full-text
//! index to fall back to. A degradation is logged once, here, rather than once
//! per arm. If the degraded read fails as well, the error travels to the caller,
//! which drops the block and runs the turn.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::recall::{
    RecallCandidates, RecallEntry, RecallRelevance, RecallRequest, RecallSearchFn, RecallTag,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgPool};

/// Hard cap on how long the recall embedding may block the start of a turn.
///
/// The same five seconds `BuiltinToolService` gives a search query's embedding:
/// a wedged embedding backend must cost recall its semantic arm, never the
/// turn's latency budget.
const EMBED_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Build the recall lookup the conversation handler calls once per turn.
///
/// `embedding_model` identifies the model behind `embed` and travels with every
/// vector it produces: both indexes scope their vector arm to it, because a
/// comparison against a row embedded by another model is a comparison across
/// vector dimensions.
pub fn build_recall_search(
    kb_store: Arc<PgKnowledgeBaseStore>,
    pool: PgPool,
    embed: EmbedFn,
    embedding_model: String,
) -> RecallSearchFn {
    Arc::new(move |request: RecallRequest| {
        let kb_store = Arc::clone(&kb_store);
        let pool = pool.clone();
        let embed = Arc::clone(&embed);
        let embedding_model = embedding_model.clone();
        Box::pin(async move { lookup(&kb_store, &pool, &embed, &embedding_model, request).await })
    })
}

/// One lookup: embed once, then ask both indexes.
async fn lookup(
    kb_store: &PgKnowledgeBaseStore,
    pool: &PgPool,
    embed: &EmbedFn,
    embedding_model: &str,
    request: RecallRequest,
) -> Result<RecallCandidates, CoreError> {
    let Some(vector) = embed_prompt(embed, &request.prompt).await else {
        // Degraded: full-text for the entries, silence for the tags.
        let entries = kb_store
            .search_text(&request.prompt, None, request.entry_limit)
            .await?
            .into_iter()
            .map(|entry| RecallEntry {
                entry,
                relevance: RecallRelevance::LexicalMatch,
            })
            .collect();
        return Ok(RecallCandidates {
            entries,
            tags: Vec::new(),
        });
    };

    // Both arms share the one vector, and neither depends on the other.
    let (entries, tags) = tokio::try_join!(
        kb_store.nearest_by_embedding(vector.clone(), embedding_model, request.entry_limit),
        desktop_assistant_storage::tag_registry::nearest_tags(
            pool,
            vector,
            embedding_model,
            request.tag_limit,
        ),
    )?;

    Ok(RecallCandidates {
        entries: entries
            .into_iter()
            .map(|(entry, distance)| RecallEntry {
                entry,
                relevance: RecallRelevance::Distance(distance),
            })
            .collect(),
        tags: tags
            .into_iter()
            .map(|(name, distance)| RecallTag {
                name,
                relevance: RecallRelevance::Distance(distance),
            })
            .collect(),
    })
}

/// Embed the prompt, bounded by [`EMBED_TIMEOUT`]. `None` means the arms
/// degrade; the reason is logged once here, not once per arm.
async fn embed_prompt(embed: &EmbedFn, prompt: &str) -> Option<Vec<f32>> {
    match tokio::time::timeout(EMBED_TIMEOUT, embed(vec![prompt.to_string()])).await {
        Ok(Ok(mut vectors)) => vectors.pop().filter(|v| !v.is_empty()),
        Ok(Err(e)) => {
            tracing::warn!(
                error = %e,
                "recall: embedding failed; degrading to full-text and skipping the tag arm"
            );
            None
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?EMBED_TIMEOUT,
                "recall: embedding timed out; degrading to full-text and skipping the tag arm"
            );
            None
        }
    }
}
