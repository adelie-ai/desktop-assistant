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
//!
//! The whole lookup carries a second ceiling, [`RECALL_CALL_CEILING`], because
//! the embedding timeout bounds only the embedding: the database round trips
//! around it are bounded by the connection pool acquire timeout, which is
//! measured in tens of seconds. Recall runs before every turn's first round, so
//! a saturated pool would otherwise hold each turn far longer than the embedding
//! timeout suggests.

use std::future::Future;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::embedding::EmbedFn;
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

/// How long one whole recall lookup may take before the turn gives up on it.
///
/// The same shape and the same value as the knowledge-base write tool's
/// `TAG_RESOLVE_CALL_CEILING`, and for the same reason: the embedding is
/// bounded on its own, and the database round trips around it are bounded only
/// by the pool acquire timeout. The value leaves [`EMBED_TIMEOUT`] its full five
/// seconds and five more for the reads around it.
///
/// Exceeding it costs the block, never the turn.
const RECALL_CALL_CEILING: std::time::Duration = std::time::Duration::from_secs(10);

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
        Box::pin(async move {
            within_ceiling(lookup(&kb_store, &pool, &embed, &embedding_model, request)).await
        })
    })
}

/// Hold one lookup to [`RECALL_CALL_CEILING`].
///
/// Separate from [`lookup`] so the ceiling can be proven without a database:
/// what it has to guarantee is that some answer comes back inside the ceiling,
/// whatever inside the lookup is slow.
async fn within_ceiling(
    call: impl Future<Output = Result<RecallCandidates, CoreError>>,
) -> Result<RecallCandidates, CoreError> {
    match tokio::time::timeout(RECALL_CALL_CEILING, call).await {
        Ok(result) => result,
        Err(_) => Err(CoreError::Storage(format!(
            "recall lookup exceeded {RECALL_CALL_CEILING:?}"
        ))),
    }
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
        //
        // `search_text_any_term`, not the trait's `search_text`: that one joins
        // the query's lexemes with AND, which asks a whole user sentence to
        // appear in one entry and answers almost nothing.
        let entries = kb_store
            .search_text_any_term(&request.prompt, request.entry_limit)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An embedding backend that answers with `answer`, after `delay`.
    fn backend(answer: Result<Vec<Vec<f32>>, CoreError>, delay: std::time::Duration) -> EmbedFn {
        let answer = Arc::new(std::sync::Mutex::new(Some(answer)));
        Arc::new(move |_texts| {
            let answer = Arc::clone(&answer);
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                answer
                    .lock()
                    .expect("the test backend is not poisoned")
                    .take()
                    .unwrap_or_else(|| Ok(vec![vec![0.0]]))
            })
        })
    }

    #[tokio::test]
    async fn a_working_backend_yields_the_vector_it_produced() {
        let embed = backend(Ok(vec![vec![0.1, 0.2, 0.3]]), std::time::Duration::ZERO);

        assert_eq!(
            embed_prompt(&embed, "where does the registry live?").await,
            Some(vec![0.1, 0.2, 0.3])
        );
    }

    #[tokio::test]
    async fn a_slow_backend_degrades_rather_than_holding_the_turn() {
        // The whole point of the ceiling: a wedged embedder must cost recall
        // its semantic arm, never the turn's latency budget.
        tokio::time::pause();
        let embed = backend(
            Ok(vec![vec![0.1]]),
            EMBED_TIMEOUT + std::time::Duration::from_secs(1),
        );

        assert_eq!(embed_prompt(&embed, "a prompt").await, None);
    }

    #[tokio::test]
    async fn a_failing_backend_degrades() {
        let embed = backend(
            Err(CoreError::Storage("backend down".into())),
            std::time::Duration::ZERO,
        );

        assert_eq!(embed_prompt(&embed, "a prompt").await, None);
    }

    #[tokio::test]
    async fn a_lookup_that_never_answers_costs_the_block_and_not_the_turn() {
        // The embedding timeout bounds only the embedding. The reads around it
        // are bounded by the pool acquire timeout, measured in tens of seconds,
        // and recall runs before every turn's first round.
        tokio::time::pause();

        let answer = within_ceiling(async {
            tokio::time::sleep(RECALL_CALL_CEILING * 2).await;
            Ok(RecallCandidates::default())
        })
        .await;

        assert!(
            answer.is_err(),
            "a lookup past the ceiling must answer with an error the caller drops"
        );
    }

    #[tokio::test]
    async fn a_lookup_that_answers_inside_the_ceiling_passes_through() {
        tokio::time::pause();

        let answer = within_ceiling(async {
            tokio::time::sleep(RECALL_CALL_CEILING / 2).await;
            Ok(RecallCandidates {
                tags: vec![RecallTag {
                    name: "topic:mine".into(),
                    relevance: RecallRelevance::Distance(0.1),
                }],
                ..RecallCandidates::default()
            })
        })
        .await
        .expect("inside the ceiling the answer travels");

        assert_eq!(answer.tags.len(), 1);
    }

    #[tokio::test]
    async fn a_backend_that_answers_with_no_vector_degrades() {
        // An empty batch, and an empty vector, are both "no embedding". Passing
        // a zero-dimension vector on to the queries would raise rather than
        // miss, which would cost the turn instead of the block.
        for answer in [vec![], vec![vec![]]] {
            let embed = backend(Ok(answer), std::time::Duration::ZERO);
            assert_eq!(embed_prompt(&embed, "a prompt").await, None);
        }
    }
}
