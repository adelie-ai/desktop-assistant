//! Adapter behind the pre-prompt recall port (#1100, #1101).
//!
//! One user prompt, one embedding, three indexes. The knowledge base answers
//! with the entries nearest the prompt and how near each is; this
//! conversation's scratchpad answers the same way about its own notes; the tag
//! registry answers with the names of the tags nearest the prompt, read from
//! vectors the near-duplicate check already built. The core decides what clears
//! its relevance floor and how the `[Recall]` block reads.
//!
//! ## Recall never fails a turn
//!
//! The embedding call is bounded by [`EMBED_TIMEOUT`], the same ceiling the
//! knowledge-base search tool already applies. On timeout, or on an embedding
//! error, the knowledge and scratchpad arms degrade to full-text search (the
//! precedent is #195) and the tag arm goes quiet, because the registry carries
//! no full-text index to fall back to. A degradation is logged once, here,
//! rather than once per arm.
//!
//! An arm that fails outright is a narrower loss than a lookup that fails. The
//! scratchpad arm reads a different table from the other two, so it can fail on
//! its own, and when it does it costs its own lines and nothing else - see
//! [`notes_or_none`]. If a degraded read fails as well, the error travels to the
//! caller, which drops the block and runs the turn.
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
use desktop_assistant_core::domain::ScratchpadNote;
use desktop_assistant_core::ports::embedding::{EMBED_TIMEOUT, EmbedFn};
use desktop_assistant_core::ports::recall::{
    RecallCandidates, RecallEntry, RecallNote, RecallRelevance, RecallRequest, RecallSearchFn,
    RecallTag,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgPool, PgScratchpadStore};

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
    // The pad adapter is a handle on the same pool, built once here rather than
    // threaded in: nothing else in the daemon holds one, and the two reads
    // behind the scratchpad arm are inherent to it.
    let pad = Arc::new(PgScratchpadStore::new(pool.clone()));
    Arc::new(move |request: RecallRequest| {
        let kb_store = Arc::clone(&kb_store);
        let pad = Arc::clone(&pad);
        let pool = pool.clone();
        let embed = Arc::clone(&embed);
        let embedding_model = embedding_model.clone();
        Box::pin(async move {
            within_ceiling(lookup(
                &kb_store,
                &pad,
                &pool,
                &embed,
                &embedding_model,
                request,
            ))
            .await
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

/// One lookup: embed once, then ask every index.
async fn lookup(
    kb_store: &PgKnowledgeBaseStore,
    pad: &PgScratchpadStore,
    pool: &PgPool,
    embed: &EmbedFn,
    embedding_model: &str,
    request: RecallRequest,
) -> Result<RecallCandidates, CoreError> {
    let Some(vector) = embed_prompt(embed, &request.prompt).await else {
        // Degraded: full-text for the entries and the notes, silence for the
        // tags.
        //
        // `search_text_any_term` on both, not the trait's `search_text`: that
        // one joins the query's lexemes with AND, which asks a whole user
        // sentence to appear in one row and answers almost nothing.
        return gather(
            async {
                Ok(kb_store
                    .search_text_any_term(&request.prompt, request.entry_limit)
                    .await?
                    .into_iter()
                    .map(|entry| RecallEntry {
                        entry,
                        relevance: RecallRelevance::LexicalMatch,
                    })
                    .collect())
            },
            async { Ok(Vec::new()) },
            async {
                Ok(pad
                    .search_text_any_term(
                        &request.conversation_id,
                        &request.prompt,
                        request.note_limit,
                    )
                    .await?
                    .into_iter()
                    .map(|note| to_recall_note(note, RecallRelevance::LexicalMatch))
                    .collect())
            },
        )
        .await;
    };

    // Every arm shares the one vector, and none depends on another.
    let vector_for_tags = vector.clone();
    let vector_for_notes = vector.clone();
    gather(
        async {
            Ok(kb_store
                .nearest_by_embedding(vector, embedding_model, request.entry_limit)
                .await?
                .into_iter()
                .map(|(entry, distance)| RecallEntry {
                    entry,
                    relevance: RecallRelevance::Distance(distance),
                })
                .collect())
        },
        async {
            Ok(desktop_assistant_storage::tag_registry::nearest_tags(
                pool,
                vector_for_tags,
                embedding_model,
                request.tag_limit,
            )
            .await?
            .into_iter()
            .map(|(name, distance)| RecallTag {
                name,
                relevance: RecallRelevance::Distance(distance),
            })
            .collect())
        },
        async {
            Ok(pad
                .nearest_by_embedding(
                    &request.conversation_id,
                    vector_for_notes,
                    embedding_model,
                    request.note_limit,
                )
                .await?
                .into_iter()
                .map(|(note, distance)| to_recall_note(note, RecallRelevance::Distance(distance)))
                .collect())
        },
    )
    .await
}

/// Run the three arms together and fold what they answered into one candidate
/// set.
///
/// Generic over the futures, and separate from [`lookup`], so both halves of
/// what it guarantees are provable without a database - which is the only way to
/// hold either to anything.
///
/// **`join!`, never `try_join!`.** The arms do not depend on each other, and one
/// arm's error must not cancel the two that were answering.
///
/// **The scratchpad arm's error is absorbed; the other two propagate.** A
/// knowledge arm that cannot read is the block's whole point failing, and the
/// caller drops the block and runs the turn anyway; losing the pad lines is the
/// smaller loss, so it is taken here rather than passed on. The absorbed arm
/// resolves first, so its failure is logged even on the turn where another
/// arm's error is about to end the lookup.
async fn gather(
    entries: impl Future<Output = Result<Vec<RecallEntry>, CoreError>>,
    tags: impl Future<Output = Result<Vec<RecallTag>, CoreError>>,
    notes: impl Future<Output = Result<Vec<RecallNote>, CoreError>>,
) -> Result<RecallCandidates, CoreError> {
    let (entries, tags, notes) = tokio::join!(entries, tags, notes);
    let notes = notes_or_none(notes);
    Ok(RecallCandidates {
        entries: entries?,
        notes,
        tags: tags?,
    })
}

/// One stored note as a recall candidate.
///
/// The key, the content and the pin travel; nothing is rendered here. How much
/// of a note a line may spend, and whether a pinned note belongs in the block at
/// all, are the core's decisions.
fn to_recall_note(note: ScratchpadNote, relevance: RecallRelevance) -> RecallNote {
    RecallNote {
        key: note.key,
        content: note.content,
        pinned: note.pinned,
        relevance,
    }
}

/// The scratchpad arm's rows, or none.
///
/// The arm reads a different table from the other two, so it fails on its own -
/// and when it does it must cost its own lines and nothing else. The knowledge
/// and tag arms still render, and the turn never sees the error.
///
/// This is deliberately narrower than the treatment the other two arms get: a
/// knowledge arm that cannot read is the block's whole point failing, so that
/// error travels to the caller, which drops the block and runs the turn anyway.
/// Losing the pad lines is a smaller loss than losing the block.
fn notes_or_none(found: Result<Vec<RecallNote>, CoreError>) -> Vec<RecallNote> {
    match found {
        Ok(notes) => notes,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "recall: the scratchpad arm failed; the other arms still render"
            );
            Vec::new()
        }
    }
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

    fn an_entry() -> RecallEntry {
        RecallEntry {
            entry: desktop_assistant_core::domain::KnowledgeEntry::new("kb-1", "body", vec![]),
            relevance: RecallRelevance::Distance(0.12),
        }
    }

    fn a_tag() -> RecallTag {
        RecallTag {
            name: "topic:mine".into(),
            relevance: RecallRelevance::Distance(0.12),
        }
    }

    fn a_note() -> RecallNote {
        RecallNote {
            key: "deploy-window".into(),
            content: "Fridays after 18:00".into(),
            pinned: false,
            relevance: RecallRelevance::Distance(0.12),
        }
    }

    /// Acceptance (#1101): the scratchpad arm reads a different table from the
    /// other two, so it fails on its own. When it does it costs its own lines
    /// and nothing else - the knowledge and tag arms still render, and the turn
    /// never sees the error.
    #[tokio::test]
    async fn recall_block_survives_the_scratchpad_arm_failing() {
        let candidates = gather(
            async { Ok(vec![an_entry()]) },
            async { Ok(vec![a_tag()]) },
            async { Err(CoreError::Storage("the pad read failed".into())) },
        )
        .await
        .expect("a failed pad read must not fail the lookup");

        assert_eq!(
            candidates.entries.len(),
            1,
            "the knowledge arm still renders"
        );
        assert_eq!(candidates.tags.len(), 1, "and so does the tag arm");
        assert!(
            candidates.notes.is_empty(),
            "the failed arm contributes none"
        );
    }

    #[tokio::test]
    async fn the_scratchpad_arm_passes_its_rows_through_when_it_answers() {
        let candidates = gather(async { Ok(vec![]) }, async { Ok(vec![]) }, async {
            Ok(vec![a_note()])
        })
        .await
        .expect("an arm that answers is not a failure");

        assert_eq!(candidates.notes.len(), 1);
        assert_eq!(candidates.notes[0].key, "deploy-window");
    }

    /// The asymmetry, stated as a test so it cannot be levelled by accident.
    /// The knowledge arm is the block's whole point, so its error ends the
    /// lookup and the caller drops the block.
    #[tokio::test]
    async fn a_failing_knowledge_arm_ends_the_lookup() {
        let answer = gather(
            async { Err(CoreError::Storage("the store is down".into())) },
            async { Ok(vec![a_tag()]) },
            async { Ok(vec![a_note()]) },
        )
        .await;

        assert!(answer.is_err());
    }

    /// The arms must run together, not one after another: three reads and an
    /// embedding sit inside a ten-second whole-lookup ceiling, and a serial
    /// fold would spend the budget three times over.
    #[tokio::test(start_paused = true)]
    async fn the_arms_run_together_rather_than_one_after_another() {
        let hold = std::time::Duration::from_secs(4);
        let started = tokio::time::Instant::now();

        gather(
            async move {
                tokio::time::sleep(hold).await;
                Ok(vec![an_entry()])
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok(vec![a_tag()])
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok(vec![a_note()])
            },
        )
        .await
        .expect("all three answered");

        assert!(
            started.elapsed() < hold * 2,
            "three arms took {:?}, which is serial rather than concurrent",
            started.elapsed()
        );
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
