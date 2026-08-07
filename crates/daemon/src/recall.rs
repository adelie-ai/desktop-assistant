//! Adapter behind the pre-prompt recall port (#1100, #1101).
//!
//! One user prompt, one embedding, two indexes. The knowledge base answers with
//! the entries nearest the prompt and how near each is; this conversation's
//! scratchpad answers the same way about its own notes. The core decides what
//! clears the bar and how the `[Recall]` block reads.
//!
//! ## What a distance is worth, and who says so
//!
//! A cosine distance means nothing on its own, so the core reads each candidate
//! against the spread of the source it came from. Only the store can say what
//! its own geometry is, so the knowledge arm's read answers with both: the
//! candidates, and the spread of the same query's distances over every row the
//! scan could reach. One scan states both, so the spread always describes the
//! query whose candidates it grades.
//!
//! ## What the use log adds, and what it costs
//!
//! A candidate the store measured also carries what the use log knows about it
//! (#698), which is the reinforcement half of its activation score (#1123). It
//! is one batched read after the scan rather than a join inside it, so that a
//! slow or broken log costs the order of the lines and not the lines - see
//! [`use_records`], which bounds it and degrades to nothing, and the comment at
//! the call site for what that placement costs. A lexical candidate carries no
//! record, because nothing ranks it by activation.
//!
//! ## Recall never fails a turn
//!
//! The embedding call is bounded by [`EMBED_TIMEOUT`], the same ceiling the
//! knowledge-base search tool already applies. On timeout, or on an embedding
//! error, both arms degrade to full-text search (the precedent is #195) and no
//! dispersion is measured, because a full-text match carries no distance to read
//! against one. A degradation is logged once, here, rather than once per arm.
//!
//! An arm that fails outright is a narrower loss than a lookup that fails. The
//! scratchpad arm reads a different table from the knowledge arm, so it can fail
//! on its own, and when it does it costs its own lines and nothing else - see
//! [`notes_or_none`]. The measurement carries its own ceiling for the same
//! reason. If a degraded read fails as well, the error travels to the caller,
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
use desktop_assistant_core::domain::ScratchpadNote;
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::situation::{SituationCue, SituationRecord};
use desktop_assistant_core::ports::embedding::{EMBED_TIMEOUT, EmbedFn};
use desktop_assistant_core::ports::knowledge_use::{KnowledgeUseLog, current_situation};
use desktop_assistant_core::ports::recall::{
    RecallCandidates, RecallDispersion, RecallEntry, RecallNote, RecallRelevance, RecallRequest,
    RecallSearchFn,
};
use desktop_assistant_storage::{
    PgKnowledgeBaseStore, PgKnowledgeUseLog, PgPool, PgScratchpadStore,
};

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

/// How long the use-log read may take before the candidates travel without it.
///
/// Two reads by primary key over at most one scan's worth of ids, so half a
/// second is already generous; what the ceiling buys is the guarantee that the
/// reinforcement half of the score can never be what makes a turn slow. Exceed
/// it and every candidate ranks on its semantic signal alone, which is how they
/// ranked before the log existed - see [`use_records`].
///
/// The read states the same figure to the database as its own
/// `statement_timeout`
/// ([`USE_LOG_READ_STATEMENT_TIMEOUT`](desktop_assistant_storage::USE_LOG_READ_STATEMENT_TIMEOUT)),
/// so giving up here also stops the backend working;
/// `the_use_log_read_gives_up_no_later_than_the_database_does` holds the two
/// together.
///
/// It fits inside what [`RECALL_CALL_CEILING`] has left after the embedding and
/// the scan, and `the_three_reads_together_stay_inside_the_lookups_ceiling`
/// holds the four constants to that. It halves the slack that ceiling keeps for
/// what none of the three bounds - pool acquisition above all, which
/// `RECALL_SCAN_STATEMENT_TIMEOUT` excludes because it is a server-side
/// statement timeout. Half a second of slack is thin, and the lookup ceiling is
/// what gives up first if it runs out, which costs the block and never the turn.
const USE_LOG_READ_CEILING: std::time::Duration = std::time::Duration::from_millis(500);

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
    // The pad adapter and the use log are handles on the same pool, built once
    // here rather than threaded in: nothing else in the daemon holds the pad
    // one, and the reads behind these arms are inherent to them.
    let pad = Arc::new(PgScratchpadStore::new(pool.clone()));
    let uses = Arc::new(PgKnowledgeUseLog::new(pool.clone()));
    Arc::new(move |request: RecallRequest| {
        let kb_store = Arc::clone(&kb_store);
        let pad = Arc::clone(&pad);
        let uses = Arc::clone(&uses);
        let embed = Arc::clone(&embed);
        let embedding_model = embedding_model.clone();
        Box::pin(async move {
            within_ceiling(lookup(
                &kb_store,
                &pad,
                &uses,
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
    uses: &PgKnowledgeUseLog,
    embed: &EmbedFn,
    embedding_model: &str,
    request: RecallRequest,
) -> Result<RecallCandidates, CoreError> {
    let Some(vector) = embed_prompt(embed, &request.prompt).await else {
        // Degraded: full-text for both arms, and no dispersion. A full-text row
        // carries no distance, so there is nothing to read against a spread.
        //
        // `search_text_any_term` on both, not the trait's `search_text`: that
        // one joins the query's lexemes with AND, which asks a whole user
        // sentence to appear in one row and answers almost nothing.
        return gather(
            async {
                Ok((
                    kb_store
                        .search_text_any_term(&request.prompt, request.entry_limit)
                        .await?
                        .into_iter()
                        // No use records are read here, and that is deliberate.
                        // A lexical candidate carries no distance, so it has no
                        // semantic term, so the core does not rank it by
                        // activation (#1123) and a record would be a query
                        // whose answer nothing reads - at exactly the moment
                        // something upstream is already failing.
                        .map(|entry| RecallEntry::new(entry, RecallRelevance::LexicalMatch))
                        .collect(),
                    None,
                    None,
                ))
            },
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

    // Both arms share the one vector, and neither depends on the other.
    let vector_for_notes = vector.clone();
    gather(
        async {
            let found = kb_store
                .nearest_by_embedding(vector, embedding_model, request.entry_limit)
                .await?;
            // One batched read after the scan, of at most one scan's worth of
            // ids, by primary key on both tables.
            //
            // The alternative was a join inside the scan's own statement, which
            // the ids are reachable from and which would cost no extra round
            // trip. This shape was chosen anyway, for two reasons. The log is a
            // separate adapter behind a separate port, and folding its two
            // tables into the recall scan would put a fourth job in a statement
            // whose whole documented virtue is that it does three in one pass.
            // And a joined read cannot degrade on its own: a slow or broken use
            // log would cost the block, where this costs only the order - see
            // `use_records`. The price is one round trip per turn, paid on every
            // turn including the many where the bar admits nothing and every
            // record read is discarded.
            let ids: Vec<String> = found.entries.iter().map(|(e, _)| e.id.clone()).collect();
            // The situation this turn arrived in, read once and against the
            // whole store. It is derived from the clock and what the client
            // reported, so it costs no model call and no extra work on the
            // write path - see `desktop_assistant_core::domain::situation`.
            let here = current_situation();
            let (mut records, mut situations, cue) = if ids.is_empty() {
                (
                    std::collections::HashMap::new(),
                    std::collections::HashMap::new(),
                    None,
                )
            } else {
                let cue_read = uses.situation_cue(here.clone());
                let (records, (situations, cue)) = tokio::join!(
                    use_records(uses.records(ids.clone())),
                    situation_signal(uses.situations(ids), cue_read),
                );
                (records, situations, cue)
            };
            // How much of the block's order the use log actually decided. An
            // operator meeting a block that looks different, or one that looks
            // exactly as it always did, cannot otherwise tell whether
            // reinforcement is in force - the same reason
            // `how_the_distances_are_read` exists. Two counts, and nothing of
            // what any entry holds: this runs on every turn.
            tracing::debug!(
                candidates = found.entries.len(),
                with_use_record = records.len(),
                with_situation_record = situations.len(),
                situation_cue = cue.as_ref().map_or(0, |cue: &SituationCue| {
                    cue.situation().iter().count()
                }),
                "recall: how many candidates the use log and the situation had something to \
                 say about"
            );
            Ok((
                found
                    .entries
                    .into_iter()
                    .map(|(entry, distance)| {
                        let record = records.remove(&entry.id);
                        let seen_in = situations.remove(&entry.id).unwrap_or_default();
                        RecallEntry::new(entry, RecallRelevance::Distance(distance))
                            .with_use_record(record)
                            .with_situation(seen_in)
                    })
                    .collect(),
                found.dispersion,
                cue,
            ))
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

/// What the use log knows about `ids`, keyed by id, and an empty map where it
/// could not be read (#1123).
///
/// **A read that fails costs the ranking and never the block.** The
/// reinforcement half of the activation score is the half retrieval worked
/// without until now, so an entry with no record ranks on its semantic signal
/// alone - which is exactly how every entry ranked before the log existed. A
/// slow or broken use log therefore degrades the order of the lines rather than
/// removing them, and it says so once in the journal.
///
/// Ids the log has never seen are simply absent, so they get the same `None` as
/// a failed read. The core does not distinguish the two and does not have to:
/// both mean "nothing to add".
/// Generic over the read, and separate from [`lookup`], so both halves of what
/// it guarantees are provable without a database - the same reason
/// [`within_ceiling`] and [`gather`] stand on their own.
async fn use_records(
    read: impl Future<Output = Result<Vec<KnowledgeUseRecord>, CoreError>>,
) -> std::collections::HashMap<String, KnowledgeUseRecord> {
    let records = match tokio::time::timeout(USE_LOG_READ_CEILING, read).await {
        Ok(Ok(records)) => records,
        Ok(Err(e)) => {
            tracing::warn!(
                error = %e,
                "recall: the use log could not be read; ranking on the semantic signal alone"
            );
            return std::collections::HashMap::new();
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?USE_LOG_READ_CEILING,
                "recall: the use log read timed out; ranking on the semantic signal alone"
            );
            return std::collections::HashMap::new();
        }
    };
    records
        .into_iter()
        .map(|record| (record.entry_id.clone(), record))
        .collect()
}

/// The situations each candidate has been seen in, and the present situation
/// read against the store (#1125).
///
/// The same bargain [`use_records`] makes, for the same reason and under the
/// same ceiling: a read that fails costs the order of the lines and never the
/// lines. Without it every entry ranks on its distance and its use log alone,
/// which is how every entry ranked before the cue existed.
///
/// Both halves degrade together. They are two views of one signal - a record
/// nothing can grade scores zero, and a cue nothing carries a record for scores
/// zero as well - so half an answer is worth exactly what no answer is worth,
/// and there is nothing to be gained by keeping one when the other is gone.
///
/// Generic over the reads, and separate from [`lookup`], so what it guarantees
/// is provable without a database.
async fn situation_signal(
    records: impl Future<Output = Result<Vec<(String, SituationRecord)>, CoreError>>,
    cue: impl Future<Output = Result<Option<SituationCue>, CoreError>>,
) -> (
    std::collections::HashMap<String, SituationRecord>,
    Option<SituationCue>,
) {
    let both = async { tokio::join!(records, cue) };
    let (records, cue) = match tokio::time::timeout(USE_LOG_READ_CEILING, both).await {
        Ok(pair) => pair,
        Err(_) => {
            tracing::warn!(
                timeout = ?USE_LOG_READ_CEILING,
                "recall: the situation read timed out; ranking without the situation cue"
            );
            return (std::collections::HashMap::new(), None);
        }
    };
    let (Ok(records), Ok(cue)) = (records, cue) else {
        tracing::warn!("recall: the situation could not be read; ranking without the cue");
        return (std::collections::HashMap::new(), None);
    };
    (records.into_iter().collect(), cue)
}

/// Run the two arms together and fold what they answered into one candidate
/// set.
///
/// Generic over the futures, and separate from [`lookup`], so both halves of
/// what it guarantees are provable without a database - which is the only way to
/// hold either to anything.
///
/// **`join!`, never `try_join!`.** The arms do not depend on each other, and one
/// arm's error must not cancel the one that was answering.
///
/// **The scratchpad arm's error is absorbed; the knowledge arm's propagates.** A
/// knowledge arm that cannot read is the block's whole point failing, and the
/// caller drops the block and runs the turn anyway; losing the pad lines is the
/// smaller loss, so it is taken here rather than passed on. The absorbed arm
/// resolves first, so its failure is logged even on the turn where the other
/// arm's error is about to end the lookup.
///
/// The knowledge arm answers with the store's own spread beside its candidates,
/// because one scan states both.
async fn gather(
    entries: impl Future<
        Output = Result<
            (
                Vec<RecallEntry>,
                Option<RecallDispersion>,
                Option<SituationCue>,
            ),
            CoreError,
        >,
    >,
    notes: impl Future<Output = Result<Vec<RecallNote>, CoreError>>,
) -> Result<RecallCandidates, CoreError> {
    let (entries, notes) = tokio::join!(entries, notes);
    let notes = notes_or_none(notes);
    let (entries, entry_dispersion, situation_cue) = entries?;
    // Which sources stated their own geometry, and which the block will read by
    // a stated estimate. Without this an operator meeting a block that is
    // quieter than expected cannot tell whether the dimensionless bar is in
    // force or whether a fixed distance is deciding, and the whole point of the
    // bar is that no fixed distance decides.
    tracing::debug!(
        knowledge = how_the_distances_are_read(entry_dispersion),
        scratchpad = how_the_distances_are_read(None),
        "recall: how each source's distances are read"
    );
    Ok(RecallCandidates {
        entries,
        notes,
        entry_dispersion,
        // The pad is read against the stated estimate. One conversation's pad
        // rarely holds enough rows for a median absolute deviation over it to
        // be a measurement rather than noise, and the pad read is already the
        // block's most expensive query - see #1146.
        note_dispersion: None,
        situation_cue,
    })
}

/// Whether the block will read a source by what that source measured, or by the
/// stated estimate it falls back to.
///
/// Two static words, and nothing of what the source holds: this line runs on
/// every turn, and a log that carried a key or a line of an entry would put
/// personal content in the journal for the sake of an operational fact.
fn how_the_distances_are_read(dispersion: Option<RecallDispersion>) -> &'static str {
    match dispersion {
        Some(_) => "measured",
        None => "estimated",
    }
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
/// The arm reads a different table from the knowledge arm, so it fails on its
/// own - and when it does it must cost its own lines and nothing else. The
/// knowledge arm still renders, and the turn never sees the error.
///
/// This is deliberately narrower than the treatment the knowledge arm gets: a
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
                "recall: embedding failed; degrading to full-text and measuring no dispersion"
            );
            None
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?EMBED_TIMEOUT,
                "recall: embedding timed out; degrading to full-text and measuring no dispersion"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_storage::{
        RECALL_SCAN_STATEMENT_TIMEOUT, USE_LOG_READ_STATEMENT_TIMEOUT,
    };

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
                entries: vec![an_entry()],
                ..RecallCandidates::default()
            })
        })
        .await
        .expect("inside the ceiling the answer travels");

        assert_eq!(answer.entries.len(), 1);
    }

    fn an_entry() -> RecallEntry {
        RecallEntry::new(
            desktop_assistant_core::domain::KnowledgeEntry::new("kb-1", "body", vec![]),
            RecallRelevance::Distance(0.12),
        )
    }

    /// A store that measured its own geometry.
    fn a_dispersion() -> RecallDispersion {
        RecallDispersion::measured(0.80, 0.06, 400).expect("a store's own statistics")
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
    /// knowledge arm, so it fails on its own. When it does it costs its own
    /// lines and nothing else - the knowledge arm still renders, and the turn
    /// never sees the error.
    #[tokio::test]
    async fn recall_block_survives_the_scratchpad_arm_failing() {
        let candidates = gather(
            async { Ok((vec![an_entry()], Some(a_dispersion()), None)) },
            async { Err(CoreError::Storage("the pad read failed".into())) },
        )
        .await
        .expect("a failed pad read must not fail the lookup");

        assert_eq!(
            candidates.entries.len(),
            1,
            "the knowledge arm still renders"
        );
        assert!(
            candidates.notes.is_empty(),
            "the failed arm contributes none"
        );
    }

    #[tokio::test]
    async fn the_scratchpad_arm_passes_its_rows_through_when_it_answers() {
        let candidates = gather(async { Ok((vec![], Some(a_dispersion()), None)) }, async {
            Ok(vec![a_note()])
        })
        .await
        .expect("an arm that answers is not a failure");

        assert_eq!(candidates.notes.len(), 1);
        assert_eq!(candidates.notes[0].key, "deploy-window");
    }

    /// The log an operator reads to tell whether the bar is in force: two
    /// static words, one per source, and nothing of what either holds.
    #[test]
    fn the_log_says_which_sources_measured_and_which_were_estimated() {
        assert_eq!(
            how_the_distances_are_read(Some(a_dispersion())),
            "measured",
            "a source that stated its own geometry is read by it"
        );
        assert_eq!(
            how_the_distances_are_read(None),
            "estimated",
            "and a source that stated none is read by a fixed distance, which is the fact \
             worth surfacing"
        );
    }

    /// The spread the core reads a distance against travels with the candidates
    /// it grades, so one turn's block is graded by one turn's geometry.
    #[tokio::test]
    async fn the_measured_dispersion_travels_with_the_candidates() {
        let candidates = gather(
            async { Ok((vec![an_entry()], Some(a_dispersion()), None)) },
            async { Ok(vec![]) },
        )
        .await
        .expect("every arm answered");

        assert_eq!(candidates.entry_dispersion, Some(a_dispersion()));
    }

    /// A store that could not be measured costs the block its unit and nothing
    /// else: the candidates still travel, and the core falls back to its stated
    /// estimate.
    #[tokio::test]
    async fn a_store_that_cannot_be_measured_still_answers_with_its_candidates() {
        let candidates = gather(async { Ok((vec![an_entry()], None, None)) }, async { Ok(vec![]) })
            .await
            .expect("a measurement is not what the lookup is for");

        assert_eq!(candidates.entries.len(), 1);
        assert_eq!(candidates.entry_dispersion, None);
    }

    /// The asymmetry, stated as a test so it cannot be levelled by accident.
    /// The knowledge arm is the block's whole point, so its error ends the
    /// lookup and the caller drops the block.
    #[tokio::test]
    async fn a_failing_knowledge_arm_ends_the_lookup() {
        let answer = gather(
            async { Err(CoreError::Storage("the store is down".into())) },
            async { Ok(vec![a_note()]) },
        )
        .await;

        assert!(answer.is_err());
    }

    /// The arms must run together, not one after another: two reads and an
    /// embedding sit inside a ten-second whole-lookup ceiling, and a serial fold
    /// would spend the budget twice over.
    #[tokio::test(start_paused = true)]
    async fn the_arms_run_together_rather_than_one_after_another() {
        let hold = std::time::Duration::from_secs(4);
        let started = tokio::time::Instant::now();

        gather(
            async move {
                tokio::time::sleep(hold).await;
                Ok((vec![an_entry()], Some(a_dispersion()), None))
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok(vec![a_note()])
            },
        )
        .await
        .expect("both arms answered");

        assert!(
            started.elapsed() < hold * 2,
            "two reads took {:?}, which is serial rather than concurrent",
            started.elapsed()
        );
    }

    // --- The ceilings the reads sit inside (#1121) --------------------------

    /// The embedding and the scan run one after the other - the arms cannot
    /// start until the vector exists - so what the whole lookup has to hold is
    /// their sum. Three constants in three crates, and nothing but this says
    /// they add up: raising either of the first two is what would break it, and
    /// neither sits beside this ceiling.
    /// The caller must not give up before the database does, or the backend
    /// goes on working on a read nobody is waiting for - the same rule the
    /// recall scan follows, and recall runs before every turn.
    #[test]
    fn the_use_log_read_gives_up_no_later_than_the_database_does() {
        assert!(
            USE_LOG_READ_CEILING >= USE_LOG_READ_STATEMENT_TIMEOUT,
            "the caller gives up at {USE_LOG_READ_CEILING:?} and the database only at \
             {USE_LOG_READ_STATEMENT_TIMEOUT:?}, so an abandoned read keeps a backend busy"
        );
    }

    #[test]
    fn the_three_reads_together_stay_inside_the_lookups_ceiling() {
        let worst = EMBED_TIMEOUT + RECALL_SCAN_STATEMENT_TIMEOUT + USE_LOG_READ_CEILING;
        assert!(
            worst < RECALL_CALL_CEILING,
            "an embedding at {EMBED_TIMEOUT:?}, a scan at {RECALL_SCAN_STATEMENT_TIMEOUT:?} and \
             a use-log read at {USE_LOG_READ_CEILING:?} spend {worst:?} of the \
             {RECALL_CALL_CEILING:?} the whole lookup has, so the lookup is what gives up first \
             and the block is dropped"
        );
    }

    // --- The use log behind the reinforcement term (#1123) ------------------

    fn a_record(entry_id: &str) -> KnowledgeUseRecord {
        KnowledgeUseRecord::unseen(entry_id, chrono::Utc::now())
    }

    #[tokio::test]
    async fn the_use_log_answers_are_keyed_by_the_entry_they_are_about() {
        let records = use_records(async { Ok(vec![a_record("kb-1"), a_record("kb-2")]) }).await;

        assert_eq!(records.len(), 2);
        assert!(records.contains_key("kb-1"));
        assert!(records.contains_key("kb-2"));
    }

    /// Acceptance (#1123): a use log that cannot be read costs the ranking, not
    /// the block. Every candidate then ranks on its semantic signal alone, which
    /// is how they all ranked before the log existed.
    #[tokio::test]
    async fn a_use_log_that_cannot_be_read_costs_the_ranking_and_not_the_block() {
        let records =
            use_records(async { Err(CoreError::Storage("the log is down".into())) }).await;

        assert!(records.is_empty());
    }

    /// The same for a log that is merely slow. The reinforcement half of the
    /// score is the half retrieval worked without until now, so it must never be
    /// what makes a turn slow.
    #[tokio::test(start_paused = true)]
    async fn a_slow_use_log_costs_the_ranking_and_not_the_block() {
        let records = use_records(async {
            tokio::time::sleep(USE_LOG_READ_CEILING * 2).await;
            Ok(vec![a_record("kb-1")])
        })
        .await;

        assert!(records.is_empty());
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
