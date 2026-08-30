//! Adapter behind the pre-prompt recall port (#1100, #1101, #1154, #1350).
//!
//! One user prompt, one embedding, four indexes. The knowledge base answers
//! with the entries nearest the prompt and how near each is; this
//! conversation's scratchpad answers the same way about its own notes; the
//! skill catalog answers with the approved procedures nearest the prompt; and
//! the episodic turn index answers with the person's own past turns, from every
//! conversation they own. The core decides what clears the bar and how the
//! `[Recall]` block reads.
//!
//! ## The episode arm answers with digests, not with lines
//!
//! An episode candidate is built by
//! [`RecallEpisode::from_digest`](desktop_assistant_core::ports::recall::RecallEpisode::from_digest),
//! which is what holds an offered line to the user's own half of a turn. This
//! adapter hands it whole digests and never a rendered string, so the rule
//! lives in the type and not in this file - see that type for why an
//! unprompted, cross-conversation line may carry nothing the turn derived.
//!
//! The arm has no degraded read: `turn_digests` carries a vector and no
//! `tsvector`, so there is no lexical index to fall back to, and it stays
//! silent on a turn whose embedding failed.
//!
//! ## The skill arm is optional, and absent is not an error
//!
//! A deployment with no skill catalog wires no skill store, and the arm then
//! contributes no candidates and no spread. That is the same absence as a
//! catalog holding nothing near the prompt, and the block renders its other
//! arms either way.
//!
//! ## What a distance is worth, and who says so
//!
//! A cosine distance means nothing on its own, so the core reads each candidate
//! against the spread of the source it came from. Only a source can say what its
//! own geometry is, so each measured arm's read answers with both: the
//! candidates, and the spread of the same query's distances over every row that
//! arm's scan could reach. One scan states both, so the spread always describes
//! the query whose candidates it grades.
//!
//! All three arms do this (#1167). The pad was the last one read by the stated
//! estimate, and it is the arm the estimate fitted worst: a note embeds
//! `"<key> <content>"`, which is terser and more telegraphic than an entry's
//! body. A source that holds too little to measure still answers `None`, and
//! one conversation's pad usually does - what the measurement buys is the long
//! conversation, whose pad is both large enough to measure and least like the
//! store.
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
//! scratchpad arm, the skill arm and the episode arm each read a different
//! table from the knowledge arm, so any of them can fail on its own, and when
//! one does it costs its own lines and nothing else - see [`notes_or_none`],
//! [`skills_or_none`] and [`episodes_or_none`].
//! The measurement carries its own ceiling for the same reason. If a degraded
//! read fails as well, the error travels to the caller, which drops the block
//! and runs the turn.
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
use desktop_assistant_core::domain::knowledge::{Disposition, KnowledgeEntry};
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::situation::{SituationCue, SituationRecord};
use desktop_assistant_core::ports::embedding::{EMBED_TIMEOUT, EmbedFn};
use desktop_assistant_core::ports::episode_use::EpisodeUseLog;
use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, SituationSignal, current_situation,
};
use desktop_assistant_core::ports::recall::RecallEpisode;
use desktop_assistant_core::ports::recall::{
    RecallCandidates, RecallDispersion, RecallEntry, RecallNote, RecallRelevance, RecallRequest,
    RecallSearchFn, RecallSkill,
};
use desktop_assistant_core::ports::skill_use::SkillUseLog;
use desktop_assistant_core::ports::turn_digest::TurnDigest;
use desktop_assistant_storage::{
    NearestSkill, PgEpisodeUseLog, PgKnowledgeBaseStore, PgKnowledgeUseLog, PgPool,
    PgScratchpadStore, PgSkillIndexStore, PgSkillUseLog, PgTurnDigestStore,
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
    skill_store: Option<Arc<PgSkillIndexStore>>,
    pool: PgPool,
    embed: EmbedFn,
    embedding_model: String,
) -> RecallSearchFn {
    // The pad adapter, the episodic store and the three use logs are handles on
    // the same pool, built once here rather than threaded in: nothing else in
    // the daemon holds the pad one, and the reads behind these arms are
    // inherent to them.
    let pad = Arc::new(PgScratchpadStore::new(pool.clone()));
    let digests = Arc::new(PgTurnDigestStore::new(pool.clone()));
    let uses = Arc::new(PgKnowledgeUseLog::new(pool.clone()));
    let skill_uses = Arc::new(PgSkillUseLog::new(pool.clone()));
    let episode_uses = Arc::new(PgEpisodeUseLog::new(pool.clone()));
    Arc::new(move |request: RecallRequest| {
        let kb_store = Arc::clone(&kb_store);
        let skill_store = skill_store.clone();
        let pad = Arc::clone(&pad);
        let digests = Arc::clone(&digests);
        let uses = Arc::clone(&uses);
        let skill_uses = Arc::clone(&skill_uses);
        let episode_uses = Arc::clone(&episode_uses);
        let embed = Arc::clone(&embed);
        let embedding_model = embedding_model.clone();
        Box::pin(async move {
            within_ceiling(lookup(
                &Sources {
                    kb_store: &kb_store,
                    skill_store: skill_store.as_deref(),
                    pad: &pad,
                    digests: &digests,
                    uses: &uses,
                    skill_uses: &skill_uses,
                    episode_uses: &episode_uses,
                },
                &embed,
                &embedding_model,
                request,
            ))
            .await
        })
    })
}

/// The stores and logs one lookup reads, gathered into one value so [`lookup`]
/// takes an argument per concern rather than one per handle.
#[derive(Clone, Copy)]
struct Sources<'a> {
    kb_store: &'a PgKnowledgeBaseStore,
    skill_store: Option<&'a PgSkillIndexStore>,
    pad: &'a PgScratchpadStore,
    digests: &'a PgTurnDigestStore,
    uses: &'a PgKnowledgeUseLog,
    skill_uses: &'a PgSkillUseLog,
    episode_uses: &'a PgEpisodeUseLog,
}

/// What the skill arm answers with: the candidates, the catalog's own spread,
/// and the catalog's own situation cue.
///
/// A named type rather than a bare tuple because it appears four times - the
/// measured read, the degraded read, [`gather`]'s parameter and
/// [`skills_or_none`]'s - and the three parts travel together by design: a
/// spread and a cue with no candidates to grade are both nothing, so nothing
/// may keep one without the others.
type SkillArm = (
    Vec<RecallSkill>,
    Option<RecallDispersion>,
    Option<SituationCue>,
);

/// The same, for the knowledge arm.
type KnowledgeArm = (
    Vec<RecallEntry>,
    Option<RecallDispersion>,
    Option<SituationCue>,
);

/// The same, for the scratchpad arm, which keeps no situation record of its
/// own.
type NoteArm = (Vec<RecallNote>, Option<RecallDispersion>);

/// The same, for the episode arm, which keeps no situation record of its own
/// either (#1350).
type EpisodeArm = (Vec<RecallEpisode>, Option<RecallDispersion>);

/// Whether the knowledge arm may offer this entry at all (#893).
///
/// **`active`, plus `refuted` rendered with the marker; nothing else.** This
/// is a narrower bar than the knowledge-base search tool's: the tool
/// resolves `superseded`/`redundant` to a successor and deranks `trivial`
/// rather than hiding it, because a person asking the tool a direct question
/// benefits from that nuance. The `[Recall]` block is different - it is
/// candidate memory offered unasked, ahead of the model's first move, so a
/// row this permissive would put a merged-away duplicate or a
/// not-worth-surfacing aside in front of every turn. Only the two values the
/// design's vocabulary marks "admitted" for `[Recall]` clear this: `active`
/// is the ordinary case, and `refuted` must stay findable so the assistant
/// can report a correction instead of silently forgetting it -
/// [`KnowledgeEntry::display_line`] is what keeps a `refuted` row from ever
/// reading as a current fact once it is offered.
///
/// **Duplicated in `desktop_assistant_storage::recall_replay::replay_admits`
/// (#1328), pinned to agree by
/// `the_two_disposition_admission_rules_agree_for_every_disposition` below.**
/// Storage cannot depend on daemon to call this directly, so the replay
/// instrument that measures this block's own ranking carries its own copy;
/// the test in this file is what stops the two from silently drifting apart.
/// Edit both functions and that test in the same change.
fn recall_admits(disposition: Disposition) -> bool {
    matches!(disposition, Disposition::Active | Disposition::Refuted)
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
    sources: &Sources<'_>,
    embed: &EmbedFn,
    embedding_model: &str,
    request: RecallRequest,
) -> Result<RecallCandidates, CoreError> {
    let Sources {
        kb_store,
        skill_store,
        pad,
        digests,
        uses,
        skill_uses,
        episode_uses,
    } = *sources;
    let Some(vector) = embed_prompt(embed, &request.prompt).await else {
        // Degraded: full-text for every arm, and no dispersion. A full-text row
        // carries no distance, so there is nothing to read against a spread.
        //
        // `search_text_any_term` on each, not the store's own `search`: that
        // one joins the query's lexemes with AND, which asks a whole user
        // sentence to appear in one row and answers almost nothing.
        return gather(
            async {
                Ok((
                    kb_store
                        .search_text_any_term(&request.prompt, request.entry_limit)
                        .await?
                        .into_iter()
                        // `active` and `refuted` only (#893) - see
                        // `recall_admits`. The degraded arm still owes the
                        // same admission bar the measured arm below does.
                        .filter(|entry: &KnowledgeEntry| recall_admits(entry.disposition))
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
                Ok((
                    pad.search_text_any_term(
                        &request.conversation_id,
                        &request.prompt,
                        request.note_limit,
                    )
                    .await?
                    .into_iter()
                    .map(|note| to_recall_note(note, RecallRelevance::LexicalMatch))
                    .collect(),
                    // A lexical row carries no distance, so there is nothing to
                    // read against a spread and nothing to measure one over.
                    None,
                ))
            },
            async {
                let Some(skills) = skill_store else {
                    return Ok((Vec::new(), None, None));
                };
                Ok((
                    skills
                        .search_text_any_term(&request.prompt, request.skill_limit)
                        .await?
                        .into_iter()
                        // No use records and no situation here either, for the
                        // reason the knowledge arm above states: a lexical
                        // candidate carries no semantic term, so nothing ranks
                        // it and every signal read for it would be discarded.
                        .map(to_recall_skill)
                        .collect(),
                    None,
                    None,
                ))
            },
            // The episode arm has no degraded read, and that is a property of
            // the store rather than a gap here. `turn_digests` carries a vector
            // and no `tsvector`, so there is no lexical index to fall back to,
            // and a scan of every digest's text on the turn where the embedder
            // is already failing is the wrong thing to spend a turn on. The arm
            // is silent for that turn and the rest of the block renders, which
            // is the same absence a deployment with no digests produces.
            async { Ok((Vec::new(), None)) },
        )
        .await;
    };

    // Every arm shares the one vector, and none depends on another.
    let vector_for_notes = vector.clone();
    let vector_for_skills = vector.clone();
    let vector_for_episodes = vector.clone();
    // The situation this turn arrived in, read once for the whole lookup. It is
    // derived from the clock and what the client reported, so it costs no model
    // call and no extra work on the write path - see
    // `desktop_assistant_core::domain::situation`. Read once rather than once
    // per arm so both arms grade the same instant: the two run together, and a
    // turn that straddled a boundary would otherwise read one part of the day
    // for facts and another for procedures.
    let here = current_situation();
    let here_for_skills = here.clone();
    gather(
        async {
            let found = kb_store
                .nearest_by_embedding(vector, embedding_model, request.entry_limit)
                .await?;
            let dispersion = found.dispersion;
            // `active` and `refuted` only (#893) - see `recall_admits`.
            // Filtered before the use log is read, so the batched read below
            // never spends a round trip on an id the block will not offer.
            let entries: Vec<(KnowledgeEntry, f64)> = found
                .entries
                .into_iter()
                .filter(|(entry, _)| recall_admits(entry.disposition))
                .collect();
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
            let ids: Vec<String> = entries.iter().map(|(e, _)| e.id.clone()).collect();
            let (mut records, mut situations, cue) = if ids.is_empty() {
                (
                    std::collections::HashMap::new(),
                    std::collections::HashMap::new(),
                    None,
                )
            } else if here.is_empty() {
                // Nothing connected, so no cue can be graded and no record can
                // score against one. The read is skipped rather than run and
                // discarded: a deployment that reports no client context pays
                // nothing per turn for a feature it is not using.
                (
                    use_records(uses.records(ids)).await,
                    std::collections::HashMap::new(),
                    None,
                )
            } else {
                let (records, (situations, cue)) = tokio::join!(
                    use_records(uses.records(ids.clone())),
                    situation_signal(uses.situation_signal(ids, here)),
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
                candidates = entries.len(),
                with_use_record = records.len(),
                with_situation_record = situations.len(),
                situation_cue = cue
                    .as_ref()
                    .map_or(0, |cue: &SituationCue| { cue.situation().iter().count() }),
                "recall: how many candidates the use log and the situation had something to \
                 say about"
            );
            Ok((
                entries
                    .into_iter()
                    .map(|(entry, distance)| {
                        let record = records.remove(&entry.id);
                        let seen_in = situations.remove(&entry.id).unwrap_or_default();
                        RecallEntry::new(entry, RecallRelevance::Distance(distance))
                            .with_use_record(record)
                            .with_situation(seen_in)
                    })
                    .collect(),
                dispersion,
                cue,
            ))
        },
        async {
            let found = pad
                .nearest_by_embedding(
                    &request.conversation_id,
                    vector_for_notes,
                    embedding_model,
                    request.note_limit,
                )
                .await?;
            Ok((
                found
                    .notes
                    .into_iter()
                    .map(|(note, distance)| {
                        to_recall_note(note, RecallRelevance::Distance(distance))
                    })
                    .collect(),
                found.dispersion,
            ))
        },
        async {
            let Some(skills) = skill_store else {
                return Ok((Vec::new(), None, None));
            };
            let found = skills
                .nearest_by_embedding(vector_for_skills, embedding_model, request.skill_limit)
                .await?;
            // One batched read after the scan, on the same terms and for the
            // same reasons as the knowledge arm's above: a slow or broken log
            // costs the order of the skill lines and never the lines.
            let names: Vec<String> = found.skills.iter().map(|s| s.name.clone()).collect();
            let here = here_for_skills;
            let (mut records, mut situations, cue) = if names.is_empty() {
                (
                    std::collections::HashMap::new(),
                    std::collections::HashMap::new(),
                    None,
                )
            } else if here.is_empty() {
                // Nothing connected, so no cue can be graded and no record can
                // score against one. The read is skipped rather than run and
                // discarded, on the same terms as the knowledge arm's.
                (
                    use_records(skill_uses.records(names)).await,
                    std::collections::HashMap::new(),
                    None,
                )
            } else {
                let (records, (situations, cue)) = tokio::join!(
                    use_records(skill_uses.records(names.clone())),
                    situation_signal(skill_uses.situation_signal(names, here)),
                );
                (records, situations, cue)
            };
            tracing::debug!(
                candidates = found.skills.len(),
                with_use_record = records.len(),
                with_situation_record = situations.len(),
                situation_cue = cue
                    .as_ref()
                    .map_or(0, |cue: &SituationCue| { cue.situation().iter().count() }),
                "recall: how many skill candidates the use log and the situation had something \
                 to say about"
            );
            Ok((
                found
                    .skills
                    .into_iter()
                    .map(|skill| {
                        let record = records.remove(&skill.name);
                        let seen_in = situations.remove(&skill.name).unwrap_or_default();
                        to_recall_skill(skill)
                            .with_use_record(record)
                            .with_situation(seen_in)
                    })
                    .collect(),
                found.dispersion,
                cue,
            ))
        },
        async {
            let found = digests
                .nearest_by_embedding(vector_for_episodes, embedding_model, request.episode_limit)
                .await?;
            // One batched read after the scan, on the same terms and for the
            // same reasons as the knowledge arm's above: a slow or broken log
            // costs the order of the episode lines and never the lines.
            //
            // No situation read beside it. Nothing records the situation an
            // episode was opened in, so there is no table to grade a cue
            // against and the term is `NO_SITUATION` for every candidate - see
            // `RecallEpisode::situation_coverage`. A read here would be a query
            // whose answer nothing reads.
            let ids: Vec<String> = found.digests.iter().map(|(d, _)| d.id.clone()).collect();
            let mut records = if ids.is_empty() {
                std::collections::HashMap::new()
            } else {
                use_records(episode_uses.records(ids)).await
            };
            // Counted before the map is consumed below, so the log line
            // reports what the log answered rather than what is left of it.
            let with_use_record = records.len();
            let episodes: Vec<RecallEpisode> = found
                .digests
                .iter()
                .filter_map(|(digest, distance)| {
                    // A digest with no user half is skipped here rather than
                    // rendered empty: `RecallEpisode` cannot be built from one,
                    // which is what keeps the answer half off the line.
                    let mut episode =
                        RecallEpisode::from_digest(digest, RecallRelevance::Distance(*distance))?;
                    episode = episode.with_use_record(records.remove(&digest.id));
                    Some(episode)
                })
                .collect();
            unrenderable_digests(&found.digests, episodes.len());
            tracing::debug!(
                candidates = episodes.len(),
                with_use_record,
                "recall: how many episode candidates the use log had something to say about"
            );
            Ok((episodes, found.dispersion))
        },
    )
    .await
}

/// Say when the store answered with digests the arm could not build a line
/// from.
///
/// A digest with no `Asked:` half is a row written by something that is not
/// `crate::turn_capture` - a hand-inserted row, or a format that moved without
/// its reader. It is dropped silently by construction, so an operator would
/// otherwise see an arm that reads rows and offers nothing with no clue why.
/// A count and nothing of what any row holds: this runs on every turn.
fn unrenderable_digests(scanned: &[(TurnDigest, f64)], built: usize) {
    let unrenderable = scanned.len().saturating_sub(built);
    if unrenderable > 0 {
        tracing::warn!(
            unrenderable,
            scanned = scanned.len(),
            "recall: some digests carry no question to offer, so the episode arm skipped them"
        );
    }
}

/// One scanned skill as a recall candidate.
///
/// The name, the description, the provenance and the presence flag travel;
/// nothing is rendered here, and the body was never read. How much of a
/// description a line may spend, and how a skill whose files are gone or whose
/// text came from outside this machine is marked, are the core's decisions.
fn to_recall_skill(skill: NearestSkill) -> RecallSkill {
    // The row states which kind of relevance it carries, rather than the call
    // site stating it: the measured read and the degraded one answer with the
    // same type, and a call site that assumed the wrong one would render a
    // lexical row as a perfect distance - clearing any bar and being ranked by
    // activation as though it held a semantic signal.
    let relevance = match skill.distance {
        Some(distance) => RecallRelevance::Distance(distance),
        None => RecallRelevance::LexicalMatch,
    };
    RecallSkill::new(
        skill.name,
        skill.description,
        skill.trust_tier,
        skill.present_on_disk,
        relevance,
    )
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
    read: impl Future<Output = Result<SituationSignal, CoreError>>,
) -> (
    std::collections::HashMap<String, SituationRecord>,
    Option<SituationCue>,
) {
    let signal = match tokio::time::timeout(USE_LOG_READ_CEILING, read).await {
        Ok(Ok(signal)) => signal,
        Ok(Err(error)) => {
            // The error text, not just the fact: an unmigrated database, a
            // missing grant and an exhausted pool all silence the cue and all
            // need a different fix. This is a read, so no message it carries can
            // echo a stored value.
            tracing::warn!(
                %error,
                "recall: the situation could not be read; ranking without the cue"
            );
            return (std::collections::HashMap::new(), None);
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?USE_LOG_READ_CEILING,
                "recall: the situation read timed out; ranking without the situation cue"
            );
            return (std::collections::HashMap::new(), None);
        }
    };
    (signal.records.into_iter().collect(), signal.cue)
}

/// Run the four arms together and fold what they answered into one candidate
/// set.
///
/// Generic over the futures, and separate from [`lookup`], so everything it
/// guarantees is provable without a database - which is the only way to hold
/// any of it to anything.
///
/// **`join!`, never `try_join!`.** The arms do not depend on each other, and one
/// arm's error must not cancel one that was answering.
///
/// **The pad, skill and episode arms' errors are absorbed; the knowledge arm's
/// propagates.** A knowledge arm that cannot read is the block's whole point
/// failing, and the caller drops the block and runs the turn anyway; losing the
/// pad lines, the skill lines or the episode lines is the smaller loss, so it
/// is taken here rather than passed on. The absorbed arms resolve first, so
/// their failures are logged even on the turn where the knowledge arm's error
/// is about to end the lookup.
///
/// Every arm answers with its own source's spread beside its candidates,
/// because one scan states both (#1167). A source that cannot measure one
/// answers `None` and the core reads it by its stated estimate.
async fn gather(
    entries: impl Future<Output = Result<KnowledgeArm, CoreError>>,
    notes: impl Future<Output = Result<NoteArm, CoreError>>,
    skills: impl Future<Output = Result<SkillArm, CoreError>>,
    episodes: impl Future<Output = Result<EpisodeArm, CoreError>>,
) -> Result<RecallCandidates, CoreError> {
    let (entries, notes, skills, episodes) = tokio::join!(entries, notes, skills, episodes);
    let (notes, note_dispersion) = notes_or_none(notes);
    let (skills, skill_dispersion, skill_situation_cue) = skills_or_none(skills);
    let (episodes, episode_dispersion) = episodes_or_none(episodes);
    let (entries, entry_dispersion, situation_cue) = entries?;
    // Which sources stated their own geometry, and which the block will read by
    // a stated estimate. Without this an operator meeting a block that is
    // quieter than expected cannot tell whether the dimensionless bar is in
    // force or whether a fixed distance is deciding, and the whole point of the
    // bar is that no fixed distance decides.
    tracing::debug!(
        knowledge = how_the_distances_are_read(entry_dispersion),
        scratchpad = how_the_distances_are_read(note_dispersion),
        skills = how_the_distances_are_read(skill_dispersion),
        episodes = how_the_distances_are_read(episode_dispersion),
        "recall: how each source's distances are read"
    );
    Ok(RecallCandidates {
        entries,
        notes,
        skills,
        episodes,
        entry_dispersion,
        note_dispersion,
        situation_cue,
        skill_dispersion,
        skill_situation_cue,
        episode_dispersion,
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
        after_outside_read: note.after_outside_read,
        relevance,
    }
}

/// The scratchpad arm's rows and its spread, or neither.
///
/// The arm reads a different table from the knowledge arm, so it fails on its
/// own - and when it does it must cost its own lines and nothing else. The
/// knowledge arm still renders, and the turn never sees the error.
///
/// This is deliberately narrower than the treatment the knowledge arm gets: a
/// knowledge arm that cannot read is the block's whole point failing, so that
/// error travels to the caller, which drops the block and runs the turn anyway.
/// Losing the pad lines is a smaller loss than losing the block.
///
/// The spread goes with the rows, for the reason [`skills_or_none`] gives: a
/// spread with no candidates to grade is nothing, and it must not be left
/// standing as though the pad had been measured.
fn notes_or_none(found: Result<NoteArm, CoreError>) -> NoteArm {
    match found {
        Ok(answer) => answer,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "recall: the scratchpad arm failed; the other arms still render"
            );
            (Vec::new(), None)
        }
    }
}

/// The skill arm's rows, its spread and its cue, or none of the three.
///
/// The same treatment [`notes_or_none`] gives the pad, and for the same reason:
/// the arm reads its own table, so it fails on its own, and losing the skill
/// lines is a smaller loss than losing the block. All three go together - a
/// spread with no candidates to grade is nothing, and so is a cue, and neither
/// must be left standing as though the catalog had been measured.
fn skills_or_none(found: Result<SkillArm, CoreError>) -> SkillArm {
    match found {
        Ok(answer) => answer,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "recall: the skill arm failed; the other arms still render"
            );
            (Vec::new(), None, None)
        }
    }
}

/// The episode arm's rows and its spread, or neither (#1350).
///
/// The same treatment [`notes_or_none`] gives the pad, and for the same reason:
/// the arm reads its own table, so it fails on its own, and losing the episode
/// lines is a smaller loss than losing the block. The spread goes with the
/// rows, because a spread with no candidates to grade is nothing and must not
/// be left standing as though the store had been measured.
fn episodes_or_none(found: Result<EpisodeArm, CoreError>) -> EpisodeArm {
    match found {
        Ok(answer) => answer,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "recall: the episode arm failed; the other arms still render"
            );
            (Vec::new(), None)
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

    /// The seam the provenance mark has to cross, and the one place nothing
    /// watched (#1175).
    ///
    /// The whole case for letting an installed skill into a system prompt is
    /// that its line carries `[installed: ...]`. Storage proves the tier leaves
    /// the row and the core proves every tier marks its line, but the adapter
    /// between them is a plain field copy - and a copy that names one tier
    /// instead of the row's compiles, passes both of those suites, and renders
    /// third-party text as the assistant's own memory.
    ///
    /// Written over every variant rather than one, because a mistake here is a
    /// constant, and a constant matches whichever variant the fixture happened
    /// to use.
    #[test]
    fn every_trust_tier_survives_the_walk_from_a_row_to_a_recall_candidate() {
        use desktop_assistant_core::domain::TrustTier;

        for tier in [
            TrustTier::Local,
            TrustTier::Github,
            TrustTier::WellKnown,
            TrustTier::Unknown,
        ] {
            let candidate = to_recall_skill(NearestSkill {
                name: "publish-a-crate".to_string(),
                description: "Cut a release and push it to the registry.".to_string(),
                trust_tier: tier,
                present_on_disk: true,
                distance: Some(0.20),
            });
            assert_eq!(
                candidate.provenance, tier,
                "the row's own tier must reach the candidate that renders it, or a \
                 skill written outside this machine renders unmarked"
            );
        }
    }

    /// Acceptance (#893): the `[Recall]` block's admission bar is `active`,
    /// plus `refuted`, and nothing else - stated exhaustively over every
    /// disposition, so the four refusals are checked as hard as the two
    /// permits and a new variant added later has to answer this.
    #[test]
    fn the_recall_block_admits_only_active_and_marked_refuted_entries() {
        use desktop_assistant_core::domain::knowledge::Disposition;

        for disposition in Disposition::ALL {
            let expected = matches!(disposition, Disposition::Active | Disposition::Refuted);
            assert_eq!(
                recall_admits(disposition),
                expected,
                "{disposition:?} must {} the recall block's admission bar",
                if expected { "clear" } else { "not clear" }
            );
        }
    }

    /// Acceptance (#1328): `recall_admits` here and
    /// `desktop_assistant_storage::recall_replay::replay_admits` are two
    /// copies of one rule - storage cannot depend on daemon to share it, so
    /// daemon (which depends on storage) is the one side that can hold them
    /// to the same answer. Enumerated from `Disposition::ALL` rather than a
    /// hand-written list, so a disposition added to the enum later and
    /// wired into only one copy fails this test instead of silently
    /// ranking wrong in whichever copy was missed.
    #[test]
    fn the_two_disposition_admission_rules_agree_for_every_disposition() {
        use desktop_assistant_core::domain::knowledge::Disposition;
        use desktop_assistant_storage::recall_replay::replay_admits;

        for disposition in Disposition::ALL {
            assert_eq!(
                recall_admits(disposition),
                replay_admits(disposition),
                "{disposition:?}: the live block's admission rule and replay's copy of it \
                 must agree, or replay can rank a candidate production would never show"
            );
        }
    }

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

    /// A pad that measured its own, and put its distances somewhere else: a
    /// note embeds `"<key> <content>"`, which is terser than an entry's body.
    ///
    /// A spread the bar can actually read. The first value written here spread
    /// a ninth of its median, which puts `distance_at(RECALL_BAR)` below zero,
    /// so `admission_dispersion` would read that pad against the estimate - and
    /// this test is about a measurement travelling, not about which scale the
    /// bar ends up using.
    fn a_pad_dispersion() -> RecallDispersion {
        RecallDispersion::measured(0.55, 0.07, 40).expect("a pad's own statistics")
    }

    fn a_skill() -> RecallSkill {
        RecallSkill::new(
            "publish-a-crate",
            "Cut a release and push it to the registry.",
            desktop_assistant_core::domain::TrustTier::Local,
            true,
            RecallRelevance::Distance(0.12),
        )
    }

    /// The skill arm answering with nothing, for a test whose subject is one of
    /// the others.
    async fn no_skills() -> Result<SkillArm, CoreError> {
        Ok((Vec::new(), None, None))
    }

    /// The episode arm answering with nothing, on the same terms.
    async fn no_episodes() -> Result<EpisodeArm, CoreError> {
        Ok((Vec::new(), None))
    }

    /// One stored digest as a candidate, built the only way a candidate can be
    /// built - from a digest, through the construction that keeps the answer
    /// half off the line.
    fn an_episode(id: &str, asked: &str) -> RecallEpisode {
        RecallEpisode::from_digest(
            &a_digest(
                id,
                &format!("Asked: {asked}\n\nAnswered: it is on the storage host."),
            ),
            RecallRelevance::Distance(0.12),
        )
        .expect("a digest with a question has a candidate")
    }

    /// A stored digest with `content` as its text.
    fn a_digest(id: &str, content: &str) -> TurnDigest {
        TurnDigest {
            id: id.to_string(),
            conversation_id: "c-earlier".to_string(),
            opening_message_id: "m-1".to_string(),
            content: content.to_string(),
            after_outside_read: false,
            disposition: Disposition::Active,
            disposition_reason: None,
            superseded_by: None,
            created_at: "2026-08-01 09:00:00".to_string(),
            updated_at: "2026-08-01 09:00:00".to_string(),
        }
    }

    fn a_note() -> RecallNote {
        RecallNote {
            key: "deploy-window".into(),
            content: "Fridays after 18:00".into(),
            pinned: false,
            after_outside_read: false,
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
            no_skills(),
            no_episodes(),
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
        let candidates = gather(
            async { Ok((vec![], Some(a_dispersion()), None)) },
            async { Ok((vec![a_note()], None)) },
            no_skills(),
            no_episodes(),
        )
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
            async { Ok((vec![], None)) },
            no_skills(),
            no_episodes(),
        )
        .await
        .expect("every arm answered");

        assert_eq!(candidates.entry_dispersion, Some(a_dispersion()));
    }

    /// Acceptance (#1167): the pad's own measured dispersion travels with its
    /// notes, so the note arm is read against the pad's own geometry.
    ///
    /// This is what says the stated estimate is no longer what the arm is read
    /// by. Before this the field was written `None` at this call site whatever
    /// the pad answered, so a measurement could not reach the block however
    /// many rows the pad held.
    #[tokio::test]
    async fn the_pads_measured_dispersion_travels_with_its_notes() {
        let candidates = gather(
            async { Ok((vec![], Some(a_dispersion()), None)) },
            async { Ok((vec![a_note()], Some(a_pad_dispersion()))) },
            no_skills(),
            no_episodes(),
        )
        .await
        .expect("every arm answered");

        assert_eq!(candidates.note_dispersion, Some(a_pad_dispersion()));
        assert_ne!(
            candidates.note_dispersion, candidates.entry_dispersion,
            "the pad is its own source, so its spread is not the store's"
        );
    }

    /// Acceptance (#1167): a pad too small to measure states nothing, and the
    /// core then falls back to its stated estimate - which is the ordinary case
    /// for one conversation's pad.
    #[tokio::test]
    async fn a_pad_that_states_no_dispersion_leaves_the_field_unmeasured() {
        let candidates = gather(
            async { Ok((vec![], Some(a_dispersion()), None)) },
            async { Ok((vec![a_note()], None)) },
            no_skills(),
            no_episodes(),
        )
        .await
        .expect("every arm answered");

        assert_eq!(candidates.notes.len(), 1, "the notes still travel");
        assert_eq!(candidates.note_dispersion, None);
    }

    /// A store that could not be measured costs the block its unit and nothing
    /// else: the candidates still travel, and the core falls back to its stated
    /// estimate.
    #[tokio::test]
    async fn a_store_that_cannot_be_measured_still_answers_with_its_candidates() {
        let candidates = gather(
            async { Ok((vec![an_entry()], None, None)) },
            async { Ok((vec![], None)) },
            no_skills(),
            no_episodes(),
        )
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
            async { Ok((vec![a_note()], None)) },
            no_skills(),
            no_episodes(),
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
                Ok((vec![an_entry()], Some(a_dispersion()), None))
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok((vec![a_note()], None))
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok((vec![a_skill()], Some(a_dispersion()), None))
            },
            async move {
                tokio::time::sleep(hold).await;
                Ok((
                    vec![an_episode("ep-1", "where does the registry live?")],
                    None,
                ))
            },
        )
        .await
        .expect("every arm answered");

        assert!(
            started.elapsed() < hold * 2,
            "four reads took {:?}, which is serial rather than concurrent",
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

    // --- The situation behind the third term (#1125) ------------------------

    /// A situation the store answers with is keyed by the entry it is about, and
    /// travels beside the cue that grades it.
    #[tokio::test]
    async fn the_situation_answers_are_keyed_by_the_entry_they_are_about() {
        let (records, cue) = situation_signal(async {
            Ok(SituationSignal {
                records: vec![
                    ("kb-1".to_string(), SituationRecord::new()),
                    ("kb-2".to_string(), SituationRecord::new()),
                ],
                cue: None,
            })
        })
        .await;

        assert_eq!(records.len(), 2);
        assert!(records.contains_key("kb-1"));
        assert!(records.contains_key("kb-2"));
        assert!(cue.is_none());
    }

    /// A situation that cannot be read costs the ranking, not the block - the
    /// same bargain the use log makes, and the reason the read is separate from
    /// it.
    ///
    /// Both halves go, and that is deliberate: a record nothing can grade scores
    /// zero and a cue nothing carries a record for scores zero, so half an
    /// answer is worth what no answer is worth.
    #[tokio::test]
    async fn a_situation_that_cannot_be_read_costs_the_ranking_and_not_the_block() {
        let (records, cue) =
            situation_signal(async { Err(CoreError::Storage("the table is not there".into())) })
                .await;

        assert!(records.is_empty());
        assert!(cue.is_none());
    }

    /// The same for a read that is merely slow. The situation is the cheapest
    /// signal in the score, so it must never be what makes a turn slow.
    #[tokio::test(start_paused = true)]
    async fn a_slow_situation_read_costs_the_ranking_and_not_the_block() {
        let (records, cue) = situation_signal(async {
            tokio::time::sleep(USE_LOG_READ_CEILING * 2).await;
            Ok(SituationSignal {
                records: vec![("kb-1".to_string(), SituationRecord::new())],
                cue: None,
            })
        })
        .await;

        assert!(records.is_empty());
        assert!(cue.is_none());
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
