use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use crate::domain::{DEFAULT_NOTE_TYPE, ScratchpadNote};
use crate::ports::embedding::{EMBED_TIMEOUT, EmbedFn};

/// One note's vectors and the model that produced them (#717).
///
/// One type rather than two loose fields because a search scopes its vector arm
/// to the model that produced the stored vector. A vector paired with another
/// model's name is compared against rows of another dimension, which pgvector
/// answers with an error rather than a miss -- so the two may only ever be set,
/// carried and replaced together.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteEmbedding {
    /// One vector per content chunk, in chunk order. A note is usually a single
    /// chunk; [`MAX_NOTE_BYTES`] is what makes more than one possible.
    pub chunks: Vec<Vec<f32>>,
    /// Identifier of the model that produced `chunks`.
    pub model: String,
}

/// A note to upsert into the scratchpad. Carries the structured fields that
/// don't fit a bare `(key, content)` pair: a free-text `note_type`
/// (default `note`), an optional `sequence` (sorted within a type), a
/// `done` flag, and the note's own vector when the writer embedded it inline.
/// Construct via [`NewScratchpadNote::new`] and the field setters, or as a
/// struct literal.
#[derive(Debug, Clone, PartialEq)]
pub struct NewScratchpadNote {
    pub key: String,
    pub content: String,
    pub note_type: String,
    pub sequence: Option<i32>,
    pub done: bool,
    /// The note's vector, when the writer embedded it before the write (see
    /// [`embed_notes`]). `None` stores the note unembedded and leaves it for
    /// the background backfill, which is the normal degraded state when no
    /// embedding backend is configured or the backend stalled.
    pub embedding: Option<NoteEmbedding>,
    /// The knowledge entry this note attaches, when it carries one (#1104).
    ///
    /// `None` on an upsert **preserves** whatever the stored note already
    /// attaches, exactly as `source` and `summary` behave on a knowledge write:
    /// a caller that rewrites a note's text and knows nothing about the
    /// attachment must not silently drop it.
    pub knowledge_entry_id: Option<String>,
}

impl NewScratchpadNote {
    /// A `note`-typed, unsequenced, not-done, unembedded upsert for
    /// `key` / `content`, attaching no knowledge entry.
    pub fn new(key: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            content: content.into(),
            note_type: DEFAULT_NOTE_TYPE.to_string(),
            sequence: None,
            done: false,
            embedding: None,
            knowledge_entry_id: None,
        }
    }

    /// The text embedded for this note: its key and its content.
    ///
    /// The key is part of it because the table's `tsv` covers
    /// `note_key || ' ' || content`, and because a key like `outcome-1.2` is
    /// often the most compact statement of what the note is about. Both the
    /// inline path here and `backfill_scratchpad_embeddings` build this same
    /// string -- a vector produced from a different one is not comparable with
    /// the vectors it would be ranked against.
    pub fn embed_text(&self) -> String {
        format!("{} {}", self.key, self.content)
    }
}

/// Embed a batch of notes in place, so a note written now is semantically
/// findable now (#717).
///
/// The background backfill runs on a several-minute cadence, and the case that
/// matters for a scratchpad is the agent looking for what it wrote moments ago
/// -- exactly the window that cadence leaves open. So the write path embeds,
/// and the backfill is the safety net rather than the only path.
///
/// Bounded by [`EMBED_TIMEOUT`]: a wedged backend must not hang the turn. On a
/// timeout, an error, or an answer that does not carry one vector per chunk,
/// every note is left unembedded and the write still lands. Those rows carry a
/// NULL vector, stay reachable through the full-text arm, and are picked up by
/// the next backfill pass.
///
/// All-or-nothing on purpose: a short answer from the embedder would otherwise
/// be zipped chunk-to-note out of step, pairing a note with another note's
/// vector. A wrong vector is worse than no vector, because nothing later
/// detects it.
pub async fn embed_notes(embed: &EmbedFn, model: &str, notes: &mut [NewScratchpadNote]) {
    if notes.is_empty() {
        return;
    }

    // Chunk every note, remembering which note each chunk belongs to, so one
    // backend round trip covers the whole batch.
    let mut owners: Vec<usize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (index, note) in notes.iter().enumerate() {
        for chunk in chunk_text(&note.embed_text(), CHUNK_MAX_CHARS, CHUNK_OVERLAP) {
            owners.push(index);
            texts.push(chunk);
        }
    }

    let expected = texts.len();
    let vectors = match tokio::time::timeout(EMBED_TIMEOUT, embed(texts)).await {
        Ok(Ok(vectors)) if vectors.len() == expected => vectors,
        Ok(Ok(vectors)) => {
            tracing::warn!(
                returned = vectors.len(),
                expected,
                "embedder answered with the wrong number of vectors; \
                 writing the notes unembedded for the backfill"
            );
            return;
        }
        Ok(Err(e)) => {
            tracing::warn!("failed to embed scratchpad notes: {e}");
            return;
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?EMBED_TIMEOUT,
                "embedding scratchpad notes timed out; writing them unembedded for the backfill"
            );
            return;
        }
    };

    for note in notes.iter_mut() {
        note.embedding = Some(NoteEmbedding {
            chunks: Vec::new(),
            model: model.to_string(),
        });
    }
    for (index, vector) in owners.into_iter().zip(vectors) {
        if let Some(embedding) = notes[index].embedding.as_mut() {
            embedding.chunks.push(vector);
        }
    }
}

/// Reserved note key whose content the service auto-surfaces as the
/// conversation's task anchor each turn (see `crate::service`). The model
/// sets/updates/clears it like any other note; its only special-ness is the
/// auto-surfacing, so an evolving goal survives windowing/compaction.
pub const SCRATCHPAD_GOAL_KEY: &str = "goal";

/// Maximum byte length of a single note's content. Notes larger than this
/// are rejected at the tool boundary — the scratchpad is for small,
/// high-signal working notes, not large blobs (those belong in a tool
/// result or the KB).
pub const MAX_NOTE_BYTES: usize = 8 * 1024;

/// Maximum number of notes accepted in a single `write` call. Excess notes
/// are not written and reported back as truncated, so one call can't grow
/// unboundedly.
pub const MAX_NOTES_PER_WRITE: usize = 32;

/// Maximum number of keys accepted in a single `get`/`delete` call. Excess
/// keys are processed up to the cap and the remainder reported as truncated.
pub const MAX_KEYS_PER_CALL: usize = 64;

/// Upper clamp on a search/list `max_results`. The tool requires the caller
/// to pass `max_results`; whatever they pass is clamped to this ceiling so a
/// single read can't return an unbounded row count.
pub const MAX_RESULTS_CEILING: usize = 100;

/// Soft byte budget for a single read response's serialized entries. Once
/// accumulated entries exceed this, the response is truncated and flagged so
/// one tool call can't blow out the model's context window.
pub const RESPONSE_BYTE_BUDGET: usize = 20 * 1024;

/// Maximum number of notes that may be pinned at once (#597).
///
/// A pinned note's full content is re-injected every single turn, so pinning
/// is the model spending its own context budget. The cap is deliberately small:
/// hitting it is the signal to unpin something that has stopped mattering, not
/// an obstacle to route around. Exceeding it is a hard error rather than a
/// silent eviction — the model must not lose a fact it believes is pinned.
pub const MAX_PINNED_NOTES: usize = 5;

/// Soft byte budget for the rendered `[Pinned]` block (#597). Pinned content
/// is repeated every turn, so the block is truncated (with an explicit marker,
/// never silently) once it exceeds this, bounding the per-turn cost even if
/// individual notes are near [`MAX_NOTE_BYTES`].
pub const PINNED_BLOCK_BYTE_BUDGET: usize = 4 * 1024;

/// How much of a referenced knowledge entry one pinned note may render, in
/// characters (#1104).
///
/// A note is capped at [`MAX_NOTE_BYTES`]; a knowledge entry has no equivalent
/// bound, so a single long entry would otherwise spend the whole
/// [`PINNED_BLOCK_BYTE_BUDGET`] by itself.
///
/// Why this size: four times
/// [`SUMMARY_MAX_CHARS`](desktop_assistant_protocol::SUMMARY_MAX_CHARS). A
/// summary line answers "is this the entry I want?", and the point of a pin is
/// having the fact itself, so a pinned entry must render more than a headline.
///
/// This is a per-entry cap, not a share of the block. It is measured in
/// characters and [`PINNED_BLOCK_BYTE_BUDGET`] in bytes, and the notes render
/// beside the entries, so [`MAX_PINNED_NOTES`] entries at this size do NOT fit
/// inside the block - a few long entries, or one entry outside ASCII, and the
/// block budget bites first. That is the intended order: this cap stops one
/// entry from spending everything, and the block budget decides how much
/// survives, with the cut marked and the notes that did not fit counted.
pub const PINNED_ENTRY_MAX_CHARS: usize = 4 * desktop_assistant_protocol::SUMMARY_MAX_CHARS;

/// Decide which keys a pin request may set, enforcing [`MAX_PINNED_NOTES`].
///
/// Pure so the cap is testable without a database. `currently_pinned` is the
/// conversation's pinned keys as storage sees them now; `keys` is the request.
///
/// Unpinning (`pinned == false`) is always allowed — it can only free budget.
/// Pinning is rejected as a whole when the resulting set would exceed the cap,
/// and the error names the keys already pinned so the model can choose what to
/// release. Keys already in the requested state are counted, not double-added,
/// so re-pinning an already-pinned note is a no-op rather than a cap error.
pub fn plan_pin(
    currently_pinned: &[String],
    keys: &[String],
    pinned: bool,
) -> Result<Vec<String>, String> {
    if !pinned {
        return Ok(keys.to_vec());
    }
    let mut resulting: Vec<&str> = currently_pinned.iter().map(String::as_str).collect();
    for key in keys {
        if !resulting.contains(&key.as_str()) {
            resulting.push(key.as_str());
        }
    }
    if resulting.len() > MAX_PINNED_NOTES {
        let mut already: Vec<&str> = currently_pinned.iter().map(String::as_str).collect();
        already.sort_unstable();
        return Err(format!(
            "at most {MAX_PINNED_NOTES} notes can be pinned at once; already pinned: [{}]. \
             Unpin one you no longer need (builtin_scratchpad_pin with pinned: false), \
             then pin this.",
            already.join(", ")
        ));
    }
    Ok(keys.to_vec())
}

/// Outbound port for the per-conversation scratchpad (ephemeral notes).
///
/// All methods are scoped to a single `conversation_id`; the adapter
/// additionally scopes by the task-local `UserId` (see [`crate::ports::auth`])
/// so cross-user reads cannot leak. Single-entity operations are expressed
/// through the multi-entity forms (`get_many`/`delete_many`) — the goal
/// anchor reads via `get_many(conv, &["goal"], 1)`.
pub trait ScratchpadStore: Send + Sync {
    /// Upsert a batch of notes for a conversation, replacing the content (and
    /// `note_type` / `sequence` / `done`) of any existing note with the same
    /// key. Returns the saved notes (with populated timestamps).
    fn write(
        &self,
        conversation_id: &str,
        notes: &[NewScratchpadNote],
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Fetch the notes for the given keys (in `updated_at DESC` order),
    /// capped at `limit`. Missing keys are simply absent from the result.
    fn get_many(
        &self,
        conversation_id: &str,
        keys: &[String],
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// List a conversation's notes, capped at `limit`. Ordered by `note_type`,
    /// then `sequence` ascending (nulls last), then `updated_at` descending —
    /// so a sequenced plan of `todo`s reads in order. When `note_type` is
    /// `Some`, only notes of that type are returned.
    fn list(
        &self,
        conversation_id: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Hybrid search over a conversation's notes (key + content), ranked,
    /// capped at `limit`. When `note_type` is `Some`, results are restricted
    /// to that type.
    ///
    /// `query_embedding` is the query embedded by `embedding_model`, as
    /// [`embed_notes`] embeds the notes themselves. An empty vector means the
    /// caller had no embedding backend, or its backend stalled; the search then
    /// takes the full-text path alone.
    ///
    /// The vector arm only reads rows stamped with `embedding_model`, because a
    /// vector of another dimension raises rather than missing. The full-text arm
    /// is never model-scoped, so changing the embedding model costs recall
    /// quality and not all recall.
    fn search(
        &self,
        conversation_id: &str,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Delete the notes for the given keys. Returns the number deleted.
    fn delete_many(
        &self,
        conversation_id: &str,
        keys: &[String],
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;

    /// Set (or clear) the `pinned` flag on the notes for the given keys (#597),
    /// returning the number of notes changed. Keys with no matching note are
    /// skipped rather than erroring — the caller reports them back.
    ///
    /// Why a dedicated write rather than a field on [`NewScratchpadNote`]:
    /// [`Self::write`] is an upsert, so carrying `pinned` there would make an
    /// ordinary content rewrite silently clear a pin the model is relying on.
    /// Keeping the flag on its own path makes that impossible, and makes
    /// pin/unpin cheap — it never restates the note's content.
    fn set_pinned(
        &self,
        conversation_id: &str,
        keys: &[String],
        pinned: bool,
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;

    /// Drop the knowledge-entry attachment from the given notes (by note id)
    /// and release any pin those notes held, returning the number changed
    /// (#1104).
    ///
    /// Why it exists: a reference must never outlive its entry. The render path
    /// resolves each pinned note's entry every round, and an entry that no
    /// longer resolves leaves a pin that asserts nothing — a fact the model
    /// believes it has and does not. This is the repair.
    ///
    /// Why note ids and not keys: a key is unique per `(conversation,
    /// owner_todo)`, and the render reads a whole subagent subtree, so a key
    /// alone could name a row in a different namespace. `conversation_id`
    /// still travels, so the repair cannot reach outside the pad it read.
    ///
    /// Confined to the caller's own `owner_todo` **subtree**, which is wider
    /// than [`Self::set_pinned`] and [`Self::delete_many`] (own namespace only)
    /// and narrower than [`Self::list`] (namespace-blind at the top level).
    /// Both extremes fail:
    ///
    /// - Unconfined, a subagent round releases a pin in an ancestor's
    ///   namespace, and the line saying so renders into the subagent's block
    ///   where the parent never sees it.
    /// - Own-namespace-only, a note the top-level read can see but not repair
    ///   keeps a dead attachment for the life of the conversation, holds a slot
    ///   of the pin cap, and costs a knowledge read every round - and the model
    ///   cannot clear it either, because the pin and delete verbs are confined
    ///   the same way.
    ///
    /// Own-subtree fits both: the root namespace repairs anything, matching its
    /// namespace-blind read, and a subagent repairs itself and its descendants
    /// but never an ancestor or a sibling.
    ///
    /// Scoped by the task-local `UserId`, which is the tenant guard.
    ///
    /// Idempotent: a second call over the same ids changes nothing and
    /// returns 0.
    fn release_knowledge_references(
        &self,
        conversation_id: &str,
        note_ids: &[String],
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;

    /// Delete every note for a conversation. Returns the number deleted.
    fn clear(&self, conversation_id: &str) -> impl Future<Output = Result<u64, CoreError>> + Send;

    /// Delete an `owner_todo` namespace AND all its descendants (the whole
    /// subtree), returning the number deleted. User-scoped via the task-local
    /// `UserId`, fail-closed, and idempotent (a second call returns 0).
    ///
    /// Why: the hard-coded roll-up cascade (#287) frees a completed step's
    /// descendant subagent namespaces in one shot when the enclosing step
    /// completes. Distinct from [`Self::delete_many`]/[`Self::clear`], which are
    /// confined to a single namespace; this deliberately spans the subtree.
    fn delete_owner_subtree(
        &self,
        conversation_id: &str,
        owner_todo: &str,
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;
}

/// Boxed async closure for batch-upserting scratchpad notes through
/// non-generic boundaries (`mcp-client` doesn't depend on `storage`).
pub type ScratchpadWriteFn = Arc<
    dyn Fn(
            String,
            Vec<NewScratchpadNote>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for fetching notes by key (also backs the goal anchor).
pub type ScratchpadGetManyFn = Arc<
    dyn Fn(
            String,
            Vec<String>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for listing notes (optionally filtered by `note_type`),
/// ordered by type then sequence.
pub type ScratchpadListFn = Arc<
    dyn Fn(
            String,
            Option<String>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for hybrid (vector + full-text) searching of notes.
///
/// Args: `(conversation_id, query, query_embedding, embedding_model,
/// note_type, limit)`. The query vector and the model that produced it travel
/// together for the reason [`NoteEmbedding`] gives. An empty vector takes the
/// full-text path alone.
pub type ScratchpadSearchFn = Arc<
    dyn Fn(
            String,
            String,
            Vec<f32>,
            String,
            Option<String>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for deleting notes by key. Returns the count deleted.
pub type ScratchpadDeleteManyFn = Arc<
    dyn Fn(String, Vec<String>) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for clearing all of a conversation's notes.
pub type ScratchpadClearFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>> + Send + Sync,
>;

/// Boxed async closure for pinning / unpinning notes by key (#597).
pub type ScratchpadSetPinnedFn = Arc<
    dyn Fn(
            String,
            Vec<String>,
            bool,
        ) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for releasing dangling knowledge-entry attachments by
/// note id (#1104). Returns the count changed. See
/// [`ScratchpadStore::release_knowledge_references`].
pub type ScratchpadReleaseReferencesFn = Arc<
    dyn Fn(String, Vec<String>) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for cascade-deleting an `owner_todo` subtree (the
/// namespace and all its descendants). Args: `(conversation_id, owner_todo)`;
/// returns the count deleted. Used by the #287 roll-up cascade through
/// non-generic boundaries.
pub type ScratchpadDeleteSubtreeFn = Arc<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::embedding::EmbedFn;

    fn keys(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    const MODEL: &str = "nomic-embed-text@1111111111111111111111111111111111111111";

    /// An embedding backend that records every text it is handed and answers
    /// each with `vector`.
    fn recording_embed(seen: Arc<std::sync::Mutex<Vec<String>>>, vector: Vec<f32>) -> EmbedFn {
        Arc::new(move |texts: Vec<String>| {
            let seen = Arc::clone(&seen);
            let vector = vector.clone();
            Box::pin(async move {
                let n = texts.len();
                seen.lock().expect("record texts").extend(texts);
                Ok(vec![vector; n])
            })
        })
    }

    /// Acceptance (#717): a wedged embedding backend must not block the write.
    /// A stuck backend is the case `EMBED_TIMEOUT` exists for -- the note has to
    /// land, unembedded, for the background backfill to pick up.
    #[tokio::test(start_paused = true)]
    async fn a_wedged_embedding_backend_does_not_block_the_write() {
        let embed: EmbedFn = Arc::new(|_| {
            Box::pin(async {
                // Never answers, as a stuck Ollama does not.
                std::future::pending::<()>().await;
                unreachable!("pending never resolves")
            })
        });
        let mut notes = vec![NewScratchpadNote::new(
            "finding",
            "the pool leaks connections under load",
        )];

        embed_notes(&embed, MODEL, &mut notes).await;

        assert!(
            notes[0].embedding.is_none(),
            "a wedged backend must leave the note unembedded so the write still lands"
        );
    }

    /// A backend that fails outright degrades the same way a wedged one does.
    #[tokio::test]
    async fn a_failing_embedding_backend_leaves_the_note_unembedded() {
        let embed: EmbedFn =
            Arc::new(|_| Box::pin(async { Err(CoreError::Storage("backend down".into())) }));
        let mut notes = vec![NewScratchpadNote::new("finding", "the pool leaks")];

        embed_notes(&embed, MODEL, &mut notes).await;

        assert!(notes[0].embedding.is_none());
    }

    /// The vector and the model that produced it travel together, so a note can
    /// never be stamped with a model that did not embed it.
    #[tokio::test]
    async fn embed_notes_stamps_each_note_with_the_model_that_produced_its_vector() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let embed = recording_embed(Arc::clone(&seen), vec![0.25, 0.5, 0.75]);
        let mut notes = vec![
            NewScratchpadNote::new("a", "alpha"),
            NewScratchpadNote::new("b", "bravo"),
        ];

        embed_notes(&embed, MODEL, &mut notes).await;

        for note in &notes {
            let embedding = note
                .embedding
                .as_ref()
                .unwrap_or_else(|| panic!("note {} must carry a vector", note.key));
            assert_eq!(embedding.model, MODEL);
            assert_eq!(embedding.chunks, vec![vec![0.25, 0.5, 0.75]]);
        }
    }

    /// The inline embed and the background backfill must embed the same text,
    /// or a note embedded by one is not comparable with a query matched against
    /// the other. The text is `key + content`, matching the table's `tsv`.
    #[tokio::test]
    async fn embed_notes_embeds_the_key_together_with_the_content() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let embed = recording_embed(Arc::clone(&seen), vec![1.0]);
        let mut notes = vec![NewScratchpadNote::new("deploy", "ship it on Friday")];

        embed_notes(&embed, MODEL, &mut notes).await;

        let texts = seen.lock().expect("read recorded texts").clone();
        assert_eq!(texts, vec!["deploy ship it on Friday".to_string()]);
    }

    /// An embedder that returns fewer vectors than it was given chunks has
    /// broken its contract. Zipping a short answer would pair a note with
    /// another note's vector, which is worse than not embedding at all.
    #[tokio::test]
    async fn a_short_answer_from_the_embedder_leaves_every_note_unembedded() {
        let embed: EmbedFn = Arc::new(|_| Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) }));
        let mut notes = vec![
            NewScratchpadNote::new("a", "alpha"),
            NewScratchpadNote::new("b", "bravo"),
        ];

        embed_notes(&embed, MODEL, &mut notes).await;

        assert!(notes.iter().all(|n| n.embedding.is_none()));
    }

    /// An empty batch never reaches the backend.
    #[tokio::test]
    async fn embed_notes_does_not_call_the_backend_for_an_empty_batch() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let embed = recording_embed(Arc::clone(&seen), vec![1.0]);
        let mut notes: Vec<NewScratchpadNote> = Vec::new();

        embed_notes(&embed, MODEL, &mut notes).await;

        assert!(seen.lock().expect("read recorded texts").is_empty());
    }

    #[test]
    fn pin_respects_max_pinned_notes() {
        // At the cap, one more pin is refused as a whole and the error names
        // what is already pinned so the model can choose what to release.
        let at_cap = keys(&["a", "b", "c", "d", "e"]);
        assert_eq!(at_cap.len(), MAX_PINNED_NOTES, "precondition: at the cap");
        let err = plan_pin(&at_cap, &keys(&["f"]), true).expect_err("must refuse past the cap");
        assert!(err.contains("at most 5"), "{err}");
        for k in ["a", "b", "c", "d", "e"] {
            assert!(err.contains(k), "error must name already-pinned {k}: {err}");
        }
    }

    #[test]
    fn pin_below_the_cap_is_allowed() {
        let planned = plan_pin(&keys(&["a", "b"]), &keys(&["c"]), true).expect("under the cap");
        assert_eq!(planned, keys(&["c"]));
    }

    #[test]
    fn repinning_an_already_pinned_note_is_not_a_cap_error() {
        // Otherwise a harmless no-op would look like the model overspending.
        let at_cap = keys(&["a", "b", "c", "d", "e"]);
        let planned =
            plan_pin(&at_cap, &keys(&["c"]), true).expect("re-pin is a no-op, not a cap breach");
        assert_eq!(planned, keys(&["c"]));
    }

    #[test]
    fn unpinning_is_always_allowed_even_at_the_cap() {
        // Unpinning can only free budget, so it must never be blocked.
        let at_cap = keys(&["a", "b", "c", "d", "e"]);
        let planned = plan_pin(&at_cap, &keys(&["a"]), false).expect("unpin must be allowed");
        assert_eq!(planned, keys(&["a"]));
    }

    #[test]
    fn pinning_a_whole_batch_past_the_cap_is_refused_atomically() {
        // Partial application would leave the model unsure what is pinned.
        let err = plan_pin(&keys(&["a", "b", "c"]), &keys(&["d", "e", "f"]), true)
            .expect_err("3 + 3 exceeds the cap of 5");
        assert!(err.contains("at most 5"), "{err}");
    }

    #[test]
    fn pinning_from_empty_up_to_the_cap_is_allowed() {
        let planned = plan_pin(&[], &keys(&["a", "b", "c", "d", "e"]), true)
            .expect("exactly the cap must fit");
        assert_eq!(planned.len(), MAX_PINNED_NOTES);
    }

    struct MockScratchpadStore;

    impl ScratchpadStore for MockScratchpadStore {
        async fn write(
            &self,
            conversation_id: &str,
            notes: &[NewScratchpadNote],
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(notes
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let mut note =
                        ScratchpadNote::new(format!("id-{i}"), conversation_id, &n.key, &n.content);
                    note.note_type = n.note_type.clone();
                    note.sequence = n.sequence;
                    note.done = n.done;
                    note
                })
                .collect())
        }

        async fn get_many(
            &self,
            _conversation_id: &str,
            _keys: &[String],
            _limit: usize,
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(vec![])
        }

        async fn list(
            &self,
            _conversation_id: &str,
            _note_type: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(vec![])
        }

        async fn search(
            &self,
            _conversation_id: &str,
            _query: &str,
            _query_embedding: Vec<f32>,
            _embedding_model: &str,
            _note_type: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(vec![])
        }

        async fn delete_many(
            &self,
            _conversation_id: &str,
            keys: &[String],
        ) -> Result<u64, CoreError> {
            Ok(keys.len() as u64)
        }

        async fn release_knowledge_references(
            &self,
            _conversation_id: &str,
            note_ids: &[String],
        ) -> Result<u64, CoreError> {
            Ok(note_ids.len() as u64)
        }

        async fn clear(&self, _conversation_id: &str) -> Result<u64, CoreError> {
            Ok(0)
        }

        async fn set_pinned(
            &self,
            _conversation_id: &str,
            keys: &[String],
            _pinned: bool,
        ) -> Result<u64, CoreError> {
            Ok(keys.len() as u64)
        }

        async fn delete_owner_subtree(
            &self,
            _conversation_id: &str,
            _owner_todo: &str,
        ) -> Result<u64, CoreError> {
            Ok(0)
        }
    }

    #[tokio::test]
    async fn mock_store_write_roundtrips_batch() {
        let store = MockScratchpadStore;
        let mut todo = NewScratchpadNote::new("step-1", "wire it");
        todo.note_type = "todo".to_string();
        todo.sequence = Some(1);
        let notes = vec![NewScratchpadNote::new("goal", "ship it"), todo];
        let saved = store.write("conv-1", &notes).await.unwrap();
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].key, "goal");
        assert_eq!(saved[0].note_type, DEFAULT_NOTE_TYPE);
        assert_eq!(saved[1].content, "wire it");
        assert_eq!(saved[1].note_type, "todo");
        assert_eq!(saved[1].sequence, Some(1));
    }

    #[tokio::test]
    async fn mock_store_delete_many_returns_count() {
        let store = MockScratchpadStore;
        let deleted = store
            .delete_many("conv-1", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(deleted, 2);
    }

    fn _assert_scratchpad_store<T: ScratchpadStore>() {}
}
