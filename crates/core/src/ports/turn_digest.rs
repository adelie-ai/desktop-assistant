//! The episodic turn index: one digest per turn, scoped to the person (#1349).
//!
//! ## What a digest is
//!
//! What the harness kept of one turn, built by [`crate::turn_capture`] with no
//! model call: the user's own words, what the assistant answered, and the tool
//! calls with their outcomes. The construction and its rules live there. This
//! module is the store the digest lands in and the shape it is stored as.
//!
//! ## Why the person and not the conversation
//!
//! A digest on the conversation's own scratchpad duplicates the transcript
//! sitting beside it, and it can never answer "when did I last deal with
//! this": a turn in one conversation is invisible from every other. Scoping
//! the store to the person makes the episodic record reachable by relevance
//! across the whole account, which is what recognition needs and what the
//! transcript, the rolling summary and a lexical search each fail to give.
//!
//! The store holds the home conversation's turns too, so a later read has the
//! material for within-conversation recall as well as cross-conversation.
//!
//! ## How a digest is read back, and what each read may show
//!
//! Two paths, and they show different halves of a digest on purpose (#1350).
//!
//! The past-turns arm of `[Recall]` offers episodes unprompted, ahead of the
//! prompt, and its line carries [`TurnDigest::marked_asked_text`] - the user's
//! own words and the disposition marker, never the assistant's half. The block
//! makes no tool call, so nothing folds an offered line's provenance into the
//! reading turn, and the assistant's half of a turn can quote a page an outside
//! party controls.
//!
//! The fetch path is a read of the digest itself, by [`TurnDigestStore::get`],
//! which carries the whole of [`TurnDigest::marked_text`] and MARKS the reading
//! turn's provenance where [`TurnDigest::after_outside_read`] says the writing
//! turn had read outside content. It is the digest that opens and never the
//! transcript: a transcript read is scoped to the active conversation and fails
//! closed, so a transcript pointer in a cross-conversation line would name a
//! body that cannot be opened from where it was offered.
//!
//! ## Disposition, and what it is for
//!
//! A knowledge entry carries a [`Disposition`] and a soft-deletion story, so
//! the machinery that withholds and retires a claim can reach it. A digest
//! that outlives its conversation needs the same hooks, or that machinery
//! cannot reach episodes at all. [`TurnDigest::marked_text`] is how a reader
//! should meet a dispositioned digest: the disposition's own marker is joined
//! to the text there, in one place, so a render path has one call to make
//! rather than a rule to remember. [`TurnDigest::content`] is still public and
//! still reachable, so this is a convention and not an enforced property.
//!
//! ## Deletion
//!
//! A digest's lifecycle is no longer its conversation's, so the cascade is
//! stated in the schema rather than left to a caller: deleting a conversation
//! deletes its digests. Without it, deleting a conversation would be a promise
//! the product does not keep.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::Disposition;
use crate::ports::embedding::{ChunkEmbeddable, ChunkedEmbedding};

/// The tool that reads one digest back (#1350).
///
/// Owned here rather than declared in the adapter, the way
/// [`crate::ports::transcript::TRANSCRIPT_GET_TOOL`] is: the name is what
/// [`crate::tool_provenance`] classifies and what the standing instruction
/// names, so it is stated once.
pub const TURN_DIGEST_GET_TOOL: &str = "builtin_episode_get";

/// Maximum byte length of one digest's content.
///
/// The digest is a bounded record, not a second transcript: a turn can carry a
/// megabyte of tool output, and every byte of it is already in `messages`.
/// [`crate::turn_capture`] spends this budget in priority order and says so
/// when it cut.
pub const MAX_DIGEST_BYTES: usize = 8 * 1024;

/// A digest to upsert, identified by the message that opened its turn.
#[derive(Debug, Clone, PartialEq)]
pub struct NewTurnDigest {
    /// The id of the message that opened the turn. The digest's identity
    /// within its conversation, so re-running the capture for the same turn
    /// writes the same row rather than a second one.
    pub opening_message_id: String,
    /// The digest text, as [`crate::turn_capture::capture_turn`] built it.
    pub content: String,
    /// Whether the turn that produced this text had already read content from
    /// outside the trust boundary (#1247). Carried from the writing turn, and
    /// re-derived from the stored tool traffic when a digest is backfilled.
    pub after_outside_read: bool,
    /// The digest's vector, when the writer embedded it before the write.
    /// `None` stores it unembedded and leaves it for the background backfill,
    /// which is the normal degraded state when no embedding backend is
    /// configured or the backend stalled.
    pub embedding: Option<ChunkedEmbedding>,
}

impl NewTurnDigest {
    /// A digest for `opening_message_id`, unembedded and unstamped.
    pub fn new(opening_message_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            opening_message_id: opening_message_id.into(),
            content: content.into(),
            after_outside_read: false,
            embedding: None,
        }
    }
}

impl ChunkEmbeddable for NewTurnDigest {
    /// The digest's own text, and nothing else.
    ///
    /// The opening message id is deliberately left out: it is a UUID, it says
    /// nothing about what the turn was about, and a store's backfill has to
    /// build the same string as its write path or the two produce vectors that
    /// are not comparable. `crate::storage`'s backfill selects `content` for
    /// exactly this reason.
    fn embed_text(&self) -> String {
        self.content.clone()
    }

    fn set_embedding(&mut self, embedding: ChunkedEmbedding) {
        self.embedding = Some(embedding);
    }
}

/// A stored turn digest.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnDigest {
    /// The row's own id.
    pub id: String,
    /// The conversation the turn happened in. A digest is readable from every
    /// conversation the person owns; this says where it came from.
    pub conversation_id: String,
    /// The message that opened the turn - the handle a reader follows back
    /// into the transcript.
    pub opening_message_id: String,
    /// The digest text as stored, WITHOUT this digest's disposition marker.
    ///
    /// A render path wants [`Self::marked_text`] instead, which joins the
    /// marker on: a `refuted` or `superseded` digest shown from this field
    /// alone reads as a current record of what happened. Nothing enforces
    /// that - the field is public, like
    /// [`crate::domain::KnowledgeEntry::content`], which pairs with its own
    /// `marked_text` the same way. Treat this sentence as a convention the
    /// reader has to keep, not a guarantee the type makes.
    pub content: String,
    /// Whether the turn that produced this text had already read outside
    /// content.
    pub after_outside_read: bool,
    /// What a person or the consolidation machinery has judged this episode to
    /// be.
    pub disposition: Disposition,
    /// The stated reason for a non-default disposition.
    pub disposition_reason: Option<String>,
    /// The digest that replaced this one, where one did.
    pub superseded_by: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl TurnDigest {
    /// [`Self::content`] as a reader should see it: carrying this digest's own
    /// [`Disposition::marker`].
    ///
    /// The one place the marker is joined, for the same reason
    /// [`crate::domain::KnowledgeEntry::marked_text`] is: two call sites
    /// agreeing to remember the rule is the shape that lets one of them go out
    /// unmarked. It is the one place, not the only reachable one - see
    /// [`Self::content`].
    #[must_use]
    pub fn marked_text(&self) -> String {
        format!("{}{}", self.disposition.marker(), self.content)
    }

    /// The user's own half of this digest, carrying this digest's own
    /// [`Disposition::marker`], and `None` where there is no such half (#1350).
    ///
    /// **What an unprompted render may show, and the whole of it.** The line
    /// the `[Recall]` episode arm offers is built from this and from nothing
    /// else - see [`crate::turn_capture::asked_half`] for why the assistant's
    /// half may not cross a conversation unasked, and
    /// [`crate::ports::recall::RecallEpisode`] for the type that has no other
    /// way to be built.
    ///
    /// The marker is joined here rather than through [`Self::marked_text`]
    /// because that method marks the whole content, and this line is half of
    /// it: marking through it would put the answer half on the line, which is
    /// the one thing this path must not do. Both methods read one field and
    /// one marker, so a disposition added to the vocabulary reaches both.
    #[must_use]
    pub fn marked_asked_text(&self) -> Option<String> {
        crate::turn_capture::asked_half(&self.content)
            .map(|asked| format!("{}{}", self.disposition.marker(), asked))
    }
}

/// The episodic turn index, scoped to the person.
///
/// Every method reads the caller's own user id from the ambient scope, the
/// same way every other personal-data store does; nothing here takes a user as
/// a parameter, and a read for one person never answers with another's rows.
#[async_trait::async_trait]
pub trait TurnDigestStore: Send + Sync {
    /// Upsert `digests` for `conversation_id`, returning the stored rows.
    ///
    /// Keyed on `(conversation_id, opening_message_id)`, so capturing the same
    /// turn twice leaves one row (AGENTS.md 8.4).
    async fn write(
        &self,
        conversation_id: &str,
        digests: &[NewTurnDigest],
    ) -> Result<Vec<TurnDigest>, CoreError>;

    /// The person's most recent digests, newest first, across every
    /// conversation they own.
    ///
    /// `include_dispositioned` admits the digests an ordinary read leaves out
    /// - see [`Disposition::Obsolete`], which no longer applies and is not
    /// offered unless it is asked for.
    async fn recent(
        &self,
        limit: usize,
        include_dispositioned: bool,
    ) -> Result<Vec<TurnDigest>, CoreError>;

    /// One digest by its row id, or `None` when this person owns no such row.
    async fn get(&self, id: &str) -> Result<Option<TurnDigest>, CoreError>;

    /// Record what this digest is judged to be.
    ///
    /// `superseded_by` is required for [`Disposition::Superseded`] and
    /// [`Disposition::Redundant`], which resolve through the link and cannot
    /// be carried without one. Answers whether a row changed.
    async fn set_disposition(
        &self,
        id: &str,
        disposition: Disposition,
        reason: Option<&str>,
        superseded_by: Option<&str>,
    ) -> Result<bool, CoreError>;
}

/// Boxed async closure for reading one digest back by its row id, for wiring
/// the fetch path through non-generic boundaries (#1350).
///
/// One id rather than a batch, unlike the knowledge base's. A digest runs to
/// [`MAX_DIGEST_BYTES`], the `[Recall]` block offers at most a handful of
/// episode lines a turn, and a batch of whole turns would fill a response
/// budget with material the model has not decided it wants - which is the
/// economy the whole index rests on.
pub type TurnDigestGetFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<Option<TurnDigest>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for writing digests through non-generic boundaries.
pub type TurnDigestWriteFn = Arc<
    dyn Fn(
            String,
            Vec<NewTurnDigest>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<TurnDigest>, CoreError>> + Send>>
        + Send
        + Sync,
>;
