use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::{DEFAULT_NOTE_TYPE, ScratchpadNote};

/// A note to upsert into the scratchpad. Carries the structured fields that
/// don't fit a bare `(key, content)` pair: a free-text `note_type`
/// (default `note`), an optional `sequence` (sorted within a type), and a
/// `done` flag. Construct via [`NewScratchpadNote::new`] and the field
/// setters, or as a struct literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewScratchpadNote {
    pub key: String,
    pub content: String,
    pub note_type: String,
    pub sequence: Option<i32>,
    pub done: bool,
}

impl NewScratchpadNote {
    /// A `note`-typed, unsequenced, not-done upsert for `key` / `content`.
    pub fn new(key: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            content: content.into(),
            note_type: DEFAULT_NOTE_TYPE.to_string(),
            sequence: None,
            done: false,
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

    /// Full-text search over a conversation's notes (key + content), ranked,
    /// capped at `limit`. When `note_type` is `Some`, results are restricted
    /// to that type.
    fn search(
        &self,
        conversation_id: &str,
        query: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Delete the notes for the given keys. Returns the number deleted.
    fn delete_many(
        &self,
        conversation_id: &str,
        keys: &[String],
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

/// Boxed async closure for full-text searching notes (optionally filtered by
/// `note_type`).
pub type ScratchpadSearchFn = Arc<
    dyn Fn(
            String,
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

        async fn clear(&self, _conversation_id: &str) -> Result<u64, CoreError> {
            Ok(0)
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
