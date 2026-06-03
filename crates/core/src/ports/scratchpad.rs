use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::ScratchpadNote;

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
    /// Upsert a batch of `(key, content)` notes for a conversation, replacing
    /// the content of any existing note with the same key. Returns the saved
    /// notes (with populated timestamps).
    fn write(
        &self,
        conversation_id: &str,
        notes: &[(String, String)],
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Fetch the notes for the given keys (in `updated_at DESC` order),
    /// capped at `limit`. Missing keys are simply absent from the result.
    fn get_many(
        &self,
        conversation_id: &str,
        keys: &[String],
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// List all notes for a conversation, newest first, capped at `limit`.
    fn list(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Full-text search over a conversation's notes (key + content), ranked,
    /// capped at `limit`.
    fn search(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send;

    /// Delete the notes for the given keys. Returns the number deleted.
    fn delete_many(
        &self,
        conversation_id: &str,
        keys: &[String],
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;

    /// Delete every note for a conversation. Returns the number deleted.
    fn clear(
        &self,
        conversation_id: &str,
    ) -> impl Future<Output = Result<u64, CoreError>> + Send;
}

/// Boxed async closure for batch-upserting scratchpad notes through
/// non-generic boundaries (`mcp-client` doesn't depend on `storage`).
pub type ScratchpadWriteFn = Arc<
    dyn Fn(
            String,
            Vec<(String, String)>,
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

/// Boxed async closure for listing all notes (newest first).
pub type ScratchpadListFn = Arc<
    dyn Fn(
            String,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for full-text searching notes.
pub type ScratchpadSearchFn = Arc<
    dyn Fn(
            String,
            String,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ScratchpadNote>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for deleting notes by key. Returns the count deleted.
pub type ScratchpadDeleteManyFn = Arc<
    dyn Fn(
            String,
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for clearing all of a conversation's notes.
pub type ScratchpadClearFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<u64, CoreError>> + Send>> + Send + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockScratchpadStore;

    impl ScratchpadStore for MockScratchpadStore {
        async fn write(
            &self,
            conversation_id: &str,
            notes: &[(String, String)],
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(notes
                .iter()
                .enumerate()
                .map(|(i, (k, c))| ScratchpadNote::new(format!("id-{i}"), conversation_id, k, c))
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
            _limit: usize,
        ) -> Result<Vec<ScratchpadNote>, CoreError> {
            Ok(vec![])
        }

        async fn search(
            &self,
            _conversation_id: &str,
            _query: &str,
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
    }

    #[tokio::test]
    async fn mock_store_write_roundtrips_batch() {
        let store = MockScratchpadStore;
        let notes = vec![
            ("goal".to_string(), "ship it".to_string()),
            ("q".to_string(), "which db".to_string()),
        ];
        let saved = store.write("conv-1", &notes).await.unwrap();
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].key, "goal");
        assert_eq!(saved[1].content, "which db");
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
