use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::Role;

/// A single match returned by [`ConversationSearchStore::search_messages`].
///
/// Carries enough context for a tool result to render usefully without a
/// follow-up fetch: the conversation id and title for grouping, the
/// message ordinal for re-anchoring inside a conversation, the matched
/// content with a highlighted snippet, and the rank score so callers can
/// truncate or re-rank.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageHit {
    pub conversation_id: String,
    pub conversation_title: String,
    pub ordinal: i32,
    pub role: Role,
    pub content: String,
    /// `ts_headline`-formatted snippet around the matched terms.
    pub snippet: String,
    /// Postgres `ts_rank_cd` score; higher is more relevant.
    pub rank: f32,
    /// Conversation `updated_at` as an RFC3339 string for chronological
    /// secondary sort / display.
    pub updated_at: String,
}

/// Outbound port for full-text searching past conversations. Backed by
/// the Postgres tsvector columns added in migration #013. The JSON store
/// has no equivalent and intentionally omits this trait.
pub trait ConversationSearchStore: Send + Sync {
    /// Run a full-text query over message content (with conversation
    /// title/summary contributing as a secondary axis). `role_filter`
    /// scopes hits to a specific message role; `None` searches all
    /// roles. Hits are ordered by `ts_rank_cd` descending.
    fn search_messages(
        &self,
        query: &str,
        limit: usize,
        role_filter: Option<Role>,
    ) -> impl Future<Output = Result<Vec<MessageHit>, CoreError>> + Send;
}

/// Boxed async closure for searching past conversations through
/// non-generic boundaries (mirrors [`KnowledgeSearchFn`]).
///
/// [`KnowledgeSearchFn`]: super::knowledge::KnowledgeSearchFn
pub type ConversationSearchFn = Arc<
    dyn Fn(
            String,
            usize,
            Option<Role>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<MessageHit>, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSearch;

    impl ConversationSearchStore for MockSearch {
        async fn search_messages(
            &self,
            _query: &str,
            _limit: usize,
            _role_filter: Option<Role>,
        ) -> Result<Vec<MessageHit>, CoreError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn mock_search_returns_empty() {
        let s = MockSearch;
        let hits = s.search_messages("hello", 10, None).await.unwrap();
        assert!(hits.is_empty());
    }

    fn _assert_search_store<T: ConversationSearchStore>() {}
}
