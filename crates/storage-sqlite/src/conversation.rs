//! SQLite adapter for [`ConversationStore`] (increment 1).

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
use desktop_assistant_core::ports::inbound::ConversationModelSelection;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::prompts::PersonalityOverride;
use sqlx::SqlitePool;

/// SQLite adapter for the `conversations` / `messages` / `message_summaries`
/// tables.
pub struct SqliteConversationStore {
    pool: SqlitePool,
}

impl SqliteConversationStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Set (or clear) the stored model selection for a conversation.
    pub async fn set_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        let _ = (&self.pool, conversation_id, selection);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    /// Read the stored model selection for a conversation.
    pub async fn get_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        let _ = (&self.pool, conversation_id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    /// Set (or clear) the stored personality override for a conversation.
    pub async fn set_conversation_personality(
        &self,
        conversation_id: &ConversationId,
        personality: Option<&PersonalityOverride>,
    ) -> Result<(), CoreError> {
        let _ = (&self.pool, conversation_id, personality);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    /// Read the stored personality override for a conversation.
    pub async fn get_conversation_personality(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<PersonalityOverride>, CoreError> {
        let _ = (&self.pool, conversation_id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    /// Read the conversation's tags.
    pub async fn get_conversation_tags(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<String>, CoreError> {
        let _ = (&self.pool, conversation_id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}

impl ConversationStore for SqliteConversationStore {
    async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
        let _ = (&self.pool, conv);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        let _ = &self.pool;
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
        let _ = (&self.pool, conv);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn create_summary(
        &self,
        conversation_id: &ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        let _ = (
            &self.pool,
            conversation_id,
            summary,
            start_ordinal,
            end_ordinal,
        );
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
        let _ = (&self.pool, summary_id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
