use crate::CoreError;
use crate::domain::{Conversation, ConversationId};

/// Outbound port for persisting conversations.
pub trait ConversationStore: Send + Sync {
    fn create(
        &self,
        conv: Conversation,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    fn list(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<Conversation>, CoreError>> + Send;

    fn update(
        &self,
        conv: Conversation,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn delete(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Mark a conversation as archived.
    fn archive(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Remove the archived flag from a conversation.
    fn unarchive(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Collapse a range of messages behind a summary. Returns the new summary ID.
    fn create_summary(
        &self,
        conversation_id: &ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;

    /// Expand (undo) a summary — deletes the summary row; ON DELETE SET NULL
    /// clears summary_id on all linked messages.
    fn expand_summary(
        &self,
        summary_id: &str,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Message, MessageSummary, Role};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MockStore {
        data: Mutex<HashMap<String, Conversation>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl ConversationStore for MockStore {
        async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
            self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
            Ok(())
        }

        async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            self.data
                .lock()
                .unwrap()
                .get(&id.0)
                .cloned()
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
        }

        async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
            Ok(self.data.lock().unwrap().values().cloned().collect())
        }

        async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            if data.contains_key(&conv.id.0) {
                data.insert(conv.id.0.clone(), conv);
                Ok(())
            } else {
                Err(CoreError::ConversationNotFound(conv.id.0.clone()))
            }
        }

        async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
            self.data
                .lock()
                .unwrap()
                .remove(&id.0)
                .map(|_| ())
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
        }

        async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            let conv = data
                .get_mut(&id.0)
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
            conv.archived_at = Some("2026-01-01 00:00:00".to_string());
            Ok(())
        }

        async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            let conv = data
                .get_mut(&id.0)
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
            conv.archived_at = None;
            Ok(())
        }

        async fn create_summary(
            &self,
            conversation_id: &ConversationId,
            summary: String,
            start_ordinal: usize,
            end_ordinal: usize,
        ) -> Result<String, CoreError> {
            let mut data = self.data.lock().unwrap();
            let conv = data
                .get_mut(&conversation_id.0)
                .ok_or_else(|| CoreError::ConversationNotFound(conversation_id.0.clone()))?;
            let id = format!("summary-{}", conv.summaries.len() + 1);
            for (i, msg) in conv.messages.iter_mut().enumerate() {
                if i >= start_ordinal && i <= end_ordinal {
                    msg.summary_id = Some(id.clone());
                }
            }
            conv.summaries.push(MessageSummary {
                id: id.clone(),
                summary,
            });
            Ok(id)
        }

        async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            for conv in data.values_mut() {
                if let Some(pos) = conv.summaries.iter().position(|s| s.id == summary_id) {
                    conv.summaries.remove(pos);
                    for msg in conv.messages.iter_mut() {
                        if msg.summary_id.as_deref() == Some(summary_id) {
                            msg.summary_id = None;
                        }
                    }
                    return Ok(());
                }
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn store_create_and_get() {
        let store = MockStore::new();
        let conv = Conversation::new("c1", "Test");
        store.create(conv).await.unwrap();

        let retrieved = store.get(&ConversationId::from("c1")).await.unwrap();
        assert_eq!(retrieved.title, "Test");
    }

    #[tokio::test]
    async fn store_list_returns_all() {
        let store = MockStore::new();
        store.create(Conversation::new("c1", "A")).await.unwrap();
        store.create(Conversation::new("c2", "B")).await.unwrap();

        let all = store.list().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn store_delete_removes() {
        let store = MockStore::new();
        store.create(Conversation::new("c1", "A")).await.unwrap();
        store.delete(&ConversationId::from("c1")).await.unwrap();

        let result = store.get(&ConversationId::from("c1")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn store_update_persists() {
        let store = MockStore::new();
        let mut conv = Conversation::new("c1", "Original");
        store.create(conv.clone()).await.unwrap();

        conv.messages.push(Message::new(Role::User, "hello"));
        store.update(conv).await.unwrap();

        let retrieved = store.get(&ConversationId::from("c1")).await.unwrap();
        assert_eq!(retrieved.messages.len(), 1);
    }

    #[tokio::test]
    async fn store_get_nonexistent_fails() {
        let store = MockStore::new();
        let result = store.get(&ConversationId::from("nope")).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }
}
