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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Message, Role};
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
