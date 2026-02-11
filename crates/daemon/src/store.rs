use std::collections::HashMap;
use std::sync::Mutex;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId};
use desktop_assistant_core::ports::store::ConversationStore;

/// In-memory conversation store backed by a `Mutex<HashMap>`.
/// Suitable for development and testing; swap for a persistent backend later.
pub struct InMemoryConversationStore {
    data: Mutex<HashMap<String, Conversation>>,
}

impl InMemoryConversationStore {
    pub fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
        }
    }
}

impl ConversationStore for InMemoryConversationStore {
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

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::{Message, Role};

    #[tokio::test]
    async fn create_and_get() {
        let store = InMemoryConversationStore::new();
        let conv = Conversation::new("c1", "Test");
        store.create(conv).await.unwrap();

        let retrieved = store.get(&ConversationId::from("c1")).await.unwrap();
        assert_eq!(retrieved.title, "Test");
        assert!(retrieved.messages.is_empty());
    }

    #[tokio::test]
    async fn list_returns_all() {
        let store = InMemoryConversationStore::new();
        store.create(Conversation::new("c1", "A")).await.unwrap();
        store.create(Conversation::new("c2", "B")).await.unwrap();

        let all = store.list().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn delete_removes() {
        let store = InMemoryConversationStore::new();
        store.create(Conversation::new("c1", "A")).await.unwrap();
        store.delete(&ConversationId::from("c1")).await.unwrap();

        let result = store.get(&ConversationId::from("c1")).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn update_persists_changes() {
        let store = InMemoryConversationStore::new();
        let mut conv = Conversation::new("c1", "Original");
        store.create(conv.clone()).await.unwrap();

        conv.messages.push(Message::new(Role::User, "hello"));
        store.update(conv).await.unwrap();

        let retrieved = store.get(&ConversationId::from("c1")).await.unwrap();
        assert_eq!(retrieved.messages.len(), 1);
        assert_eq!(retrieved.messages[0].content, "hello");
    }

    #[tokio::test]
    async fn get_nonexistent_fails() {
        let store = InMemoryConversationStore::new();
        let result = store.get(&ConversationId::from("nope")).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn update_nonexistent_fails() {
        let store = InMemoryConversationStore::new();
        let conv = Conversation::new("nope", "X");
        let result = store.update(conv).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn delete_nonexistent_fails() {
        let store = InMemoryConversationStore::new();
        let result = store.delete(&ConversationId::from("nope")).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }
}
