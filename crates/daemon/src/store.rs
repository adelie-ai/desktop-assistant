use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::Local;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId};
use desktop_assistant_core::ports::store::ConversationStore;

fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Default persistent conversation file path following XDG base directories.
///
/// - `$XDG_DATA_HOME/desktop-assistant/conversations.json`, or
/// - `$HOME/.local/share/desktop-assistant/conversations.json` when
///   `XDG_DATA_HOME` is not set.
pub fn default_conversation_store_path() -> PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.local/share")
    });

    PathBuf::from(data_home)
        .join("desktop-assistant")
        .join("conversations.json")
}

/// Persistent conversation store backed by a JSON file.
pub struct PersistentConversationStore {
    data: Mutex<HashMap<String, Conversation>>,
    path: PathBuf,
}

impl PersistentConversationStore {
    pub fn new(path: PathBuf) -> Result<Self, CoreError> {
        let mut data = HashMap::new();
        let mut needs_migration = false;

        if path.exists() {
            let content = fs::read_to_string(&path).map_err(|e| {
                CoreError::Storage(format!("failed reading store file {}: {e}", path.display()))
            })?;

            if !content.trim().is_empty() {
                let conversations: Vec<Conversation> =
                    serde_json::from_str(&content).map_err(|e| {
                        CoreError::Storage(format!(
                            "failed parsing store file {}: {e}",
                            path.display()
                        ))
                    })?;

                for mut conversation in conversations {
                    let created = conversation.created_at.trim().to_string();
                    let updated = conversation.updated_at.trim().to_string();
                    if created.is_empty() && updated.is_empty() {
                        let timestamp = now_timestamp();
                        conversation.created_at = timestamp.clone();
                        conversation.updated_at = timestamp;
                        needs_migration = true;
                    } else if created.is_empty() {
                        conversation.created_at = updated;
                        needs_migration = true;
                    } else if updated.is_empty() {
                        conversation.updated_at = created;
                        needs_migration = true;
                    }

                    data.insert(conversation.id.0.clone(), conversation);
                }
            }
        }

        let store = Self {
            data: Mutex::new(data),
            path,
        };

        if needs_migration {
            let data = store.data.lock().unwrap();
            store.persist(&data)?;
        }

        Ok(store)
    }

    pub fn from_default_path() -> Result<Self, CoreError> {
        Self::new(default_conversation_store_path())
    }

    fn persist(&self, data: &HashMap<String, Conversation>) -> Result<(), CoreError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                CoreError::Storage(format!(
                    "failed creating store directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        let conversations: Vec<Conversation> = data.values().cloned().collect();
        let serialized = serde_json::to_string_pretty(&conversations)
            .map_err(|e| CoreError::Storage(format!("failed serializing conversations: {e}")))?;

        let tmp_path = self.path.with_extension("json.tmp");
        fs::write(&tmp_path, serialized).map_err(|e| {
            CoreError::Storage(format!(
                "failed writing temporary store file {}: {e}",
                tmp_path.display()
            ))
        })?;
        fs::rename(&tmp_path, &self.path).map_err(|e| {
            CoreError::Storage(format!(
                "failed replacing store file {}: {e}",
                self.path.display()
            ))
        })?;

        Ok(())
    }
}

impl ConversationStore for PersistentConversationStore {
    async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        data.insert(conv.id.0.clone(), conv);
        self.persist(&data)
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
            self.persist(&data)
        } else {
            Err(CoreError::ConversationNotFound(conv.id.0.clone()))
        }
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        if data.remove(&id.0).is_some() {
            self.persist(&data)
        } else {
            Err(CoreError::ConversationNotFound(id.0.clone()))
        }
    }
}

/// In-memory conversation store backed by a `Mutex<HashMap>`.
/// Suitable for development and testing; swap for a persistent backend later.
#[cfg_attr(not(test), allow(dead_code))]
pub struct InMemoryConversationStore {
    data: Mutex<HashMap<String, Conversation>>,
}

impl InMemoryConversationStore {
    #[cfg_attr(not(test), allow(dead_code))]
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

    fn temp_store_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "desktop-assistant-store-test-{}.json",
            uuid::Uuid::new_v4()
        ))
    }

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

    #[tokio::test]
    async fn persistent_store_survives_restart() {
        let path = temp_store_path();

        let mut conversation = Conversation::new("c1", "Persisted");
        conversation
            .messages
            .push(Message::new(Role::User, "hello there"));

        {
            let store = PersistentConversationStore::new(path.clone()).unwrap();
            store.create(conversation).await.unwrap();
        }

        let reopened = PersistentConversationStore::new(path.clone()).unwrap();
        let loaded = reopened.get(&ConversationId::from("c1")).await.unwrap();
        assert_eq!(loaded.title, "Persisted");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "hello there");

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn persistent_store_delete_persists() {
        let path = temp_store_path();

        {
            let store = PersistentConversationStore::new(path.clone()).unwrap();
            store
                .create(Conversation::new("c1", "Will be deleted"))
                .await
                .unwrap();
            store.delete(&ConversationId::from("c1")).await.unwrap();
        }

        let reopened = PersistentConversationStore::new(path.clone()).unwrap();
        let missing = reopened.get(&ConversationId::from("c1")).await;
        assert!(matches!(missing, Err(CoreError::ConversationNotFound(_))));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn persistent_store_migrates_missing_timestamps() {
        let path = temp_store_path();

        let legacy = serde_json::json!([
            {
                "id": "legacy-1",
                "title": "Legacy Chat",
                "messages": []
            }
        ]);
        fs::write(&path, serde_json::to_string(&legacy).unwrap()).unwrap();

        let store = PersistentConversationStore::new(path.clone()).unwrap();
        let migrated = store.get(&ConversationId::from("legacy-1")).await.unwrap();

        assert!(!migrated.created_at.is_empty());
        assert!(!migrated.updated_at.is_empty());
        assert_eq!(migrated.created_at.len(), 19);
        assert_eq!(migrated.updated_at.len(), 19);

        let reopened = PersistentConversationStore::new(path.clone()).unwrap();
        let migrated_again = reopened.get(&ConversationId::from("legacy-1")).await.unwrap();
        assert!(!migrated_again.created_at.is_empty());
        assert!(!migrated_again.updated_at.is_empty());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn default_store_path_uses_desktop_assistant_data_dir() {
        let path = default_conversation_store_path();
        let path_str = path.to_string_lossy();
        assert!(path_str.contains("desktop-assistant"));
        assert!(path_str.ends_with("conversations.json"));
    }
}
