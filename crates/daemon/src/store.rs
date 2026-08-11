use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::Local;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
use desktop_assistant_core::ports::store::ConversationStore;

fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn should_persist(conversation: &Conversation) -> bool {
    !conversation.messages.is_empty()
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
                    if !should_persist(&conversation) {
                        needs_migration = true;
                        continue;
                    }

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

        let conversations: Vec<Conversation> = data
            .values()
            .filter(|conversation| should_persist(conversation))
            .cloned()
            .collect();
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

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(self
            .data
            .lock()
            .unwrap()
            .values()
            .map(ConversationSummary::from)
            .collect())
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

    async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        let conv = data
            .get_mut(&id.0)
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
        if conv.archived_at.is_none() {
            conv.archived_at = Some(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string());
        }
        self.persist(&data)
    }

    async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        let conv = data
            .get_mut(&id.0)
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
        conv.archived_at = None;
        self.persist(&data)
    }

    async fn create_summary(
        &self,
        conversation_id: &ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        use desktop_assistant_core::domain::MessageSummary;
        let mut data = self.data.lock().unwrap();
        let conv = data
            .get_mut(&conversation_id.0)
            .ok_or_else(|| CoreError::ConversationNotFound(conversation_id.0.clone()))?;
        let id = uuid::Uuid::now_v7().to_string();
        for (i, msg) in conv.messages.iter_mut().enumerate() {
            if i >= start_ordinal && i <= end_ordinal {
                msg.summary_id = Some(id.clone());
            }
        }
        conv.summaries.push(MessageSummary {
            id: id.clone(),
            summary,
        });
        self.persist(&data)?;
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
                return self.persist(&data);
            }
        }
        Ok(())
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

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(self
            .data
            .lock()
            .unwrap()
            .values()
            .map(ConversationSummary::from)
            .collect())
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
        if conv.archived_at.is_none() {
            conv.archived_at = Some(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string());
        }
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
        use desktop_assistant_core::domain::MessageSummary;
        let mut data = self.data.lock().unwrap();
        let conv = data
            .get_mut(&conversation_id.0)
            .ok_or_else(|| CoreError::ConversationNotFound(conversation_id.0.clone()))?;
        let id = uuid::Uuid::now_v7().to_string();
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
    async fn persistent_store_does_not_persist_empty_conversation() {
        let path = temp_store_path();

        {
            let store = PersistentConversationStore::new(path.clone()).unwrap();
            store
                .create(Conversation::new("empty-1", "Empty Chat"))
                .await
                .unwrap();
        }

        let reopened = PersistentConversationStore::new(path.clone()).unwrap();
        let all = reopened.list().await.unwrap();
        assert!(all.is_empty());

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn persistent_store_load_drops_empty_conversations() {
        let path = temp_store_path();

        let legacy = serde_json::json!([
            {
                "id": "legacy-empty",
                "title": "Legacy Empty",
                "messages": []
            },
            {
                "id": "legacy-non-empty",
                "title": "Legacy Non Empty",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello"
                    }
                ]
            }
        ]);
        fs::write(&path, serde_json::to_string(&legacy).unwrap()).unwrap();

        let store = PersistentConversationStore::new(path.clone()).unwrap();
        let all = store.list().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id.as_str(), "legacy-non-empty");

        let reopened = PersistentConversationStore::new(path.clone()).unwrap();
        let all_reopened = reopened.list().await.unwrap();
        assert_eq!(all_reopened.len(), 1);
        assert_eq!(all_reopened[0].id.as_str(), "legacy-non-empty");

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn persistent_store_migrates_missing_timestamps() {
        let path = temp_store_path();

        let legacy = serde_json::json!([
            {
                "id": "legacy-1",
                "title": "Legacy Chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello"
                    }
                ]
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
        let migrated_again = reopened
            .get(&ConversationId::from("legacy-1"))
            .await
            .unwrap();
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

/// Acceptance suite for the remote-door tenancy guard (#773).
///
/// The daemon runs on one of two conversation stores. One of them keeps a
/// user's conversations apart from every other user's, and one of them does
/// not. The remote WebSocket door admits more than one authenticated person, so
/// the two facts have to be checked together, at startup, before anybody
/// connects.
#[cfg(test)]
mod remote_door_store_tenancy {
    use super::{ConversationStoreKind, check_remote_door_store_tenancy};

    /// A daemon selects PostgreSQL when a database URL is configured, and the
    /// JSON file when none is. This is the fact the guard reasons about, so it
    /// is pinned on its own.
    #[test]
    fn a_configured_database_selects_postgres_and_no_database_selects_the_json_file() {
        assert_eq!(
            ConversationStoreKind::for_database_configured(true),
            ConversationStoreKind::Postgres
        );
        assert_eq!(
            ConversationStoreKind::for_database_configured(false),
            ConversationStoreKind::JsonFile
        );
    }

    /// The JSON file holds no owner column and has no partition, so it cannot
    /// keep one user's conversations from another's. PostgreSQL can.
    #[test]
    fn only_the_database_store_is_user_scoped() {
        assert!(ConversationStoreKind::Postgres.is_user_scoped());
        assert!(!ConversationStoreKind::JsonFile.is_user_scoped());
    }

    /// #773 acceptance: a daemon configured for remote access on a store with
    /// no per-user partition refuses to start. Two authenticated people would
    /// otherwise share one unpartitioned file, and each would read the other's
    /// conversations.
    #[test]
    fn a_remote_door_on_a_store_with_no_user_partition_is_refused() {
        let result = check_remote_door_store_tenancy(true, ConversationStoreKind::JsonFile);
        assert!(
            result.is_err(),
            "the remote door on an unpartitioned store must be refused, not served"
        );
    }

    /// #773 acceptance: the single-user local install is the common case, it is
    /// safe, and it must not gain a new error. The remote door is off, so
    /// nothing about the store's tenancy can leak between users.
    #[test]
    fn a_local_only_daemon_on_the_json_file_store_still_starts() {
        assert!(
            check_remote_door_store_tenancy(false, ConversationStoreKind::JsonFile).is_ok(),
            "a local-only daemon on the JSON store is the common desktop case and must start"
        );
    }

    /// The remote door on a store that does partition by user is exactly what
    /// the deployed configuration does, and it must still start.
    #[test]
    fn a_remote_door_on_a_user_scoped_store_starts() {
        assert!(check_remote_door_store_tenancy(true, ConversationStoreKind::Postgres).is_ok());
        assert!(check_remote_door_store_tenancy(false, ConversationStoreKind::Postgres).is_ok());
    }

    /// The refusal has to be actionable: an operator reading it must learn what
    /// is wrong and both ways to fix it, without reading the source.
    #[test]
    fn the_refusal_names_the_cause_and_both_ways_to_fix_it() {
        let message = check_remote_door_store_tenancy(true, ConversationStoreKind::JsonFile)
            .expect_err("the configuration is refused")
            .to_string();
        assert!(
            message.contains("DESKTOP_ASSISTANT_DATABASE_URL"),
            "the refusal must name the database setting: {message}"
        );
        assert!(
            message.contains("ws_enabled"),
            "the refusal must name the transport switch: {message}"
        );
        assert!(
            message.contains("conversation"),
            "the refusal must say what is shared: {message}"
        );
    }
}
