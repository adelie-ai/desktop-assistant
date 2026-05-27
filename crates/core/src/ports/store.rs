use crate::CoreError;
use crate::domain::{Conversation, ConversationId};
use serde::{Deserialize, Serialize};

/// Lifecycle status of a conversation turn persisted to the DB (#107).
///
/// The turn state machine drives transitions through these states so that
/// a turn suspended on a client-local tool call can be resumed when the
/// client posts the result back. Server-side tool dispatch transits
/// `pending_llm` → `pending_tool_dispatch` → `pending_llm` → `complete`
/// inside a single tokio task; client-side dispatch is the same path
/// except the turn parks at `pending_client_tool` and the row is the
/// only durable record while the client executes the tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    /// Daemon is calling the LLM (or about to).
    PendingLlm,
    /// LLM returned tool calls; daemon is dispatching server-side tools.
    PendingToolDispatch,
    /// Daemon emitted a `ClientToolCall` and is waiting for the client's
    /// `ClientToolResult`. Hot-path resumption uses an in-memory
    /// `oneshot::Sender`; the DB row exists for observability and so
    /// a crashed daemon's pending rows can be marked `failed` on
    /// restart instead of accumulating.
    PendingClientTool,
    /// Turn finished normally.
    Complete,
    /// Turn ended in an error (cancelled, LLM failure, daemon restart,
    /// invalid client tool result, etc.). The reason is recorded in
    /// `last_error`.
    Failed,
}

impl TurnStatus {
    /// Canonical lowercase key used in SQL and JSON. Mirrors the serde
    /// snake_case serialization so the DB column matches the wire shape.
    pub fn as_key(self) -> &'static str {
        match self {
            Self::PendingLlm => "pending_llm",
            Self::PendingToolDispatch => "pending_tool_dispatch",
            Self::PendingClientTool => "pending_client_tool",
            Self::Complete => "complete",
            Self::Failed => "failed",
        }
    }

    pub fn from_key(s: &str) -> Option<Self> {
        match s {
            "pending_llm" => Some(Self::PendingLlm),
            "pending_tool_dispatch" => Some(Self::PendingToolDispatch),
            "pending_client_tool" => Some(Self::PendingClientTool),
            "complete" => Some(Self::Complete),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }

    /// Returns `true` for states the daemon should not resume on restart.
    /// Pending states pile up if the daemon crashes mid-turn; the
    /// startup scan marks them `failed` so they don't shadow the user's
    /// next request.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Failed)
    }
}

/// A pending client-side tool call recorded in the turn's `state_json`.
///
/// Carries everything the client needs to execute the tool plus the
/// `tool_call_id` the LLM generated, which the client echoes back in
/// its `ClientToolResult` so the daemon can correlate the response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingClientToolCall {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// JSON payload stored in `turns.state_json`. The shape is internal to
/// the application + storage layers; the wire protocol's `Event::ClientToolCall`
/// carries only the subset the client needs.
///
/// Why a single JSON column rather than a normalized schema: the turn
/// state's contents (history, partial responses, retry counters, ...)
/// evolve as the in-memory state machine evolves. A schema migration on
/// every internal-state tweak is friction we don't need, and the
/// `(user_id, status)` index already covers every hot query path. The
/// JSON shape is versioned so we can grow it without losing pre-restart
/// rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnStateJson {
    /// Bumped on every breaking shape change. Readers default to v1
    /// when the column existed before this field was added (it didn't,
    /// but defaulting keeps downgrades clean).
    #[serde(default = "default_state_version")]
    pub version: u32,
    /// The currently outstanding client-side tool call, if any. `Some`
    /// iff `status == PendingClientTool`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_client_tool: Option<PendingClientToolCall>,
}

fn default_state_version() -> u32 {
    1
}

impl Default for TurnStateJson {
    fn default() -> Self {
        Self {
            version: 1,
            pending_client_tool: None,
        }
    }
}

/// A single turn row read from the DB. Mirrors the columns in the
/// `turns` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRow {
    pub id: String,
    pub user_id: String,
    pub conversation_id: String,
    pub status: TurnStatus,
    pub state: TurnStateJson,
    pub last_error: Option<String>,
}

/// Outbound port for persisting conversation turn state (#107).
///
/// Implementations are responsible for `(user_id, …)` scoping; callers
/// inside the per-request scope rely on `current_user_id()`. Cross-user
/// reads MUST behave like the row doesn't exist (don't leak presence).
///
/// The default no-op implementation makes the trait opt-in for tests and
/// legacy non-turn-state code paths: a service that doesn't care about
/// turn state never needs to construct a concrete store.
pub trait TurnStateStore: Send + Sync {
    /// Insert a new turn row. Implementations stamp `created_at` /
    /// `updated_at` themselves. Returns `Err` if a row with this id
    /// already exists under the same user_id — the caller chose a
    /// duplicate task_id, which is a programming error.
    fn create_turn(
        &self,
        row: TurnRow,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Read a turn row by id, scoped to the current user. Returns `Ok(None)`
    /// when the row doesn't exist OR exists under a different user_id —
    /// the same opacity rule as `PgConversationStore::get_conversation_model_selection`.
    fn get_turn(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<TurnRow>, CoreError>> + Send;

    /// Atomically update a turn row's status, state_json, and last_error.
    /// Implementations bump `updated_at`. Returns `Err` if the row does
    /// not exist for this user_id.
    fn update_turn(
        &self,
        id: &str,
        status: TurnStatus,
        state: &TurnStateJson,
        last_error: Option<&str>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Scan every turn row whose status is non-terminal across ALL users.
    /// Called once at daemon startup so abandoned rows can be marked
    /// `failed("daemon_restarted")` instead of accumulating. Skips the
    /// `current_user_id()` scope because the caller is a system task
    /// (no JWT context); implementations explicitly bypass scoping.
    fn scan_non_terminal(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<TurnRow>, CoreError>> + Send;
}

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

    // ---- #107: turn state machine port ------------------------------------

    #[test]
    fn turn_status_round_trips_via_key() {
        for status in [
            TurnStatus::PendingLlm,
            TurnStatus::PendingToolDispatch,
            TurnStatus::PendingClientTool,
            TurnStatus::Complete,
            TurnStatus::Failed,
        ] {
            let key = status.as_key();
            let back = TurnStatus::from_key(key).expect("known key parses");
            assert_eq!(status, back, "key {key} must round-trip");
        }
        assert_eq!(TurnStatus::from_key("nonsense"), None);
    }

    #[test]
    fn turn_status_is_terminal_matches_complete_or_failed() {
        assert!(TurnStatus::Complete.is_terminal());
        assert!(TurnStatus::Failed.is_terminal());
        assert!(!TurnStatus::PendingLlm.is_terminal());
        assert!(!TurnStatus::PendingToolDispatch.is_terminal());
        assert!(!TurnStatus::PendingClientTool.is_terminal());
    }

    #[test]
    fn turn_state_json_serializes_with_version_field() {
        let state = TurnStateJson {
            version: 1,
            pending_client_tool: Some(PendingClientToolCall {
                tool_call_id: "call-1".into(),
                tool_name: "fs_read".into(),
                arguments: serde_json::json!({"path": "/etc/hosts"}),
            }),
        };
        let v = serde_json::to_value(&state).unwrap();
        assert_eq!(v.get("version").unwrap(), 1);
        let back: TurnStateJson = serde_json::from_value(v).unwrap();
        assert_eq!(state, back);
    }

    #[test]
    fn turn_state_json_default_version_is_1_when_missing() {
        // Roll-forward safety: a row written before the `version` field
        // existed must still parse. The default factory produces 1.
        let state: TurnStateJson = serde_json::from_str("{}").unwrap();
        assert_eq!(state.version, 1);
        assert!(state.pending_client_tool.is_none());
    }

    /// In-memory `TurnStateStore` for trait-impl tests. Mirrors the
    /// `MockStore` pattern above — keyed by id, no user_id scoping
    /// (callers exercise scoping at higher layers).
    struct MockTurnStore {
        data: Mutex<HashMap<String, TurnRow>>,
    }

    impl MockTurnStore {
        fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl TurnStateStore for MockTurnStore {
        async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            if data.contains_key(&row.id) {
                return Err(CoreError::Storage(format!(
                    "turn id already exists: {}",
                    row.id
                )));
            }
            data.insert(row.id.clone(), row);
            Ok(())
        }

        async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
            Ok(self.data.lock().unwrap().get(id).cloned())
        }

        async fn update_turn(
            &self,
            id: &str,
            status: TurnStatus,
            state: &TurnStateJson,
            last_error: Option<&str>,
        ) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            let row = data
                .get_mut(id)
                .ok_or_else(|| CoreError::Storage(format!("turn not found: {id}")))?;
            row.status = status;
            row.state = state.clone();
            row.last_error = last_error.map(String::from);
            Ok(())
        }

        async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
            Ok(self
                .data
                .lock()
                .unwrap()
                .values()
                .filter(|r| !r.status.is_terminal())
                .cloned()
                .collect())
        }
    }

    #[tokio::test]
    async fn turn_state_store_create_get_update_round_trip() {
        let store = MockTurnStore::new();
        let row = TurnRow {
            id: "task-1".into(),
            user_id: "alice".into(),
            conversation_id: "conv-1".into(),
            status: TurnStatus::PendingLlm,
            state: TurnStateJson::default(),
            last_error: None,
        };
        store.create_turn(row.clone()).await.unwrap();

        let read = store.get_turn("task-1").await.unwrap().unwrap();
        assert_eq!(read.status, TurnStatus::PendingLlm);

        let updated_state = TurnStateJson {
            version: 1,
            pending_client_tool: Some(PendingClientToolCall {
                tool_call_id: "c1".into(),
                tool_name: "fs_read".into(),
                arguments: serde_json::json!({"path": "/tmp/x"}),
            }),
        };
        store
            .update_turn(
                "task-1",
                TurnStatus::PendingClientTool,
                &updated_state,
                None,
            )
            .await
            .unwrap();

        let read = store.get_turn("task-1").await.unwrap().unwrap();
        assert_eq!(read.status, TurnStatus::PendingClientTool);
        assert!(read.state.pending_client_tool.is_some());
    }

    #[tokio::test]
    async fn turn_state_store_scan_non_terminal_excludes_complete_and_failed() {
        let store = MockTurnStore::new();
        // Two finished rows, two pending — only the pending ones should
        // surface in the scan, since the startup hook uses this to
        // mark abandoned turns `failed` without churning completed ones.
        for (id, status) in [
            ("t-c", TurnStatus::Complete),
            ("t-f", TurnStatus::Failed),
            ("t-llm", TurnStatus::PendingLlm),
            ("t-ct", TurnStatus::PendingClientTool),
        ] {
            store
                .create_turn(TurnRow {
                    id: id.into(),
                    user_id: "u".into(),
                    conversation_id: "c".into(),
                    status,
                    state: TurnStateJson::default(),
                    last_error: None,
                })
                .await
                .unwrap();
        }

        let pending = store.scan_non_terminal().await.unwrap();
        let mut ids: Vec<_> = pending.iter().map(|r| r.id.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["t-ct".to_string(), "t-llm".to_string()]);
    }

    #[tokio::test]
    async fn turn_state_store_duplicate_create_fails() {
        // Duplicate task_ids signal a caller bug — the dispatcher should
        // generate unique ids per turn. Surfacing a clear error catches
        // this in tests rather than silently overwriting in production.
        let store = MockTurnStore::new();
        let row = TurnRow {
            id: "task-1".into(),
            user_id: "u".into(),
            conversation_id: "c".into(),
            status: TurnStatus::PendingLlm,
            state: TurnStateJson::default(),
            last_error: None,
        };
        store.create_turn(row.clone()).await.unwrap();
        let err = store.create_turn(row).await.unwrap_err();
        assert!(matches!(err, CoreError::Storage(_)));
    }

    #[tokio::test]
    async fn turn_state_store_get_unknown_returns_none() {
        // Cross-user / unknown reads return `Ok(None)` rather than an
        // error — the application layer interprets this as "no pending
        // turn" without needing a special error variant.
        let store = MockTurnStore::new();
        let read = store.get_turn("nope").await.unwrap();
        assert!(read.is_none());
    }
}
