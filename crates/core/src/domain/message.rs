use serde::{Deserialize, Serialize};
use uuid::{ContextV7, Timestamp, Uuid};

use super::tool::ToolCall;

thread_local! {
    /// Per-thread monotonic UUIDv7 counter context for message ids (#1 live
    /// multi-client sync). The counter makes ids minted in the same millisecond
    /// on a thread order deterministically, so a v7 id can serve as a message's
    /// identity AND a sortable high-water cursor: `max(id)` is the latest
    /// message, `id > since` is an exact "everything after" filter.
    ///
    /// Thread-local (not a global `Mutex`) because `Message::new` is hot — a
    /// shared lock per construction would contend. Within a single conversation
    /// appends are serialized by the turn lock and paced seconds apart by the
    /// LLM/human, so the millisecond timestamp alone separates them across
    /// threads; the counter only needs to disambiguate same-thread same-ms
    /// bursts, which it does. `ContextV7` is `!Sync` anyway, so it cannot be a
    /// `static`.
    static MESSAGE_ID_CTX: ContextV7 = const { ContextV7::new() };
}

/// Mint a fresh monotonic UUIDv7 message id.
pub fn new_message_id() -> String {
    MESSAGE_ID_CTX.with(|ctx| Uuid::new_v7(Timestamp::now(ctx)).to_string())
}

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// A single chat message within a conversation.
///
/// `id` is a monotonic UUIDv7 ([`new_message_id`]) assigned at creation — the
/// message's stable identity, ordering key, and resume cursor for live
/// multi-client sync. It is deliberately NOT part of `PartialEq`/`Eq`: equality
/// compares message *content* (role/content/tool calls), so existing
/// value-comparisons and the storage structural diff are unaffected by ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Stable monotonic UUIDv7 identity, assigned at creation and preserved
    /// across load/clone. `serde(default)` mints one so messages persisted
    /// before ids were carried still get a stable id when deserialized.
    #[serde(default = "new_message_id")]
    pub id: String,
    pub role: Role,
    pub content: String,
    /// Tool calls requested by the assistant (only set for Role::Assistant).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The tool call ID this message is a response to (only set for Role::Tool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// If set, this message is collapsed behind a `MessageSummary` with this ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_id: Option<String>,
    /// The client-supplied idempotency key for this message (#570 Phase 1b).
    ///
    /// Carried on USER rows only — the message that initiated a
    /// client-retryable send — so a transcript reload or reconnect returns the
    /// key and clients dedup an echoed `UserMessageAdded` by exact match rather
    /// than a content compare. `None` for assistant/tool rows and for keyless
    /// sends. Stamped at the single user-message persist site in `send_prompt`
    /// from the [`crate::ports::llm::current_idempotency_key`] task-local.
    ///
    /// Deliberately excluded from [`PartialEq`]/[`Eq`] (like `id`): it is
    /// carried-through metadata, not content, so the storage structural diff
    /// (`ExistingMsgRow::matches`) stays unaffected and a re-diff on load causes
    /// no update churn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            summary_id: None,
            idempotency_key: None,
        }
    }

    /// Create an assistant message that requests tool calls.
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: new_message_id(),
            role: Role::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
            summary_id: None,
            idempotency_key: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            summary_id: None,
            idempotency_key: None,
        }
    }
}

/// Equality compares message *content*, deliberately excluding `id` AND
/// `idempotency_key`: a fresh monotonic id is minted on every construction, and
/// the idempotency key is carried-through metadata (not content), so two
/// `Message::new` calls with the same content must still compare equal for the
/// storage structural diff (`ExistingMsgRow::matches`) and the many
/// value-comparison tests. Excluding the key also keeps a re-diff on load equal,
/// so surfacing the persisted key causes no update churn.
impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role
            && self.content == other.content
            && self.tool_calls == other.tool_calls
            && self.tool_call_id == other.tool_call_id
            && self.summary_id == other.summary_id
    }
}

impl Eq for Message {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_creation() {
        let msg = Message::new(Role::User, "hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
        assert!(msg.tool_calls.is_empty());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::Assistant, Role::System);
        assert_ne!(Role::System, Role::Tool);
    }

    #[test]
    fn message_clone() {
        let msg = Message::new(Role::Assistant, "response");
        let cloned = msg.clone();
        assert_eq!(msg, cloned);
    }

    /// Equality deliberately excludes `id`: two *independently constructed*
    /// messages with identical content must compare equal even though each
    /// minted its own fresh id. The storage structural diff
    /// (`ExistingMsgRow::matches`) and value-comparison tests depend on this.
    #[test]
    fn message_equality_ignores_id() {
        let a = Message::new(Role::User, "hello");
        let b = Message::new(Role::User, "hello");
        assert_ne!(a.id, b.id, "each construction must mint a distinct id");
        assert_eq!(
            a, b,
            "same content must compare equal despite differing ids"
        );

        // A content difference must still break equality (equality isn't a
        // constant-true).
        let c = Message::new(Role::User, "goodbye");
        assert_ne!(a, c);
    }

    /// `new_message_id` mints monotonically increasing UUIDv7 strings on a
    /// thread, so `max(id)` is the latest message and `id > since` is an exact
    /// "everything after" cursor. Lexicographic string order matches mint order.
    #[test]
    fn message_ids_are_monotonic() {
        let id_a = new_message_id();
        let id_b = new_message_id();
        assert!(id_b > id_a, "ids must increase: {id_a} !< {id_b}");

        // The ids carried by successive Message constructions are ordered too.
        let m1 = Message::new(Role::User, "one");
        let m2 = Message::new(Role::Assistant, "two");
        assert!(
            m2.id > m1.id,
            "message ids must increase: {} !< {}",
            m1.id,
            m2.id
        );
    }

    #[test]
    fn role_serialization_roundtrip() {
        let role = Role::User;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"user\"");
        let deserialized: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, role);
    }

    #[test]
    fn message_serialization_roundtrip() {
        let msg = Message::new(Role::Assistant, "hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, msg);
    }

    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn assistant_with_tool_calls() {
        let calls = vec![
            ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/a.txt"}"#),
            ToolCall::new("call-2", "read_file", r#"{"path": "/tmp/b.txt"}"#),
        ];
        let msg = Message::assistant_with_tool_calls(calls.clone());
        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.content.is_empty());
        assert_eq!(msg.tool_calls, calls);
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn tool_result_message() {
        let msg = Message::tool_result("call-1", "file contents");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.content, "file contents");
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn message_without_tools_omits_fields_in_json() {
        let msg = Message::new(Role::User, "hi");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));
    }

    #[test]
    fn message_with_tools_serialization_roundtrip() {
        let calls = vec![ToolCall::new("c1", "test_tool", "{}")];
        let msg = Message::assistant_with_tool_calls(calls);
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, msg);
    }

    #[test]
    fn tool_result_serialization_roundtrip() {
        let msg = Message::tool_result("c1", "result data");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, msg);
    }
}
