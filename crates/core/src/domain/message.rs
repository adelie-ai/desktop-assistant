use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
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
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            summary_id: None,
        }
    }

    /// Create an assistant message that requests tool calls.
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
            summary_id: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            summary_id: None,
        }
    }
}

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
