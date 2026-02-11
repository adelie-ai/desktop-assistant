use serde::{Deserialize, Serialize};

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// A single chat message within a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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
    }

    #[test]
    fn role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::Assistant, Role::System);
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
    }
}
