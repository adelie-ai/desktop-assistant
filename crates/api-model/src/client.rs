//! Client-facing, digested conversation views + their projections from the wire
//! types.
//!
//! These are the small structs the shared client core (`client-ui-common`) and
//! every UI consume — distinct from the richer wire types in the crate root
//! (e.g. the root [`crate::ConversationSummary`] vs the digested
//! [`ConversationSummary`] here). They lived in `client-common` but moved here
//! (#377) so the wasm-targeting client cores can build/convert them without
//! `client-common`'s native transport tail. `client-common` re-exports them, so
//! existing `client_common::{ChatMessage, ConversationDetail, ConversationSummary}`
//! paths are unchanged.

use crate as api;

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
    pub archived: bool,
}

/// Presentation metadata for a [`ChatMessage`] — explicit so a UI never has to
/// parse `content` to know what a bubble is (voice#126). Daemon-sourced messages
/// are always `Normal`; clients tag the lines they generate locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MessageKind {
    /// An ordinary user / assistant / system / tool message.
    #[default]
    Normal,
    /// A line Adele spoke aloud via the `say_this` voice tool (on-demand mode).
    /// A real transcript entry, rendered with a "Spoken" marker.
    Spoken,
    /// A `say_this` the client did not speak because voice output is off — shown
    /// as an inline "(speech mode disabled)" note.
    SpeechDisabled,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Stable monotonic UUIDv7 id (#1) — the message's identity, ordering key,
    /// and the cursor a client uses to dedupe live vs snapshot, subscribe
    /// forward, and back-page. Empty only when talking to a pre-id daemon.
    pub id: String,
    pub role: String,
    pub content: String,
    /// Presentation metadata (voice#126); `Normal` for daemon-sourced messages.
    pub kind: MessageKind,
    /// The client-minted idempotency key stamped on a locally-drawn optimistic
    /// user bubble (#570), so the echoed-back `UserMessageAdded` carrying the
    /// same key can be deduped by exact match rather than a content compare.
    /// `None` for every daemon-sourced message (they carry a real `id`) and for
    /// keyless send paths.
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConversationDetail {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub model_selection: Option<api::ConversationModelSelectionView>,
    /// The conversation's stored personality override (#227), or `None` when it
    /// uses the global personality. A picker pre-fills its sliders from this.
    pub conversation_personality: Option<api::ConversationPersonalityView>,
}

impl From<api::ConversationSummary> for ConversationSummary {
    fn from(value: api::ConversationSummary) -> Self {
        Self {
            id: value.id,
            title: value.title,
            message_count: value.message_count,
            archived: value.archived,
        }
    }
}

impl From<api::MessageView> for ChatMessage {
    fn from(value: api::MessageView) -> Self {
        Self {
            id: value.id,
            role: value.role,
            content: value.content,
            // Daemon-sourced messages are always ordinary; clients tag the lines
            // they generate locally (voice#126).
            kind: MessageKind::Normal,
            // Surface the persisted idempotency key (#570 Phase 1b): a USER row
            // carries the client's key, so a transcript reload/reconnect dedups
            // an echoed `UserMessageAdded` by exact match rather than a
            // content compare. `None` for assistant/tool rows and keyless sends.
            idempotency_key: value.idempotency_key,
        }
    }
}

impl From<api::ConversationView> for ConversationDetail {
    fn from(value: api::ConversationView) -> Self {
        Self {
            id: value.id,
            title: value.title,
            messages: value.messages.into_iter().map(ChatMessage::from).collect(),
            model_selection: value.model_selection,
            conversation_personality: value.conversation_personality,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_kind_defaults_to_normal() {
        assert_eq!(MessageKind::default(), MessageKind::Normal);
    }

    #[test]
    fn daemon_messages_convert_as_normal_kind() {
        // A wire MessageView -> client ChatMessage is always Normal; only clients
        // tag Spoken / SpeechDisabled locally (voice#126).
        let m = ChatMessage::from(api::MessageView {
            id: "m1".into(),
            role: "assistant".into(),
            content: "hi".into(),
            idempotency_key: None,
        });
        assert_eq!(m.kind, MessageKind::Normal);
        assert_eq!(m.content, "hi");
        assert!(
            m.idempotency_key.is_none(),
            "a daemon-sourced message never carries a client idempotency stamp"
        );
    }

    /// #570 Phase 1b: a persisted idempotency key on the wire `MessageView` is
    /// carried onto the `ChatMessage` on reload, so a reconnecting client can
    /// dedupe an echoed `UserMessageAdded` by exact key match instead of a
    /// content compare (the Phase 1 limitation this slice removes).
    #[test]
    fn persisted_idempotency_key_surfaces_on_reload() {
        let m = ChatMessage::from(api::MessageView {
            id: "m1".into(),
            role: "user".into(),
            content: "hi".into(),
            idempotency_key: Some("k1".into()),
        });
        assert_eq!(
            m.idempotency_key.as_deref(),
            Some("k1"),
            "a persisted key must pass through From<MessageView> onto the ChatMessage"
        );
    }
}
