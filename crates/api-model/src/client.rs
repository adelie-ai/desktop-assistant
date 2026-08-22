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
    /// When the message was created, as Unix epoch milliseconds — recovered
    /// from `id`, not stored (see [`uuidv7_millis`]). `None` when the id is not
    /// a UUIDv7 (a pre-id daemon's row), so a client shows no time rather than
    /// a fabricated epoch.
    ///
    /// Epoch millis rather than a formatted string because this crate is
    /// deliberately dependency-minimal to stay wasm-clean (no date library), and
    /// because every client must localise for display anyway — handing over a
    /// number avoids a parse-then-reformat round trip and leaves no room for two
    /// clients to disagree about timezone rendering.
    pub created_at_ms: Option<u64>,
}

/// Recover a UUIDv7's embedded creation time as Unix epoch milliseconds.
///
/// A UUIDv7's first 48 bits *are* the creation timestamp, so a message's time is
/// already carried by its id — no timestamp column, no backfill, and it works on
/// every row ever stored. Message ids have been UUIDv7 since the daemon's
/// migration 005.
///
/// Returns `None` for anything that is not a UUIDv7 — an empty id from a pre-id
/// daemon, a v4 id, or a malformed string — so callers degrade to "unknown time"
/// instead of inventing one. Hand-rolled rather than pulling in the `uuid` crate
/// because this crate stays dependency-minimal for the wasm client (#377).
pub fn uuidv7_millis(id: &str) -> Option<u64> {
    let b = id.as_bytes();
    // 8-4-4-4-12 with dashes; the version nibble sits at index 14.
    if b.len() != 36 || b[8] != b'-' || b[13] != b'-' || b[14] != b'7' {
        return None;
    }
    let mut millis: u64 = 0;
    // The timestamp is the first 48 bits: 12 hex digits spanning [0..8) and
    // [9..13), i.e. either side of the first dash.
    for &c in b[0..8].iter().chain(&b[9..13]) {
        let digit = (c as char).to_digit(16)?;
        millis = (millis << 4) | u64::from(digit);
    }
    Some(millis)
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
    /// The conversation's stored tool-provenance-gate override (#1007).
    /// `true` means the gate is disabled for every turn in this
    /// conversation; `false` (the default) means it stays enforced.
    pub tool_gate_disabled: bool,
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
            created_at_ms: uuidv7_millis(&value.id),
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
            // Derived at the one projection every client shares, rather than in
            // each client: four independent UUIDv7 decoders would be four
            // chances to disagree about what time a message happened.
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
            tool_gate_disabled: value.tool_gate_disabled,
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
            content_total_bytes: None,
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
            content_total_bytes: None,
        });
        assert_eq!(
            m.idempotency_key.as_deref(),
            Some("k1"),
            "a persisted key must pass through From<MessageView> onto the ChatMessage"
        );
    }
}

#[cfg(test)]
mod created_at_tests {
    use super::*;

    #[test]
    fn chat_message_carries_created_at_from_uuidv7() {
        // A UUIDv7 whose first 48 bits are 0x0193C7B2A400 ms since the epoch.
        // The time is recovered from the id, so no column and no backfill.
        let id = "0193c7b2-a400-7abc-8def-0123456789ab";
        assert_eq!(uuidv7_millis(id), Some(0x0193_C7B2_A400));
    }

    #[test]
    fn created_at_absent_for_non_uuidv7_id() {
        // Each must yield None, never a fabricated epoch: an empty id from a
        // pre-id daemon, a v4 id, a malformed string, and a non-hex body.
        for id in [
            "",
            "not-a-uuid",
            "0193c7b2-a400-4abc-8def-0123456789ab", // v4, not v7
            "0193c7b2-a400-7abc-8def-0123456789",   // too short
            "0193c7b2xa400-7abc-8def-0123456789ab", // dash in the wrong place
            "zzzzzzzz-a400-7abc-8def-0123456789ab", // non-hex timestamp
        ] {
            assert_eq!(uuidv7_millis(id), None, "must not invent a time for {id:?}");
        }
    }

    #[test]
    fn created_at_is_monotonic_with_uuidv7_ordering() {
        // UUIDv7 ids sort by time, so the recovered times must agree with that
        // ordering — otherwise a transcript sorted by id would show times out
        // of order.
        let earlier = uuidv7_millis("0193c7b2-a400-7abc-8def-0123456789ab").unwrap();
        let later = uuidv7_millis("0193c7b2-a401-7abc-8def-0123456789ab").unwrap();
        assert!(later > earlier);
    }

    #[test]
    fn created_at_is_stable_across_reprojection() {
        // Derived from the id, so the same message re-fetched can never drift.
        let view = api::MessageView {
            id: "0193c7b2-a400-7abc-8def-0123456789ab".to_string(),
            role: "user".to_string(),
            content: "hi".to_string(),
            idempotency_key: None,
            content_total_bytes: None,
        };
        let first = ChatMessage::from(view.clone()).created_at_ms;
        let second = ChatMessage::from(view).created_at_ms;
        assert_eq!(first, second);
        assert!(first.is_some());
    }
}
