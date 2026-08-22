use serde::{Deserialize, Serialize};

use super::Message;
use crate::CoreError;

/// Opaque identifier for a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ConversationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ConversationId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Conversation tag marking a subagent's private working conversation (#609).
///
/// A subagent runs its turns in a conversation of its own (for history and LLM
/// context), but that conversation is an implementation detail: it clutters the
/// user's conversation list and pollutes message search with the subagent's raw
/// working transcript. Tagging it with this reserved value lets `list` and
/// conversation search filter it out while `get` still resolves it directly.
/// The double-underscore namespacing keeps it clear of any user-authored tag.
pub const RESERVED_SUBAGENT_TAG: &str = "__subagent__";

/// The most serialized bytes a conversation title may take (#1303).
///
/// A title reaches storage two ways. A client writes one with
/// `CreateConversation` or `RenameConversation`, and the daemon generates one
/// for itself - `Standalone: <name>`, `Subagent: <name>`, and the name an LLM
/// writes after the first message of a conversation. Every title then rides in
/// every conversation view, every list row and the title-changed event, so one
/// unbounded title could push a response past the transport cap on its own and
/// make that conversation, or the whole conversation list, unreadable.
///
/// The bound is on SERIALIZED bytes, not raw bytes, because JSON escaping is
/// the difficulty: one control byte costs six bytes on the wire, so a title
/// well inside a raw cap can be far past it once escaped. The read path
/// measures the same unit against the same number, so a title accepted here is
/// never cut when it is read back.
///
/// # Why 4 KiB
///
/// The number is chosen for the person who types a title, not for the response
/// envelope. 4096 serialized bytes hold about 4090 Latin characters, about
/// 1360 characters of a three-byte script such as Japanese, or about 680
/// characters if every one of them is a control byte that escapes to six. That
/// is far more than a one-line label needs in any script, so no title a person
/// would type or paste is refused - which matters, because a refusal is
/// visible to that person and a truncation would be silent loss.
///
/// It stays small against the budget it must fit. A list answer carries
/// 3 MiB, so several hundred rows whose titles are ALL at this cap still fit
/// one response; an ordinary title is nearer 50 bytes, so a real list of tens
/// of thousands of rows fits. The response bound cuts the list if it ever does
/// not.
pub const MAX_TITLE_BYTES: usize = 4096;

/// Byte length of `title` as a JSON string, without building the JSON.
///
/// [`MAX_TITLE_BYTES`] is a bound on what the encoder emits, so this counts
/// exactly that and allocates nothing. The two quote bytes of the JSON string
/// are included, so a title of `MAX_TITLE_BYTES - 2` plain bytes is the
/// longest that fits.
pub fn title_serialized_len(title: &str) -> usize {
    struct ByteCounter(usize);

    impl std::io::Write for ByteCounter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut counter = ByteCounter(0);
    match serde_json::to_writer(&mut counter, title) {
        Ok(()) => counter.0,
        // A `&str` cannot fail to serialize. Reporting the largest possible
        // size keeps the failure from becoming a silent hole in the bound.
        Err(_) => usize::MAX,
    }
}

/// Refuse a title past [`MAX_TITLE_BYTES`] (#1303).
///
/// A refusal, not a truncation. Silently rewriting what a person typed is the
/// loss this bound exists to remove, and a title cut on the way IN cannot be
/// recovered, while a title refused on the way in costs one retry. Base rule
/// 8.2: this is an operational decline, not a failure - the request was
/// understood and a fixed rule refuses it, so repeating it cannot succeed.
///
/// The description names the size only. It never quotes the title, so a
/// refusal cannot echo what the caller wrote into a log line.
///
/// Used for a title a CLIENT supplies. A title the daemon composes for itself
/// goes through [`bound_generated_title`] instead, because refusing there
/// would fail an operation the user did not ask for.
pub fn check_title_bound(title: &str) -> Result<(), CoreError> {
    let bytes = title_serialized_len(title);
    if bytes <= MAX_TITLE_BYTES {
        return Ok(());
    }
    Err(CoreError::InvalidInput {
        code: "conversation_title_too_long",
        description: format!(
            "a conversation title is {bytes} serialized bytes; the limit is {MAX_TITLE_BYTES}"
        ),
        message: "That title is too long. Use a shorter one.".to_string(),
    })
}

/// The marker a cut generated title ends with, so a reader can see that the
/// name is shortened rather than oddly chosen.
const GENERATED_TITLE_CUT_MARK: &str = "...";

/// Cut a title the DAEMON generates for itself to [`MAX_TITLE_BYTES`] (#1303).
///
/// Bounded, not refused. The user did not supply this title and often did not
/// ask for the operation that produces it, so a refusal would fail their turn
/// over a name they never saw. Three titles come through here: the name an LLM
/// writes after the first message of a conversation, and the `Standalone: ` /
/// `Subagent: ` labels the daemon composes from a name a client or a model
/// supplies.
///
/// A cut is visible in the value: what is kept ends with
/// `GENERATED_TITLE_CUT_MARK`. Where a composed title is cut, the label leads
/// and the supplied name is what a cut removes, so the title still says what
/// the conversation is.
///
/// The cut lands on a `char` boundary, so the result is valid UTF-8.
pub fn bound_generated_title(title: &str) -> String {
    if title_serialized_len(title) <= MAX_TITLE_BYTES {
        return title.to_string();
    }
    // Both lengths count the two quote bytes of their own JSON string, and the
    // joined string has one pair, so this reserves two bytes more than the
    // mark needs. Erring high can only make the result smaller.
    let room = MAX_TITLE_BYTES.saturating_sub(title_serialized_len(GENERATED_TITLE_CUT_MARK));
    format!(
        "{}{GENERATED_TITLE_CUT_MARK}",
        head_title_to_bytes(title, room)
    )
}

/// The longest prefix of `text` whose serialized length fits `budget`, cut on
/// a `char` boundary (#1303).
///
/// Serialized, not raw: JSON escaping is the whole difficulty, so a raw length
/// is not a bound. Serialized length never shrinks as the prefix grows, so the
/// binary search is exact. Returns an empty string when not even one character
/// fits.
fn head_title_to_bytes(text: &str, budget: usize) -> &str {
    if title_serialized_len(text) <= budget {
        return text;
    }
    let mut low = 0usize;
    // Escaping only ever makes a prefix longer than its raw bytes, so the raw
    // length can never be under the budget once the serialized one is over it.
    let mut high = text.len().min(budget);
    while low < high {
        // Bias up so the loop always makes progress towards `high`.
        let mut mid = low + (high - low).div_ceil(2);
        while mid > low && !text.is_char_boundary(mid) {
            mid -= 1;
        }
        if mid == low {
            break;
        }
        if title_serialized_len(&text[..mid]) <= budget {
            low = mid;
        } else {
            high = mid - 1;
            while high > low && !text.is_char_boundary(high) {
                high -= 1;
            }
        }
    }
    &text[..low]
}

/// A collapsed range of messages replaced by a summary text.
///
/// Why: the range covered by a summary is recovered at render time from
/// the positions of `Message`s whose `summary_id` matches `id`. Storing
/// vec-index ordinals on the summary itself duplicates information already
/// carried by `Message::summary_id` and breaks if any message in the
/// conversation is deleted (the recorded indices would silently drift).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSummary {
    pub id: String,
    pub summary: String,
}

/// A conversation aggregate containing its messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: ConversationId,
    pub title: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    pub messages: Vec<Message>,
    /// Rolling summary of messages dropped by context windowing.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context_summary: String,
    /// Message index up to which compaction has been performed.
    #[serde(default)]
    pub compacted_through: usize,
    /// Collapsed message ranges with their summary text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summaries: Vec<MessageSummary>,
    /// When the conversation was archived (None = active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<String>,
    /// The user's most recent prompt, captured at the start of each
    /// `send_prompt`. Re-injected into the message stream as a
    /// `[Current task]` system message when the original user message
    /// has been windowed out, summarised, or buried under many tool
    /// rounds.
    ///
    /// Why: long agentic tool loops drift away from the goal when the
    /// rolling summary buries it; an explicit anchor keeps the model
    /// on-task across compaction and windowing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_task: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl Conversation {
    pub fn new(id: impl Into<ConversationId>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            created_at: String::new(),
            updated_at: String::new(),
            messages: Vec::new(),
            context_summary: String::new(),
            compacted_through: 0,
            summaries: Vec::new(),
            archived_at: None,
            active_task: None,
            tags: Vec::new(),
        }
    }
}

/// Lightweight summary for listing conversations.
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: ConversationId,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: usize,
    pub archived: bool,
    pub tags: Vec<String>,
}

impl From<&Conversation> for ConversationSummary {
    fn from(conv: &Conversation) -> Self {
        Self {
            id: conv.id.clone(),
            title: conv.title.clone(),
            created_at: conv.created_at.clone(),
            updated_at: conv.updated_at.clone(),
            message_count: conv.messages.len(),
            archived: conv.archived_at.is_some(),
            tags: conv.tags.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Message, Role};

    #[test]
    fn conversation_id_from_string() {
        let id = ConversationId::from("abc-123".to_string());
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn conversation_id_from_str() {
        let id = ConversationId::from("abc-123");
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn conversation_id_equality() {
        let a = ConversationId::from("same");
        let b = ConversationId::from("same");
        assert_eq!(a, b);
    }

    #[test]
    fn conversation_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ConversationId::from("a"));
        set.insert(ConversationId::from("a"));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn new_conversation_has_empty_messages() {
        let conv = Conversation::new("id-1", "Test Chat");
        assert_eq!(conv.id.as_str(), "id-1");
        assert_eq!(conv.title, "Test Chat");
        assert!(conv.created_at.is_empty());
        assert!(conv.updated_at.is_empty());
        assert!(conv.messages.is_empty());
    }

    #[test]
    fn conversation_summary_from_conversation() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.messages.push(Message::new(Role::User, "hi"));
        conv.messages.push(Message::new(Role::Assistant, "hello"));

        let summary = ConversationSummary::from(&conv);
        assert_eq!(summary.id.as_str(), "id-1");
        assert_eq!(summary.title, "Chat");
        assert_eq!(summary.created_at, "");
        assert_eq!(summary.updated_at, "");
        assert_eq!(summary.message_count, 2);
        assert!(!summary.archived);
    }

    #[test]
    fn conversation_serialization_roundtrip() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.messages.push(Message::new(Role::User, "test"));
        let json = serde_json::to_string(&conv).unwrap();
        let deserialized: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, conv.id);
        assert_eq!(deserialized.title, conv.title);
        assert_eq!(deserialized.created_at, conv.created_at);
        assert_eq!(deserialized.updated_at, conv.updated_at);
        assert_eq!(deserialized.messages.len(), 1);
    }

    #[test]
    fn conversation_deserializes_without_timestamps() {
        let json = r#"{"id":"id-1","title":"Chat","messages":[]}"#;
        let conv: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conv.created_at, "");
        assert_eq!(conv.updated_at, "");
    }

    #[test]
    fn conversation_deserializes_without_compaction_fields() {
        let json = r#"{"id":"id-1","title":"Chat","messages":[]}"#;
        let conv: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conv.context_summary, "");
        assert_eq!(conv.compacted_through, 0);
    }

    #[test]
    fn conversation_serialization_roundtrip_with_compaction() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.context_summary = "User asked about Rust lifetimes.".to_string();
        conv.compacted_through = 25;
        conv.messages.push(Message::new(Role::User, "test"));

        let json = serde_json::to_string(&conv).unwrap();
        let deserialized: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.context_summary,
            "User asked about Rust lifetimes."
        );
        assert_eq!(deserialized.compacted_through, 25);
    }

    #[test]
    fn conversation_skips_empty_context_summary_in_serialization() {
        let conv = Conversation::new("id-1", "Chat");
        let json = serde_json::to_string(&conv).unwrap();
        assert!(!json.contains("context_summary"));
    }

    #[test]
    fn message_summary_tolerates_legacy_ordinal_fields() {
        // Persisted JSON from before the ordinal fields were dropped must
        // still deserialize. Serde tolerates unknown keys by default
        // (no #[serde(deny_unknown_fields)] on the struct).
        let json = r#"{
            "id": "s1",
            "summary": "First batch.",
            "start_ordinal": 1,
            "end_ordinal": 3
        }"#;
        let summary: MessageSummary = serde_json::from_str(json).unwrap();
        assert_eq!(summary.id, "s1");
        assert_eq!(summary.summary, "First batch.");
    }

    #[test]
    fn new_conversation_has_no_active_task() {
        let conv = Conversation::new("id-1", "Test Chat");
        assert!(conv.active_task.is_none());
    }

    #[test]
    fn conversation_deserializes_without_active_task() {
        // Backwards compatibility: persisted state from before the active_task
        // column existed must continue to deserialize cleanly.
        let json = r#"{"id":"id-1","title":"Chat","messages":[]}"#;
        let conv: Conversation = serde_json::from_str(json).expect("deserialize legacy json");
        assert!(conv.active_task.is_none());
    }

    #[test]
    fn conversation_serialization_roundtrip_with_active_task() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.active_task = Some("Refactor the authentication module".to_string());
        let json = serde_json::to_string(&conv).expect("serialize");
        let deserialized: Conversation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            deserialized.active_task.as_deref(),
            Some("Refactor the authentication module")
        );
    }

    #[test]
    fn conversation_skips_none_active_task_in_serialization() {
        let conv = Conversation::new("id-1", "Chat");
        let json = serde_json::to_string(&conv).expect("serialize");
        assert!(!json.contains("active_task"));
    }
}
