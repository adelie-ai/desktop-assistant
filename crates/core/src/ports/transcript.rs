//! Turn-scoped read-back of the conversation transcript (#1226).
//!
//! ## The problem
//!
//! A sizeable tool result leaves the model's working view by two paths, and
//! both keep every byte in the conversation's transcript. A completed step
//! distils the result into a scratchpad note and leaves a pointer
//! ([`crate::planning`]); overflow recovery replaces the oldest tool groups
//! with a notice ([`crate::context`]). Neither path used to leave the model a
//! way back to the bytes, so re-running the tool was the only advertised
//! route - and that is the wrong instruction when the tool has side effects,
//! when its answer varies with time, or when the first call was expensive.
//!
//! This module is the way back. [`TRANSCRIPT_GET_TOOL`] returns the stored
//! content of one message, addressed by [`crate::domain::Message::id`], which
//! is already a stable monotonic UUIDv7 preserved across load and clone.
//!
//! ## Why a task-local
//!
//! The read has to see the turn's own messages, not only what storage holds:
//! a result read on round 2 and evicted on round 5 is fetched back on round 9,
//! and the turn does not persist its rows until it ends. The turn loop owns
//! `Conversation::messages`; the tool runs deep inside the MCP executor, which
//! has no conversation parameter. So the loop installs a [`TranscriptView`]
//! around each server-side tool execution, exactly as it installs
//! [`crate::ports::conversation_ctx`]'s `ConversationId`. See AGENTS.md
//! ("cross-cutting context propagates via `tokio::task_local!`").
//!
//! The view is built once per turn and grows as the turn appends messages.
//! The first [`TranscriptView::absorb`] of a turn copies the content of every
//! message the conversation already holds, because that whole list is what the
//! turn loaded; each later one copies only what the round appended. Installing
//! the view around a tool costs nothing beyond that: the messages sit behind an
//! `Arc`, so cloning a view copies the two scope values and a pointer.
//!
//! ## Scope, and how it fails
//!
//! Every read is scoped to the calling user AND to the active conversation,
//! and fails closed. Three things hold that:
//!
//! - The view only ever holds one conversation's messages, loaded by the turn
//!   under the calling user's id, so no other conversation's message is
//!   reachable to begin with.
//! - The view records the user and the conversation it was minted for, and
//!   [`read_transcript_message`] refuses when either disagrees with the
//!   task-local in force. A task-local does not cross `tokio::spawn`, so this
//!   is defence in depth rather than the primary boundary - but a leaked or
//!   stale view then returns nothing instead of another scope's bytes.
//! - No view installed means no read. There is no fallback to storage.
//!
//! A refusal carries no bytes, and it never says whether a message id exists
//! somewhere else. Two codes separate the two shapes of refusal, and neither
//! describes another scope's data:
//!
//! - `TRANSCRIPT_OUT_OF_SCOPE` says nothing was readable here - no view
//!   installed, or a view minted for another user or another conversation.
//! - `MESSAGE_NOT_FOUND` says the transcript in scope holds no such id. That
//!   is the one answer for an id in another conversation, an id belonging to
//!   another user, and an id nobody holds, so the caller cannot tell those
//!   three apart.
//!
//! ## Provenance
//!
//! This is a new way for old bytes to re-enter a turn, so it must not launder
//! them. A web page read on round 2, evicted on round 5 and fetched back on
//! round 9 is still externally-controlled text. The read resolves the tool
//! that produced the fetched message from its `tool_call_id`, asks
//! [`crate::tool_provenance::result_is_externally_controlled`] the same
//! question the turn asked when the result first arrived, and stamps
//! [`crate::tool_provenance::EXTERNAL_CONTENT_MARKER`] into its own payload
//! when the answer is yes. [`TRANSCRIPT_GET_TOOL`] is classified
//! `Declared(ExternalContentMarker)`, so the marker closes the gate exactly as
//! the original result did.
//!
//! A tool result whose producing tool cannot be resolved is graded as outside
//! content, because the other way round fails open: an unattributable result
//! would come back with no marker and read as the assistant's own bytes. Only
//! a tool result is graded that way. The rest of the transcript is the
//! conversation itself, which every turn replays untainted, so reading one of
//! its own messages back changes nothing.

use std::collections::HashMap;
use std::sync::Arc;

use crate::domain::{ConversationId, Message, Role};
use crate::ports::auth::{UserId, current_user_id};
use crate::ports::conversation_ctx::current_conversation_id;
use crate::tool_provenance::{EXTERNAL_CONTENT_MARKER, result_is_externally_controlled};

/// LLM-visible name of the transcript read-back tool.
///
/// Declared here, in `core`, rather than in the crate that implements it:
/// `crate::planning::compaction_pointer` and
/// `crate::context::overflow_compaction_notice` both name the tool in the text
/// the model reads, and neither may depend on the MCP layer.
pub const TRANSCRIPT_GET_TOOL: &str = "builtin_transcript_get";

/// Most content bytes one read may return.
///
/// The fetch is partial on purpose. The result was evicted because it was
/// large, so returning it whole re-inflates the context the eviction freed.
/// The value matches
/// [`crate::ports::scratchpad::RESPONSE_BYTE_BUDGET`], which bounds the other
/// read a model pages through, so the two feel the same to a caller.
pub const TRANSCRIPT_READ_MAX_BYTES: usize = 20 * 1024;

/// One message of the transcript, as a read-back needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptEntry {
    /// The message's stable id - what a caller addresses it by.
    pub id: String,
    /// Who said it.
    pub role: Role,
    /// The stored content, whole. Never the round's replacement.
    pub content: String,
    /// The tool that produced this content, resolved from the message's
    /// `tool_call_id` against the assistant tool-call requests before it.
    /// `None` for every message that is not a tool result, and for a tool
    /// result whose request is no longer in view.
    pub tool_name: Option<String>,
}

/// The messages a [`TranscriptView`] carries, behind one `Arc` so a view is
/// cheap to clone into a task-local and can still grow in place.
#[derive(Debug, Clone, Default)]
struct TranscriptData {
    /// One entry per message, in transcript order.
    entries: Vec<TranscriptEntry>,
    /// Message id -> index into `entries`.
    by_id: HashMap<String, usize>,
    /// `tool_call_id` -> the tool that was asked to run, read from the
    /// assistant requests as they are absorbed.
    call_names: HashMap<String, String>,
}

/// One turn's readable transcript, scoped to a user and a conversation.
///
/// Cloning is cheap: the messages sit behind an `Arc` and only the two scope
/// values are copied.
#[derive(Debug, Clone)]
pub struct TranscriptView {
    user_id: UserId,
    conversation_id: ConversationId,
    data: Arc<TranscriptData>,
}

impl TranscriptView {
    /// An empty view for `user_id`'s `conversation_id`.
    #[must_use]
    pub fn new(user_id: UserId, conversation_id: ConversationId) -> Self {
        Self {
            user_id,
            conversation_id,
            data: Arc::new(TranscriptData::default()),
        }
    }

    /// Take in every message the view does not already carry.
    ///
    /// The turn's message list is append-only while the turn runs, so the
    /// usual call copies only what the last round added. That property is
    /// checked rather than assumed: when `messages` is shorter than what the
    /// view holds, or when the message at the boundary is no longer the one
    /// absorbed there, the view is rebuilt from scratch. A stale index would
    /// otherwise return one message's bytes under another's id.
    pub fn absorb(&mut self, messages: &[Message]) {
        let data = Arc::make_mut(&mut self.data);
        let held = data.entries.len();
        let diverged = messages.len() < held
            || (held > 0 && messages[held - 1].id != data.entries[held - 1].id);
        if diverged {
            *data = TranscriptData::default();
        }
        for message in &messages[data.entries.len()..] {
            if message.role == Role::Assistant {
                for call in &message.tool_calls {
                    data.call_names.insert(call.id.clone(), call.name.clone());
                }
            }
            let tool_name = message
                .tool_call_id
                .as_deref()
                .and_then(|id| data.call_names.get(id))
                .cloned();
            data.by_id.insert(message.id.clone(), data.entries.len());
            data.entries.push(TranscriptEntry {
                id: message.id.clone(),
                role: message.role.clone(),
                content: message.content.clone(),
                tool_name,
            });
        }
    }

    /// How many messages the view carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.entries.len()
    }

    /// Whether the view carries no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.entries.is_empty()
    }

    /// The entry with `message_id`, or `None`.
    #[must_use]
    fn get(&self, message_id: &str) -> Option<&TranscriptEntry> {
        self.data
            .by_id
            .get(message_id)
            .and_then(|i| self.data.entries.get(*i))
    }

    /// Whether this view was minted for the scope now in force.
    fn matches_current_scope(&self) -> bool {
        current_user_id() == self.user_id
            && current_conversation_id().as_ref() == Some(&self.conversation_id)
    }
}

tokio::task_local! {
    /// The transcript the running tool may read. Installed by the turn loop
    /// around each server-side tool execution; read by
    /// [`read_transcript_message`]. Unset for background workers, client-side
    /// tools and tests, where a read returns nothing.
    static TRANSCRIPT: TranscriptView;
}

/// Run `fut` with `view` installed as the readable transcript.
pub async fn with_transcript<F, T>(view: TranscriptView, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TRANSCRIPT.scope(view, fut).await
}

/// The transcript installed for this task, or `None` when there is none.
#[must_use]
pub fn current_transcript() -> Option<TranscriptView> {
    TRANSCRIPT.try_with(Clone::clone).ok()
}

/// One read of the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptReadRequest {
    /// The message to read.
    pub message_id: String,
    /// Where to start, in bytes from the beginning of the content.
    pub offset: usize,
    /// How many bytes to return. `None` reads up to
    /// [`TRANSCRIPT_READ_MAX_BYTES`]; a larger value is cut to it.
    pub length: Option<usize>,
}

impl TranscriptReadRequest {
    /// Read `message_id` from the beginning, up to the cap.
    #[must_use]
    pub fn new(message_id: impl Into<String>) -> Self {
        Self {
            message_id: message_id.into(),
            offset: 0,
            length: None,
        }
    }
}

/// Business code for a read that returned no content.
///
/// A decline is a normal outcome, not a failure (AGENTS.md 8.2), and it is
/// machine-readable so a caller never has to match English text (8.3).
const CODE_OUT_OF_SCOPE: &str = "TRANSCRIPT_OUT_OF_SCOPE";
/// Business code for a message id the transcript in scope does not hold.
const CODE_NOT_FOUND: &str = "MESSAGE_NOT_FOUND";

/// Read one message of the transcript in scope, as the tool's JSON payload.
///
/// Returns a structured decline - never an error - when there is no transcript
/// in scope, when the view in scope belongs to another user or conversation,
/// or when no message in it holds `message_id`. See the module header for what
/// the caller is and is not told.
#[must_use]
pub fn read_transcript_message(request: &TranscriptReadRequest) -> String {
    let Some(view) = current_transcript().filter(TranscriptView::matches_current_scope) else {
        return decline(
            CODE_OUT_OF_SCOPE,
            "no conversation transcript is readable here",
            "There is no transcript to read from in this context.",
        );
    };
    let Some(entry) = view.get(&request.message_id) else {
        return decline(
            CODE_NOT_FOUND,
            "no message in this conversation has that id",
            "This conversation holds no message with that id. Check the id in the pointer \
             you read it from.",
        );
    };
    let total = entry.content.len();
    let requested = request.length.unwrap_or(TRANSCRIPT_READ_MAX_BYTES);
    let capped = requested.min(TRANSCRIPT_READ_MAX_BYTES);
    let start = entry.content.floor_char_boundary(request.offset);
    let end = slice_end(&entry.content, start, capped);
    let slice = &entry.content[start..end];

    // `next_offset` is an offset the caller can follow, so it is offered only
    // when following it reads bytes this response did not return. A read that
    // returned nothing while bytes remain - which only a `length` of zero
    // does - would otherwise hand back the offset it started at, and a caller
    // told to page by `next_offset` would ask the same question until its
    // rounds ran out.
    let advanced = end > start;
    let unread_remain = end < total;
    let mut payload = serde_json::json!({
        "ok": true,
        "message_id": entry.id,
        "role": entry.role,
        "produced_by": entry.tool_name,
        "total_bytes": total,
        "offset": start,
        "returned_bytes": slice.len(),
        "next_offset": if unread_remain && advanced { Some(end) } else { None },
        "content": slice,
    });
    if requested > TRANSCRIPT_READ_MAX_BYTES {
        payload["truncated"] = serde_json::Value::Bool(true);
        payload["message"] = serde_json::json!(format!(
            "`length` was cut to the {TRANSCRIPT_READ_MAX_BYTES}-byte cap on one read. Read \
             the rest from `next_offset`."
        ));
    } else if unread_remain && !advanced {
        payload["message"] = serde_json::json!(format!(
            "`length` was 0, so this read returned no bytes and there is no `next_offset` to \
             follow. Read again from offset {start} with a `length` of at least one byte."
        ));
    }
    // Provenance is decided from the WHOLE stored result, not from the slice
    // returned: a range that happens to miss the marker is still a range of
    // externally-controlled bytes.
    //
    // A tool result whose producing tool cannot be resolved is graded as
    // outside content. The other way round fails open: an unattributable
    // result would come back with no marker and read as the assistant's own
    // bytes. Only a tool result is graded this way - the rest of the
    // transcript is the conversation itself, which every turn replays
    // untainted.
    let external = match entry.tool_name.as_deref() {
        Some(name) => result_is_externally_controlled(name, &entry.content),
        None => entry.role == Role::Tool,
    };
    if external {
        payload["provenance"] = serde_json::json!(EXTERNAL_CONTENT_MARKER);
    }
    payload.to_string()
}

/// A refusal that carries no bytes: a stable code, a description, a line for
/// the reader, and whether trying again could help.
fn decline(code: &str, description: &str, message: &str) -> String {
    serde_json::json!({
        "ok": false,
        "code": code,
        "description": description,
        "message": message,
        "retryable": false,
    })
    .to_string()
}

/// Where a read starting at `start` and asking for `len` bytes ends.
///
/// Snapped back to a char boundary, because a range that splits a character is
/// not a string. When snapping back would return nothing at all - the next
/// character is wider than the whole request - the end moves forward to the
/// next boundary instead, so a read that asked for bytes always returns some
/// rather than an empty slice at the same offset.
///
/// That widening applies only to a read that asked for at least one byte. A
/// read of zero bytes returns zero: handing back a character nobody asked for
/// would make the response disagree with its own request. Such a read reports
/// no `next_offset`, which is what keeps paging terminating - see
/// [`read_transcript_message`].
fn slice_end(s: &str, start: usize, len: usize) -> usize {
    let end = s.floor_char_boundary(start.saturating_add(len));
    if end > start || start >= s.len() || len == 0 {
        return end;
    }
    // `start` is inside the string here, so `start + 1` is a byte index it
    // holds and the widening never asks past the end.
    s.ceil_char_boundary(start + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ToolCall;
    use crate::ports::auth::with_user_id;
    use crate::ports::conversation_ctx::with_conversation_id;
    use crate::tool_provenance::{GateChange, TurnProvenance};

    fn tool_result(call_id: &str, content: &str) -> Message {
        Message::tool_result(call_id, content)
    }

    fn requested(call_id: &str, tool: &str) -> Message {
        Message::assistant_with_tool_calls(vec![ToolCall::new(call_id, tool, "{}")])
    }

    /// A view over `messages`, scoped to `user`/`conversation`.
    fn view(user: &str, conversation: &str, messages: &[Message]) -> TranscriptView {
        let mut v = TranscriptView::new(
            UserId::new(user.to_string()),
            ConversationId::from(conversation.to_string()),
        );
        v.absorb(messages);
        v
    }

    /// Run `body` with `view` installed under `user`/`conversation`.
    async fn scoped<T>(
        user: &str,
        conversation: &str,
        view: TranscriptView,
        body: impl std::future::Future<Output = T>,
    ) -> T {
        with_user_id(
            UserId::new(user.to_string()),
            with_conversation_id(
                ConversationId::from(conversation.to_string()),
                with_transcript(view, body),
            ),
        )
        .await
    }

    fn parse(payload: &str) -> serde_json::Value {
        serde_json::from_str(payload).expect("the tool payload must be JSON")
    }

    #[tokio::test]
    async fn a_read_returns_the_whole_stored_content_when_it_fits() {
        let messages = vec![requested("c1", "read_file"), tool_result("c1", "the bytes")];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&id))
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true);
        assert_eq!(got["content"], "the bytes");
        assert_eq!(got["produced_by"], "read_file");
        assert_eq!(got["total_bytes"], 9);
        assert!(got["next_offset"].is_null(), "{payload}");
    }

    /// AC: a read with an offset and a length returns that range, and the
    /// response states the total size.
    #[tokio::test]
    async fn a_read_with_an_offset_and_a_length_returns_that_range_and_states_the_total_size() {
        let body = "0123456789abcdef";
        let messages = vec![requested("c1", "read_file"), tool_result("c1", body)];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 4,
                length: Some(6),
            })
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true);
        assert_eq!(got["content"], "456789", "{payload}");
        assert_eq!(got["offset"], 4);
        assert_eq!(got["returned_bytes"], 6);
        assert_eq!(
            got["total_bytes"], 16,
            "the response must state the whole size, not the slice: {payload}"
        );
        assert_eq!(got["next_offset"], 10, "{payload}");
    }

    /// AC: a length above the cap is truncated to the cap, and the response
    /// says so.
    #[tokio::test]
    async fn a_length_above_the_cap_is_truncated_to_the_cap_and_the_response_says_so() {
        let body = "x".repeat(TRANSCRIPT_READ_MAX_BYTES * 2);
        let messages = vec![requested("c1", "read_file"), tool_result("c1", &body)];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 0,
                length: Some(TRANSCRIPT_READ_MAX_BYTES * 2),
            })
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true);
        assert_eq!(
            got["returned_bytes"],
            TRANSCRIPT_READ_MAX_BYTES,
            "a read may not exceed the cap: {}",
            &payload[..200.min(payload.len())]
        );
        assert_eq!(got["truncated"], true, "the response must say it was cut");
        assert!(
            got["message"]
                .as_str()
                .is_some_and(|m| m.contains(&TRANSCRIPT_READ_MAX_BYTES.to_string())),
            "the message must name the cap"
        );
        assert_eq!(got["next_offset"], TRANSCRIPT_READ_MAX_BYTES);
    }

    /// AC: a message id owned by another user is refused.
    ///
    /// Both directions, because two different things hold them. The primary
    /// boundary is the view itself: the turn builds it from the conversation it
    /// loaded under the calling user's id, so another user's message is not in
    /// it to begin with. The second is the user stamped on the view, which
    /// refuses a view minted for another user - defence in depth, for a view
    /// that leaked or went stale.
    #[tokio::test]
    async fn a_message_id_owned_by_another_user_is_refused() {
        let alices = vec![
            requested("c1", "read_file"),
            tool_result("c1", "alice's bytes"),
        ];
        let alices_id = alices[1].id.clone();
        // The active turn is bob's, and holds his own, unrelated message.
        let bobs = vec![
            requested("c9", "read_file"),
            tool_result("c9", "bob's bytes"),
        ];

        let payload = scoped("bob", "conv", view("bob", "conv", &bobs), async {
            read_transcript_message(&TranscriptReadRequest::new(&alices_id))
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(
            got["code"], CODE_NOT_FOUND,
            "bob's own view holds no such id, which is the boundary this case \
             exercises: {payload}"
        );
        assert!(
            got.get("content").is_none(),
            "a refusal must carry no bytes: {payload}"
        );
        assert!(!payload.contains("alice's bytes"), "{payload}");

        // And the other direction: alice's own view, in force while bob is the
        // user, reads nothing either.
        let payload = scoped("bob", "conv", view("alice", "conv", &alices), async {
            read_transcript_message(&TranscriptReadRequest::new(&alices_id))
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(
            got["code"], CODE_OUT_OF_SCOPE,
            "the view's own user stamp is the boundary this case exercises: {payload}"
        );
        assert!(
            got.get("content").is_none(),
            "a refusal must carry no bytes: {payload}"
        );
        assert!(!payload.contains("alice's bytes"), "{payload}");
    }

    /// AC: a message id in another conversation is refused.
    #[tokio::test]
    async fn a_message_id_in_another_conversation_is_refused() {
        let other = vec![
            requested("c1", "read_file"),
            tool_result("c1", "other bytes"),
        ];
        let other_id = other[1].id.clone();
        // The active conversation holds its own, unrelated message.
        let mine = vec![requested("c9", "read_file"), tool_result("c9", "my bytes")];
        let v = view("u", "conv-mine", &mine);

        let payload = scoped("u", "conv-mine", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&other_id))
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(
            got["code"], CODE_NOT_FOUND,
            "the view holds no such id, and says only that: {payload}"
        );
        assert!(got.get("content").is_none(), "{payload}");
        assert!(!payload.contains("other bytes"), "{payload}");

        // And the other direction: the other conversation's own view, in force
        // while a different conversation is active, reads nothing either.
        let leaked = view("u", "conv-other", &other);
        let payload = scoped("u", "conv-mine", leaked, async {
            read_transcript_message(&TranscriptReadRequest::new(&other_id))
        })
        .await;
        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(
            got["code"], CODE_OUT_OF_SCOPE,
            "the view's own conversation stamp is the boundary this case exercises: {payload}"
        );
        assert!(!payload.contains("other bytes"), "{payload}");
    }

    /// AC: an unknown message id returns a structured decline, not an error.
    #[tokio::test]
    async fn an_unknown_message_id_returns_a_structured_decline_not_an_error() {
        let messages = vec![requested("c1", "read_file"), tool_result("c1", "the bytes")];
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new("nobody-holds-this"))
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(got["code"], CODE_NOT_FOUND, "{payload}");
        assert!(got["description"].is_string(), "{payload}");
        assert!(got["message"].is_string(), "{payload}");
        assert_eq!(got["retryable"], false, "{payload}");
    }

    /// AC: reading back a result from an externally-controlled tool taints the
    /// turn exactly as the original result did.
    #[tokio::test]
    async fn reading_back_an_externally_controlled_result_taints_the_turn_like_the_original() {
        let page = "a web page the model fetched";
        let messages = vec![
            requested("c1", "weather_get_current"),
            tool_result("c1", page),
        ];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&id))
        })
        .await;

        // What the original result did to a fresh turn.
        let mut original = TurnProvenance::new();
        assert_eq!(
            original.observe_result("weather_get_current", page),
            GateChange::JustClosed
        );

        // What the read-back does to a fresh turn: the same.
        let mut readback = TurnProvenance::new();
        assert_eq!(
            readback.observe_result(TRANSCRIPT_GET_TOOL, &payload),
            GateChange::JustClosed,
            "the read-back must not launder the bytes: {payload}"
        );
        assert!(readback.ingested_external());
    }

    /// AC: reading back a result from a trusted tool does not taint the turn.
    #[tokio::test]
    async fn reading_back_a_trusted_result_does_not_taint_the_turn() {
        let messages = vec![
            requested("c1", "builtin_knowledge_base_get"),
            tool_result("c1", "the user prefers dark roast"),
        ];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&id))
        })
        .await;

        let mut turn = TurnProvenance::new();
        assert_eq!(
            turn.observe_result(TRANSCRIPT_GET_TOOL, &payload),
            GateChange::Unchanged,
            "a trusted result must stay trusted on the way back: {payload}"
        );
        assert!(!turn.ingested_external());
    }

    /// A tool result whose request is not in view cannot be graded, so it is
    /// taken as outside content rather than as trusted.
    ///
    /// The alternative fails open: an unattributable result would come back
    /// with no marker and read as the assistant's own bytes, which is the one
    /// thing a read-back may not do. A non-tool message is a separate case and
    /// is not covered by this - see
    /// `a_read_of_a_message_that_is_not_a_tool_result_does_not_taint_the_turn`.
    #[tokio::test]
    async fn a_read_of_a_tool_result_with_no_resolvable_tool_taints_the_turn() {
        // The result is in the view; the assistant request that named the tool
        // is not, so nothing says which tool produced these bytes.
        let orphan = Message::tool_result("c-gone", "bytes nobody can attribute");
        let messages = vec![orphan];
        let id = messages[0].id.clone();
        let v = view("u", "conv", &messages);
        assert_eq!(
            v.get(&id).and_then(|e| e.tool_name.clone()),
            None,
            "the fixture must leave the producing tool unresolvable"
        );

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&id))
        })
        .await;

        let mut turn = TurnProvenance::new();
        assert_eq!(
            turn.observe_result(TRANSCRIPT_GET_TOOL, &payload),
            GateChange::JustClosed,
            "a result that cannot be graded must not read as trusted: {payload}"
        );
    }

    /// The other half of the rule above: the ordinary transcript is not a tool
    /// result, and every turn replays it untainted, so reading one of its own
    /// messages back changes nothing.
    #[tokio::test]
    async fn a_read_of_a_message_that_is_not_a_tool_result_does_not_taint_the_turn() {
        let messages = vec![Message::new(Role::User, "what the user typed")];
        let id = messages[0].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest::new(&id))
        })
        .await;

        let mut turn = TurnProvenance::new();
        assert_eq!(
            turn.observe_result(TRANSCRIPT_GET_TOOL, &payload),
            GateChange::Unchanged,
            "the user's own words are not outside content: {payload}"
        );
    }

    #[tokio::test]
    async fn a_read_with_no_transcript_in_scope_is_refused() {
        let payload = with_user_id(UserId::new("u".to_string()), async {
            read_transcript_message(&TranscriptReadRequest::new("anything"))
        })
        .await;
        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(got["code"], CODE_OUT_OF_SCOPE, "{payload}");
    }

    #[tokio::test]
    async fn absorb_takes_in_only_what_it_does_not_already_hold() {
        let mut messages = vec![requested("c1", "read_file"), tool_result("c1", "one")];
        let mut v = TranscriptView::new(
            UserId::new("u".to_string()),
            ConversationId::from("conv".to_string()),
        );
        v.absorb(&messages);
        assert_eq!(v.len(), 2);

        messages.push(requested("c2", "read_file"));
        messages.push(tool_result("c2", "two"));
        v.absorb(&messages);
        assert_eq!(v.len(), 4);
        assert_eq!(
            v.get(&messages[3].id).map(|e| e.content.as_str()),
            Some("two")
        );
        assert_eq!(
            v.get(&messages[1].id).map(|e| e.content.as_str()),
            Some("one"),
            "the earlier round must survive the second absorb"
        );
    }

    /// The index maps an id to a message, so it may never survive a transcript
    /// that stopped being an extension of what was absorbed.
    #[tokio::test]
    async fn absorb_rebuilds_when_the_transcript_is_no_longer_an_extension() {
        let first = vec![requested("c1", "read_file"), tool_result("c1", "one")];
        let mut v = TranscriptView::new(
            UserId::new("u".to_string()),
            ConversationId::from("conv".to_string()),
        );
        v.absorb(&first);
        let stale_id = first[1].id.clone();

        let second = vec![requested("c9", "read_file"), tool_result("c9", "nine")];
        v.absorb(&second);

        assert_eq!(v.len(), 2, "the view must hold the new transcript only");
        assert!(
            v.get(&stale_id).is_none(),
            "an id from the replaced transcript must not resolve"
        );
        assert_eq!(
            v.get(&second[1].id).map(|e| e.content.as_str()),
            Some("nine")
        );
    }

    #[tokio::test]
    async fn a_read_never_splits_a_character() {
        // Four-byte characters, so every naive byte cut lands mid-character.
        let body = "\u{1F600}".repeat(8);
        let messages = vec![requested("c1", "read_file"), tool_result("c1", &body)];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 3,
                length: Some(6),
            })
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true, "{payload}");
        let content = got["content"].as_str().expect("content is a string");
        let offset = got["offset"].as_u64().expect("offset is a number") as usize;
        let returned = got["returned_bytes"]
            .as_u64()
            .expect("returned_bytes is a number") as usize;
        assert!(
            !content.is_empty(),
            "a read that asked for bytes must return some: {payload}"
        );
        assert_eq!(
            content,
            &body[offset..offset + returned],
            "the content must be exactly the range the response describes: {payload}"
        );
        assert_eq!(
            content, "\u{1F600}",
            "a 6-byte request over 4-byte characters returns the one whole \
             character that fits: {payload}"
        );
        assert_eq!(
            offset, 0,
            "the offset must snap back to a character boundary: {payload}"
        );
    }

    /// A response that carries a `next_offset` carries one the caller can
    /// follow: reading from it returns bytes the caller does not already hold.
    /// A response that cannot offer that carries no `next_offset` at all, so a
    /// caller told to page by it can never repeat the same read forever.
    #[tokio::test]
    async fn paging_always_advances() {
        let body = "\u{1F600}\u{1F600}";
        let messages = vec![requested("c1", "read_file"), tool_result("c1", body)];
        let id = messages[1].id.clone();

        // A character wider than the whole request: snapping the end back would
        // return nothing and leave `next_offset` where it started.
        let payload = scoped("u", "conv", view("u", "conv", &messages), async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 0,
                length: Some(1),
            })
        })
        .await;
        let got = parse(&payload);
        assert_eq!(got["returned_bytes"], 4, "{payload}");
        assert_eq!(got["next_offset"], 4, "{payload}");

        // A request for no bytes. There are bytes left, but the offset that
        // would read them is the offset this read started at, so following it
        // asks the same question again.
        let payload = scoped("u", "conv", view("u", "conv", &messages), async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 0,
                length: Some(0),
            })
        })
        .await;
        let got = parse(&payload);
        assert_eq!(got["returned_bytes"], 0, "{payload}");
        assert!(
            got["next_offset"].is_null(),
            "a read that returned nothing must not offer an offset to repeat it: {payload}"
        );
        assert!(
            got["message"]
                .as_str()
                .is_some_and(|m| m.contains("length")),
            "the response must say how to read the rest: {payload}"
        );
    }

    /// The progress rule widens a read that asked for at least one byte and
    /// would otherwise return none. A read that asked for none is not that
    /// case, and must return none: handing back a byte nobody asked for is a
    /// response that does not match its own request.
    #[tokio::test]
    async fn a_read_of_zero_bytes_returns_nothing() {
        let messages = vec![requested("c1", "read_file"), tool_result("c1", "hello")];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 2,
                length: Some(0),
            })
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true, "{payload}");
        assert_eq!(got["content"], "", "{payload}");
        assert_eq!(got["returned_bytes"], 0, "{payload}");
        assert_eq!(
            got["offset"], 2,
            "the read must not move on from where it was asked to start: {payload}"
        );
    }

    #[tokio::test]
    async fn a_read_past_the_end_returns_nothing_and_says_so() {
        let messages = vec![requested("c1", "read_file"), tool_result("c1", "short")];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async {
            read_transcript_message(&TranscriptReadRequest {
                message_id: id.clone(),
                offset: 9_000,
                length: None,
            })
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], true, "{payload}");
        assert_eq!(got["content"], "");
        assert_eq!(got["offset"], 5);
        assert!(got["next_offset"].is_null(), "{payload}");
    }

    #[tokio::test]
    async fn a_view_does_not_cross_tokio_spawn() {
        let messages = vec![requested("c1", "read_file"), tool_result("c1", "the bytes")];
        let id = messages[1].id.clone();
        let v = view("u", "conv", &messages);

        let payload = scoped("u", "conv", v, async move {
            tokio::spawn(async move { read_transcript_message(&TranscriptReadRequest::new(&id)) })
                .await
                .expect("join")
        })
        .await;

        let got = parse(&payload);
        assert_eq!(got["ok"], false, "{payload}");
        assert_eq!(got["code"], CODE_OUT_OF_SCOPE, "{payload}");
    }
}
