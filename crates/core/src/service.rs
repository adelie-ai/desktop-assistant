use crate::CoreError;
use crate::context::{
    COMPACTION_TOKEN_RATIO, ConversationView, DEFAULT_MAX_TOOL_RESULT_BYTES, MAX_CONTEXT_MESSAGES,
    MAX_OVERFLOW_RETRIES, MIN_CONTEXT_MESSAGES, ToolContext, ToolLocalityContext, TurnAnchors,
    assemble_turn_within_budget, cap_tool_result, compaction_range, generate_context_summary,
    recover_from_overflow,
};
use crate::domain::{
    Conversation, ConversationId, ConversationSummary, Message, Role, ToolCall, ToolDefinition,
    ToolNamespace,
};
use crate::planning::{self, StepStack};
use crate::ports::client_tools::current_client_tools;
use crate::ports::conversation_ctx::with_conversation_id;
use crate::ports::inbound::ConversationService;
use crate::ports::llm::{
    ChunkCallback, LlmClient, ReasoningConfig, StatusCallback, current_cancellation_token,
    current_context_budget, current_tool_allowlist,
};
use crate::ports::scratchpad::{
    MAX_NOTE_BYTES, NewScratchpadNote, SCRATCHPAD_GOAL_KEY, ScratchpadGetManyFn, ScratchpadListFn,
    ScratchpadWriteFn,
};
use crate::ports::store::ConversationStore;
use crate::ports::tool_observer::{ToolEvent, notify_tool_event};
use crate::ports::tools::ToolExecutor;
use crate::ports::transport::{current_client_label, current_co_location, current_transport_kind};
use crate::sanitize::sanitize_assistant_text;
use crate::tools::{
    NoopToolExecutor, categorize_tool_namespaces, summarize_tool_text, summarize_tool_value,
    tool_set_hash,
};
use chrono::{Duration, Local};
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use tokio_util::sync::CancellationToken;

/// Return `Err(CoreError::Cancelled)` if the current task's cancellation
/// token (installed by [`crate::ports::llm::with_cancellation_token`]) has
/// been tripped. `None` (no token installed) is treated as "never
/// cancelled" so legacy call sites — tests, dreaming jobs, anything that
/// doesn't route through `send_prompt_with_override` — keep their
/// pre-#109 behaviour.
fn bail_if_cancelled() -> Result<(), CoreError> {
    if let Some(token) = current_cancellation_token()
        && token.is_cancelled()
    {
        return Err(CoreError::Cancelled);
    }
    Ok(())
}

/// Return the per-turn cancellation token, falling back to a fresh
/// never-cancelled token (via `Default::default()`) when no scope is
/// installed. Used by the chunk callback so streaming code can call
/// `token.is_cancelled()` without having to special-case the
/// absent-scope path on every chunk.
fn cancellation_token_or_default() -> CancellationToken {
    current_cancellation_token().unwrap_or_default()
}

/// Maximum number of tool-calling rounds before giving up.
const MAX_TOOL_ROUNDS: usize = 200;

/// How often to emit a keepalive status while a server-side tool (or a subagent,
/// which runs as a tool) executes silently, so the client's stall watchdog
/// (90s, `EVENT_STALL_TIMEOUT`) does not false-abandon a turn the daemon is
/// actively servicing (#584). Comfortably under the stall window, leaving margin
/// for several resets.
const SERVER_TOOL_KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Transient instruction shown to the model for the #453 wind-down completion
/// only (never persisted): the tool budget is spent, so it must close out in
/// prose rather than request more tools.
const WIND_DOWN_INSTRUCTION: &str = "You've reached this turn's limit on tool \
    calls, so you can't run any more tools right now. Wrap up now in a brief, \
    natural reply: what you accomplished, what's still left, and how we can \
    continue from here.";

/// Closing persisted when the #453 wind-down completion itself fails or comes
/// back empty — so a round-budget-exhausted turn is never silently lost.
const WIND_DOWN_FALLBACK: &str = "I reached the limit on tool calls for this \
    turn before I could finish. I've kept the work so far — send another \
    message and I'll pick up where I left off.";

fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn cutoff_timestamp(max_age_days: u32) -> String {
    (Local::now() - Duration::days(i64::from(max_age_days)))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

/// Translate a [`CoreError`] into a user-visible explanation suitable
/// for surfacing in chat. Each LLM-domain variant maps to a tailored
/// message; non-LLM variants and the bare `Llm(detail)` fallback share a
/// generic "I hit an LLM backend error..." line that includes the raw
/// detail for debugging.
fn user_visible_llm_error_message(error: &CoreError) -> String {
    match error {
        CoreError::ContextOverflow { detail, .. } => format!(
            "The conversation exceeded the model's context window. We'll truncate older content and retry. Details: {detail}"
        ),
        CoreError::RateLimited { detail, .. } => format!(
            "The API rate limit was exceeded. Please wait a moment and try again. Details: {detail}"
        ),
        CoreError::QuotaExceeded { detail } => format!(
            "Your API quota is exhausted. Top up the account or switch to a different API key. Details: {detail}"
        ),
        CoreError::ModelLoading { detail } => format!(
            "The model is still downloading or loading. Please wait a moment and try again. Details: {detail}"
        ),
        CoreError::ToolsUnsupported { detail } => format!(
            "This model does not support tool use. Please switch to a tool-capable model or disable tools for this chat. Details: {detail}"
        ),
        // Bare LLM error and any non-LLM variant share the generic
        // fallback. This intentionally does NOT enumerate every
        // CoreError variant — `Display` already produces a readable
        // string and the surrounding service layer is the right place
        // to add tailored messages for non-LLM domains.
        _ => format!(
            "I hit an LLM backend error and could not complete this request. Details: {error}"
        ),
    }
}

/// Strip surrounding quotes/backticks and trailing punctuation from a raw LLM title,
/// then limit to at most 8 words as a guard-rail.
fn sanitize_generated_title(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or("").trim();
    let stripped = first_line
        .trim_matches(|c| matches!(c, '"' | '\'' | '`'))
        .trim_end_matches(['.', ',', ';', '!', '?']);
    stripped
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Ask the LLM for a concise 3-5-word channel name based on the initial prompt.
/// Returns an empty string on failure so the caller can keep the existing title.
async fn generate_conversation_title<L: LlmClient>(initial_prompt: &str, llm: &L) -> String {
    let messages = vec![
        Message::new(
            Role::System,
            "Generate a concise channel name for a new conversation. \
             Use 3-5 words. Front-load the most specific and meaningful words first — \
             the name may be truncated at the end. Use title case. No punctuation at \
             the edges, no quotes, no explanation. Respond with ONLY the channel name.",
        ),
        Message::new(
            Role::User,
            format!("First message in the conversation: {initial_prompt}"),
        ),
    ];
    match llm
        .stream_completion(
            messages,
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        )
        .await
    {
        Ok(response) => sanitize_generated_title(&response.text),
        Err(e) => {
            tracing::warn!("conversation title generation failed: {e}");
            String::new()
        }
    }
}

/// Core service implementing conversation management.
/// Generic over store, LLM, and tool executor backends for testability.
pub struct ConversationHandler<S, L, T = NoopToolExecutor> {
    store: S,
    llm: L,
    backend_llm: Option<L>,
    tools: T,
    id_generator: Box<dyn Fn() -> String + Send + Sync>,
    /// Memoized result of `categorize_tool_namespaces`, keyed by `tool_set_hash`.
    ///
    /// Why: Categorization is an LLM call carrying the full tool manifest
    /// (often ≥1K input tokens). Re-running it every turn is wasteful when
    /// the underlying tools have not changed. Lifetime is per-handler
    /// (process-lifetime — there is no eviction); invalidation happens
    /// implicitly when the hash of the current tool set differs from the
    /// stored one. The hash covers tool names AND descriptions, so any
    /// edit to either triggers a fresh categorization.
    namespace_cache: std::sync::Mutex<Option<(u64, Vec<ToolNamespace>)>>,
    /// Single-flight guard for the categorization LLM call (issue #305 item 8).
    ///
    /// `namespace_cache` answers cache *hits* with a cheap sync lock, but two
    /// concurrent first turns (cold cache) would both miss and each pay the
    /// categorization round-trip — the "thundering herd". This async mutex
    /// serializes the *miss path*: the winner runs categorization and populates
    /// the cache; losers wait here, then re-check the cache and find the result
    /// already there. Held only across the categorization await on a miss, never
    /// on a hit, so steady-state turns are unaffected. A single guard suffices —
    /// the tool set has one hash at a time, and a hash change just means the next
    /// miss recomputes under the same guard.
    categorize_lock: tokio::sync::Mutex<()>,
    /// Optional reader for the reserved scratchpad `goal` note. When set, the
    /// dispatch loop reads it each round and prefers it over the verbatim
    /// user prompt as the task anchor, so a model-maintained goal survives
    /// windowing/compaction. `None` (the default) preserves the prior
    /// verbatim-prompt-only anchor behaviour.
    scratchpad_goal_read: Option<ScratchpadGetManyFn>,
    /// Optional writer for scratchpad notes. When set, the planning tools
    /// (`begin_step`/`complete_step`, #240) are advertised each turn and the
    /// dispatch loop uses this to record plan todos + distilled step outcomes.
    /// `None` (the default) leaves the planning tools off and the loop behaves
    /// exactly as before. Wire the daemon's *event-emitting* write closure so
    /// plan changes reach clients via `ScratchpadChanged`.
    scratchpad_write: Option<ScratchpadWriteFn>,
    /// Optional lister for scratchpad notes. When set, the dispatch loop reads
    /// the conversation's `todo` notes each round and surfaces the open plan
    /// as a compact `[Plan]` system message so it stays in view while raw work
    /// is evicted. `None` disables per-round plan surfacing.
    scratchpad_list: Option<ScratchpadListFn>,
    /// Maximum byte length a single tool result may occupy before it is
    /// truncated at ingestion (issue #174). Defaults to
    /// [`DEFAULT_MAX_TOOL_RESULT_BYTES`]; override via
    /// [`Self::with_max_tool_result_bytes`].
    max_tool_result_bytes: usize,
    /// The daemon's self-identity label, used as the `host` of a server-side
    /// [`crate::domain::ToolLocality`] in the per-turn tool note (issue #243).
    /// The daemon sets this to its hostname via [`Self::with_host`]; the
    /// follow-up phase will replace it with a stable machine-id. Defaults to
    /// [`DEFAULT_HOST_LABEL`] so callers that don't set it (tests, background
    /// jobs) still produce a coherent note.
    host: String,
    /// Per-conversation turn serialization (#282). Maps a conversation id to a
    /// `Weak`-referenced async mutex; a turn upgrades-or-inserts the entry, holds
    /// the `Arc<Mutex<()>>` guard across its whole body, then drops it. Entries
    /// are `Weak`, so once no turn holds the `Arc` the entry dangles and is
    /// pruned opportunistically on the next get-or-insert — the map stays bounded
    /// by the number of *concurrently active* conversations (typically single
    /// digits). The outer `std::sync::Mutex` is only ever held for the
    /// upgrade/insert/prune of the `Arc` — never across an `.await`. Different
    /// conversation ids never contend; same-id turns serialize FIFO.
    turn_locks: std::sync::Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

/// Fallback `host` label for [`ConversationHandler`] when the daemon does not
/// set one via [`ConversationHandler::with_host`] (issue #243). The live daemon
/// always sets its hostname; this keeps tests and background jobs coherent.
pub const DEFAULT_HOST_LABEL: &str = "this machine";

impl<S, L> ConversationHandler<S, L, NoopToolExecutor> {
    pub fn new(store: S, llm: L, id_generator: Box<dyn Fn() -> String + Send + Sync>) -> Self {
        Self {
            store,
            llm,
            backend_llm: None,
            tools: NoopToolExecutor,
            id_generator,
            namespace_cache: std::sync::Mutex::new(None),
            categorize_lock: tokio::sync::Mutex::new(()),
            scratchpad_goal_read: None,
            scratchpad_write: None,
            scratchpad_list: None,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            host: DEFAULT_HOST_LABEL.to_string(),
            turn_locks: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl<S, L, T> ConversationHandler<S, L, T> {
    pub fn with_tools(
        store: S,
        llm: L,
        tools: T,
        id_generator: Box<dyn Fn() -> String + Send + Sync>,
    ) -> Self {
        Self {
            store,
            llm,
            backend_llm: None,
            tools,
            id_generator,
            namespace_cache: std::sync::Mutex::new(None),
            categorize_lock: tokio::sync::Mutex::new(()),
            scratchpad_goal_read: None,
            scratchpad_write: None,
            scratchpad_list: None,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            host: DEFAULT_HOST_LABEL.to_string(),
            turn_locks: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Set the daemon's self-identity `host` label used for server-side tool
    /// localities in the per-turn tool note (issue #243). The daemon wires its
    /// hostname here; the follow-up phase replaces it with a stable machine-id.
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set a separate LLM for backend tasks (title generation, context summary).
    /// Falls back to the primary LLM when not set.
    pub fn with_backend_llm(mut self, llm: L) -> Self {
        self.backend_llm = Some(llm);
        self
    }

    /// Wire a reader for the reserved scratchpad `goal` note. The dispatch
    /// loop reads it once per tool round (a bounded single-key fetch) and,
    /// when present, surfaces it as the conversation's task anchor in
    /// preference to the verbatim user prompt — so a model-maintained,
    /// evolving goal keeps showing up even after history is compacted away.
    pub fn with_scratchpad_goal(mut self, goal_read: ScratchpadGetManyFn) -> Self {
        self.scratchpad_goal_read = Some(goal_read);
        self
    }

    /// Wire a writer for scratchpad notes, enabling the step-planning +
    /// context-compaction tools (`begin_step`/`complete_step`, #240). The
    /// dispatch loop advertises those tools each turn and uses this closure to
    /// record plan todos and distilled step outcomes. Wire the daemon's
    /// *event-emitting* write closure so plan changes reach clients.
    pub fn with_scratchpad_write(mut self, write: ScratchpadWriteFn) -> Self {
        self.scratchpad_write = Some(write);
        self
    }

    /// Wire a lister for scratchpad notes, enabling per-round surfacing of the
    /// open plan (the conversation's `todo` notes) as a `[Plan]` system message.
    pub fn with_scratchpad_list(mut self, list: ScratchpadListFn) -> Self {
        self.scratchpad_list = Some(list);
        self
    }

    /// Override the per-tool-result ingestion cap (issue #174). Results
    /// larger than this are truncated with a notice before being stored so a
    /// single runaway tool call can't wedge the conversation or the database.
    pub fn with_max_tool_result_bytes(mut self, max_bytes: usize) -> Self {
        self.max_tool_result_bytes = max_bytes;
        self
    }

    /// Get-or-insert the per-conversation turn lock (#282), pruning dangling
    /// weak entries in the same critical section so the map stays bounded by the
    /// number of *concurrently active* conversations. Returns an owned
    /// `Arc<tokio::sync::Mutex<()>>`; the caller `.lock().await`s it and holds
    /// the guard across the turn body. The `std::sync::Mutex` is held only for
    /// this upgrade/insert/prune — never across an `.await`.
    fn turn_lock_for(&self, conversation_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut map = self.turn_locks.lock().expect("turn_locks mutex poisoned");
        // Opportunistic prune: drop entries whose Arc is gone. Bounded work —
        // the map only ever holds entries for concurrently-active conversations.
        map.retain(|_, weak| weak.strong_count() > 0);
        if let Some(existing) = map.get(conversation_id).and_then(Weak::upgrade) {
            return existing;
        }
        let arc = Arc::new(tokio::sync::Mutex::new(()));
        map.insert(conversation_id.to_string(), Arc::downgrade(&arc));
        arc
    }

    /// Test-only: current number of entries in the turn-lock map (#282), used to
    /// assert the weak-entry map does not grow unboundedly.
    #[cfg(test)]
    fn turn_lock_map_len(&self) -> usize {
        self.turn_locks.lock().unwrap().len()
    }

    /// Handle a `begin_step` / `complete_step` control call (#240).
    ///
    /// These are core-loop tools, not tool-executor tools: only the dispatch
    /// loop owns `conv.messages` (for eviction) and the per-turn [`StepStack`].
    /// `begin_step` pushes a step and records its goal as an ordered `todo`
    /// note; `complete_step` pops the step, writes the distilled outcome as a
    /// carry-forward note, marks the todo done, and evicts the step's raw tool
    /// results from working context (replacing them with a searchable pointer
    /// to the note). Returns the JSON ack the model sees as the tool result —
    /// for `begin_step` it carries the assigned dotted step number.
    ///
    /// Note writes are best-effort: a failed write is logged and the turn
    /// continues (the plan note is simply missing) rather than aborting.
    async fn handle_step_control(
        &self,
        conv: &mut Conversation,
        stack: &mut StepStack,
        call: &ToolCall,
        args: &serde_json::Value,
        conversation_id: &ConversationId,
    ) -> String {
        let Some(write) = self.scratchpad_write.clone() else {
            return r#"{"ok":false,"error":"planning is not available in this turn"}"#.to_string();
        };
        let conv_id = conversation_id.0.clone();

        if call.name == planning::BEGIN_STEP_TOOL {
            let goal = args
                .get("goal")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if goal.is_empty() {
                return r#"{"ok":false,"error":"begin_step requires a non-empty 'goal'"}"#
                    .to_string();
            }
            // Capture the scope start BEFORE this call's own ack is pushed, so
            // complete_step evicts the work done *within* the step.
            let watermark = conv.messages.len();
            let (key, sequence) = stack.begin(goal, watermark);
            let note = NewScratchpadNote {
                key: key.clone(),
                content: planning::truncate_on_char_boundary(goal, MAX_NOTE_BYTES),
                note_type: planning::STEP_NOTE_TYPE.to_string(),
                sequence: Some(sequence),
                done: false,
            };
            if let Err(e) = write(conv_id, vec![note]).await {
                tracing::warn!(step = %key, error = %e, "failed to record plan step note");
            }
            return serde_json::json!({
                "ok": true,
                "action": "begin_step",
                "step": key,
                "depth": stack.depth(),
                "goal": goal,
            })
            .to_string();
        }

        // complete_step
        let outcome = args
            .get("outcome")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let abandoned = args
            .get("status")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("abandoned"));

        let Some(frame) = stack.complete() else {
            // No active step to close. Still record a standalone note if the
            // model handed us an outcome, so the finding isn't lost.
            if let Some(o) = outcome {
                let key = format!("note-{}", (self.id_generator)());
                let body = if abandoned {
                    format!("Abandoned: {o}")
                } else {
                    o.to_string()
                };
                let note = NewScratchpadNote {
                    key: key.clone(),
                    content: planning::truncate_on_char_boundary(&body, MAX_NOTE_BYTES),
                    note_type: planning::OUTCOME_NOTE_TYPE.to_string(),
                    sequence: None,
                    done: false,
                };
                if let Err(e) = write(conv_id, vec![note]).await {
                    tracing::warn!(error = %e, "failed to record standalone outcome note");
                }
                return serde_json::json!({
                    "ok": true,
                    "action": "complete_step",
                    "note": "no active step; recorded a standalone note",
                    "outcome_note": key,
                })
                .to_string();
            }
            return r#"{"ok":true,"action":"complete_step","note":"no active step to complete"}"#
                .to_string();
        };

        // One write for the done-todo plus the optional carry-forward outcome.
        let mut notes = vec![NewScratchpadNote {
            key: frame.key.clone(),
            content: planning::truncate_on_char_boundary(&frame.goal, MAX_NOTE_BYTES),
            note_type: planning::STEP_NOTE_TYPE.to_string(),
            sequence: Some(frame.sequence),
            done: true,
        }];
        let mut note_keys: Vec<String> = Vec::new();
        if let Some(o) = outcome {
            let okey = format!("{}{}", planning::OUTCOME_KEY_PREFIX, frame.key);
            let body = if abandoned {
                format!("Abandoned: {o}")
            } else {
                o.to_string()
            };
            notes.push(NewScratchpadNote {
                key: okey.clone(),
                content: planning::truncate_on_char_boundary(&body, MAX_NOTE_BYTES),
                note_type: planning::OUTCOME_NOTE_TYPE.to_string(),
                sequence: None,
                done: false,
            });
            note_keys.push(okey);
        }
        if let Err(e) = write(conv_id, notes).await {
            tracing::warn!(step = %frame.key, error = %e, "failed to record step completion notes");
        }

        // Evict the step's raw tool results, leaving a pointer to the outcome
        // note. This is what stops the per-round `msg_chars` growth (#239).
        let (evicted, freed) =
            planning::evict_tool_results(&mut conv.messages, frame.watermark, &note_keys);
        tracing::info!(
            step = %frame.key,
            evicted_results = evicted,
            freed_bytes = freed,
            abandoned,
            "completed step — compacted scope to scratchpad"
        );

        serde_json::json!({
            "ok": true,
            "action": "complete_step",
            "step": frame.key,
            "status": if abandoned { "abandoned" } else { "done" },
            "evicted_results": evicted,
            "freed_bytes": freed,
            "outcome_note": note_keys.first(),
        })
        .to_string()
    }

    /// Render the open plan (#240) for per-round surfacing: read the
    /// conversation's notes and render the step tree with each completed step's
    /// finding nested under it (findings drop from view once a parent rolls them
    /// up). Marks the live step. Returns `None` when no lister is wired or there
    /// are no steps to show.
    async fn render_current_plan(
        &self,
        conversation_id: &ConversationId,
        current_key: Option<&str>,
    ) -> Option<String> {
        let list = self.scratchpad_list.clone()?;
        // Fetch all notes (todo steps + their outcome notes), a bit beyond the
        // render cap so a step's finding isn't dropped before the step itself.
        let notes = list(
            conversation_id.0.clone(),
            None,
            planning::MAX_PLAN_ITEMS.saturating_mul(3),
        )
        .await
        .ok()?;
        let raw: Vec<planning::RawNote> = notes
            .iter()
            .map(|n| planning::RawNote {
                key: n.key.as_str(),
                content: n.content.as_str(),
                note_type: n.note_type.as_str(),
                done: n.done,
            })
            .collect();
        planning::render_plan_from_notes(&raw, current_key, planning::MAX_PLAN_ITEMS)
    }

    /// Render the free-form scratchpad index (#340) for per-round surfacing: the
    /// keys of `note`-typed notes that aren't already shown as `[Current task]`
    /// (`goal`) or `[Plan]` (`outcome:*` findings and `todo` steps). These notes
    /// are durable in storage but otherwise invisible once the message that wrote
    /// them is windowed/compacted away, so the context builder advertises their
    /// keys (gated on the same "context is dropping" trigger as `[Current task]`)
    /// to remind the model what it can `builtin_scratchpad_search` for. Returns
    /// `None` when no lister is wired or there are no free-form notes.
    async fn render_current_scratchpad_index(
        &self,
        conversation_id: &ConversationId,
    ) -> Option<String> {
        let list = self.scratchpad_list.clone()?;
        // No type filter — `goal` and `outcome:*` are also `note`-typed, so the
        // free-form set is carved out by key in `freeform_note_keys`, not by a
        // storage-side type filter.
        let notes = list(
            conversation_id.0.clone(),
            None,
            planning::MAX_SCRATCHPAD_INDEX_KEYS.saturating_mul(3),
        )
        .await
        .ok()?;
        let raw: Vec<planning::RawNote> = notes
            .iter()
            .map(|n| planning::RawNote {
                key: n.key.as_str(),
                content: n.content.as_str(),
                note_type: n.note_type.as_str(),
                done: n.done,
            })
            .collect();
        let keys = planning::freeform_note_keys(&raw);
        planning::render_scratchpad_index(&keys, planning::MAX_SCRATCHPAD_INDEX_KEYS)
    }

    /// Build the per-turn [`StepStack`], seeding its top-level numbering from the
    /// conversation's existing `todo` notes (DA-7 / #292).
    ///
    /// Each turn used to start a fresh `StepStack` numbering from `"1"`, but the
    /// scratchpad `write` is upsert-by-key — so a second turn's step `"1"`
    /// silently overwrote the first turn's note (resetting its content and
    /// `done`). Seeding the root counter from the highest existing top-level key
    /// makes a new turn mint the next number instead. Without a lister wired (or
    /// on a read error), falls back to a fresh stack — the prior behaviour.
    async fn build_step_stack(&self, conversation_id: &ConversationId) -> StepStack {
        let Some(list) = self.scratchpad_list.clone() else {
            return StepStack::new();
        };
        // Only `todo`-typed notes are plan steps. Cap generously; only their
        // keys matter, and a conversation never accrues that many top-level
        // steps.
        match list(
            conversation_id.0.clone(),
            Some(planning::STEP_NOTE_TYPE.to_string()),
            planning::MAX_PLAN_ITEMS.saturating_mul(3),
        )
        .await
        {
            Ok(notes) => {
                let max = planning::max_top_level_key(notes.iter().map(|n| n.key.as_str()));
                StepStack::with_root_counter(max)
            }
            Err(_) => StepStack::new(),
        }
    }
}

impl<S, L: LlmClient, T> ConversationHandler<S, L, T> {
    /// Returns the backend-tasks LLM if configured, otherwise the primary LLM.
    fn task_llm(&self) -> &L {
        self.backend_llm.as_ref().unwrap_or(&self.llm)
    }
}

#[async_trait::async_trait]
impl<S: ConversationStore, L: LlmClient, T: ToolExecutor> ConversationService
    for ConversationHandler<S, L, T>
{
    async fn create_conversation(
        &self,
        title: String,
        tags: Vec<String>,
    ) -> Result<Conversation, CoreError> {
        let id = (self.id_generator)();
        let mut conv = Conversation::new(id, title);
        let timestamp = now_timestamp();
        conv.created_at = timestamp.clone();
        conv.updated_at = timestamp;
        conv.tags = tags;
        self.store.create(conv.clone()).await?;
        Ok(conv)
    }

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        // DS-6 (#295): `store.list()` already returns the light
        // `ConversationSummary` projection (no message bodies); this method
        // only filters and sorts it.
        let mut convs = self.store.list().await?;

        if !include_archived {
            convs.retain(|conv| !conv.archived);
        }

        if let Some(days) = max_age_days.filter(|days| *days > 0) {
            let cutoff = cutoff_timestamp(days);
            convs.retain(|conv| !conv.updated_at.is_empty() && conv.updated_at >= cutoff);
        }

        convs.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| left.title.cmp(&right.title))
                .then_with(|| left.id.0.cmp(&right.id.0))
        });

        Ok(convs)
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.store.get(id).await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.delete(id).await
    }

    async fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
    ) -> Result<(), CoreError> {
        // Rename is itself a whole-conversation read-modify-write (get → set
        // title → full `store.update`), so a rename racing an active turn would
        // load a stale snapshot and clobber the turn's messages. Take the same
        // per-conversation lock as `send_prompt` (#282); it's quick, so queueing
        // it behind a turn is invisible. `archive`/`unarchive`/`delete` don't
        // load-and-rewrite message rows, so they need no lock.
        let turn_lock = self.turn_lock_for(&id.0);
        let _turn_guard = turn_lock.lock().await;
        let mut conv = self.store.get(id).await?;
        conv.title = title;
        conv.updated_at = now_timestamp();
        self.store.update(conv).await
    }

    async fn archive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.archive(id).await
    }

    async fn unarchive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.unarchive(id).await
    }

    async fn clear_all_history(&self) -> Result<u32, CoreError> {
        let conversations = self.store.list().await?;
        let mut deleted = 0u32;

        for conversation in conversations {
            self.store.delete(&conversation.id).await?;
            deleted += 1;
        }

        Ok(deleted)
    }

    async fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
        mut on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        // Cooperative cancellation checkpoint (issue #109): bail out
        // before any I/O if the caller has already tripped the token.
        bail_if_cancelled()?;

        // Per-conversation turn serialization (#282). Concurrent turns on the
        // SAME conversation are a read-modify-write race: each does
        // `store.get` → mutate `conv.messages` → `store.update`, so a late
        // `update` clobbers a turn that completed in between, silently losing
        // its user prompt + reply. We serialize turn bodies per conversation
        // id by holding a per-conversation async mutex across the WHOLE turn.
        //
        // The guard is the first local, so RAII releases it on every return
        // path — `?`, error arms, and panics alike (no poisoning:
        // `tokio::sync::Mutex` guards are plain RAII). Different conversation
        // ids take different mutexes and never contend.
        //
        // INVARIANT (deadlock-freedom): a turn holding conversation X's lock
        // must never dispatch another turn to X. The only re-entrant turn path
        // is `spawn_subagent`, which always targets a FRESH child conversation
        // (lock order is strictly parent→fresh-child, acyclic). `begin_step` /
        // `complete_step` are handled inline in this dispatch loop and never
        // re-enter `send_prompt`.
        //
        // The wait itself is cancellable so a turn QUEUED behind a long
        // agentic turn can be cancelled while it waits (not just at the next
        // checkpoint): we `select!` the lock acquisition against the
        // cancellation token. Dropping the losing `lock()` future removes the
        // waiter from the mutex's FIFO queue without disturbing the running
        // turn.
        let turn_lock = self.turn_lock_for(&conversation_id.0);
        let _turn_guard = match current_cancellation_token() {
            Some(token) => {
                tokio::select! {
                    biased;
                    _ = token.cancelled() => return Err(CoreError::Cancelled),
                    guard = turn_lock.lock() => guard,
                }
            }
            None => turn_lock.lock().await,
        };
        // Re-check after acquiring: the wait may have been long (a multi-minute
        // agentic turn ahead of us), and the token may have tripped just as we
        // won the lock.
        bail_if_cancelled()?;

        // The chunk callback must survive every tool round: each round's
        // stream wrapper gets a proxy into this shared slot instead of
        // consuming the callback, so the final answer of a tool-calling turn
        // still streams (DA-9 — rounds after the first used to replace the
        // callback with a noop and stream nothing).
        let on_chunk: Arc<std::sync::Mutex<ChunkCallback>> =
            Arc::new(std::sync::Mutex::new(on_chunk));

        let mut conv = self.store.get(conversation_id).await?;
        let is_first_message = conv.messages.is_empty();
        // Stamp the client's idempotency key (#570 Phase 1b) onto this — the
        // single user-message persist site. Read from the task-local the
        // foreground dispatch wrapper installs; `None` for agent runs and any
        // caller not routing through that wrapper. Assistant rows pushed later
        // in this turn stay `None`.
        let mut user_msg = Message::new(Role::User, &prompt);
        user_msg.idempotency_key = crate::ports::llm::current_idempotency_key();
        conv.messages.push(user_msg);
        // Capture the prompt as the active-task anchor for this turn. It is
        // re-injected in `assemble_turn` when conditions indicate
        // the original message has drifted out of the model's view.
        conv.active_task = Some(prompt.clone());
        // Persist the user prompt eagerly, before any cancellable work (#585).
        // Otherwise the prompt lives only in memory until the terminal
        // `store.update`, and every cancellation checkpoint returns
        // `Err(Cancelled)` without saving — so a cancel/crash mid-turn would
        // lose the user's message. Writing it now — inside the turn-lock, so no
        // read-modify-write race (#282) — guarantees the prompt survives even if
        // the turn is abandoned; the eventual terminal update overwrites this
        // row with the full turn (prompt + reply). The clone is the cost of
        // keeping `conv` for the rest of the turn (one extra write per turn).
        self.store.update(conv.clone()).await?;

        // Effective window size for this turn. May shrink further if the
        // provider reports input-token usage above COMPACTION_TOKEN_RATIO.
        let mut target_window = MAX_CONTEXT_MESSAGES;

        // Count of in-turn ContextOverflow recoveries. Bounded so a
        // persistently-oversized request doesn't loop indefinitely.
        let mut overflow_retries: u32 = 0;

        // Run compaction if enough messages have been dropped by windowing.
        if let Some((from, to)) = compaction_range(&conv, target_window) {
            let summary = generate_context_summary(
                &conv.context_summary,
                &conv.messages[from..to],
                self.task_llm(),
            )
            .await;
            conv.context_summary = summary;
            conv.compacted_through = to;
        }

        // Dynamic tool discovery: start with core tools, activate more via tool_search.
        let use_hosted_search = self.llm.supports_hosted_tool_search();
        let namespaces: Vec<ToolNamespace> = if use_hosted_search {
            let raw_namespaces = self.tools.tool_namespaces().await;
            if raw_namespaces.is_empty() {
                vec![]
            } else {
                let hash = tool_set_hash(&raw_namespaces);
                // Fast path: a populated cache for this hash answers without
                // touching the single-flight guard, so steady-state turns never
                // serialize.
                let cached_hit = {
                    let cached = self.namespace_cache.lock().unwrap();
                    cached
                        .as_ref()
                        .filter(|(h, _)| *h == hash)
                        .map(|(_, ns)| ns.clone())
                };
                if let Some(ns) = cached_hit {
                    tracing::debug!(
                        hash,
                        namespace_count = ns.len(),
                        "tool categorization cache hit"
                    );
                    ns
                } else {
                    // Miss: take the single-flight guard so concurrent cold
                    // turns coalesce into one categorization LLM call (issue
                    // #305 item 8). Only the winner runs the call; losers wake,
                    // re-check the cache, and reuse its result.
                    let _flight = self.categorize_lock.lock().await;
                    // Double-check: a peer may have populated the cache for this
                    // hash while we waited for the guard.
                    let recheck = {
                        let cached = self.namespace_cache.lock().unwrap();
                        cached
                            .as_ref()
                            .filter(|(h, _)| *h == hash)
                            .map(|(_, ns)| ns.clone())
                    };
                    if let Some(ns) = recheck {
                        tracing::debug!(
                            hash,
                            namespace_count = ns.len(),
                            "tool categorization cache hit after single-flight wait"
                        );
                        ns
                    } else {
                        tracing::debug!(hash, "tool categorization cache miss; invoking LLM");
                        let result = categorize_tool_namespaces(
                            raw_namespaces,
                            self.task_llm(),
                            current_context_budget(),
                        )
                        .await;
                        *self.namespace_cache.lock().unwrap() = Some((hash, result.clone()));
                        result
                    }
                }
            }
        } else {
            vec![]
        };

        let core_tools = self.tools.core_tools().await;
        // When hosted search is active and we have namespaces, remove
        // builtin_tool_search from core tools — the provider handles discovery.
        let core_tools_for_llm: Vec<ToolDefinition> = if use_hosted_search && !namespaces.is_empty()
        {
            core_tools
                .iter()
                .filter(|t| t.name != "builtin_tool_search")
                .cloned()
                .collect()
        } else {
            core_tools.clone()
        };

        let mut activated_tools: std::collections::HashMap<String, ToolDefinition> =
            std::collections::HashMap::new();
        // Track whether hosted search has been demoted to local fallback.
        let mut hosted_search_demoted = false;

        // No turn-start filler. A quick/direct answer narrates nothing and just
        // streams its reply. Progress is narrated only when the model declares a
        // logical step (`begin_step`, in the dispatch loop below) — a step spans
        // multiple tool calls, so we narrate the step, not the turn start or each
        // tool. A slow turn that declares no step is covered by the voice
        // client's delayed-liveness safety net, not an unconditional status here.

        // Client-side tool execution (#107 / #234). When the connection
        // registered client-local tools, the application installs a per-turn
        // adapter as a task-local. Resolve it once: its `tool_definitions()`
        // are merged into every round's tool set so the LLM can pick them, and
        // a call to a registered name is routed through `port.execute(..)`
        // (which suspends the turn) instead of the server-side `ToolExecutor`.
        // Unset (no client tools registered, tests, background workers) leaves
        // the loop's behaviour exactly as before — every tool is server-side.
        let client_tool_port = current_client_tools();
        let client_tool_defs: Vec<ToolDefinition> = match &client_tool_port {
            Some(port) => port.tool_definitions().await,
            None => Vec::new(),
        };

        // Tool execution-locality context (issue #243, refined in #248).
        // Resolve the turn's co-location signal once: the authoritative
        // per-machine system-id match (#248) when the client reported an id, and
        // the connection's transport (UDS/D-Bus ⇒ same machine, WebSocket ⇒
        // possibly remote) as the fallback for older clients that sent none.
        // Plus the daemon's host label, an optional client-reported host label,
        // the server-side tool names, and the client-local tool names. The
        // tool-note builder uses it to tag each tool with where it runs and to
        // route a capability that exists on both the server and a remote client.
        // The server set is the full core set plus every namespaced tool —
        // activated tools are always a subset of these (they come from
        // server-side search), so a tool that isn't in this set is client-only.
        let server_tool_names: Vec<String> = core_tools
            .iter()
            .map(|t| t.name.clone())
            .chain(
                namespaces
                    .iter()
                    .flat_map(|ns| ns.tools.iter().map(|t| t.name.clone())),
            )
            .collect();
        // Prefer a client-reported host label for the remote tool note; fall
        // back to the generic "your device" when none was sent (#248).
        let client_label = current_client_label()
            .filter(|l| !l.trim().is_empty())
            .unwrap_or_else(|| "your device".to_string());
        let tool_locality = ToolLocalityContext {
            co_located: current_co_location(),
            transport: current_transport_kind(),
            host: self.host.clone(),
            client_label,
            server_tool_names,
            client_tool_names: client_tool_defs.iter().map(|d| d.name.clone()).collect(),
        };

        // Per-turn step stack for the planning + compaction tools (#240).
        // Frames hold watermarks into `conv.messages`; `complete_step` evicts a
        // scope's raw tool results down to a searchable scratchpad pointer.
        // Seeded from the conversation's existing `todo` keys so a later turn
        // continues the numbering instead of clobbering an earlier turn's note
        // via the scratchpad's upsert-by-key write (DA-7 / #292).
        let mut step_stack = self.build_step_stack(conversation_id).await;

        for round in 0..MAX_TOOL_ROUNDS {
            // Between-turns cancellation checkpoint (issue #109): if the
            // caller cancelled while the previous tool round was
            // executing, surface `Cancelled` before we dispatch the next
            // LLM call. This is the contract tested by
            // `send_prompt_returns_cancelled_when_token_fires_between_turns`
            // and `cancellation_during_tool_dispatch_aborts_before_next_llm_call`.
            bail_if_cancelled()?;

            // Build the tool set: core + dynamically activated.
            // When hosted search has been demoted, use the full core set
            // (which includes builtin_tool_search) instead of the filtered one.
            let mut tool_defs: Vec<ToolDefinition> = if hosted_search_demoted {
                core_tools.clone()
            } else {
                core_tools_for_llm.clone()
            };
            tool_defs.extend(activated_tools.values().cloned());
            // Offer the connection's registered client-local tools alongside
            // the server-side set so the LLM can invoke them (#234). Skip any
            // whose name already collides with a server-side tool — the
            // server-side definition wins to keep dispatch unambiguous.
            for def in &client_tool_defs {
                if !tool_defs.iter().any(|t| t.name == def.name) {
                    tool_defs.push(def.clone());
                }
            }
            // Advertise the step-planning + compaction tools (#240) when a
            // scratchpad writer is wired. They are core-loop tools — intercepted
            // by name in the dispatch loop below rather than routed to the tool
            // executor — so they're appended here, after the server/client sets,
            // every round. Without a writer wired they stay off entirely.
            if self.scratchpad_write.is_some() {
                tool_defs.push(planning::begin_step_tool());
                tool_defs.push(planning::complete_step_tool());
            }

            // Restrict the advertised tool set to the caller's allowlist
            // (issues #291 / #133) so a restricted subagent's LLM only ever
            // sees the tools it may use. `None` ⇒ no restriction; an empty
            // allowlist ⇒ no tools. The core-loop step-planning tools are
            // exempt — they're the loop's own control surface, not delegable
            // capabilities, and dispatch also re-checks the allowlist below.
            if let Some(allowed) = current_tool_allowlist() {
                tool_defs.retain(|t| {
                    t.name == planning::BEGIN_STEP_TOOL
                        || t.name == planning::COMPLETE_STEP_TOOL
                        || allowed.iter().any(|a| a == &t.name)
                });
            }

            let deferred_ns: &[ToolNamespace] = if !hosted_search_demoted {
                &namespaces
            } else {
                &[]
            };
            // `tool_rounds_since_anchor` doubles as "how many tool rounds
            // have we executed in this turn". Each completed round increments
            // the count, and the anchor was just (re)set at the start of
            // `send_prompt` — so this is exactly the round counter we want
            // to thread into the active-task injection check.
            let tool_rounds_since_anchor = u32::try_from(round).unwrap_or(u32::MAX);

            // Auto-surface the evolving goal. When a scratchpad goal reader is
            // wired, read the reserved `goal` note (a bounded single-key fetch)
            // and prefer it over the verbatim user prompt as the task anchor —
            // a model-maintained goal then keeps showing up even after history
            // is windowed/compacted away. Reading per round means a goal the
            // model wrote mid-turn surfaces on the next round.
            let goal = match &self.scratchpad_goal_read {
                Some(read) => read(
                    conversation_id.0.clone(),
                    vec![SCRATCHPAD_GOAL_KEY.to_string()],
                    1,
                )
                .await
                .ok()
                .and_then(|mut notes| notes.pop())
                .map(|note| note.content)
                .filter(|content| !content.trim().is_empty()),
                None => None,
            };
            let anchor = goal.as_deref().or(conv.active_task.as_deref());

            // Surface the open plan (#240): read the conversation's `todo`
            // notes and render them as a compact tree so the plan stays in view
            // across rounds while the expensive raw work is evicted. Reading per
            // round means a step the model just began or completed shows up
            // (with its updated done-mark) on the next round.
            let current_step = step_stack.current_key().map(str::to_string);
            let plan = self
                .render_current_plan(conversation_id, current_step.as_deref())
                .await;

            // Advertise the free-form scratchpad note keys (#340) so a note the
            // model stashed earlier survives windowing/compaction as recognition
            // (it can search for the key) even after the writing message is gone.
            // Gated context-builder-side on the same trigger as [Current task].
            let scratchpad_index = self.render_current_scratchpad_index(conversation_id).await;

            // The estimator borrows `&self.llm` so the closure is built
            // each iteration; constructing it is cheap (no allocation).
            let estimate = |text: &str| self.llm.estimate_tokens(text);
            let llm_messages = assemble_turn_within_budget(
                &ConversationView {
                    messages: &conv.messages,
                    summaries: &conv.summaries,
                    context_summary: &conv.context_summary,
                },
                &ToolContext {
                    tool_defs: &tool_defs,
                    deferred_namespaces: deferred_ns,
                    locality: Some(&tool_locality),
                },
                &TurnAnchors {
                    active_task: anchor,
                    plan: plan.as_deref(),
                    scratchpad_index: scratchpad_index.as_deref(),
                    tool_rounds_since_anchor,
                },
                target_window,
                current_context_budget(),
                &estimate,
            );
            // Incremental sanitizer: carries think-block parser state across
            // chunks so each byte is scanned once, instead of re-sanitizing
            // the full accumulated stream on every chunk (O(n²) per turn).
            let mut sanitizer = crate::sanitize::StreamSanitizer::new();
            let visible_chunk_callback = Arc::clone(&on_chunk);
            // Capture a clone of the per-turn cancellation token so the
            // wrapped callback can short-circuit mid-stream by returning
            // `false` — the contract LLM adapters already obey to abort
            // the SSE/NDJSON body. The adapter's own `tokio::select!`
            // against `token.cancelled()` is the primary signal; this
            // callback-side check covers callbacks that fire after the
            // adapter has already buffered a chunk but before the next
            // `select!` poll.
            let cancellation_token = cancellation_token_or_default();
            let filtered_chunk_callback: ChunkCallback = Box::new(move |chunk| {
                if cancellation_token.is_cancelled() {
                    return false;
                }
                let visible = sanitizer.push(&chunk);

                if visible.is_empty() {
                    true
                } else {
                    (visible_chunk_callback.lock().unwrap())(visible)
                }
            });

            // Reasoning config is threaded through a task-local set by
            // the daemon-side routing wrapper (`RoutingConversationHandler`)
            // before it calls `send_prompt`. In tests / standalone uses
            // with no wrapper, the slot is unset and we pass the default
            // empty config, matching the pre-issue-18 behaviour.
            let reasoning = crate::ports::llm::current_reasoning_config();

            let response = match if use_hosted_search
                && !namespaces.is_empty()
                && !hosted_search_demoted
            {
                self.llm
                    .stream_completion_with_namespaces(
                        llm_messages,
                        &tool_defs,
                        &namespaces,
                        reasoning,
                        filtered_chunk_callback,
                    )
                    .await
            } else {
                self.llm
                    .stream_completion(llm_messages, &tool_defs, reasoning, filtered_chunk_callback)
                    .await
            } {
                Ok(r) => r,
                Err(CoreError::ContextOverflow {
                    prompt_tokens,
                    max_tokens,
                    detail: _,
                }) if overflow_retries < MAX_OVERFLOW_RETRIES => {
                    // The provider rejected this turn's prompt for
                    // exceeding its context window. Run the recovery
                    // ladder (truncate large tool result → trim old
                    // pairs → summarise-and-shrink) and retry. The
                    // counter bounds total attempts across all steps
                    // so persistently-oversized requests can't loop.
                    overflow_retries += 1;
                    tracing::warn!(
                        attempt = overflow_retries,
                        max_attempts = MAX_OVERFLOW_RETRIES,
                        prompt_tokens = ?prompt_tokens,
                        max_tokens = ?max_tokens,
                        "context overflow — running recovery ladder"
                    );
                    recover_from_overflow(
                        &mut conv,
                        prompt_tokens,
                        max_tokens,
                        &mut target_window,
                        self.task_llm(),
                        &estimate,
                    )
                    .await;
                    // Overflow recovery can drain messages (trim_tool_pairs),
                    // which invalidates the step stack's absolute watermarks.
                    // Drop the frames so no later complete_step evicts the wrong
                    // range — the plan todos persist on the scratchpad regardless.
                    step_stack.clear();
                    continue;
                }
                Err(CoreError::Cancelled) => {
                    // Cancellation is the user's explicit signal to
                    // stop — surface it verbatim instead of converting
                    // to a friendly "LLM backend error" string. The
                    // partial assistant message is dropped on purpose:
                    // the user asked us to abandon this turn.
                    tracing::info!(
                        conversation_id = %conversation_id.0,
                        "send_prompt cancelled mid-stream"
                    );
                    return Err(CoreError::Cancelled);
                }
                Err(e) => {
                    // Anything else — including exhausted overflow
                    // retries — surfaces as a user-visible message.
                    // Non-context errors are no longer trimmed-and-prayed
                    // through old path C; that swallowed transient
                    // failures (rate limits, server errors, malformed
                    // tool calls) by mutating conversation state.
                    let friendly = user_visible_llm_error_message(&e);
                    conv.messages.push(Message::new(Role::Assistant, &friendly));
                    conv.updated_at = now_timestamp();
                    self.store.update(conv).await?;
                    return Ok(friendly);
                }
            };

            // Post-stream cancellation check (issue #109): the adapter
            // may have returned a partial response because the chunk
            // callback returned `false` after observing cancellation
            // (the cooperative-shutdown contract). In that case the
            // adapter returns `Ok(...)` with whatever it had streamed
            // so far, but we want to surface `Cancelled` to the caller
            // — the partial text is discarded.
            bail_if_cancelled()?;

            // Token-pressure check: if the provider reports input tokens
            // above COMPACTION_TOKEN_RATIO of its context window, shrink the
            // effective message window and compact the newly-dropped range
            // before building the next turn's prompt.
            //
            // The budget is resolved once at dispatch entry by the daemon's
            // routing wrapper (issue #63) and read here via the
            // `CONTEXT_BUDGET` task-local. When the slot is unset (test
            // contexts, background jobs that don't route through the
            // wrapper), token-based compaction skips — same behaviour as
            // when the connector previously returned `None` from
            // `max_context_tokens()`.
            if let (Some(budget), Some(usage)) = (current_context_budget(), response.usage.as_ref())
                && let Some(input_tokens) = usage.input_tokens
            {
                let max_tokens = budget.max_input_tokens;
                let threshold = (max_tokens as f64 * COMPACTION_TOKEN_RATIO) as u64;
                // Whether proactive compaction actually ran this turn — set
                // only when we both crossed the threshold AND were able to
                // shrink the window. Reported to clients so the indicator can
                // show that summarization is active (#341).
                let mut compaction_active = false;
                if input_tokens > threshold {
                    let new_window = (target_window / 2).max(MIN_CONTEXT_MESSAGES);
                    if new_window < target_window {
                        tracing::info!(
                            input_tokens,
                            max_tokens,
                            prev_window = target_window,
                            new_window,
                            "context pressure — shrinking window and compacting"
                        );
                        target_window = new_window;
                        if let Some((from, to)) = compaction_range(&conv, target_window) {
                            let summary = generate_context_summary(
                                &conv.context_summary,
                                &conv.messages[from..to],
                                self.task_llm(),
                            )
                            .await;
                            conv.context_summary = summary;
                            conv.compacted_through = to;
                        }
                        compaction_active = true;
                    } else {
                        tracing::debug!(
                            input_tokens,
                            max_tokens,
                            window = target_window,
                            "context pressure with window already at minimum"
                        );
                    }
                }
                // Surface the fill to subscribed clients (#341). Token counts
                // only — no message content crosses this boundary.
                crate::ports::llm::emit_context_usage(crate::ports::llm::ContextUsage {
                    used_tokens: input_tokens,
                    budget_tokens: max_tokens,
                    compaction_active,
                });
            }

            if !response.has_tool_calls() {
                // Hosted-search fallback: if the model returned text-only
                // while hosted search was active, it likely couldn't invoke
                // deferred tools.  Demote to local builtin_tool_search and
                // let the model try again with the classic tool-discovery path.
                if use_hosted_search
                    && !namespaces.is_empty()
                    && !hosted_search_demoted
                    && round < 2
                {
                    tracing::warn!(
                        round,
                        "hosted tool search produced no tool calls — \
                         falling back to builtin_tool_search"
                    );
                    hosted_search_demoted = true;
                    // Keep the assistant text so the model has context,
                    // then inject a system nudge to use builtin_tool_search.
                    if !response.text.is_empty() {
                        conv.messages
                            .push(Message::new(Role::Assistant, &response.text));
                    }
                    conv.messages.push(Message::new(
                        Role::System,
                        "The server-side tool search was unable to surface the \
                         tools you need. You now have access to `builtin_tool_search` \
                         — call it with a query describing what you need.",
                    ));
                    continue;
                }

                // Text-only response — we're done
                let mut visible_text = sanitize_assistant_text(&response.text);
                if visible_text.is_empty() {
                    tracing::warn!(
                        raw_len = response.text.len(),
                        raw_first_100 = %response.text.chars().take(100).collect::<String>(),
                        round,
                        "LLM returned empty visible text after sanitization"
                    );
                    if round > 0 {
                        visible_text =
                            "I wasn't able to complete this request — the tools I tried \
                             returned errors. Please check the conversation log or try again."
                                .to_string();
                    }
                }
                conv.messages
                    .push(Message::new(Role::Assistant, &visible_text));
                // On the first message, generate a descriptive title via the LLM
                // so the conversation list shows meaningful names rather than
                // timestamp-based placeholders.
                if is_first_message {
                    let generated = generate_conversation_title(&prompt, self.task_llm()).await;
                    if !generated.is_empty() {
                        conv.title = generated;
                    }
                }
                conv.updated_at = now_timestamp();
                self.store.update(conv).await?;
                return Ok(visible_text);
            }

            // LLM wants to call tools — record the assistant message with tool calls
            tracing::info!(
                "LLM requested {} tool call(s) (round {}/{})",
                response.tool_calls.len(),
                round + 1,
                MAX_TOOL_ROUNDS
            );
            conv.messages.push(Message::assistant_with_tool_calls(
                response.tool_calls.clone(),
            ));

            // Execute each tool call and append results
            for tool_call in &response.tool_calls {
                // Per-tool cancellation checkpoint (issue #109): if the
                // caller cancelled between tool dispatches we must stop
                // here rather than fire more tool side-effects. The
                // between-turns check above protects the next LLM
                // round; this one protects the inner per-tool loop.
                bail_if_cancelled()?;

                // Parse the model-supplied argument JSON. An empty string is
                // tolerated as "no arguments" (some providers emit it for
                // zero-arg calls), but otherwise-malformed JSON must NOT be
                // silently defaulted to `null` — the tool would run with
                // garbage arguments and the model would get a confusing
                // tool-specific error instead of the real cause (DA-13).
                let arguments: serde_json::Value = if tool_call.arguments.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    match serde_json::from_str(&tool_call.arguments) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!(
                                tool = %tool_call.name,
                                error = %e,
                                "tool call arguments were not valid JSON"
                            );
                            conv.messages.push(Message::tool_result(
                                &tool_call.id,
                                format!(
                                    "Error: the arguments for this tool call were not valid \
                                     JSON ({e}). Emit valid JSON and call the tool again."
                                ),
                            ));
                            continue;
                        }
                    }
                };
                tracing::info!(tool = %tool_call.name, %arguments, "executing tool");

                // Step-planning + compaction control (#240) is handled here in
                // the loop, not by the tool executor: only the loop owns
                // `conv.messages` (for eviction) and the per-turn step stack.
                // Every tool call still needs a tool_result for provider
                // pairing, so we push the (small) ack and move to the next call.
                // Gate on the same condition that advertises these tools, so when
                // planning is off the names aren't shadowed (an MCP tool could
                // otherwise share one) and dispatch falls through as normal.
                if self.scratchpad_write.is_some()
                    && (tool_call.name == planning::BEGIN_STEP_TOOL
                        || tool_call.name == planning::COMPLETE_STEP_TOOL)
                {
                    // Step-level narration: announce the logical step the model
                    // just declared, once, as its goal. This is the only progress
                    // narration now (turn-start filler and per-tool chatter were
                    // removed); a step spans multiple tool calls. complete_step
                    // stays silent.
                    if tool_call.name == planning::BEGIN_STEP_TOOL
                        && let Some(goal) = arguments.get("goal").and_then(|v| v.as_str())
                    {
                        let goal = goal.trim();
                        if !goal.is_empty() {
                            on_status(goal.to_string());
                        }
                    }
                    let ack = self
                        .handle_step_control(
                            &mut conv,
                            &mut step_stack,
                            tool_call,
                            &arguments,
                            conversation_id,
                        )
                        .await;
                    conv.messages
                        .push(Message::tool_result(&tool_call.id, &ack));
                    continue;
                }

                // Tool allowlist enforcement (issues #291 / #133). A subagent
                // (or any caller) may install a `TOOL_ALLOWLIST` task-local
                // restricting which tools it can invoke. The allowlist is also
                // applied at advertisement time below (the LLM only sees the
                // permitted set), but enforce it here at the dispatch
                // chokepoint too: a call to a non-allowlisted name — whether the
                // model hallucinated it or it leaked in from history — is
                // rejected with a recoverable error folded into the tool_result,
                // and no executor runs. `None` means "no restriction"; an empty
                // allowlist means "no tools". The core-loop step-planning tools
                // handled above are intentionally exempt (they aren't real tool
                // work and were never advertised through the allowlist).
                if let Some(allowed) = current_tool_allowlist()
                    && !allowed.iter().any(|t| t == &tool_call.name)
                {
                    tracing::warn!(
                        tool = %tool_call.name,
                        "tool call rejected: not on the subagent's allowlist"
                    );
                    let rejection = format!(
                        "Error: the tool '{}' is not permitted for this subagent — it is not \
                         on the configured tool allowlist. Choose a tool from your available \
                         set, or answer without it.",
                        tool_call.name
                    );
                    notify_tool_event(ToolEvent::Started {
                        name: tool_call.name.clone(),
                        args: summarize_tool_value(&arguments),
                    });
                    notify_tool_event(ToolEvent::Finished {
                        name: tool_call.name.clone(),
                        ok: false,
                        output: "rejected: not on allowlist".to_string(),
                    });
                    conv.messages
                        .push(Message::tool_result(&tool_call.id, &rejection));
                    continue;
                }

                // Report the call to any installed tool observer (the task
                // panel's activity feed). Emitted here — after the step-control
                // fast path, before either execution branch — so it covers real
                // tool work (server-side and client-local alike) exactly once.
                notify_tool_event(ToolEvent::Started {
                    name: tool_call.name.clone(),
                    args: summarize_tool_value(&arguments),
                });

                // Route client-local tools to the client (#107 / #234): if a
                // per-turn client-tool port is installed and the called name is
                // registered for this user, suspend the turn and await the
                // client's result instead of running a server-side executor.
                // A registered client tool whose name collides with a
                // server-side one never reaches here — the tool-set merge above
                // gives the server-side definition precedence, so the LLM was
                // offered (and called) the server-side tool.
                let client_exec = match &client_tool_port {
                    Some(port) if port.is_registered(&tool_call.name).await => Some(port),
                    _ => None,
                };

                // `tool_ok` is tracked alongside the result so the observer can
                // distinguish a successful call from an error the loop folds
                // into the tool result (and keeps looping on).
                let (result, tool_ok) = if let Some(port) = client_exec {
                    match port
                        .execute(&tool_call.id, &tool_call.name, arguments)
                        .await
                    {
                        Ok(output) => {
                            tracing::debug!(tool = %tool_call.name, output = %output, "client tool result");
                            (output, true)
                        }
                        // Cancellation while a client tool was suspended (e.g.
                        // the user pressed Cancel) must abort the turn, not be
                        // folded into a tool result the LLM would keep looping
                        // on. The observer already saw `Started` above; emit a
                        // matching `Finished{ok:false}` before the early return
                        // so the activity feed never strands a started-but-never
                        // -finished row on the cancel path (issue #252).
                        Err(CoreError::Cancelled) => {
                            notify_tool_event(ToolEvent::Finished {
                                name: tool_call.name.clone(),
                                ok: false,
                                output: "cancelled".to_string(),
                            });
                            return Err(CoreError::Cancelled);
                        }
                        Err(e) => {
                            tracing::warn!(tool = %tool_call.name, error = %e, "client tool execution failed");
                            (format!("Error: {e}"), false)
                        }
                    }
                } else {
                    // Install the conversation as a task-local for the duration
                    // of tool execution so conversation-scoped builtins (the
                    // scratchpad) can resolve which pad they operate on without
                    // the `ToolExecutor` port growing a conversation parameter.
                    let exec = self.tools.execute_tool(&tool_call.name, arguments);
                    let exec = with_conversation_id(conversation_id.clone(), exec);
                    // Keepalive during long server-side tool execution (#584): a
                    // tool — or a subagent, which runs as a tool — can execute
                    // silently for longer than the client's 90s stall watchdog,
                    // which would then false-abandon a turn the daemon is still
                    // servicing. Emit a periodic status so the client's watchdog
                    // keeps resetting. Client tools don't need this (their
                    // suspension parks the watchdog via the `ClientToolCall`
                    // event). Cancellation is unaffected: the pinned `exec` future
                    // still resolves `Cancelled` and breaks the loop.
                    tokio::pin!(exec);
                    let outcome = loop {
                        tokio::select! {
                            r = &mut exec => break r,
                            _ = tokio::time::sleep(SERVER_TOOL_KEEPALIVE_INTERVAL) => {
                                on_status(format!("Still working on {}", tool_call.name));
                            }
                        }
                    };
                    match outcome {
                        Ok(output) => {
                            tracing::debug!(tool = %tool_call.name, output = %output, "tool result");
                            (output, true)
                        }
                        Err(e) => {
                            tracing::warn!(tool = %tool_call.name, error = %e, "tool execution failed");
                            (format!("Error: {e}"), false)
                        }
                    }
                };

                // Cap the result at ingestion (issue #174): a runaway tool can
                // return a multi-megabyte payload that, stored verbatim, wedges
                // the conversation against the model's context window on every
                // later turn and stalls the messages INSERT. Truncate with a
                // notice so the model still sees what ran and how to narrow it.
                // Computed before the `Finished` event so the activity feed
                // mirrors exactly what the model is shown (issue #257), rather
                // than summarizing a pre-cap payload the turn never used.
                let stored = match cap_tool_result(&result, self.max_tool_result_bytes) {
                    Some(truncated) => {
                        tracing::warn!(
                            tool = %tool_call.name,
                            original_bytes = result.len(),
                            kept_bytes = truncated.len(),
                            cap_bytes = self.max_tool_result_bytes,
                            "tool result exceeded the ingestion cap — truncated"
                        );
                        truncated
                    }
                    None => result.clone(),
                };

                notify_tool_event(ToolEvent::Finished {
                    name: tool_call.name.clone(),
                    ok: tool_ok,
                    output: summarize_tool_text(&stored),
                });

                // Dynamic activation: if tool_search returned results,
                // activate the discovered tools for subsequent rounds.
                // Skip when hosted search is active (unless demoted to local fallback).
                if (!use_hosted_search || hosted_search_demoted)
                    && tool_call.name == "builtin_tool_search"
                    && let Ok(found) = serde_json::from_str::<serde_json::Value>(&result)
                    && let Some(tools_arr) = found.get("tools").and_then(|v| v.as_array())
                {
                    for tool_entry in tools_arr {
                        if let Some(name) = tool_entry.get("name").and_then(|v| v.as_str())
                            && !activated_tools.contains_key(name)
                            && !core_tools.iter().any(|t| t.name == name)
                            && let Ok(Some(def)) = self.tools.tool_definition(name).await
                        {
                            tracing::info!("dynamically activated tool: {}", def.name);
                            activated_tools.insert(def.name.clone(), def);
                        }
                    }
                }

                // When hosted search is active and the model calls a
                // deferred namespace tool, activate the entire namespace
                // so full schemas are available in subsequent rounds.
                if use_hosted_search
                    && !hosted_search_demoted
                    && !activated_tools.contains_key(&tool_call.name)
                    && !core_tools.iter().any(|t| t.name == tool_call.name)
                {
                    for ns in &namespaces {
                        if ns.tools.iter().any(|t| t.name == tool_call.name) {
                            for t in &ns.tools {
                                if !activated_tools.contains_key(&t.name)
                                    && !core_tools.iter().any(|ct| ct.name == t.name)
                                {
                                    tracing::info!(
                                        "activated deferred tool from namespace {:?}: {}",
                                        ns.name,
                                        t.name
                                    );
                                    activated_tools.insert(t.name.clone(), t.clone());
                                }
                            }
                            break;
                        }
                    }
                }

                conv.messages
                    .push(Message::tool_result(&tool_call.id, &stored));
            }
        }

        // #453: the tool-round budget is spent. Rather than returning an error
        // and dropping the entire turn (the user's prompt plus every tool
        // round), do a bounded, tool-free wind-down: ask the model — in full
        // context, with NO tools offered — for a fluent closing that says what
        // it got done, what's left, and how to continue. Then persist the turn
        // so it can be picked up later. A canned message is the fallback if
        // that final call fails or returns nothing, so the turn is never lost.
        tracing::warn!(
            conversation_id = %conversation_id.0,
            max_rounds = MAX_TOOL_ROUNDS,
            "tool-round budget exhausted — winding down and persisting the turn"
        );
        bail_if_cancelled()?;

        // Recompute the light task anchors so the wind-down prompt carries the
        // same [Current task]/[Plan] context the loop rounds did.
        let goal = match &self.scratchpad_goal_read {
            Some(read) => read(
                conversation_id.0.clone(),
                vec![SCRATCHPAD_GOAL_KEY.to_string()],
                1,
            )
            .await
            .ok()
            .and_then(|mut notes| notes.pop())
            .map(|note| note.content)
            .filter(|content| !content.trim().is_empty()),
            None => None,
        };
        let wind_down_anchor = goal
            .as_deref()
            .or(conv.active_task.as_deref())
            .map(str::to_string);
        let wind_down_plan = self.render_current_plan(conversation_id, None).await;
        let wind_down_index = self.render_current_scratchpad_index(conversation_id).await;

        // Show the model a transient wrap-up instruction for THIS call only,
        // then drop it so only its closing reply is persisted.
        conv.messages
            .push(Message::new(Role::User, WIND_DOWN_INSTRUCTION));
        let wind_down_messages = {
            let estimate = |text: &str| self.llm.estimate_tokens(text);
            assemble_turn_within_budget(
                &ConversationView {
                    messages: &conv.messages,
                    summaries: &conv.summaries,
                    context_summary: &conv.context_summary,
                },
                &ToolContext {
                    tool_defs: &[],
                    deferred_namespaces: &[],
                    locality: None,
                },
                &TurnAnchors {
                    active_task: wind_down_anchor.as_deref(),
                    plan: wind_down_plan.as_deref(),
                    scratchpad_index: wind_down_index.as_deref(),
                    tool_rounds_since_anchor: u32::MAX,
                },
                target_window,
                current_context_budget(),
                &estimate,
            )
        };
        conv.messages.pop(); // the transient instruction is never persisted

        // Stream the closing through the shared callback, sanitizing think
        // blocks and honoring cancellation exactly like a normal round.
        let mut wind_down_sanitizer = crate::sanitize::StreamSanitizer::new();
        let wind_down_callback_slot = Arc::clone(&on_chunk);
        let wind_down_token = cancellation_token_or_default();
        let wind_down_stream: ChunkCallback = Box::new(move |chunk| {
            if wind_down_token.is_cancelled() {
                return false;
            }
            let visible = wind_down_sanitizer.push(&chunk);
            if visible.is_empty() {
                true
            } else {
                (wind_down_callback_slot.lock().unwrap())(visible)
            }
        });
        let reasoning = crate::ports::llm::current_reasoning_config();
        let closing = match self
            .llm
            .stream_completion(wind_down_messages, &[], reasoning, wind_down_stream)
            .await
        {
            Ok(response) => {
                let visible = sanitize_assistant_text(&response.text);
                if visible.trim().is_empty() {
                    WIND_DOWN_FALLBACK.to_string()
                } else {
                    visible
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "wind-down completion failed; using canned closing");
                WIND_DOWN_FALLBACK.to_string()
            }
        };

        // Persist the whole turn: prompt + tool transcript + closing.
        conv.messages.push(Message::new(Role::Assistant, &closing));
        if is_first_message {
            let generated = generate_conversation_title(&prompt, self.task_llm()).await;
            if !generated.is_empty() {
                conv.title = generated;
            }
        }
        conv.updated_at = now_timestamp();
        self.store.update(conv).await?;
        Ok(closing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MIN_TRUNCATION_TOKENS;
    use crate::domain::{ToolCall, ToolDefinition, TransportKind};
    use crate::ports::llm::{BudgetSource, ContextBudget, LlmResponse, TokenUsage};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    // --- Mock Store ---
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
            _conversation_id: &ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("mock-summary".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    // --- Mock LLM ---
    struct MockLlm {
        response_chunks: Vec<String>,
    }

    impl MockLlm {
        fn new(chunks: Vec<&str>) -> Self {
            Self {
                response_chunks: chunks.into_iter().map(String::from).collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for MockLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let mut full = String::new();
            for chunk in &self.response_chunks {
                full.push_str(chunk);
                if !on_chunk(chunk.clone()) {
                    return Ok(LlmResponse::text(full));
                }
            }
            Ok(LlmResponse::text(full))
        }
    }

    fn make_handler(chunks: Vec<&str>) -> ConversationHandler<MockStore, MockLlm> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::new(
            MockStore::new(),
            MockLlm::new(chunks),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    fn noop_callback() -> ChunkCallback {
        Box::new(|_| true)
    }

    fn noop_status() -> StatusCallback {
        Box::new(|_| {})
    }

    /// A [`StatusCallback`] that records every emitted status message into the
    /// returned shared buffer, so a test can assert what the turn emitted.
    fn recording_status() -> (StatusCallback, Arc<std::sync::Mutex<Vec<String>>>) {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_for_cb = Arc::clone(&log);
        let cb: StatusCallback = Box::new(move |msg| log_for_cb.lock().unwrap().push(msg));
        (cb, log)
    }

    struct ListOnlyStore {
        conversations: Vec<Conversation>,
    }

    impl ConversationStore for ListOnlyStore {
        async fn create(&self, _conv: Conversation) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get(&self, _id: &ConversationId) -> Result<Conversation, CoreError> {
            Err(CoreError::ConversationNotFound("unused".to_string()))
        }

        async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(self
                .conversations
                .iter()
                .map(ConversationSummary::from)
                .collect())
        }

        async fn update(&self, _conv: Conversation) -> Result<(), CoreError> {
            Ok(())
        }

        async fn delete(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn archive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn unarchive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn create_summary(
            &self,
            _conversation_id: &ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("mock-summary".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn create_assigns_unique_ids() {
        let handler = make_handler(vec![]);
        let c1 = handler
            .create_conversation("A".into(), vec![])
            .await
            .unwrap();
        let c2 = handler
            .create_conversation("B".into(), vec![])
            .await
            .unwrap();
        assert_ne!(c1.id, c2.id);
        assert_eq!(c1.id.as_str(), "conv-1");
        assert_eq!(c2.id.as_str(), "conv-2");
    }

    #[tokio::test]
    async fn create_sets_human_readable_timestamps() {
        let handler = make_handler(vec![]);
        let conv = handler
            .create_conversation("A".into(), vec![])
            .await
            .unwrap();
        assert!(!conv.created_at.is_empty());
        assert!(!conv.updated_at.is_empty());
        assert_eq!(conv.created_at.len(), 19);
        assert_eq!(conv.updated_at.len(), 19);
        assert_eq!(conv.created_at, conv.updated_at);
    }

    #[tokio::test]
    async fn create_stores_conversation() {
        let handler = make_handler(vec![]);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let retrieved = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(retrieved.title, "Test");
    }

    #[tokio::test]
    async fn list_returns_summaries() {
        let handler = make_handler(vec![]);
        handler
            .create_conversation("A".into(), vec![])
            .await
            .unwrap();
        handler
            .create_conversation("B".into(), vec![])
            .await
            .unwrap();

        let summaries = handler.list_conversations(None, false).await.unwrap();
        assert_eq!(summaries.len(), 2);
        for s in &summaries {
            assert_eq!(s.message_count, 0);
        }
    }

    #[tokio::test]
    async fn list_filters_by_age_and_sorts_descending() {
        let now = Local::now();

        let mut old_conv = Conversation::new("old", "Old");
        old_conv.created_at = (now - Duration::days(30))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        old_conv.updated_at = old_conv.created_at.clone();

        let mut newer_conv = Conversation::new("newer", "Newer");
        newer_conv.created_at = (now - Duration::days(2))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        newer_conv.updated_at = newer_conv.created_at.clone();

        let mut newest_conv = Conversation::new("newest", "Newest");
        newest_conv.created_at = (now - Duration::hours(1))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        newest_conv.updated_at = newest_conv.created_at.clone();

        let handler = ConversationHandler::new(
            ListOnlyStore {
                conversations: vec![old_conv, newer_conv, newest_conv],
            },
            MockLlm::new(vec![]),
            Box::new(|| "unused".to_string()),
        );

        let filtered = handler.list_conversations(Some(7), false).await.unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id.as_str(), "newest");
        assert_eq!(filtered[1].id.as_str(), "newer");
    }

    #[tokio::test]
    async fn delete_removes_conversation() {
        let handler = make_handler(vec![]);
        let conv = handler
            .create_conversation("Gone".into(), vec![])
            .await
            .unwrap();
        handler.delete_conversation(&conv.id).await.unwrap();

        let result = handler.get_conversation(&conv.id).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn clear_all_history_removes_all_conversations() {
        let handler = make_handler(vec![]);
        handler
            .create_conversation("A".into(), vec![])
            .await
            .unwrap();
        handler
            .create_conversation("B".into(), vec![])
            .await
            .unwrap();

        let deleted = handler.clear_all_history().await.unwrap();
        assert_eq!(deleted, 2);

        let summaries = handler.list_conversations(None, false).await.unwrap();
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn send_prompt_adds_messages_to_history() {
        let handler = make_handler(vec!["Hello", " there"]);
        let conv = handler
            .create_conversation("Chat".into(), vec![])
            .await
            .unwrap();

        let response = handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(response, "Hello there");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 2);
        assert_eq!(updated.messages[0].role, Role::User);
        assert_eq!(updated.messages[0].content, "Hi");
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert_eq!(updated.messages[1].content, "Hello there");
    }

    #[tokio::test]
    async fn send_prompt_streams_chunks() {
        let handler = make_handler(vec!["a", "b", "c"]);
        let conv = handler
            .create_conversation("Chat".into(), vec![])
            .await
            .unwrap();

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);
        let response = handler
            .send_prompt(
                &conv.id,
                "test".into(),
                Box::new(move |chunk| {
                    chunks_clone.lock().unwrap().push(chunk);
                    true
                }),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(response, "abc");
        assert_eq!(*chunks.lock().unwrap(), vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn send_prompt_hides_thinking_blocks_in_final_response() {
        let handler = make_handler(vec!["<think>internal reasoning</think>\n\nVisible answer"]);
        let conv = handler
            .create_conversation("Chat".into(), vec![])
            .await
            .unwrap();

        let response = handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(response, "Visible answer");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert_eq!(updated.messages[1].content, "Visible answer");
    }

    #[tokio::test]
    async fn send_prompt_hides_thinking_blocks_in_streamed_chunks() {
        let handler = make_handler(vec!["Visible ", "<th", "ink>internal</think>", "answer"]);
        let conv = handler
            .create_conversation("Chat".into(), vec![])
            .await
            .unwrap();

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);
        let response = handler
            .send_prompt(
                &conv.id,
                "Hi".into(),
                Box::new(move |chunk| {
                    chunks_clone.lock().unwrap().push(chunk);
                    true
                }),
                noop_status(),
            )
            .await
            .unwrap();

        assert_eq!(response, "Visible answer");
        assert_eq!(*chunks.lock().unwrap(), vec!["Visible ", "answer"]);
    }

    #[test]
    fn sanitize_assistant_text_handles_unclosed_think_block() {
        let input = "Visible before <think>internal";
        let output = sanitize_assistant_text(input);
        assert_eq!(output, "Visible before");
    }

    #[tokio::test]
    async fn send_prompt_nonexistent_conversation_fails() {
        let handler = make_handler(vec![]);
        let result = handler
            .send_prompt(
                &ConversationId::from("nope"),
                "hi".into(),
                noop_callback(),
                noop_status(),
            )
            .await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    // --- Tool calling tests ---

    /// Mock LLM that returns tool calls on first invocation, then text.
    struct ToolCallingLlm {
        /// Responses to return in sequence. Each call to stream_completion
        /// pops the first response.
        responses: Mutex<Vec<LlmResponse>>,
    }

    impl ToolCallingLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for ToolCallingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Ok(LlmResponse::text("fallback"));
                }
                responses.remove(0)
            };
            // Stream any text content
            if !response.text.is_empty() {
                on_chunk(response.text.clone());
            }
            Ok(response)
        }
    }

    /// Mock tool executor that returns predictable results.
    struct MockToolExecutor {
        tools: Vec<ToolDefinition>,
        results: Mutex<HashMap<String, String>>,
    }

    impl MockToolExecutor {
        fn new(tools: Vec<ToolDefinition>, results: HashMap<String, String>) -> Self {
            Self {
                tools,
                results: Mutex::new(results),
            }
        }
    }

    impl ToolExecutor for MockToolExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
        }

        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }

        async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(self.tools.iter().find(|t| t.name == name).cloned())
        }

        async fn execute_tool(
            &self,
            name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            self.results
                .lock()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(|| CoreError::ToolExecution(format!("unknown tool: {name}")))
        }
    }

    // #584: a tool executor that sleeps a configurable duration, to test the
    // keepalive emitted during long server-side tool execution.
    struct SlowToolExecutor {
        tools: Vec<ToolDefinition>,
        result: String,
        delay: std::time::Duration,
    }

    impl ToolExecutor for SlowToolExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
        }
        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }
        async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(self.tools.iter().find(|t| t.name == name).cloned())
        }
        async fn execute_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            tokio::time::sleep(self.delay).await;
            Ok(self.result.clone())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn long_server_tool_emits_keepalive_status_within_stall_window() {
        // #584: a server-side tool that runs longer than the keepalive interval
        // must emit periodic status keepalives so the client's 90s stall watchdog
        // does not false-abandon a turn the daemon is still servicing. (Subagents
        // run as a tool, so this also covers "actively working in the background".)
        use std::sync::atomic::{AtomicU64, Ordering};
        let tools = vec![ToolDefinition::new(
            "slow",
            "slow tool",
            serde_json::json!({"type": "object"}),
        )];
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "slow", "{}")]),
            LlmResponse::text("done"),
        ];
        let executor = SlowToolExecutor {
            tools,
            result: "ok".to_string(),
            // Longer than several keepalive intervals.
            delay: std::time::Duration::from_secs(120),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(responses),
            executor,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("c".into(), vec![])
            .await
            .unwrap();
        let (status_cb, status_log) = recording_status();
        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), status_cb)
            .await
            .expect("turn completes");
        let statuses = status_log.lock().unwrap();
        let keepalives = statuses
            .iter()
            .filter(|s| s.contains("Still working"))
            .count();
        assert!(
            keepalives >= 2,
            "a long server tool must emit periodic keepalive statuses; got: {statuses:?}"
        );
    }

    fn make_tool_handler(
        responses: Vec<LlmResponse>,
        tools: Vec<ToolDefinition>,
        tool_results: HashMap<String, String>,
    ) -> ConversationHandler<MockStore, ToolCallingLlm, MockToolExecutor> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(responses),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    #[tokio::test]
    async fn tool_loop_executes_tool_and_returns_final_text() {
        let tool_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let tool_call = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/test"}"#);

        let responses = vec![
            // First: LLM requests a tool call
            LlmResponse::with_tool_calls("", vec![tool_call]),
            // Second: LLM returns final text after seeing tool result
            LlmResponse::text("The file contains: hello world"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "hello world".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Read /tmp/test".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "The file contains: hello world");

        // Verify conversation history has all messages
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 4);
        assert_eq!(updated.messages[0].role, Role::User);
        assert_eq!(updated.messages[1].role, Role::Assistant); // tool call request
        assert!(!updated.messages[1].tool_calls.is_empty());
        assert_eq!(updated.messages[2].role, Role::Tool); // tool result
        assert_eq!(updated.messages[2].content, "hello world");
        assert_eq!(updated.messages[3].role, Role::Assistant); // final response
        assert_eq!(
            updated.messages[3].content,
            "The file contains: hello world"
        );
    }

    // --- TOOL_ALLOWLIST dispatch enforcement (issue #291 / #133) --------
    //
    // The `TOOL_ALLOWLIST` task-local (#113) is read at the dispatch
    // chokepoint: a tool call whose name is NOT on the allowlist is
    // rejected with a recoverable tool_result error and the executor is
    // never invoked. `None` means "no restriction"; an empty allowlist
    // means "no tools".

    #[tokio::test]
    async fn dispatch_rejects_tool_not_on_allowlist() {
        // A subagent is given `tools: ["read_file"]` but the LLM tries to
        // call `delete_file`. The call must be rejected with a recoverable
        // error folded into the tool_result, and the executor must NOT run
        // the disallowed tool (it would have returned "boom" if it had).
        let read_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let delete_def = ToolDefinition::new(
            "delete_file",
            "Delete a file",
            serde_json::json!({"type": "object"}),
        );
        let bad_call = ToolCall::new("call-1", "delete_file", r#"{"path": "/etc/passwd"}"#);

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![bad_call]),
            LlmResponse::text("done"),
        ];
        let mut tool_results = HashMap::new();
        // If dispatch wrongly executes the disallowed tool, this is what it
        // would return — its absence from the history proves enforcement.
        tool_results.insert("delete_file".to_string(), "boom".to_string());

        let handler = make_tool_handler(responses, vec![read_def, delete_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = crate::ports::llm::with_tool_allowlist(
            vec!["read_file".to_string()],
            handler.send_prompt(
                &conv.id,
                "delete the file".into(),
                noop_callback(),
                noop_status(),
            ),
        )
        .await
        .unwrap();
        assert_eq!(result, "done");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool_result must be recorded for the rejected call");
        assert!(
            tool_msg.content.contains("not permitted")
                || tool_msg.content.to_lowercase().contains("not allowed"),
            "rejection text should explain the tool is not on the allowlist, got: {}",
            tool_msg.content
        );
        assert!(
            tool_msg.content.contains("delete_file"),
            "rejection should name the disallowed tool, got: {}",
            tool_msg.content
        );
        assert!(
            !tool_msg.content.contains("boom"),
            "the disallowed tool must NOT have executed, got: {}",
            tool_msg.content
        );
    }

    #[tokio::test]
    async fn dispatch_allows_tool_on_allowlist() {
        // Baseline: an allowed tool dispatches normally under an allowlist.
        let read_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let good_call = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/ok"}"#);
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![good_call]),
            LlmResponse::text("read it"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "hello world".to_string());

        let handler = make_tool_handler(responses, vec![read_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = crate::ports::llm::with_tool_allowlist(
            vec!["read_file".to_string()],
            handler.send_prompt(&conv.id, "read it".into(), noop_callback(), noop_status()),
        )
        .await
        .unwrap();
        assert_eq!(result, "read it");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool_result must be recorded");
        assert_eq!(
            tool_msg.content, "hello world",
            "an allowlisted tool must execute normally"
        );
    }

    #[tokio::test]
    async fn dispatch_empty_allowlist_rejects_every_tool() {
        // An empty allowlist (distinct from None) means "no tools": every
        // tool call is rejected.
        let read_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let call = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/ok"}"#);
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![call]),
            LlmResponse::text("done"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "hello world".to_string());

        let handler = make_tool_handler(responses, vec![read_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        // Re-run under an empty allowlist via a fresh conversation/handler so
        // the assertion is unambiguous.
        let read_def2 = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let call2 = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/ok"}"#);
        let responses2 = vec![
            LlmResponse::with_tool_calls("", vec![call2]),
            LlmResponse::text("done"),
        ];
        let mut tr2 = HashMap::new();
        tr2.insert("read_file".to_string(), "hello world".to_string());
        let handler2 = make_tool_handler(responses2, vec![read_def2], tr2);
        let conv2 = handler2
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        crate::ports::llm::with_tool_allowlist(
            Vec::new(),
            handler2.send_prompt(&conv2.id, "go".into(), noop_callback(), noop_status()),
        )
        .await
        .unwrap();

        let updated = handler2.get_conversation(&conv2.id).await.unwrap();
        let tool_msg = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool_result must be recorded for the rejected call");
        assert!(
            !tool_msg.content.contains("hello world"),
            "an empty allowlist must reject every tool, got: {}",
            tool_msg.content
        );
        assert!(
            tool_msg.content.contains("not permitted")
                || tool_msg.content.to_lowercase().contains("not allowed"),
            "rejection text expected, got: {}",
            tool_msg.content
        );
    }

    #[tokio::test]
    async fn final_answer_streams_after_a_tool_round() {
        // DA-9: the user-facing chunk callback must keep streaming after the
        // first tool round — the final answer of a tool-calling turn used to
        // stream nothing because later rounds replaced the callback with a
        // noop.
        let tool_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "read_file", "{}")]),
            LlmResponse::text("final answer after tools"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "data".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let streamed = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&streamed);
        let cb: ChunkCallback = Box::new(move |chunk| {
            sink.lock().unwrap().push_str(&chunk);
            true
        });

        handler
            .send_prompt(&conv.id, "go".into(), cb, noop_status())
            .await
            .unwrap();

        let streamed = streamed.lock().unwrap();
        assert!(
            streamed.contains("final answer after tools"),
            "the final answer must be streamed to the caller, got: {streamed:?}"
        );
    }

    #[tokio::test]
    async fn final_answer_streams_after_multiple_tool_rounds() {
        // DA-9 unhappy path: two consecutive tool rounds, then text. Streaming
        // must survive every round transition, not just the first.
        let tool_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "read_file", "{}")]),
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-2", "read_file", "{}")]),
            LlmResponse::text("done at last"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "data".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let streamed = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&streamed);
        let cb: ChunkCallback = Box::new(move |chunk| {
            sink.lock().unwrap().push_str(&chunk);
            true
        });

        handler
            .send_prompt(&conv.id, "go".into(), cb, noop_status())
            .await
            .unwrap();

        let streamed = streamed.lock().unwrap();
        assert!(
            streamed.contains("done at last"),
            "the final answer must be streamed after multiple tool rounds, got: {streamed:?}"
        );
    }

    #[tokio::test]
    async fn malformed_tool_call_arguments_surface_parse_error_to_model() {
        // DA-13: when the model emits tool-call arguments that are not valid
        // JSON, the tool must NOT run with defaulted (null) arguments; the
        // tool result must tell the model its arguments were invalid JSON so
        // it can correct itself.
        let tool_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let bad_call = ToolCall::new("call-1", "read_file", "{ this is not json");

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![bad_call]),
            LlmResponse::text("done"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "hello world".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "done");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[2].role, Role::Tool);
        let content = &updated.messages[2].content;
        assert!(
            content.contains("not valid JSON"),
            "tool result must report invalid-JSON arguments, got: {content}"
        );
        assert!(
            !content.contains("hello world"),
            "tool must not execute with defaulted arguments, got: {content}"
        );
    }

    #[tokio::test]
    async fn empty_tool_call_arguments_are_treated_as_empty_object() {
        // DA-13 unhappy-path guard: some providers emit an empty string for
        // no-argument tool calls. That must keep executing (as `{}`), not be
        // rejected as malformed JSON.
        let tool_def = ToolDefinition::new(
            "list_files",
            "List files",
            serde_json::json!({"type": "object"}),
        );
        let empty_call = ToolCall::new("call-1", "list_files", "");

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![empty_call]),
            LlmResponse::text("done"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("list_files".to_string(), "a.txt".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[2].role, Role::Tool);
        assert_eq!(
            updated.messages[2].content, "a.txt",
            "empty-string arguments must execute the tool with an empty object"
        );
    }

    // --- Planning + compaction (#240) ---

    /// An in-memory scratchpad backing the write/list closures, plus a handle
    /// to inspect what was written.
    fn in_memory_scratchpad() -> (
        ScratchpadWriteFn,
        ScratchpadListFn,
        Arc<Mutex<HashMap<String, crate::domain::ScratchpadNote>>>,
    ) {
        use crate::domain::ScratchpadNote;
        let store: Arc<Mutex<HashMap<String, ScratchpadNote>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let w = Arc::clone(&store);
        let write: ScratchpadWriteFn =
            Arc::new(move |_conv: String, notes: Vec<NewScratchpadNote>| {
                let w = Arc::clone(&w);
                Box::pin(async move {
                    let mut map = w.lock().unwrap();
                    let saved: Vec<ScratchpadNote> = notes
                        .into_iter()
                        .map(|n| {
                            let mut note = ScratchpadNote::new(
                                format!("id-{}", n.key),
                                "conv",
                                &n.key,
                                &n.content,
                            );
                            note.note_type = n.note_type;
                            note.sequence = n.sequence;
                            note.done = n.done;
                            map.insert(n.key.clone(), note.clone());
                            note
                        })
                        .collect();
                    Ok(saved)
                })
            });

        let l = Arc::clone(&store);
        let list: ScratchpadListFn = Arc::new(move |_conv, note_type: Option<String>, _limit| {
            let l = Arc::clone(&l);
            Box::pin(async move {
                let map = l.lock().unwrap();
                let mut out: Vec<ScratchpadNote> = map
                    .values()
                    .filter(|n| note_type.as_deref().is_none_or(|t| n.note_type == t))
                    .cloned()
                    .collect();
                out.sort_by(|a, b| a.key.cmp(&b.key));
                Ok(out)
            })
        });

        (write, list, store)
    }

    fn id_gen() -> Box<dyn Fn() -> String + Send + Sync> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        Box::new(move || {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            format!("conv-{n}")
        })
    }

    #[tokio::test]
    async fn complete_step_evicts_raw_tool_result_into_scratchpad_pointer() {
        // The headline of #240: begin a step, run a tool that returns a big
        // payload, complete the step — and the raw result leaves working context,
        // replaced by a searchable pointer to the distilled outcome note, while
        // the message structure (role + tool_call_id) is preserved.
        let big = "DATA".repeat(2000); // ~8 KB, well above the eviction threshold
        let tools = vec![ToolDefinition::new(
            "weather_forecast",
            "Get a forecast",
            serde_json::json!({"type": "object"}),
        )];
        let mut tool_results = HashMap::new();
        tool_results.insert("weather_forecast".to_string(), big.clone());

        let responses = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    "b1",
                    "begin_step",
                    r#"{"goal":"get the forecast"}"#,
                )],
            ),
            LlmResponse::with_tool_calls("", vec![ToolCall::new("t1", "weather_forecast", "{}")]),
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    "c1",
                    "complete_step",
                    r#"{"outcome":"Cary NC 7-day: highs low-80s, rain Tue"}"#,
                )],
            ),
            LlmResponse::text("All done — it'll be warm with rain Tuesday."),
        ];

        let (write, list, sp) = in_memory_scratchpad();
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(responses),
            MockToolExecutor::new(tools, tool_results),
            id_gen(),
        )
        .with_scratchpad_write(write)
        .with_scratchpad_list(list);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result = handler
            .send_prompt(&conv.id, "weather?".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "All done — it'll be warm with rain Tuesday.");

        // The big tool result must have been compacted in place: still a Tool
        // message bound to its call, but the payload is gone and replaced by a
        // pointer naming the tool and the outcome note.
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let big_result = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("t1"))
            .expect("the weather tool result message must still exist");
        assert!(
            big_result.content.starts_with("<compacted to scratchpad"),
            "raw result should be replaced by a pointer, got: {}",
            big_result.content
        );
        assert!(big_result.content.contains("weather_forecast"));
        assert!(big_result.content.contains("outcome:1"));
        assert!(
            !big_result.content.contains("DATADATA"),
            "the raw payload must be gone from working context"
        );

        // The scratchpad holds the done todo + the distilled outcome note.
        let notes = sp.lock().unwrap();
        let todo = notes.get("1").expect("step todo must exist");
        assert_eq!(todo.note_type, "todo");
        assert!(todo.done, "the step todo must be checked off");
        let outcome = notes.get("outcome:1").expect("outcome note must exist");
        assert_eq!(outcome.content, "Cary NC 7-day: highs low-80s, rain Tue");
    }

    #[tokio::test]
    async fn second_turn_step_keys_do_not_clobber_first_turns_notes() {
        // DA-7 (#292): a step in turn 2 must continue the numbering ("2"), not
        // restart at "1" and overwrite turn 1's still-persisted todo via the
        // scratchpad's upsert-by-key write.
        let (write, list, sp) = in_memory_scratchpad();

        // Turn 1: one begin_step (mints "1") then a final answer.
        let turn1 = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    "b1",
                    "begin_step",
                    r#"{"goal":"first step"}"#,
                )],
            ),
            LlmResponse::text("done one"),
        ];
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(turn1),
            MockToolExecutor::new(vec![], HashMap::new()),
            id_gen(),
        )
        .with_scratchpad_write(Arc::clone(&write))
        .with_scratchpad_list(Arc::clone(&list));
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        // After turn 1 the scratchpad has a "1" todo with the first goal.
        {
            let notes = sp.lock().unwrap();
            assert_eq!(notes.get("1").unwrap().content, "first step");
        }

        // Turn 2 on the SAME conversation: another begin_step. With seeding it
        // must mint "2"; without the fix it would mint "1" and overwrite the
        // first goal.
        let turn2 = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    "b2",
                    "begin_step",
                    r#"{"goal":"second step"}"#,
                )],
            ),
            LlmResponse::text("done two"),
        ];
        let handler2 = ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(turn2),
            MockToolExecutor::new(vec![], HashMap::new()),
            id_gen(),
        )
        .with_scratchpad_write(Arc::clone(&write))
        .with_scratchpad_list(Arc::clone(&list));
        // Re-create the conversation in handler2's store and pre-seed nothing;
        // the scratchpad (the source of step keys) is shared via the closures.
        let conv2 = handler2
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        handler2
            .send_prompt(&conv2.id, "again".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let notes = sp.lock().unwrap();
        assert_eq!(
            notes.get("1").unwrap().content,
            "first step",
            "turn 1's note must NOT be clobbered"
        );
        assert_eq!(
            notes.get("2").unwrap().content,
            "second step",
            "turn 2's step must mint the next key"
        );
    }

    /// Capturing LLM that records the message list it is handed each round, then
    /// returns the next scripted response.
    struct PlanContextCapturingLlm {
        responses: Mutex<Vec<LlmResponse>>,
        captured: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    #[async_trait::async_trait]
    impl LlmClient for PlanContextCapturingLlm {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.captured.lock().unwrap().push(messages);
            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Ok(LlmResponse::text("fallback"));
                }
                responses.remove(0)
            };
            if !response.text.is_empty() {
                on_chunk(response.text.clone());
            }
            Ok(response)
        }
    }

    #[tokio::test]
    async fn open_plan_is_surfaced_into_the_next_round() {
        // After begin_step records a todo, the next round's assembled context
        // must carry a [Plan] system message so the plan stays in view.
        let captured: Arc<Mutex<Vec<Vec<Message>>>> = Arc::new(Mutex::new(Vec::new()));
        let llm = PlanContextCapturingLlm {
            responses: Mutex::new(vec![
                LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new(
                        "b1",
                        "begin_step",
                        r#"{"goal":"map the plan"}"#,
                    )],
                ),
                LlmResponse::text("done"),
            ]),
            captured: Arc::clone(&captured),
        };

        let (write, list, _sp) = in_memory_scratchpad();
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            MockToolExecutor::new(vec![], HashMap::new()),
            id_gen(),
        )
        .with_scratchpad_write(write)
        .with_scratchpad_list(list);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(
                &conv.id,
                "do a multi-step thing".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        let rounds = captured.lock().unwrap();
        // Once begin_step records a todo, a later round's assembled context must
        // carry the [Plan] surface. (Round 0 — before any todo — does not; a
        // separate title-generation call also has none, so search all rounds.)
        let plan_msg = rounds
            .iter()
            .flatten()
            .find(|m| m.role == Role::System && m.content.starts_with("[Plan]"))
            .expect("the open plan must be surfaced once a todo exists");
        assert!(plan_msg.content.contains("map the plan"));
        assert!(plan_msg.content.contains("← you are here"));
    }

    #[tokio::test]
    async fn tool_calls_without_steps_emit_no_status() {
        // New narration model: no turn-start filler and no per-tool chatter. A
        // turn that calls tools but declares no plan steps narrates nothing —
        // progress is reserved for logical steps (`begin_step`).
        let tools = vec![
            ToolDefinition::new("calendar_list", "List calendar", serde_json::json!({})),
            ToolDefinition::new("notes_search", "Search notes", serde_json::json!({})),
        ];
        let responses = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![
                    ToolCall::new("c1", "calendar_list", "{}"),
                    ToolCall::new("c2", "notes_search", "{}"),
                ],
            ),
            LlmResponse::text("All set"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("calendar_list".to_string(), "ok".to_string());
        tool_results.insert("notes_search".to_string(), "ok".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let (status_cb, status_log) = recording_status();
        let result = handler
            .send_prompt(&conv.id, "Do it".into(), noop_callback(), status_cb)
            .await
            .unwrap();
        assert_eq!(result, "All set");

        let statuses = status_log.lock().unwrap().clone();
        assert!(
            statuses.is_empty(),
            "tool calls without declared steps must emit no status; got {statuses:?}"
        );
    }

    #[tokio::test]
    async fn begin_step_narrates_its_goal() {
        // A declared logical step IS narrated — once, as its goal — so clients
        // (text + voice) get meaningful progress on multi-step work.
        let llm = PlanContextCapturingLlm {
            responses: Mutex::new(vec![
                LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new(
                        "b1",
                        "begin_step",
                        r#"{"goal":"map the plan"}"#,
                    )],
                ),
                LlmResponse::text("done"),
            ]),
            captured: Arc::new(Mutex::new(Vec::new())),
        };

        let (write, list, _sp) = in_memory_scratchpad();
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            MockToolExecutor::new(vec![], HashMap::new()),
            id_gen(),
        )
        .with_scratchpad_write(write)
        .with_scratchpad_list(list);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let (status_cb, status_log) = recording_status();
        handler
            .send_prompt(
                &conv.id,
                "do a multi-step thing".into(),
                noop_callback(),
                status_cb,
            )
            .await
            .unwrap();

        let statuses = status_log.lock().unwrap().clone();
        assert_eq!(
            statuses,
            vec!["map the plan".to_string()],
            "the begin_step goal must be narrated once; got {statuses:?}"
        );
    }

    /// Fake [`ClientToolPort`] (#234) for the core turn-loop integration
    /// tests. Records the names it was asked to execute and returns a
    /// canned result so the loop can feed it back to the LLM. A parking
    /// variant (held behind a oneshot) is used to prove the loop suspends.
    struct FakeClientToolPort {
        defs: Vec<ToolDefinition>,
        executed: Arc<Mutex<Vec<(String, String)>>>,
        result: String,
        /// When set, `execute` returns this error instead of `result` — used to
        /// drive the cancel/error paths through the dispatch loop.
        error: Option<CoreError>,
    }

    impl FakeClientToolPort {
        fn ok(
            defs: Vec<ToolDefinition>,
            executed: Arc<Mutex<Vec<(String, String)>>>,
            result: impl Into<String>,
        ) -> Self {
            Self {
                defs,
                executed,
                result: result.into(),
                error: None,
            }
        }

        fn failing(
            defs: Vec<ToolDefinition>,
            executed: Arc<Mutex<Vec<(String, String)>>>,
            error: CoreError,
        ) -> Self {
            Self {
                defs,
                executed,
                result: String::new(),
                error: Some(error),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::ports::client_tools::ClientToolPort for FakeClientToolPort {
        async fn tool_definitions(&self) -> Vec<ToolDefinition> {
            self.defs.clone()
        }
        async fn is_registered(&self, name: &str) -> bool {
            self.defs.iter().any(|d| d.name == name)
        }
        async fn execute(
            &self,
            tool_call_id: &str,
            tool_name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            self.executed
                .lock()
                .unwrap()
                .push((tool_call_id.to_string(), tool_name.to_string()));
            match &self.error {
                Some(CoreError::Cancelled) => Err(CoreError::Cancelled),
                Some(other) => Err(CoreError::Llm(other.to_string())),
                None => Ok(self.result.clone()),
            }
        }
    }

    /// Install a recording tool observer around `fut` and return its result
    /// alongside the events the dispatch loop emitted (issue #252/#257 tests).
    async fn capture_tool_events<F, T>(fut: F) -> (T, Vec<ToolEvent>)
    where
        F: std::future::Future<Output = T>,
    {
        use crate::ports::tool_observer::{ToolObserver, with_tool_observer};
        let events: Arc<Mutex<Vec<ToolEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let events = Arc::clone(&events);
            Arc::new(move |e: ToolEvent| events.lock().unwrap().push(e)) as ToolObserver
        };
        let out = with_tool_observer(sink, fut).await;
        let captured = events.lock().unwrap().clone();
        (out, captured)
    }

    #[tokio::test]
    async fn turn_routes_registered_client_tool_through_port_and_feeds_result_back() {
        use crate::ports::client_tools::with_client_tools;

        // The LLM first calls `fs_read` (a client-local tool the server-side
        // executor knows nothing about), then returns final text after seeing
        // the client's result.
        let responses = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "fs_read",
                    r#"{"path":"/etc/hosts"}"#,
                )],
            ),
            LlmResponse::text("The file says: 127.0.0.1 localhost"),
        ];
        // No server-side tools and no server-side result for `fs_read`: if the
        // loop tried to run it server-side it would error, proving the client
        // path is the one taken.
        let handler = make_tool_handler(responses, vec![], HashMap::new());
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let executed = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn crate::ports::client_tools::ClientToolPort> =
            Arc::new(FakeClientToolPort::ok(
                vec![ToolDefinition::new(
                    "fs_read",
                    "Read a file on the client",
                    serde_json::json!({"type": "object"}),
                )],
                Arc::clone(&executed),
                "127.0.0.1 localhost",
            ));

        let result = with_client_tools(
            port,
            handler.send_prompt(
                &conv.id,
                "Read /etc/hosts".into(),
                noop_callback(),
                noop_status(),
            ),
        )
        .await
        .unwrap();

        assert_eq!(result, "The file says: 127.0.0.1 localhost");
        // The client-tool port — not the server-side executor — ran `fs_read`.
        let ran = executed.lock().unwrap().clone();
        assert_eq!(ran, vec![("call-1".to_string(), "fs_read".to_string())]);

        // The client's result was threaded into history as the tool result so
        // the LLM saw it on the next round.
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool result message");
        assert_eq!(tool_msg.content, "127.0.0.1 localhost");
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[tokio::test]
    async fn cancelled_client_tool_emits_matched_started_and_finished() {
        // Issue #252: when a suspended client tool is cancelled, the dispatch
        // loop aborts the turn with `Err(Cancelled)`. The activity feed must
        // still see exactly one `Started` and one `Finished{ok:false}` for that
        // call — never a started-but-never-finished row.
        use crate::ports::client_tools::with_client_tools;

        let responses = vec![LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(
                "call-1",
                "fs_read",
                r#"{"path":"/etc/hosts"}"#,
            )],
        )];
        let handler = make_tool_handler(responses, vec![], HashMap::new());
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let executed = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn crate::ports::client_tools::ClientToolPort> =
            Arc::new(FakeClientToolPort::failing(
                vec![ToolDefinition::new(
                    "fs_read",
                    "Read a file on the client",
                    serde_json::json!({"type": "object"}),
                )],
                Arc::clone(&executed),
                CoreError::Cancelled,
            ));

        let (result, events) = capture_tool_events(with_client_tools(
            port,
            handler.send_prompt(
                &conv.id,
                "Read /etc/hosts".into(),
                noop_callback(),
                noop_status(),
            ),
        ))
        .await;

        assert!(matches!(result, Err(CoreError::Cancelled)));

        let starts = events
            .iter()
            .filter(|e| matches!(e, ToolEvent::Started { name, .. } if name == "fs_read"))
            .count();
        let finishes: Vec<bool> = events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Finished { name, ok, .. } if name == "fs_read" => Some(*ok),
                _ => None,
            })
            .collect();
        assert_eq!(starts, 1, "exactly one Started; events={events:?}");
        assert_eq!(
            finishes,
            vec![false],
            "exactly one Finished{{ok:false}}; events={events:?}"
        );
    }

    #[tokio::test]
    async fn errored_client_tool_emits_one_started_finished_pair() {
        // Server-error (non-cancel) path: the loop folds the error into a tool
        // result and keeps going, but the observer must still see exactly one
        // Started/Finished pair with ok=false for that call.
        use crate::ports::client_tools::with_client_tools;

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "fs_read", "{}")]),
            LlmResponse::text("recovered"),
        ];
        let handler = make_tool_handler(responses, vec![], HashMap::new());
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let executed = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn crate::ports::client_tools::ClientToolPort> =
            Arc::new(FakeClientToolPort::failing(
                vec![ToolDefinition::new(
                    "fs_read",
                    "Read a file on the client",
                    serde_json::json!({"type": "object"}),
                )],
                Arc::clone(&executed),
                CoreError::Llm("boom".into()),
            ));

        let (result, events) = capture_tool_events(with_client_tools(
            port,
            handler.send_prompt(&conv.id, "go".into(), noop_callback(), noop_status()),
        ))
        .await;

        assert_eq!(result.unwrap(), "recovered");
        let starts = events
            .iter()
            .filter(|e| matches!(e, ToolEvent::Started { name, .. } if name == "fs_read"))
            .count();
        let finishes: Vec<bool> = events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Finished { name, ok, .. } if name == "fs_read" => Some(*ok),
                _ => None,
            })
            .collect();
        assert_eq!(starts, 1, "events={events:?}");
        assert_eq!(finishes, vec![false], "events={events:?}");
    }

    #[tokio::test]
    async fn finished_event_summarizes_capped_result() {
        // Issue #257: the Finished event must summarize the same (post-cap)
        // value the model is shown, not the pre-cap payload. Drive a tool that
        // returns more than the cap and assert the observer's output reflects
        // the truncated/stored text (it contains the "truncated" notice rather
        // than the full original body).
        let tool_def = ToolDefinition::new("dump", "Dumps a lot", serde_json::json!({}));
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "dump", "{}")]),
            LlmResponse::text("ok"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("dump".to_string(), "A".repeat(5_000));

        // Cap so small the truncation notice surfaces at the front of the
        // stored value (the kept prefix collapses to ~nothing). The pre-cap
        // payload is 5000 'A's and contains no "truncated" notice, so seeing
        // the notice in the Finished summary proves we summarized `stored`,
        // not `result`.
        let handler = make_tool_handler(responses, vec![tool_def], tool_results)
            .with_max_tool_result_bytes(16);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let (_result, events) = capture_tool_events(handler.send_prompt(
            &conv.id,
            "dump it".into(),
            noop_callback(),
            noop_status(),
        ))
        .await;

        let output = events
            .iter()
            .find_map(|e| match e {
                ToolEvent::Finished { name, output, .. } if name == "dump" => Some(output.clone()),
                _ => None,
            })
            .expect("a Finished event for dump");
        assert!(
            output.contains("truncated"),
            "Finished output should mirror the capped result; got: {output}"
        );
    }

    #[tokio::test]
    async fn turn_without_client_tool_port_runs_server_side_only() {
        // Same tool name, but no port installed: the loop must fall through to
        // the server-side executor (which here supplies the result). This pins
        // that the client-tool hook is strictly opt-in and never changes the
        // server-side path when unset.
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "fs_read", "{}")]),
            LlmResponse::text("done"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("fs_read".to_string(), "server output".to_string());
        let handler = make_tool_handler(
            responses,
            vec![ToolDefinition::new(
                "fs_read",
                "server tool",
                serde_json::json!({}),
            )],
            tool_results,
        );
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "done");
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool result message");
        assert_eq!(tool_msg.content, "server output");
    }

    #[tokio::test]
    async fn turn_with_no_tools_emits_no_status() {
        // A plain text turn (no tools, no steps) is a "quick answer": it
        // narrates nothing and just streams its reply.
        let handler = make_handler(vec!["Hello there"]);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let (status_cb, status_log) = recording_status();
        handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), status_cb)
            .await
            .unwrap();

        let statuses = status_log.lock().unwrap().clone();
        assert!(
            statuses.is_empty(),
            "a no-tool quick answer must emit no status; got {statuses:?}"
        );
    }

    #[tokio::test]
    async fn oversized_tool_result_is_truncated_at_ingestion_and_stays_paired() {
        // Issue #174: a tool returning a huge payload must be truncated before
        // it is stored, so it can't wedge the conversation on later turns. The
        // tool_call_id pairing must survive truncation.
        let tool_def = ToolDefinition::new("dump", "Dumps a lot", serde_json::json!({}));
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("call-1", "dump", "{}")]),
            LlmResponse::text("ok"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("dump".to_string(), "A".repeat(5_000));

        let handler = make_tool_handler(responses, vec![tool_def], tool_results)
            .with_max_tool_result_bytes(1_024);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(&conv.id, "dump it".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = &updated.messages[2];
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call-1"));
        assert!(
            tool_msg.content.len() <= 1_024,
            "stored tool result {} exceeds cap",
            tool_msg.content.len()
        );
        assert!(tool_msg.content.contains("truncated"));
        assert!(tool_msg.content.starts_with("AAAA"));
    }

    #[tokio::test]
    async fn tool_loop_handles_multiple_tool_calls() {
        let tools = vec![
            ToolDefinition::new("tool_a", "Tool A", serde_json::json!({})),
            ToolDefinition::new("tool_b", "Tool B", serde_json::json!({})),
        ];

        let responses = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![
                    ToolCall::new("c1", "tool_a", "{}"),
                    ToolCall::new("c2", "tool_b", "{}"),
                ],
            ),
            LlmResponse::text("Done with both tools"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("tool_a".to_string(), "result_a".to_string());
        tool_results.insert("tool_b".to_string(), "result_b".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(&conv.id, "Do both".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "Done with both tools");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        // User + assistant(tool_calls) + tool_result_a + tool_result_b + assistant(final)
        assert_eq!(updated.messages.len(), 5);
    }

    #[tokio::test]
    async fn tool_loop_handles_tool_error_gracefully() {
        let tools = vec![ToolDefinition::new(
            "bad_tool",
            "Fails",
            serde_json::json!({}),
        )];

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "bad_tool", "{}")]),
            LlmResponse::text("Tool failed, but I can continue"),
        ];

        // No results configured — tool will return error
        let handler = make_tool_handler(responses, tools, HashMap::new());
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Try bad tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "Tool failed, but I can continue");

        // The tool error should be in the conversation as a tool result message
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[2].role, Role::Tool);
        assert!(updated.messages[2].content.starts_with("Error:"));
    }

    #[tokio::test]
    async fn tool_loop_respects_max_rounds() {
        let tools = vec![ToolDefinition::new(
            "loop_tool",
            "Loops",
            serde_json::json!({}),
        )];

        // LLM always returns tool calls — never text
        let responses: Vec<LlmResponse> = (0..MAX_TOOL_ROUNDS + 1)
            .map(|i| {
                LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new(format!("c{i}"), "loop_tool", "{}")],
                )
            })
            .collect();

        let mut tool_results = HashMap::new();
        tool_results.insert("loop_tool".to_string(), "ok".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        // #453: the loop is still bounded — it stops at MAX_TOOL_ROUNDS rather
        // than looping forever — but now winds down and persists a closing
        // instead of returning an error, so the turn isn't lost.
        let closing = handler
            .send_prompt(
                &conv.id,
                "Loop forever".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .expect("bounded loop winds down to Ok rather than erroring");
        assert!(!closing.is_empty(), "a wind-down closing is produced");
    }

    // --- Context recovery test ---

    /// Mock LLM that fails on a specific call index.
    struct FailingLlm {
        responses: Mutex<Vec<LlmResponse>>,
        fail_on_call: usize,
        call_count: Mutex<usize>,
        error_factory: Box<dyn Fn() -> CoreError + Send + Sync>,
    }

    impl FailingLlm {
        fn new(responses: Vec<LlmResponse>, fail_on_call: usize) -> Self {
            Self {
                responses: Mutex::new(responses),
                fail_on_call,
                call_count: Mutex::new(0),
                // Default to a generic LLM error; tests that need a
                // specific structured variant call `with_error_variant`.
                error_factory: Box::new(|| CoreError::Llm("context_length_exceeded".into())),
            }
        }

        /// Substitute the variant produced on the failing call. Used by
        /// tests that exercise control-flow paths keyed on the specific
        /// `CoreError` variant (e.g. `RateLimited` skipping the trim
        /// branch).
        fn with_error_variant<F>(mut self, factory: F) -> Self
        where
            F: Fn() -> CoreError + Send + Sync + 'static,
        {
            self.error_factory = Box::new(factory);
            self
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for FailingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let call_idx = {
                let mut count = self.call_count.lock().unwrap();
                let idx = *count;
                *count += 1;
                idx
            };

            if call_idx == self.fail_on_call {
                return Err((self.error_factory)());
            }

            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Ok(LlmResponse::text("fallback"));
                }
                responses.remove(0)
            };
            if !response.text.is_empty() {
                on_chunk(response.text.clone());
            }
            Ok(response)
        }
    }

    #[tokio::test]
    async fn non_context_error_after_round_zero_surfaces_directly() {
        // Old path C trimmed-and-retried any non-retryable, non-rate-limit
        // error after round 0 — including transient or malformed-call
        // failures that had nothing to do with context size. Now that the
        // recovery ladder is gated on `CoreError::ContextOverflow`, those
        // errors must surface to the user immediately instead of mutating
        // the conversation state.
        let tools = vec![ToolDefinition::new(
            "my_tool",
            "A tool",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests tool call.
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "my_tool", "{}")]),
            // Round 1: fails with a generic LLM error (call index 1 below).
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("my_tool".to_string(), "result".to_string());

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 1)
                .with_error_variant(|| CoreError::Llm("context_length_exceeded".into())),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Use my tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        // The user-visible error mentions the underlying detail; what
        // matters is that we don't pretend to have recovered.
        assert!(result.contains("LLM backend error"));
        assert!(result.contains("context_length_exceeded"));

        // No system trim notice was injected — path C is gone.
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_trim_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(
            !has_trim_msg,
            "non-context errors must not trigger context trimming"
        );
    }

    #[tokio::test]
    async fn first_round_llm_error_is_saved_as_assistant_message() {
        // If the first LLM call fails, return a user-visible assistant message
        let tools = vec![];

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(vec![], 0), // fail on 1st call
            MockToolExecutor::new(tools, HashMap::new()),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert!(result.contains("LLM backend error"));

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 2);
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert!(updated.messages[1].content.contains("LLM backend error"));
    }

    #[test]
    fn user_visible_error_for_unsupported_tools() {
        let err = CoreError::ToolsUnsupported {
            detail: "phi4:14b does not support tools".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("does not support tool use"));
    }

    #[test]
    fn user_visible_error_for_loading_model() {
        let err = CoreError::ModelLoading {
            detail: "model is currently loading".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("still downloading or loading"));
    }

    #[test]
    fn user_visible_error_for_rate_limit_429() {
        let err = CoreError::RateLimited {
            retry_after: None,
            detail: "Rate limited".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
    }

    #[test]
    fn user_visible_error_for_overloaded_529_uses_generic_fallback() {
        // Repaired (issue #441): the prior version built `CoreError::RateLimited`
        // and asserted the rate-limit message — byte-for-byte the same arm as
        // `user_visible_error_for_rate_limit_429`, so it proved nothing new. A
        // 529 "overloaded" is a *transient* server error: `error_classify` maps
        // "overloaded" to `NormalizedCause::Transient`, which `cause_to_core_error`
        // leaves unmapped, so it surfaces as a bare `CoreError::Llm` and lands on
        // the generic fallback arm — NOT the rate-limit arm. This asserts that
        // distinct arm.
        let err = CoreError::Llm("Overloaded (529): the model is overloaded".into());
        let msg = user_visible_llm_error_message(&err);
        assert!(
            msg.contains("LLM backend error"),
            "a 529/overloaded transient error must use the generic fallback, got: {msg}"
        );
        assert!(
            msg.contains("overloaded"),
            "the underlying detail must be surfaced"
        );
        assert!(
            !msg.contains("rate limit was exceeded"),
            "must NOT reuse the rate-limit (429) arm"
        );
    }

    #[test]
    fn user_visible_error_for_quota_exceeded() {
        let err = CoreError::QuotaExceeded {
            detail: "insufficient_quota".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("quota is exhausted"));
    }

    #[test]
    fn user_visible_error_for_context_overflow() {
        let err = CoreError::ContextOverflow {
            prompt_tokens: Some(203_524),
            max_tokens: Some(200_000),
            detail: "prompt is too long".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("context window"));
    }

    #[test]
    fn user_visible_error_for_generic_llm() {
        let err = CoreError::Llm("invalid API key".into());
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("LLM backend error"));
        assert!(msg.contains("invalid API key"));
    }

    #[tokio::test]
    async fn rate_limit_error_mid_loop_does_not_trim_context() {
        let tools = vec![ToolDefinition::new(
            "my_tool",
            "A tool",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests tool call
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "my_tool", "{}")]),
            // Round 1: fails with 429 (simulated by FailingLlm, call index 1)
            // — should NOT trim, should surface as user-visible error
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("my_tool".to_string(), "result".to_string());

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 1).with_error_variant(|| CoreError::RateLimited {
                retry_after: None,
                detail: "Anthropic API error (HTTP 429 Too Many Requests): rate_limit_error".into(),
            }),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Use my tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        // Should get a rate-limit user-visible message, not "adjusted my approach"
        assert!(result.contains("rate limit was exceeded"));

        // Verify NO system message about trimming was added
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_trim_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(
            !has_trim_msg,
            "rate limit error should not trigger context trimming"
        );
    }

    #[tokio::test]
    async fn noop_executor_returns_empty_tools() {
        let executor = NoopToolExecutor;
        assert!(executor.core_tools().await.is_empty());
    }

    #[tokio::test]
    async fn noop_executor_returns_error() {
        let executor = NoopToolExecutor;
        let result = executor
            .execute_tool("anything", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    struct CapturingLlm {
        seen_messages: Arc<Mutex<Vec<Message>>>,
    }

    #[async_trait::async_trait]
    impl LlmClient for CapturingLlm {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            // Only capture the first call (the main LLM turn). The second call
            // triggered by title generation must not overwrite the captured state
            // that the test assertions rely on.
            let mut seen = self.seen_messages.lock().unwrap();
            if seen.is_empty() {
                *seen = messages;
            }
            Ok(LlmResponse::text("ok"))
        }
    }

    #[tokio::test]
    async fn llm_input_includes_runtime_instruction_message() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let handler = ConversationHandler::new(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let _ = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let messages = seen.lock().unwrap();
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0].content.contains(
            "You are Adele, a desktop assistant named in reference to the Adélie penguin"
        ));
        assert!(messages[0].content.contains("Your name is Adele"));
        assert!(
            messages[0]
                .content
                .contains("Follow these rules in priority order")
        );
        assert!(
            messages[0]
                .content
                .contains("Current-turn instructions override stored data")
        );
        assert!(
            messages[0]
                .content
                .contains("Search the knowledge base for each piece")
        );
        assert!(
            messages[0]
                .content
                .contains("ask one brief question rather than guess")
        );
        assert!(
            messages[0]
                .content
                .contains("Don't guess user-specific details")
        );
        assert!(
            messages[0]
                .content
                .contains("Validate temporally variable facts")
        );
        assert!(
            messages[0]
                .content
                .contains("No tools are available in this turn.")
        );
        assert!(messages[0].content.contains("non-blocking pattern"));
        assert!(messages[0].content.contains("PATH"));
        assert!(messages[0].content.contains("Flatpak/Snap"));
        assert!(
            messages[0]
                .content
                .contains("builtin_knowledge_base_write/search/list/delete")
        );
        assert!(messages[0].content.contains("builtin_sys_props"));
        assert!(messages[0].content.contains("builtin_tool_search"));
        assert!(messages[0].content.contains("Never fabricate outputs"));
    }

    #[test]
    fn runtime_instruction_enforces_kb_first_for_user_specific_requests() {
        use crate::prompts;

        let instruction = prompts::assemble(&prompts::static_sections());

        // Behavioral invariants: each of these must be expressed somewhere in
        // the assembled prompt. Exact wording is the prompt files' concern;
        // this test exists to catch silent drops of a load-bearing rule.
        let priority_rule = "Current-turn instructions override stored data";
        let kb_search = "Search the knowledge base for each piece";
        let ambiguity_guard = "ask one brief question rather than guess";
        let no_guessing = "Don't guess user-specific details";
        let verify_facts = "Validate temporally variable facts";
        let no_fabrication = "Never fabricate outputs";
        let tool_search_discovery = "builtin_tool_search";
        let skill_search_discovery = "skills_search_skills";

        assert!(
            instruction.contains(priority_rule),
            "missing: {priority_rule}"
        );
        assert!(instruction.contains(kb_search), "missing: {kb_search}");
        assert!(
            instruction.contains(ambiguity_guard),
            "missing: {ambiguity_guard}"
        );
        assert!(instruction.contains(no_guessing), "missing: {no_guessing}");
        assert!(
            instruction.contains(verify_facts),
            "missing: {verify_facts}"
        );
        assert!(
            instruction.contains(no_fabrication),
            "missing: {no_fabrication}"
        );
        assert!(
            instruction.contains(tool_search_discovery),
            "missing: {tool_search_discovery}"
        );
        assert!(
            instruction.contains(skill_search_discovery),
            "missing: {skill_search_discovery}"
        );
    }

    // --- Title generation tests ---

    #[test]
    fn sanitize_generated_title_basic() {
        assert_eq!(
            sanitize_generated_title("Weather Forecast Today"),
            "Weather Forecast Today"
        );
    }

    #[test]
    fn sanitize_generated_title_strips_quotes_and_punctuation() {
        assert_eq!(
            sanitize_generated_title("\"Fix Broken Build Pipeline\""),
            "Fix Broken Build Pipeline"
        );
        assert_eq!(
            sanitize_generated_title("'Deploy to Production.'"),
            "Deploy to Production"
        );
    }

    #[test]
    fn sanitize_generated_title_takes_first_line_only() {
        assert_eq!(
            sanitize_generated_title("Rust Memory Debug\nSome explanation here"),
            "Rust Memory Debug"
        );
    }

    #[test]
    fn sanitize_generated_title_limits_to_eight_words() {
        let long = "One Two Three Four Five Six Seven Eight Nine Ten";
        assert_eq!(
            sanitize_generated_title(long),
            "One Two Three Four Five Six Seven Eight"
        );
    }

    #[tokio::test]
    async fn send_prompt_generates_title_on_first_message() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("That sounds great!"),   // main response
                LlmResponse::text("Plan Weekend Getaway"), // title generation
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("New Chat".into(), vec![])
            .await
            .unwrap();
        assert_eq!(conv.title, "New Chat");

        handler
            .send_prompt(
                &conv.id,
                "Let's plan a trip!".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.title, "Plan Weekend Getaway");
    }

    #[tokio::test]
    async fn send_prompt_does_not_overwrite_title_on_second_message() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("First response"), // main response round 1
                LlmResponse::text("Generated Title Here"), // title generation round 1
                LlmResponse::text("Second response"), // main response round 2
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("New Chat".into(), vec![])
            .await
            .unwrap();

        // First prompt — sets the title
        handler
            .send_prompt(&conv.id, "Hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        let after_first = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(after_first.title, "Generated Title Here");

        // Second prompt — title must remain unchanged
        handler
            .send_prompt(
                &conv.id,
                "Follow-up question".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        let after_second = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(after_second.title, "Generated Title Here");
    }

    #[tokio::test]
    async fn send_prompt_keeps_original_title_when_generation_returns_empty() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("Sure, I can help."), // main response
                LlmResponse::text(""),                  // title generation returns empty
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("My Chat".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.title, "My Chat");
    }

    #[tokio::test]
    async fn llm_input_runtime_instruction_lists_available_tools() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let tools = vec![ToolDefinition::new(
            "terminal",
            "Run terminal command",
            serde_json::json!({"type": "object"}),
        )];
        let tool_results = HashMap::new();

        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let _ = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let messages = seen.lock().unwrap();
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::System);
        assert!(
            messages[0]
                .content
                .contains("Available tools in this turn: terminal.")
        );
    }

    #[tokio::test]
    async fn remote_ws_turn_twins_duplicated_capability_with_locality_labels() {
        // Issue #243 end-to-end through the dispatch loop: a server-side
        // `terminal` plus a client-registered `terminal` over a REMOTE
        // (WebSocket) connection. The per-turn tool note must expose BOTH with
        // locality labels (server host vs your device) and a routing hint —
        // i.e. the remote case does not collapse the capability.
        use crate::ports::client_tools::with_client_tools;
        use crate::ports::transport::with_transport_kind;
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let server_tools = vec![ToolDefinition::new(
            "terminal",
            "Run terminal command on the daemon host",
            serde_json::json!({"type": "object"}),
        )];
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            MockToolExecutor::new(server_tools, HashMap::new()),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
        .with_host("daemon-host");

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        // Client registers a tool with the SAME name as the server-side one.
        let port: Arc<dyn crate::ports::client_tools::ClientToolPort> =
            Arc::new(FakeClientToolPort::ok(
                vec![ToolDefinition::new(
                    "terminal",
                    "Run terminal command on the user's device",
                    serde_json::json!({"type": "object"}),
                )],
                Arc::new(Mutex::new(Vec::new())),
                "",
            ));

        // Drive the turn as if it arrived over a WebSocket connection.
        with_transport_kind(
            TransportKind::WebSocket,
            with_client_tools(
                port,
                handler.send_prompt(&conv.id, "hi".into(), noop_callback(), noop_status()),
            ),
        )
        .await
        .unwrap();

        let messages = seen.lock().unwrap();
        let system = &messages[0].content;
        assert!(
            system.contains("terminal — server 'daemon-host'"),
            "remote note must label the server tool: {system}"
        );
        assert!(
            system.contains("terminal — your device"),
            "remote note must label the client twin: {system}"
        );
        assert!(
            system.contains("ask which machine"),
            "remote duplicated capability must carry the routing hint: {system}"
        );
    }

    #[tokio::test]
    async fn local_uds_turn_collapses_duplicated_capability_to_plain_list() {
        // Companion to the remote test: the SAME server+client `terminal` over
        // a co-located (UDS) connection collapses to a single plain `terminal`
        // entry — no locality labels, no routing hint.
        use crate::ports::client_tools::with_client_tools;
        use crate::ports::transport::with_transport_kind;
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            MockToolExecutor::new(
                vec![ToolDefinition::new(
                    "terminal",
                    "Run terminal command",
                    serde_json::json!({"type": "object"}),
                )],
                HashMap::new(),
            ),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
        .with_host("daemon-host");

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let port: Arc<dyn crate::ports::client_tools::ClientToolPort> =
            Arc::new(FakeClientToolPort::ok(
                vec![ToolDefinition::new(
                    "terminal",
                    "Run terminal command on the user's device",
                    serde_json::json!({"type": "object"}),
                )],
                Arc::new(Mutex::new(Vec::new())),
                "",
            ));

        with_transport_kind(
            TransportKind::Uds,
            with_client_tools(
                port,
                handler.send_prompt(&conv.id, "hi".into(), noop_callback(), noop_status()),
            ),
        )
        .await
        .unwrap();

        let messages = seen.lock().unwrap();
        let system = &messages[0].content;
        // Inspect the tool-availability line specifically: the static prompt
        // mentions "server"/"your device" as guidance, so assert against the
        // generated tool listing rather than the whole system message.
        let tool_line = system
            .lines()
            .find(|l| l.starts_with("Available tools in this turn:"))
            .expect("a tool-availability line");
        assert!(
            tool_line.contains("Available tools in this turn: terminal."),
            "co-located note must be a plain single entry: {tool_line}"
        );
        assert!(
            !tool_line.contains("your device") && !tool_line.contains("server 'daemon-host'"),
            "co-located note must omit locality labels: {tool_line}"
        );
        assert!(
            !tool_line.contains("ask which machine"),
            "co-located note must omit the routing hint: {tool_line}"
        );
    }

    #[tokio::test]
    async fn recovery_picks_largest_by_token_estimate_not_bytes() {
        // Two tool results with the same byte length but different
        // token-estimate weights (using the chars/4 default):
        //
        //  - `ascii`  = 256 ASCII bytes = 256 chars → 64 estimated tokens
        //  - `emoji`  = 64 emoji × 4 bytes = 256 bytes / 64 chars → 16 tokens
        //
        // With the byte-length picker (the pre-#65 logic) both ties: the
        // first one to enumerate would win. With token-estimate ranking
        // the ASCII result wins unambiguously.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "ok".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        // Build two payloads where byte length and estimated tokens
        // give different rankings. Both clear MIN_TRUNCATION_TOKENS so
        // either could be picked by step 1; the picker must choose by
        // token estimate, not bytes.
        //
        //   ASCII:  8192 chars × 1 byte  =  8192 bytes / 2048 est. tokens
        //   Emoji:  4096 chars × 4 bytes = 16384 bytes / 1024 est. tokens
        //
        // Bytes alone would pick emoji (more bytes); tokens pick ASCII
        // (more estimated cost). That's the regression this guards.
        let ascii_payload: String = "A".repeat(8192);
        let emoji_one = "\u{1F600}"; // 4 bytes, 1 char
        let emoji_payload: String = emoji_one.repeat(4096);
        assert!(
            emoji_payload.len() > ascii_payload.len(),
            "emoji payload must have more bytes so byte-picker would mis-target"
        );
        assert!(
            ascii_payload.chars().count() > emoji_payload.chars().count(),
            "ASCII payload must have more chars so token-picker prefers it"
        );

        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "ascii", "t", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result("ascii", &ascii_payload));
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "emoji", "t", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result("emoji", &emoji_payload));
        handler.store.update(stored).await.unwrap();

        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let after = handler.get_conversation(&conv.id).await.unwrap();
        let ascii_after = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("ascii"))
            .expect("ascii result preserved");
        let emoji_after = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("emoji"))
            .expect("emoji result preserved");
        assert!(
            ascii_after.content.starts_with("<tool output omitted"),
            "token-estimate picker should target the ASCII result, got: {:?}",
            &ascii_after.content
        );
        assert_eq!(
            emoji_after.content, emoji_payload,
            "emoji result must be preserved verbatim — fewer estimated tokens"
        );
    }

    // --- Token-pressure compaction tests ---

    /// Mock LLM that reports configurable token usage and a declared
    /// `max_context_tokens`, used to drive the token-pressure path in
    /// `send_prompt`.
    struct TokenReportingLlm {
        text: String,
        input_tokens: u64,
        max_context: Option<u64>,
    }

    #[async_trait::async_trait]
    impl LlmClient for TokenReportingLlm {
        fn max_context_tokens(&self) -> Option<u64> {
            self.max_context
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            on_chunk(self.text.clone());
            let usage = TokenUsage {
                input_tokens: Some(self.input_tokens),
                output_tokens: Some(10),
                ..Default::default()
            };
            Ok(LlmResponse::text(self.text.clone()).with_usage(usage))
        }
    }

    #[tokio::test]
    async fn send_prompt_shrinks_window_on_token_pressure() {
        use crate::ports::llm::{BudgetSource, ContextBudget, with_context_budget};
        use std::sync::atomic::{AtomicU64, Ordering};

        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens: 180_000, // 90% of 200K — above 85% threshold
                max_context: Some(200_000),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        // Prime the conversation with enough messages to exceed the default
        // window, so shrinking it triggers a new compaction range.
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..(MAX_CONTEXT_MESSAGES + 20) {
            if i % 2 == 0 {
                stored
                    .messages
                    .push(Message::new(Role::User, format!("u-{i}")));
            } else {
                stored
                    .messages
                    .push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        handler.store.update(stored).await.unwrap();

        let before = handler.get_conversation(&conv.id).await.unwrap();
        let baseline_compacted = before.compacted_through;

        // Install the resolved budget the daemon's wrapper would set
        // (issue #63) so the token-pressure check fires. Without the
        // wrapper, `current_context_budget()` returns `None` and the
        // token-pressure branch skips.
        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            // Drive a turn that will receive high token usage and trigger
            // the token-pressure shrink + compaction path.
            handler
                .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                .await
                .unwrap();
        })
        .await;

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            after.compacted_through > baseline_compacted,
            "token pressure should have advanced compacted_through"
        );
    }

    #[tokio::test]
    async fn send_prompt_no_shrink_when_tokens_under_threshold() {
        use crate::ports::llm::{BudgetSource, ContextBudget, with_context_budget};
        use std::sync::atomic::{AtomicU64, Ordering};

        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens: 100_000, // 50% — below threshold
                max_context: Some(200_000),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        // Small conversation: no windowing, no compaction expected.
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for _ in 0..5 {
            stored.messages.push(Message::new(Role::User, "hi"));
        }
        handler.store.update(stored).await.unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                .await
                .unwrap();
        })
        .await;

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(
            after.compacted_through, 0,
            "no compaction expected when token usage is below threshold"
        );
    }

    // --- Context-usage emission tests (issue #341) ----------------------

    /// Run one turn with a [`ContextUsage`](crate::ports::llm::ContextUsage)
    /// sink + the given budget installed, returning every usage report the
    /// dispatch loop emitted. `input_tokens` is what the mock LLM reports;
    /// `prime_messages` seeds the conversation so the window-shrink path can
    /// be exercised when desired.
    async fn capture_context_usage(
        input_tokens: u64,
        max_context: u64,
        prime_messages: usize,
    ) -> Vec<crate::ports::llm::ContextUsage> {
        use crate::ports::llm::{
            BudgetSource, ContextBudget, ContextUsage, ContextUsageSink, with_context_budget,
            with_context_usage_sink,
        };

        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens,
                max_context: Some(max_context),
            },
            Box::new(|| "conv-1".to_string()),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        if prime_messages > 0 {
            let mut stored = handler.get_conversation(&conv.id).await.unwrap();
            for i in 0..prime_messages {
                let role = if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                };
                stored.messages.push(Message::new(role, format!("m-{i}")));
            }
            handler.store.update(stored).await.unwrap();
        }

        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_sink = Arc::clone(&captured);
        let sink: ContextUsageSink = Arc::new(move |u: ContextUsage| {
            captured_for_sink.lock().unwrap().push(u);
        });

        let budget = ContextBudget {
            max_input_tokens: max_context,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            with_context_usage_sink(sink, async {
                handler
                    .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                    .await
                    .unwrap();
            })
            .await
        })
        .await;

        captured.lock().unwrap().clone()
    }

    #[tokio::test]
    async fn emits_context_usage_with_correct_used_and_budget() {
        // A modest fill well under the 0.85 line: report used/budget verbatim,
        // compaction not active.
        let reports = capture_context_usage(12_000, 32_000, 4).await;
        // One report for THIS single-round turn. Per-round cadence (a turn with
        // N tool rounds emits N reports) is covered by
        // `multi_round_turn_emits_one_usage_report_per_round`.
        assert_eq!(reports.len(), 1, "one usage report for a single-round turn");
        let r = reports[0];
        assert_eq!(r.used_tokens, 12_000);
        assert_eq!(r.budget_tokens, 32_000);
        assert!(!r.compaction_active);
    }

    #[tokio::test]
    async fn emits_context_usage_at_0_85_boundary_without_compaction() {
        // Exactly at the threshold: the pressure branch uses `>` (strictly
        // greater), so being *at* 0.85 does NOT trigger compaction. The
        // 0.85 amber colour decision is the client's; the daemon only flags
        // compaction when it actually ran. 27_200 == 0.85 * 32_000.
        let reports = capture_context_usage(27_200, 32_000, 4).await;
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].used_tokens, 27_200);
        assert_eq!(reports[0].budget_tokens, 32_000);
        assert!(
            !reports[0].compaction_active,
            "at exactly 0.85 the strict `>` threshold must not flag compaction"
        );
    }

    #[tokio::test]
    async fn emits_context_usage_flagging_compaction_when_window_shrinks() {
        // Above the threshold with enough primed history that the window can
        // actually shrink → compaction ran → flag set. used > budget here
        // (overflow), which clients render red.
        let reports = capture_context_usage(40_000, 32_000, MAX_CONTEXT_MESSAGES + 20).await;
        assert_eq!(reports.len(), 1);
        let r = reports[0];
        assert_eq!(r.used_tokens, 40_000);
        assert_eq!(r.budget_tokens, 32_000);
        assert!(
            r.used_tokens > r.budget_tokens,
            "overflow case: used exceeds budget"
        );
        assert!(
            r.compaction_active,
            "above threshold with shrinkable window must flag compaction"
        );
    }

    #[tokio::test]
    async fn no_context_usage_emitted_when_budget_unset() {
        use crate::ports::llm::{ContextUsage, ContextUsageSink, with_context_usage_sink};

        // No budget installed (foreground send / background job): the
        // token-pressure branch is gated on `current_context_budget()`, so
        // no usage is reported even though the LLM reported input tokens.
        // This is the "used==0 at turn start / budget unknown" graceful case
        // — clients simply never see a report and render nothing.
        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens: 5_000,
                max_context: Some(32_000),
            },
            Box::new(|| "conv-1".to_string()),
        );
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_sink = Arc::clone(&captured);
        let sink: ContextUsageSink = Arc::new(move |u: ContextUsage| {
            captured_for_sink.lock().unwrap().push(u);
        });
        // Sink installed, but NO `with_context_budget` wrapper.
        with_context_usage_sink(sink, async {
            handler
                .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                .await
                .unwrap();
        })
        .await;

        assert!(
            captured.lock().unwrap().is_empty(),
            "no budget installed → no context-usage report"
        );
    }

    // --- Overflow-recovery tests ---

    /// LLM that returns `ContextOverflow` for a configurable number of
    /// calls before succeeding. Tracks call count so tests can assert on it.
    struct OverflowThenSucceedLlm {
        remaining_overflows: Mutex<u32>,
        call_count: Arc<AtomicU32>,
        ok_text: String,
    }

    #[async_trait::async_trait]
    impl LlmClient for OverflowThenSucceedLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let mut remaining = self.remaining_overflows.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                return Err(CoreError::ContextOverflow {
                    prompt_tokens: Some(203_524),
                    max_tokens: Some(200_000),
                    detail: "Bedrock validation error: prompt is too long".into(),
                });
            }
            drop(remaining);
            on_chunk(self.ok_text.clone());
            Ok(LlmResponse::text(self.ok_text.clone()))
        }
    }

    #[tokio::test]
    async fn recovery_step1_truncates_largest_tool_result() {
        // Step 1 of the ladder: when there is at least one tool result
        // bigger than MIN_TRUNCATION_TOKENS (in estimated tokens), truncate
        // the largest and retry.
        // Smaller tool results stay untouched.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "all done".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        // Prime the conversation with three tool results: one tiny, one
        // medium-but-still-below-threshold, one well above the threshold.
        // Only the third should be truncated.
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1", "tiny", "{}",
            )]));
        stored.messages.push(Message::tool_result("c1", "ok"));
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c2", "medium", "{}",
            )]));
        // 2048 chars ≈ 512 tokens (chars/4 default) — below the
        // 1024-token threshold, so step 1 should leave it alone.
        let medium_content = "m".repeat((MIN_TRUNCATION_TOKENS * 2) as usize);
        stored
            .messages
            .push(Message::tool_result("c2", &medium_content));
        // 16384 chars ≈ 4096 tokens — well above the 1024-token threshold.
        let big_content = "X".repeat((MIN_TRUNCATION_TOKENS * 16) as usize);
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c3", "big", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result("c3", &big_content));
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "what happened?".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        assert_eq!(result, "all done");
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            2,
            "expected one overflow + one retry"
        );

        let after = handler.get_conversation(&conv.id).await.unwrap();
        let small = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c1"))
            .expect("small tool result present");
        assert_eq!(small.content, "ok", "small tool result must be untouched");
        let medium = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c2"))
            .expect("medium tool result present");
        assert_eq!(
            medium.content, medium_content,
            "below-threshold tool result must be untouched"
        );
        let big = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c3"))
            .expect("big tool result present");
        assert!(
            big.content.starts_with("<tool output omitted"),
            "expected truncation notice, got: {:?}",
            &big.content
        );
        assert!(
            big.content
                .contains(&format!("{} bytes", big_content.len()))
        );
    }

    #[tokio::test]
    async fn recovery_step2_trims_oldest_pairs_when_no_large_results() {
        // Step 2 of the ladder: when no tool result is large enough to be
        // worth truncating but multiple tool-pair groups exist, drop the
        // oldest groups via `trim_tool_pairs`. The most recent group must
        // survive.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "ok".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        // Four tool-pair groups, all with tiny results (below
        // MIN_TRUNCATION_TOKENS in estimated tokens) so step 1 declines.
        for i in 1..=4 {
            stored
                .messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("c{i}"),
                    "tiny",
                    "{}",
                )]));
            stored
                .messages
                .push(Message::tool_result(format!("c{i}"), format!("res-{i}")));
        }
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "ok");

        let after = handler.get_conversation(&conv.id).await.unwrap();
        // The most recent group must remain intact, regardless of how many
        // older groups got trimmed.
        let kept_recent = after
            .messages
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some("c4"));
        assert!(kept_recent, "the most recent tool group must survive");
        let dropped_oldest = !after
            .messages
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some("c1"));
        assert!(dropped_oldest, "the oldest tool group must be trimmed");
    }

    #[tokio::test]
    async fn recovery_step3_summarises_when_nothing_to_trim() {
        // Step 3 of the ladder: with no tool results to truncate and no
        // tool-pair groups to trim, recovery falls through to summarising
        // and shrinking the active window. The rolling summary on the
        // conversation should advance after recovery runs.
        use std::sync::atomic::AtomicU64;

        struct OverflowThenSucceedWithSummary {
            remaining_overflows: Mutex<u32>,
            ok_text: String,
            summary_text: String,
            call_count: Arc<AtomicU32>,
        }

        #[async_trait::async_trait]
        impl LlmClient for OverflowThenSucceedWithSummary {
            fn max_context_tokens(&self) -> Option<u64> {
                Some(200_000)
            }

            async fn stream_completion(
                &self,
                messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                // The summary-generation call passes a system prompt that
                // contains "conversation summarizer". Detect it and reply
                // with the canned summary text instead of the OK text.
                let is_summary_call = messages.iter().any(|m| {
                    m.role == Role::System && m.content.contains("conversation summarizer")
                });
                if is_summary_call {
                    on_chunk(self.summary_text.clone());
                    return Ok(LlmResponse::text(self.summary_text.clone()));
                }
                self.call_count.fetch_add(1, Ordering::Relaxed);
                let mut remaining = self.remaining_overflows.lock().unwrap();
                if *remaining > 0 {
                    *remaining -= 1;
                    return Err(CoreError::ContextOverflow {
                        prompt_tokens: Some(300_000),
                        max_tokens: Some(200_000),
                        detail: "prompt too long".into(),
                    });
                }
                drop(remaining);
                on_chunk(self.ok_text.clone());
                Ok(LlmResponse::text(self.ok_text.clone()))
            }
        }

        let call_count = Arc::new(AtomicU32::new(0));
        let counter = Arc::new(AtomicU64::new(0));
        let llm = OverflowThenSucceedWithSummary {
            remaining_overflows: Mutex::new(1),
            ok_text: "done".into(),
            summary_text: "- recovery summary".into(),
            call_count: Arc::clone(&call_count),
        };
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        // Prime with enough plain User/Assistant turns to push the window
        // past `MAX_CONTEXT_MESSAGES`, so step 3 has a non-empty range to
        // summarise. No tool calls are present, so steps 1 and 2 decline.
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..(MAX_CONTEXT_MESSAGES + 4) {
            if i % 2 == 0 {
                stored
                    .messages
                    .push(Message::new(Role::User, format!("u-{i}")));
            } else {
                stored
                    .messages
                    .push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        handler.store.update(stored).await.unwrap();
        let baseline = handler
            .get_conversation(&conv.id)
            .await
            .unwrap()
            .context_summary
            .clone();

        let result = handler
            .send_prompt(&conv.id, "follow-up".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "done");
        // One overflow + one retry from the main path; the inner summary
        // call doesn't bump call_count.
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            2,
            "expected one overflow + one retry"
        );

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            after.context_summary != baseline && !after.context_summary.is_empty(),
            "step 3 must update the rolling summary; got: {:?}",
            after.context_summary
        );
    }

    #[tokio::test]
    async fn recovery_exhausts_retries_then_surfaces() {
        // After MAX_OVERFLOW_RETRIES recoveries the loop must surface a
        // user-visible error rather than spin forever.
        struct AlwaysOverflowLlm {
            call_count: Arc<AtomicU32>,
        }
        #[async_trait::async_trait]
        impl LlmClient for AlwaysOverflowLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                self.call_count.fetch_add(1, Ordering::Relaxed);
                Err(CoreError::ContextOverflow {
                    prompt_tokens: Some(300_000),
                    max_tokens: Some(200_000),
                    detail: "prompt is too long".into(),
                })
            }
        }

        use std::sync::atomic::AtomicU64;
        let call_count = Arc::new(AtomicU32::new(0));
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            AlwaysOverflowLlm {
                call_count: Arc::clone(&call_count),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1", "t", "{}",
            )]));
        stored.messages.push(Message::tool_result(
            "c1",
            "x".repeat((MIN_TRUNCATION_TOKENS * 8) as usize),
        ));
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert!(result.to_ascii_lowercase().contains("context"));

        // MAX_OVERFLOW_RETRIES + 1 calls total: the recovered attempts plus
        // the final one whose error gets surfaced.
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            MAX_OVERFLOW_RETRIES + 1,
            "should stop after bounded retries"
        );
    }

    #[tokio::test]
    async fn active_task_anchor_set_on_user_prompt() {
        let handler = make_handler(vec!["ok"]);
        let conv = handler
            .create_conversation("Chat".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(
                &conv.id,
                "refactor the auth module".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .expect("send_prompt succeeds");

        let stored = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(
            stored.active_task.as_deref(),
            Some("refactor the auth module"),
            "the user's prompt should be captured as the active-task anchor"
        );
    }

    // --- Tool namespace categorization cache tests ---
    //
    // These exercise the `namespace_cache` on `ConversationHandler` by driving
    // `send_prompt` end-to-end with a mock LLM that supports hosted tool
    // search. The mock recognises the categorization call by its system prompt
    // and counts how many times the categorizer is invoked across calls.

    /// System-prompt fragment unique to `categorize_tool_namespaces`.
    /// Used by the mock LLM to distinguish categorization calls from
    /// regular completion calls.
    const CATEGORIZATION_SYSTEM_FRAGMENT: &str = "You organize tools into semantic categories";

    /// Mock LLM that:
    /// - Reports hosted tool search support so the cache path runs.
    /// - Counts categorization calls (system prompt fragment match) and
    ///   returns a deterministic JSON categorization for them.
    /// - For all other calls returns plain text so `send_prompt` exits.
    struct CategorizingLlm {
        categorization_calls: Arc<AtomicU32>,
        category_payload: Mutex<String>,
        /// Artificial delay applied inside the categorization branch so a test
        /// can widen the window two concurrent cold turns overlap in.
        categorization_delay: std::time::Duration,
    }

    impl CategorizingLlm {
        fn new(category_payload: String) -> Self {
            Self {
                categorization_calls: Arc::new(AtomicU32::new(0)),
                category_payload: Mutex::new(category_payload),
                categorization_delay: std::time::Duration::ZERO,
            }
        }

        fn with_categorization_delay(mut self, delay: std::time::Duration) -> Self {
            self.categorization_delay = delay;
            self
        }

        fn calls(&self) -> Arc<AtomicU32> {
            Arc::clone(&self.categorization_calls)
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for CategorizingLlm {
        fn supports_hosted_tool_search(&self) -> bool {
            true
        }

        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let is_categorization = messages.iter().any(|m| {
                matches!(m.role, Role::System) && m.content.contains(CATEGORIZATION_SYSTEM_FRAGMENT)
            });
            if is_categorization {
                self.categorization_calls.fetch_add(1, Ordering::SeqCst);
                if !self.categorization_delay.is_zero() {
                    tokio::time::sleep(self.categorization_delay).await;
                }
                let payload = self.category_payload.lock().unwrap().clone();
                return Ok(LlmResponse::text(payload));
            }
            let text = "ok".to_string();
            on_chunk(text.clone());
            Ok(LlmResponse::text(text))
        }
    }

    /// Mock tool executor with a mutable namespace set so individual tests
    /// can edit names/descriptions between `send_prompt` calls.
    struct NamespacedToolExecutor {
        namespaces: Mutex<Vec<ToolNamespace>>,
    }

    impl NamespacedToolExecutor {
        fn new(namespaces: Vec<ToolNamespace>) -> Self {
            Self {
                namespaces: Mutex::new(namespaces),
            }
        }

        fn mutate<F: FnOnce(&mut Vec<ToolNamespace>)>(&self, f: F) {
            let mut guard = self.namespaces.lock().unwrap();
            f(&mut guard);
        }
    }

    impl ToolExecutor for NamespacedToolExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            Vec::new()
        }

        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(Vec::new())
        }

        async fn tool_definition(&self, _name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(None)
        }

        async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
            self.namespaces.lock().unwrap().clone()
        }

        async fn execute_tool(
            &self,
            name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            Err(CoreError::ToolExecution(format!("unexpected exec: {name}")))
        }
    }

    /// Build a single namespace containing `count` distinct tools so the
    /// total tool count exceeds `categorize_tool_namespaces`'s skip threshold.
    fn make_oversized_namespace(count: usize) -> ToolNamespace {
        let tools: Vec<ToolDefinition> = (0..count)
            .map(|i| {
                ToolDefinition::new(
                    format!("tool_{i}"),
                    format!("Description for tool {i}"),
                    serde_json::json!({"type": "object"}),
                )
            })
            .collect();
        ToolNamespace::new("seed_namespace", "Seed namespace for tests", tools)
    }

    /// Categorization payload that puts every `tool_*` into one bucket.
    /// Construction matches `make_oversized_namespace` so the LLM-shaped
    /// JSON is internally consistent and `categorize_tool_namespaces`
    /// accepts it (every tool appears in exactly one category).
    fn make_categorization_payload(count: usize) -> String {
        let names: Vec<String> = (0..count).map(|i| format!("\"tool_{i}\"")).collect();
        format!(
            r#"[{{"name":"all","description":"All tools","tools":[{}]}}]"#,
            names.join(",")
        )
    }

    fn build_categorization_handler(
        executor: NamespacedToolExecutor,
        llm: CategorizingLlm,
    ) -> ConversationHandler<MockStore, CategorizingLlm, NamespacedToolExecutor> {
        use std::sync::atomic::{AtomicU64, Ordering as IdOrdering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            executor,
            Box::new(move || {
                let n = counter.fetch_add(1, IdOrdering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    #[tokio::test]
    async fn categorization_cache_hits_on_unchanged_tools() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt with valid conv must succeed");
        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt with valid conv must succeed");

        // Cache hit on second call: categorizer runs at most once.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "categorizer should run once and be served from cache thereafter"
        );
    }

    #[tokio::test]
    async fn categorization_cache_invalidates_on_description_change() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt must succeed");

        // Mutate a description without changing any name. Without
        // descriptions in the hash, the cache would falsely hit.
        handler.tools.mutate(|namespaces| {
            namespaces[0].tools[0].description = "Description for tool 0 (edited)".to_string();
        });

        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt must succeed");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "description change must invalidate the categorization cache"
        );
    }

    #[tokio::test]
    async fn categorization_cache_invalidates_on_tool_addition() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        // Pre-build a payload that covers the post-addition tool set so the
        // second categorization call returns valid JSON for `count + 1` tools.
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt must succeed");

        // Add a tool, then update the LLM's stored payload so the second
        // categorization succeeds (and thus actually runs end-to-end).
        handler.tools.mutate(|namespaces| {
            namespaces[0].tools.push(ToolDefinition::new(
                format!("tool_{count}"),
                "Description for added tool",
                serde_json::json!({"type": "object"}),
            ));
        });
        *handler.llm.category_payload.lock().unwrap() = make_categorization_payload(count + 1);

        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt must succeed");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "tool addition must invalidate the categorization cache"
        );
    }

    /// Item 8: two concurrent cold turns (different conversations, shared
    /// handler) must coalesce into ONE categorization LLM call. A categorization
    /// delay guarantees both turns are simultaneously past the cache-miss check;
    /// without the single-flight guard both would invoke the categorizer.
    #[tokio::test]
    async fn concurrent_cold_turns_coalesce_categorization() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count))
            .with_categorization_delay(std::time::Duration::from_millis(100));
        let calls = llm.calls();
        let handler = Arc::new(build_categorization_handler(executor, llm));

        // Two distinct conversations so the turns take different per-conversation
        // turn locks and genuinely run in parallel (only the categorization
        // single-flight may serialize them).
        let conv_a = handler
            .create_conversation("A".into(), vec![])
            .await
            .unwrap();
        let conv_b = handler
            .create_conversation("B".into(), vec![])
            .await
            .unwrap();

        let h1 = handler.clone();
        let ida = conv_a.id.clone();
        let t1 = tokio::spawn(async move {
            h1.send_prompt(&ida, "a".into(), noop_callback(), noop_status())
                .await
        });
        let h2 = handler.clone();
        let idb = conv_b.id.clone();
        let t2 = tokio::spawn(async move {
            h2.send_prompt(&idb, "b".into(), noop_callback(), noop_status())
                .await
        });

        t1.await.unwrap().expect("turn a succeeds");
        t2.await.unwrap().expect("turn b succeeds");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "concurrent cold turns must coalesce into one categorization call"
        );
    }

    #[tokio::test]
    async fn categorization_skipped_when_listing_fits_budget() {
        use crate::ports::llm::with_context_budget;

        // Generous budget + short tool descriptions — the raw listing
        // sums well below 10% of the budget, so categorization should
        // skip the LLM round-trip and return the input namespaces.
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
                .await
                .expect("invariant: first send_prompt must succeed");
        })
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "categorizer must not be called when the full listing fits the budget"
        );
    }

    #[tokio::test]
    async fn categorization_runs_when_listing_too_large() {
        use crate::ports::llm::with_context_budget;

        // Same setup but with very long tool descriptions, so the raw
        // listing pushes past the 10% threshold and categorization runs.
        let count = 12;
        let big_desc = "DESCRIPTION ".repeat(1000); // ~12 KB per tool
        let tools: Vec<ToolDefinition> = (0..count)
            .map(|i| {
                ToolDefinition::new(
                    format!("tool_{i}"),
                    big_desc.clone(),
                    serde_json::json!({"type": "object"}),
                )
            })
            .collect();
        let namespace = ToolNamespace::new("seed", "seed namespace", tools);
        let executor = NamespacedToolExecutor::new(vec![namespace]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
                .await
                .expect("invariant: first send_prompt must succeed");
        })
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "categorizer must run once when the raw listing exceeds the budget threshold"
        );
    }

    // --- Scratchpad: conversation scoping + goal anchor ---

    /// Tool executor that records the task-local conversation id observed
    /// during `execute_tool`, proving the dispatch loop installs it.
    struct ConvIdCapturingExecutor {
        tools: Vec<ToolDefinition>,
        observed: Arc<Mutex<Option<ConversationId>>>,
    }

    impl ToolExecutor for ConvIdCapturingExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
        }
        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }
        async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(self.tools.iter().find(|t| t.name == name).cloned())
        }
        async fn execute_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            *self.observed.lock().unwrap() =
                crate::ports::conversation_ctx::current_conversation_id();
            Ok("ok".to_string())
        }
    }

    #[tokio::test]
    async fn conversation_id_scoped_during_tool_execution() {
        let observed: Arc<Mutex<Option<ConversationId>>> = Arc::new(Mutex::new(None));
        let tool = ToolDefinition::new("noop", "noop", serde_json::json!({"type": "object"}));
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "noop", "{}")]),
            LlmResponse::text("done"),
        ];
        let executor = ConvIdCapturingExecutor {
            tools: vec![tool],
            observed: Arc::clone(&observed),
        };
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(responses),
            executor,
            Box::new(|| "conv-scope-1".to_string()),
        );
        let conv = handler
            .create_conversation("t".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(
            observed.lock().unwrap().clone(),
            Some(conv.id.clone()),
            "execute_tool must observe the conversation as a task-local"
        );
    }

    /// LLM that captures the messages from every invocation, so we can assert
    /// how the task anchor was assembled. (First-message title generation
    /// also calls `stream_completion`, so we record all calls and inspect
    /// them collectively rather than keeping only the last.)
    struct MessageCapturingLlm {
        captured: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    #[async_trait::async_trait]
    impl LlmClient for MessageCapturingLlm {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.captured.lock().unwrap().push(messages);
            Ok(LlmResponse::text("done"))
        }
    }

    fn goal_reader(content: &'static str) -> ScratchpadGetManyFn {
        Arc::new(move |conv: String, keys: Vec<String>, _limit: usize| {
            Box::pin(async move {
                // Only the reserved goal key resolves to a note.
                if keys.iter().any(|k| k == SCRATCHPAD_GOAL_KEY) {
                    Ok(vec![crate::domain::ScratchpadNote::new(
                        "g",
                        conv,
                        SCRATCHPAD_GOAL_KEY,
                        content,
                    )])
                } else {
                    Ok(vec![])
                }
            })
        })
    }

    /// Find a `[Current task]` anchor system message across all captured
    /// LLM invocations.
    fn find_anchor(captures: &[Vec<Message>]) -> Option<String> {
        captures
            .iter()
            .flatten()
            .find(|m| m.role == Role::System && m.content.starts_with("[Current task]"))
            .map(|m| m.content.clone())
    }

    #[tokio::test]
    async fn scratchpad_goal_is_surfaced_as_task_anchor() {
        let captured: Arc<Mutex<Vec<Vec<Message>>>> = Arc::new(Mutex::new(Vec::new()));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            MessageCapturingLlm {
                captured: Arc::clone(&captured),
            },
            NoopToolExecutor,
            Box::new(|| "conv-goal-1".to_string()),
        )
        .with_scratchpad_goal(goal_reader("Ship the scratchpad, then promote learnings"));

        let conv = handler
            .create_conversation("t".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(
                &conv.id,
                "what next?".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        let anchor = find_anchor(&captured.lock().unwrap())
            .expect("a [Current task] anchor must be injected from the goal note");
        assert!(
            anchor.contains("Ship the scratchpad, then promote learnings"),
            "anchor must carry the goal note content, got {anchor:?}"
        );
        assert!(
            !anchor.contains("what next?"),
            "the evolving goal must take precedence over the verbatim prompt"
        );
    }

    #[tokio::test]
    async fn anchor_falls_back_to_prompt_when_no_goal_note() {
        // With a goal reader that returns nothing, the verbatim prompt remains
        // the anchor source — and since it's a visible user message in a
        // single-turn conversation, no [Current task] line is injected.
        let captured: Arc<Mutex<Vec<Vec<Message>>>> = Arc::new(Mutex::new(Vec::new()));
        let empty_reader: ScratchpadGetManyFn =
            Arc::new(|_c, _k, _l| Box::pin(async { Ok(vec![]) }));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            MessageCapturingLlm {
                captured: Arc::clone(&captured),
            },
            NoopToolExecutor,
            Box::new(|| "conv-goal-2".to_string()),
        )
        .with_scratchpad_goal(empty_reader);

        let conv = handler
            .create_conversation("t".into(), vec![])
            .await
            .unwrap();
        handler
            .send_prompt(&conv.id, "just this".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        assert!(
            find_anchor(&captured.lock().unwrap()).is_none(),
            "no anchor should be injected when there's no goal and the prompt is visible"
        );
    }

    // --- Per-request system-prompt refinement --------------------------------

    /// A distinctive marker the test injects as the refinement so it can be
    /// found unambiguously in the captured system prompt.
    const REFINEMENT_MARKER: &str =
        "You are Adele, responding by voice. Keep replies to one or two sentences.";

    /// Opening of the static identity section — proves the BASE system prompt
    /// is still present alongside the refinement.
    const BASE_PROMPT_MARKER: &str = "You are Adele, a desktop assistant named in reference";

    /// Find the primary system instruction (the first `Role::System` message,
    /// which is the assembled static + tool-availability + refinement block)
    /// across all captured LLM invocations.
    fn first_system_message(captures: &[Vec<Message>]) -> Option<String> {
        captures
            .iter()
            .flatten()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone())
    }

    #[tokio::test]
    async fn system_refinement_is_appended_to_system_prompt_for_the_request() {
        let captured: Arc<Mutex<Vec<Vec<Message>>>> = Arc::new(Mutex::new(Vec::new()));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            MessageCapturingLlm {
                captured: Arc::clone(&captured),
            },
            NoopToolExecutor,
            Box::new(|| "conv-refine-1".to_string()),
        );
        let conv = handler
            .create_conversation("t".into(), vec![])
            .await
            .unwrap();

        // Install the per-request refinement the way the daemon dispatch
        // wrapper does (a task-local around the send), then send a clean
        // prompt.
        crate::ports::llm::with_system_refinement(REFINEMENT_MARKER.to_string(), async {
            handler
                .send_prompt(
                    &conv.id,
                    "what's the weather?".into(),
                    noop_callback(),
                    noop_status(),
                )
                .await
                .unwrap();
        })
        .await;

        // The system prompt sent to the LLM carries BOTH the base prompt and
        // the refinement.
        let system = first_system_message(&captured.lock().unwrap())
            .expect("a system message must be present in the LLM request");
        assert!(
            system.contains(BASE_PROMPT_MARKER),
            "system prompt must still contain the base prompt, got: {system:?}"
        );
        assert!(
            system.contains(REFINEMENT_MARKER),
            "system prompt must contain the appended refinement, got: {system:?}"
        );
        // The refinement is appended AFTER the base prompt, not prepended.
        let base_at = system.find(BASE_PROMPT_MARKER).unwrap();
        let refine_at = system.find(REFINEMENT_MARKER).unwrap();
        assert!(
            refine_at > base_at,
            "refinement must come after the base system prompt"
        );

        // The stored conversation contains ONLY the clean user prompt and the
        // assistant reply — the refinement is never persisted as a message.
        let stored = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(stored.messages.len(), 2);
        assert_eq!(stored.messages[0].role, Role::User);
        assert_eq!(stored.messages[0].content, "what's the weather?");
        assert_eq!(stored.messages[1].role, Role::Assistant);
        for m in &stored.messages {
            assert!(
                !m.content.contains(REFINEMENT_MARKER),
                "the refinement must never appear in stored conversation messages, got: {:?}",
                m.content
            );
        }
        // And it must not have been stashed on the conversation's active_task
        // anchor either — that's the user's prompt, not the refinement.
        assert_eq!(stored.active_task.as_deref(), Some("what's the weather?"));
    }

    #[tokio::test]
    async fn empty_system_refinement_leaves_system_prompt_unchanged() {
        // Capture a turn WITH a refinement installed and one WITHOUT, and
        // assert the no-refinement system prompt equals the prompt produced
        // when the refinement scope is simply absent (the default path).
        async fn capture_system_prompt(refinement: Option<&str>) -> String {
            let captured: Arc<Mutex<Vec<Vec<Message>>>> = Arc::new(Mutex::new(Vec::new()));
            let handler = ConversationHandler::with_tools(
                MockStore::new(),
                MessageCapturingLlm {
                    captured: Arc::clone(&captured),
                },
                NoopToolExecutor,
                Box::new(|| "conv-refine-2".to_string()),
            );
            let conv = handler
                .create_conversation("t".into(), vec![])
                .await
                .unwrap();
            let send = async {
                handler
                    .send_prompt(&conv.id, "hi".into(), noop_callback(), noop_status())
                    .await
                    .unwrap();
            };
            match refinement {
                Some(r) => {
                    crate::ports::llm::with_system_refinement(r.to_string(), send).await;
                }
                None => send.await,
            }
            first_system_message(&captured.lock().unwrap()).expect("system message present")
        }

        // An explicitly empty refinement must produce the identical system
        // prompt to never installing one at all.
        let no_scope = capture_system_prompt(None).await;
        let empty_scope = capture_system_prompt(Some("")).await;
        let whitespace_scope = capture_system_prompt(Some("   \n  ")).await;
        assert_eq!(
            no_scope, empty_scope,
            "an empty refinement must not change the system prompt"
        );
        assert_eq!(
            no_scope, whitespace_scope,
            "a whitespace-only refinement must not change the system prompt"
        );
        assert!(
            !no_scope.contains(REFINEMENT_MARKER),
            "no refinement marker should leak into the baseline prompt"
        );
    }

    // ================================================================
    // Round-loop fallback/branch coverage (issue #441).
    // ================================================================

    /// A batch of tool calls in one assistant turn where the middle call has
    /// malformed JSON arguments: the parse error must be folded into a
    /// `tool_result` for *that* call while the good calls on either side still
    /// execute and pair. Guards the `continue` at the parse-error arm — a
    /// `break`/`return` there would strand the later calls unpaired (a provider
    /// 400) and skip real tool work.
    #[tokio::test]
    async fn malformed_arg_in_batch_still_pairs_all_calls() {
        let tool_def = ToolDefinition::new("read_file", "Read", serde_json::json!({}));
        let good1 = ToolCall::new("c1", "read_file", r#"{"path":"/a"}"#);
        let bad = ToolCall::new("c2", "read_file", "{ this is not json");
        let good2 = ToolCall::new("c3", "read_file", r#"{"path":"/b"}"#);

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![good1, bad, good2]),
            LlmResponse::text("done"),
        ];
        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "content".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "done");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let tool_msg = |id: &str| {
            updated
                .messages
                .iter()
                .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(id))
                .unwrap_or_else(|| panic!("a tool_result must be paired for {id}"))
                .content
                .clone()
        };
        // All three calls paired.
        assert_eq!(tool_msg("c1"), "content", "first good call must execute");
        assert!(
            tool_msg("c2").contains("not valid JSON"),
            "malformed call must surface a parse error, got: {}",
            tool_msg("c2")
        );
        assert_eq!(
            tool_msg("c3"),
            "content",
            "the good call AFTER the malformed one must still execute"
        );
    }

    /// After a tool round that yields empty visible text, the loop substitutes a
    /// fixed "tools returned errors" recovery message — but ONLY when `round >
    /// 0`. An empty text-only reply on round 0 stays empty.
    #[tokio::test]
    async fn empty_after_tool_round_uses_canned_text() {
        // Case A: empty text on round 1 (after a tool round) → canned recovery.
        let tool_def = ToolDefinition::new("t", "T", serde_json::json!({}));
        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "t", "{}")]),
            LlmResponse::text(""), // empty visible text on round 1
        ];
        let mut tr = HashMap::new();
        tr.insert("t".to_string(), "ran".to_string());
        let handler = make_tool_handler(responses, vec![tool_def], tr);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert!(
            result.contains("tools I tried") && result.contains("returned errors"),
            "empty text after a tool round must use the canned recovery message, got: {result:?}"
        );

        // Case B: empty text on round 0 (no prior tool round) stays empty.
        let handler0 = make_handler(vec![]); // MockLlm returns "" for empty chunks
        let conv0 = handler0
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result0 = handler0
            .send_prompt(&conv0.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(
            result0, "",
            "an empty reply on round 0 must stay empty, not use the canned text"
        );
    }

    /// Enough tool-call responses to burn the whole round budget, then one
    /// trailing response the tool-free wind-down completion returns.
    fn exhausting_responses(closing: LlmResponse) -> Vec<LlmResponse> {
        let mut responses: Vec<LlmResponse> = (0..MAX_TOOL_ROUNDS)
            .map(|i| {
                LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new(format!("c{i}"), "loop_tool", "{}")],
                )
            })
            .collect();
        responses.push(closing);
        responses
    }

    /// #453 FIX: exhausting `MAX_TOOL_ROUNDS` no longer drops the turn. The
    /// daemon does a bounded, tool-free wind-down (one final completion with no
    /// tools offered) and persists the whole turn — the user's prompt, the tool
    /// transcript, and the model's closing summary — so the conversation can be
    /// continued instead of silently vanishing.
    #[tokio::test]
    async fn max_rounds_exhaustion_winds_down_and_persists_turn() {
        let tools = vec![ToolDefinition::new(
            "loop_tool",
            "Loops",
            serde_json::json!({}),
        )];
        let responses = exhausting_responses(LlmResponse::text(
            "I hit the tool-call limit before finishing. Done: read the files. \
             Still to do: apply the edit. Say continue and I'll pick up.",
        ));
        let mut tool_results = HashMap::new();
        tool_results.insert("loop_tool".to_string(), "ok".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "loop forever".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .expect("exhaustion now winds down to Ok, not Err");
        assert!(
            result.starts_with("I hit the tool-call limit"),
            "the fluent wind-down closing is returned, got: {result}"
        );

        // The turn is persisted, not lost: the user prompt is present, the
        // closing is the last message, and the tool transcript survived.
        let persisted = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            persisted
                .messages
                .iter()
                .any(|m| m.content == "loop forever"),
            "#453: the user's prompt MUST be persisted after exhaustion"
        );
        let last = persisted.messages.last().expect("non-empty history");
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(
            last.content, result,
            "closing summary is persisted verbatim"
        );
        assert!(
            persisted.messages.len() > 2,
            "the tool transcript must be preserved, got {} messages",
            persisted.messages.len()
        );
        // The transient wind-down instruction must never leak into history.
        assert!(
            !persisted
                .messages
                .iter()
                .any(|m| m.content.contains("Wrap up now")),
            "the transient wind-down instruction must not be persisted"
        );
    }

    /// #453: if the wind-down completion itself returns no usable text, a canned
    /// closing is persisted rather than an empty assistant turn — the turn is
    /// preserved either way.
    #[tokio::test]
    async fn max_rounds_exhaustion_falls_back_when_wind_down_is_empty() {
        let tools = vec![ToolDefinition::new(
            "loop_tool",
            "Loops",
            serde_json::json!({}),
        )];
        let responses = exhausting_responses(LlmResponse::text(""));
        let mut tool_results = HashMap::new();
        tool_results.insert("loop_tool".to_string(), "ok".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "loop forever".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .expect("exhaustion winds down to Ok even when the closing is empty");
        assert_eq!(result, WIND_DOWN_FALLBACK);

        let persisted = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            persisted
                .messages
                .iter()
                .any(|m| m.content == "loop forever"),
            "#453: the user's prompt MUST be persisted even on the fallback path"
        );
        assert_eq!(
            persisted.messages.last().unwrap().content,
            WIND_DOWN_FALLBACK
        );
    }

    /// A connector that supports hosted tool search but returns text-only on an
    /// early round is demoted to `builtin_tool_search` with a one-shot system
    /// nudge — but the demotion is gated to `round < 2`. Asserts both: the nudge
    /// is injected on a round-0 text-only reply, and no demotion happens when a
    /// text-only reply first arrives on round 2+.
    #[tokio::test]
    async fn hosted_search_demotion_injects_nudge_and_gates_round() {
        // A hosted-search-capable LLM that replays a scripted response list.
        struct HostedSearchLlm {
            responses: Mutex<Vec<LlmResponse>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for HostedSearchLlm {
            fn supports_hosted_tool_search(&self) -> bool {
                true
            }
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                let resp = {
                    let mut r = self.responses.lock().unwrap();
                    if r.is_empty() {
                        return Ok(LlmResponse::text("fallback"));
                    }
                    r.remove(0)
                };
                if !resp.text.is_empty() {
                    on_chunk(resp.text.clone());
                }
                Ok(resp)
            }
        }

        // 2-tool namespace (<=10 → no categorization LLM call), so hosted
        // search is active with a non-empty namespace set.
        fn ns() -> Vec<ToolNamespace> {
            vec![ToolNamespace::new(
                "grp",
                "a group",
                vec![
                    ToolDefinition::new("ns_tool_a", "a", serde_json::json!({})),
                    ToolDefinition::new("ns_tool_b", "b", serde_json::json!({})),
                ],
            )]
        }
        const NUDGE: &str = "server-side tool search was unable";

        // --- Case 1: round-0 text-only → demote + inject nudge. ---
        let llm = HostedSearchLlm {
            responses: Mutex::new(vec![
                LlmResponse::text("thinking out loud"), // round 0 text-only
                LlmResponse::text("final answer"),      // round 1 (demoted)
            ]),
        };
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            NamespacedToolExecutor::new(ns()),
            id_gen(),
        );
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result = handler
            .send_prompt(&conv.id, "help".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "final answer");
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            updated
                .messages
                .iter()
                .any(|m| m.role == Role::System && m.content.contains(NUDGE)),
            "a demotion nudge must be injected on an early text-only round"
        );
        // The pre-demotion assistant text is kept for context.
        assert!(
            updated
                .messages
                .iter()
                .any(|m| m.role == Role::Assistant && m.content == "thinking out loud"),
            "the pre-demotion assistant text must be preserved"
        );

        // --- Case 2: text-only first arrives on round 2 → NO demotion. ---
        // Rounds 0 and 1 make tool calls (so they're never text-only and never
        // demote); the text-only reply lands on round 2, where `round < 2` is
        // false, so no nudge is injected.
        let llm2 = HostedSearchLlm {
            responses: Mutex::new(vec![
                LlmResponse::with_tool_calls("", vec![ToolCall::new("t0", "ns_tool_a", "{}")]),
                LlmResponse::with_tool_calls("", vec![ToolCall::new("t1", "ns_tool_a", "{}")]),
                LlmResponse::text("done late"),
            ]),
        };
        let handler2 = ConversationHandler::with_tools(
            MockStore::new(),
            llm2,
            NamespacedToolExecutor::new(ns()),
            id_gen(),
        );
        let conv2 = handler2
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let result2 = handler2
            .send_prompt(&conv2.id, "help".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result2, "done late");
        let updated2 = handler2.get_conversation(&conv2.id).await.unwrap();
        assert!(
            !updated2.messages.iter().any(|m| m.content.contains(NUDGE)),
            "no demotion nudge when the text-only reply first arrives on round 2+"
        );
    }

    /// Cooperative-cancel conversion (issue #109): a connector that returns
    /// `Ok(partial)` because its chunk callback returned `false` after
    /// cancellation must still surface `Cancelled` at the post-stream
    /// `bail_if_cancelled()` — and the partial assistant text must NOT leak into
    /// history. All the other cancel tests use `Err(Cancelled)` directly; this
    /// exercises the `Ok`-then-bail conversion.
    #[tokio::test]
    async fn ok_partial_after_cancel_becomes_cancelled() {
        // Cancels the ambient turn token from inside the stream (simulating the
        // adapter observing cancellation) and then returns Ok with partial text.
        struct OkPartialThenCancelLlm;
        #[async_trait::async_trait]
        impl LlmClient for OkPartialThenCancelLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                if let Some(token) = current_cancellation_token() {
                    token.cancel();
                }
                // The real adapter would see this return `false` and stop; we
                // still hand back what was streamed so far as `Ok`.
                let _ = on_chunk("partial ".to_string());
                Ok(LlmResponse::text("partial text"))
            }
        }

        let handler = ConversationHandler::new(MockStore::new(), OkPartialThenCancelLlm, id_gen());
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();

        let token = CancellationToken::new();
        let result = crate::ports::llm::with_cancellation_token(
            token,
            handler.send_prompt(&conv.id, "go".into(), noop_callback(), noop_status()),
        )
        .await;

        assert!(
            matches!(result, Err(CoreError::Cancelled)),
            "an Ok(partial) after cancellation must convert to Cancelled, got {result:?}"
        );
        // The partial text must not have been persisted.
        let persisted = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            !persisted
                .messages
                .iter()
                .any(|m| m.content.contains("partial")),
            "partial post-cancel text must not leak into history, got {:?}",
            persisted.messages
        );
    }

    /// `step_stack.clear()` after overflow recovery (issue #240 / #441): a
    /// `begin_step` then an in-turn ContextOverflow recovery (which can drain
    /// messages and invalidate the frame's absolute watermark) must drop the
    /// step frames, so a later `complete_step` finds NO active step and does not
    /// evict via a stale watermark. Without the `clear()`, `complete_step` would
    /// pop the frame and act on the now-invalid watermark.
    #[tokio::test]
    async fn overflow_recovery_invalidates_step_watermarks() {
        // A scripted LLM that can inject a ContextOverflow on a chosen call.
        enum Step {
            Resp(LlmResponse),
            Overflow,
        }
        struct ScriptedOverflowLlm {
            steps: Mutex<Vec<Step>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for ScriptedOverflowLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                let step = {
                    let mut s = self.steps.lock().unwrap();
                    if s.is_empty() {
                        return Ok(LlmResponse::text("fallback"));
                    }
                    s.remove(0)
                };
                match step {
                    Step::Overflow => Err(CoreError::ContextOverflow {
                        prompt_tokens: Some(203_524),
                        max_tokens: Some(200_000),
                        detail: "prompt is too long".into(),
                    }),
                    Step::Resp(r) => {
                        if !r.text.is_empty() {
                            on_chunk(r.text.clone());
                        }
                        Ok(r)
                    }
                }
            }
        }

        let (write, list, _sp) = in_memory_scratchpad();
        let llm = ScriptedOverflowLlm {
            steps: Mutex::new(vec![
                Step::Resp(LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new("b1", "begin_step", r#"{"goal":"do work"}"#)],
                )),
                Step::Overflow, // triggers recover_from_overflow + step_stack.clear()
                Step::Resp(LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new("c1", "complete_step", "{}")],
                )),
                Step::Resp(LlmResponse::text("all done")),
            ]),
        };
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            MockToolExecutor::new(vec![], HashMap::new()),
            id_gen(),
        )
        .with_scratchpad_write(write)
        .with_scratchpad_list(list);

        // Prime several small tool-pair groups so: (a) is_first_message is
        // false (no title call), and (b) overflow-recovery step 2 trims the
        // oldest pairs, actually draining messages and shifting watermarks.
        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..3 {
            stored
                .messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("p{i}"),
                    "prior",
                    "{}",
                )]));
            stored
                .messages
                .push(Message::tool_result(format!("p{i}"), "ok"));
        }
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "all done");

        // The complete_step ack must report NO active step: the frame was
        // cleared by overflow recovery, so it neither marked a todo done nor
        // evicted via the stale watermark.
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let complete_ack = updated
            .messages
            .iter()
            .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c1"))
            .expect("complete_step ack must be recorded")
            .content
            .clone();
        assert!(
            complete_ack.contains("no active step to complete"),
            "after overflow recovery cleared the stack, complete_step must find no \
             active step (got: {complete_ack})"
        );
    }

    /// Context-usage cadence across a multi-round turn: every round that reports
    /// usage emits exactly one usage report (so a 2-round turn emits 2), and
    /// `compaction_active` is per-round — false on an early below-threshold
    /// round, true on a later round that crosses the threshold and shrinks.
    #[tokio::test]
    async fn multi_round_turn_emits_one_usage_report_per_round() {
        use crate::ports::llm::{
            ContextUsage, ContextUsageSink, with_context_budget, with_context_usage_sink,
        };

        // A tool-calling LLM that attaches per-call usage. Auxiliary calls
        // (summary/title) return a canned response WITHOUT consuming the script.
        struct MultiRoundUsageLlm {
            script: Mutex<Vec<(LlmResponse, u64)>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for MultiRoundUsageLlm {
            async fn stream_completion(
                &self,
                messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                let is_aux = messages.iter().any(|m| {
                    matches!(m.role, Role::System)
                        && (m.content.contains("conversation summarizer")
                            || m.content.contains("channel name"))
                });
                if is_aux {
                    return Ok(LlmResponse::text("aux"));
                }
                let (resp, tokens) = {
                    let mut s = self.script.lock().unwrap();
                    if s.is_empty() {
                        return Ok(LlmResponse::text("fallback"));
                    }
                    s.remove(0)
                };
                if !resp.text.is_empty() {
                    on_chunk(resp.text.clone());
                }
                let usage = TokenUsage {
                    input_tokens: Some(tokens),
                    output_tokens: Some(1),
                    ..Default::default()
                };
                Ok(resp.with_usage(usage))
            }
        }

        let budget_max = 32_000u64; // threshold = 0.85 * 32_000 = 27_200
        let llm = MultiRoundUsageLlm {
            script: Mutex::new(vec![
                // Round 0: tool call, below threshold → no compaction.
                (
                    LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "noop", "{}")]),
                    12_000,
                ),
                // Round 1: text, above threshold → window shrinks → compaction.
                (LlmResponse::text("final"), 40_000),
            ]),
        };
        let mut tool_results = HashMap::new();
        tool_results.insert("noop".to_string(), "ok".to_string());
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            MockToolExecutor::new(
                vec![ToolDefinition::new("noop", "N", serde_json::json!({}))],
                tool_results,
            ),
            id_gen(),
        );

        let conv = handler
            .create_conversation("Test".into(), vec![])
            .await
            .unwrap();
        // Prime 30 messages: below MAX_CONTEXT_MESSAGES (40) so no top-of-turn
        // compaction, but above the shrunk window (20) so round 1 can compact.
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..30 {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            stored.messages.push(Message::new(role, format!("m-{i}")));
        }
        handler.store.update(stored).await.unwrap();

        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_for_sink = Arc::clone(&captured);
        let sink: ContextUsageSink = Arc::new(move |u: ContextUsage| {
            captured_for_sink.lock().unwrap().push(u);
        });
        let budget = ContextBudget {
            max_input_tokens: budget_max,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            with_context_usage_sink(sink, async {
                handler
                    .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                    .await
                    .unwrap();
            })
            .await
        })
        .await;

        let reports = captured.lock().unwrap().clone();
        assert_eq!(
            reports.len(),
            2,
            "a 2-round turn must emit one usage report PER ROUND, got {reports:?}"
        );
        assert_eq!(reports[0].used_tokens, 12_000);
        assert!(
            !reports[0].compaction_active,
            "round 0 is below threshold → compaction not active"
        );
        assert_eq!(reports[1].used_tokens, 40_000);
        assert!(
            reports[1].compaction_active,
            "round 1 crosses the threshold and shrinks the window → compaction active"
        );
    }
}

/// Concurrency tests for per-conversation turn serialization (DA-1, #282).
///
/// These exercise the bug directly: two turns racing the *same* conversation
/// must both persist (no lost messages), turns on *different* conversations
/// must stay concurrent, queued turns must run FIFO, a queued turn must be
/// cancellable while it waits, an erroring turn must release the lock, a
/// rename racing a turn must not clobber messages, and the lock map must not
/// grow unboundedly.
#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use crate::domain::ToolDefinition;
    use crate::ports::llm::{LlmResponse, with_cancellation_token};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration as StdDuration;

    // In-memory store mirroring the test `MockStore`, but cloneable via Arc so
    // it can back several concurrent handler calls. Read-modify-write is the
    // same shape as the real Postgres store: `get` clones out, the caller
    // mutates, `update` replaces the whole row — so without serialization a
    // late `update` clobbers a turn that finished in between.
    #[derive(Clone)]
    struct SharedStore {
        data: Arc<StdMutex<HashMap<String, Conversation>>>,
    }

    impl SharedStore {
        fn new() -> Self {
            Self {
                data: Arc::new(StdMutex::new(HashMap::new())),
            }
        }
    }

    impl ConversationStore for SharedStore {
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

        async fn archive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn unarchive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn create_summary(
            &self,
            _conversation_id: &ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("mock-summary".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    /// LLM whose `stream_completion` blocks until released, so a test can hold
    /// several turns simultaneously *inside* their turn bodies and force the
    /// interleaving that the race needs. Each call increments `in_flight`, then
    /// waits for a `permits` token before returning the reply. Tests observe
    /// in-flight state by polling `in_flight`.
    ///
    /// `permits` is a token *count* (not a `Notify`), so `open_gate()` called
    /// before a turn parks still releases it — no notify/park race. Tests call
    /// `open_gate()` in a poll loop and each call grants one more turn passage.
    #[derive(Clone)]
    struct GatedLlm {
        reply: String,
        in_flight: Arc<AtomicUsize>,
        permits: Arc<AtomicUsize>,
    }

    impl GatedLlm {
        fn new(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                in_flight: Arc::new(AtomicUsize::new(0)),
                permits: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// Grant one more turn passage through the gate. Permit-based, so order
        /// vs a turn's parking does not matter.
        fn open_gate(&self) {
            self.permits.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for GatedLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            // Spin-wait for a permit. Cheap for tests; yields so other tasks run.
            loop {
                let cur = self.permits.load(Ordering::SeqCst);
                if cur > 0
                    && self
                        .permits
                        .compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(2)).await;
            }
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(LlmResponse::text(self.reply.clone()))
        }
    }

    /// Trivial LLM returning a fixed reply, for tests where the LLM is not the
    /// thing under test (the store's first `update` is forced to fail instead).
    struct FixedLlm(String);

    #[async_trait::async_trait]
    impl LlmClient for FixedLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Ok(LlmResponse::text(self.0.clone()))
        }
    }

    /// Store whose first `update` fails (then succeeds), so `send_prompt`
    /// returns `Err` via `?` mid-turn — exercising RAII lock release on an early
    /// error return.
    #[derive(Clone)]
    struct FailFirstUpdateStore {
        inner: SharedStore,
        fail_updates: Arc<AtomicUsize>,
    }

    impl ConversationStore for FailFirstUpdateStore {
        async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
            self.inner.create(conv).await
        }
        async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            self.inner.get(id).await
        }
        async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
            self.inner.list().await
        }
        async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
            if self.fail_updates.load(Ordering::SeqCst) > 0 {
                self.fail_updates.fetch_sub(1, Ordering::SeqCst);
                return Err(CoreError::Llm("update boom".to_string()));
            }
            self.inner.update(conv).await
        }
        async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
            self.inner.delete(id).await
        }
        async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
            self.inner.archive(id).await
        }
        async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
            self.inner.unarchive(id).await
        }
        async fn create_summary(
            &self,
            conversation_id: &ConversationId,
            summary: String,
            start_ordinal: usize,
            end_ordinal: usize,
        ) -> Result<String, CoreError> {
            self.inner
                .create_summary(conversation_id, summary, start_ordinal, end_ordinal)
                .await
        }
        async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
            self.inner.expand_summary(summary_id).await
        }
    }

    fn make_handler_with<S: ConversationStore, L: LlmClient>(
        store: S,
        llm: L,
    ) -> ConversationHandler<S, L> {
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::new(
            store,
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    fn noop_callback() -> ChunkCallback {
        Box::new(|_| true)
    }

    fn noop_status() -> StatusCallback {
        Box::new(|_| {})
    }

    /// Marker content for the pre-seeded history message (see below).
    const SEED_MARKER: &str = "__seed__";

    /// Seed a conversation that already has one prior assistant message. This
    /// makes `is_first_message` false, so a turn does NOT trigger title
    /// generation — which would otherwise be a *second* LLM call per turn and
    /// require a second gate permit, confounding the permit-based timing these
    /// tests rely on. Assertions filter out `SEED_MARKER` so the seed is
    /// invisible to message-count checks.
    fn seed_conv(store: &SharedStore, id: &str) -> ConversationId {
        let mut conv = Conversation::new(id, "Chat");
        let ts = now_timestamp();
        conv.created_at = ts.clone();
        conv.updated_at = ts;
        conv.messages
            .push(Message::new(Role::Assistant, SEED_MARKER));
        store.data.lock().unwrap().insert(id.to_string(), conv);
        ConversationId(id.to_string())
    }

    /// Messages excluding the pre-seeded history marker (see `seed_conv`).
    fn real_messages(conv: &Conversation) -> Vec<(Role, String)> {
        conv.messages
            .iter()
            .filter(|m| m.content != SEED_MARKER)
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect()
    }

    /// AC1: two concurrent turns on ONE conversation must both persist — both
    /// user prompts and both replies present afterwards. This is the data-loss
    /// bug: before serialization, the late `update` clobbers the early one.
    #[tokio::test]
    async fn concurrent_send_prompts_same_conversation_lose_nothing() {
        let store = SharedStore::new();
        let id = seed_conv(&store, "c1");
        let llm = GatedLlm::new("reply");
        let handler = Arc::new(make_handler_with(store.clone(), llm.clone()));

        let h1 = handler.clone();
        let id1 = id.clone();
        let t1 = tokio::spawn(async move {
            h1.send_prompt(&id1, "first".into(), noop_callback(), noop_status())
                .await
        });
        let h2 = handler.clone();
        let id2 = id.clone();
        let t2 = tokio::spawn(async move {
            h2.send_prompt(&id2, "second".into(), noop_callback(), noop_status())
                .await
        });

        // Repeatedly open the gate so each turn proceeds as it acquires the
        // lock (with serialization only one is in-flight at a time).
        for _ in 0..60 {
            tokio::time::sleep(StdDuration::from_millis(10)).await;
            llm.open_gate();
            if t1.is_finished() && t2.is_finished() {
                break;
            }
        }

        t1.await.unwrap().unwrap();
        t2.await.unwrap().unwrap();

        let conv = store.data.lock().unwrap().get("c1").cloned().unwrap();
        let real = real_messages(&conv);
        let users: Vec<&String> = real
            .iter()
            .filter(|(r, _)| *r == Role::User)
            .map(|(_, c)| c)
            .collect();
        let assistants = real.iter().filter(|(r, _)| *r == Role::Assistant).count();
        assert!(
            users.contains(&&"first".to_string()) && users.contains(&&"second".to_string()),
            "both user prompts must survive, got: {real:?}"
        );
        assert_eq!(
            assistants, 2,
            "both assistant replies must survive (4 real messages total), got: {real:?}"
        );
        assert_eq!(real.len(), 4);
    }

    /// AC2: turns on DIFFERENT conversations must not serialize. The gated LLM
    /// only releases once both turns are simultaneously in-flight; a global or
    /// cross-conversation lock would let only one enter and this would time out.
    #[tokio::test]
    async fn concurrent_turns_on_different_conversations_run_in_parallel() {
        let store = SharedStore::new();
        let id_a = seed_conv(&store, "a");
        let id_b = seed_conv(&store, "b");
        let llm = GatedLlm::new("reply");
        let handler = Arc::new(make_handler_with(store.clone(), llm.clone()));

        let h1 = handler.clone();
        let t1 = tokio::spawn(async move {
            h1.send_prompt(&id_a, "qa".into(), noop_callback(), noop_status())
                .await
        });
        let h2 = handler.clone();
        let t2 = tokio::spawn(async move {
            h2.send_prompt(&id_b, "qb".into(), noop_callback(), noop_status())
                .await
        });

        // Wait until BOTH turns are inside the LLM at the same time.
        let both_in_flight = async {
            loop {
                if llm.in_flight.load(Ordering::SeqCst) >= 2 {
                    return;
                }
                tokio::time::sleep(StdDuration::from_millis(5)).await;
            }
        };
        tokio::time::timeout(StdDuration::from_secs(5), both_in_flight)
            .await
            .expect("different conversations must run concurrently, not serialize");

        // Drain.
        for _ in 0..50 {
            llm.open_gate();
            if t1.is_finished() && t2.is_finished() {
                break;
            }
            tokio::time::sleep(StdDuration::from_millis(10)).await;
        }
        t1.await.unwrap().unwrap();
        t2.await.unwrap().unwrap();
    }

    /// AC: queued turns on one conversation run in submission (FIFO) order.
    #[tokio::test]
    async fn queued_turns_run_in_fifo_order() {
        let store = SharedStore::new();
        let id = seed_conv(&store, "c1");
        let llm = GatedLlm::new("r");
        let handler = Arc::new(make_handler_with(store.clone(), llm.clone()));

        let mut handles = Vec::new();
        for i in 0..3 {
            let h = handler.clone();
            let id = id.clone();
            let prompt = format!("p{i}");
            handles.push(tokio::spawn(async move {
                h.send_prompt(&id, prompt, noop_callback(), noop_status())
                    .await
            }));
            // Stagger submission so arrival order at the lock is deterministic.
            tokio::time::sleep(StdDuration::from_millis(30)).await;
        }

        for _ in 0..80 {
            llm.open_gate();
            tokio::time::sleep(StdDuration::from_millis(10)).await;
            if handles.iter().all(|h| h.is_finished()) {
                break;
            }
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        let conv = store.data.lock().unwrap().get("c1").cloned().unwrap();
        let users: Vec<String> = conv
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .collect();
        assert_eq!(
            users,
            vec!["p0", "p1", "p2"],
            "queued turns must persist in FIFO submission order"
        );
    }

    /// AC: a turn queued behind an active turn can be cancelled WHILE it waits;
    /// it returns `Cancelled` promptly, the running turn is unaffected, and only
    /// the running turn's messages persist.
    #[tokio::test]
    async fn cancelling_a_queued_turn_releases_it_while_waiting() {
        let store = SharedStore::new();
        let id = seed_conv(&store, "c1");
        let llm = GatedLlm::new("reply");
        let handler = Arc::new(make_handler_with(store.clone(), llm.clone()));

        // Turn A acquires the lock and parks inside the LLM.
        let ha = handler.clone();
        let id_a = id.clone();
        let ta = tokio::spawn(async move {
            ha.send_prompt(&id_a, "A".into(), noop_callback(), noop_status())
                .await
        });
        tokio::time::timeout(StdDuration::from_secs(5), async {
            while llm.in_flight.load(Ordering::SeqCst) < 1 {
                tokio::time::sleep(StdDuration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn A should enter the LLM");

        // Turn B queues behind A under its own cancellation token.
        let token = CancellationToken::new();
        let hb = handler.clone();
        let id_b = id.clone();
        let token_for_b = token.clone();
        let tb = tokio::spawn(async move {
            with_cancellation_token(token_for_b, async move {
                hb.send_prompt(&id_b, "B".into(), noop_callback(), noop_status())
                    .await
            })
            .await
        });

        // Give B time to reach the lock wait, then cancel it.
        tokio::time::sleep(StdDuration::from_millis(100)).await;
        token.cancel();

        let b_result = tokio::time::timeout(StdDuration::from_secs(5), tb)
            .await
            .expect("cancelled queued turn must return promptly while waiting")
            .unwrap();
        assert!(
            matches!(b_result, Err(CoreError::Cancelled)),
            "queued-then-cancelled turn must return Cancelled, got {b_result:?}"
        );

        // A still completes fine.
        llm.open_gate();
        ta.await.unwrap().unwrap();

        let conv = store.data.lock().unwrap().get("c1").cloned().unwrap();
        let users: Vec<String> = conv
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .collect();
        assert_eq!(
            users,
            vec!["A".to_string()],
            "only the running turn should persist"
        );
    }

    /// AC: an erroring turn releases the lock (RAII / no poisoning) so a queued
    /// turn proceeds normally afterwards.
    #[tokio::test]
    async fn turn_error_releases_the_lock() {
        let inner = SharedStore::new();
        let id = seed_conv(&inner, "c1");
        let store = FailFirstUpdateStore {
            inner: inner.clone(),
            fail_updates: Arc::new(AtomicUsize::new(1)),
        };
        let handler = make_handler_with(store, FixedLlm("ok".to_string()));

        // First turn errors mid-persist (store.update fails) → Err via `?`.
        let first = handler
            .send_prompt(&id, "boom".into(), noop_callback(), noop_status())
            .await;
        assert!(first.is_err(), "first turn should error, got {first:?}");

        // Second turn must proceed (lock released despite the early error).
        let second = handler
            .send_prompt(&id, "after".into(), noop_callback(), noop_status())
            .await;
        assert!(
            second.is_ok(),
            "lock must be released after an error so the next turn proceeds: {second:?}"
        );
        let conv = inner.data.lock().unwrap().get("c1").cloned().unwrap();
        assert!(
            conv.messages.iter().any(|m| m.content == "after"),
            "the post-error turn must persist"
        );
    }

    /// AC (§1.2): a rename racing an active turn must not clobber the turn's
    /// messages — the final state has the new title AND the turn's messages.
    #[tokio::test]
    async fn rename_during_active_turn_does_not_clobber_messages() {
        let store = SharedStore::new();
        let id = seed_conv(&store, "c1");
        let llm = GatedLlm::new("reply");
        let handler = Arc::new(make_handler_with(store.clone(), llm.clone()));

        // Start a turn that parks in the LLM (holding the lock).
        let h_turn = handler.clone();
        let id_turn = id.clone();
        let turn = tokio::spawn(async move {
            h_turn
                .send_prompt(&id_turn, "hello".into(), noop_callback(), noop_status())
                .await
        });
        tokio::time::timeout(StdDuration::from_secs(5), async {
            while llm.in_flight.load(Ordering::SeqCst) < 1 {
                tokio::time::sleep(StdDuration::from_millis(5)).await;
            }
        })
        .await
        .expect("turn should enter the LLM");

        // Rename queues behind the turn (load conv, set title, write).
        let h_rename = handler.clone();
        let id_rename = id.clone();
        let rename = tokio::spawn(async move {
            h_rename
                .rename_conversation(&id_rename, "New Title".into())
                .await
        });

        tokio::time::sleep(StdDuration::from_millis(100)).await;
        // Release the turn; rename should run after it, on fresh state.
        llm.open_gate();
        turn.await.unwrap().unwrap();
        rename.await.unwrap().unwrap();

        let conv = store.data.lock().unwrap().get("c1").cloned().unwrap();
        assert_eq!(conv.title, "New Title", "rename must take effect");
        let real = real_messages(&conv);
        assert!(
            real.iter().any(|(_, c)| c == "hello"),
            "the turn's user message must survive the rename, got: {real:?}"
        );
        assert_eq!(
            real.len(),
            2,
            "user + assistant must both survive the rename, got: {real:?}"
        );
    }

    /// AC: the lock map must not grow unboundedly — entries are weak and pruned,
    /// so after N sequential turns across N conversations the map is bounded
    /// (dangling weak entries removed once no turn holds the Arc).
    #[tokio::test]
    async fn lock_map_does_not_grow_unboundedly() {
        let store = SharedStore::new();
        let llm = GatedLlm::new("r");
        let handler = make_handler_with(store.clone(), llm.clone());

        for i in 0..20 {
            let cid = format!("c{i}");
            let id = seed_conv(&store, &cid);
            let fut = handler.send_prompt(&id, format!("p{i}"), noop_callback(), noop_status());
            tokio::pin!(fut);
            loop {
                tokio::select! {
                    r = &mut fut => { r.unwrap(); break; }
                    _ = tokio::time::sleep(StdDuration::from_millis(5)) => { llm.open_gate(); }
                }
            }
        }

        // After all turns complete, no Arc is held, so weak entries must have
        // been pruned: the map is far smaller than the 20 conversations touched.
        let len = handler.turn_lock_map_len();
        assert!(
            len <= 1,
            "lock map should be pruned of dangling weak entries, len = {len}"
        );
    }
}
