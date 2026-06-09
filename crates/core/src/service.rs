use crate::CoreError;
use crate::context::{
    COMPACTION_TOKEN_RATIO, DEFAULT_MAX_TOOL_RESULT_BYTES, MAX_CONTEXT_MESSAGES,
    MAX_OVERFLOW_RETRIES, MIN_CONTEXT_MESSAGES, ToolLocalityContext, cap_tool_result,
    compaction_range, generate_context_summary, llm_messages_for_turn_with_plan,
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
    current_context_budget, current_system_refinement,
};
use crate::ports::scratchpad::{
    MAX_NOTE_BYTES, NewScratchpadNote, SCRATCHPAD_GOAL_KEY, ScratchpadGetManyFn, ScratchpadListFn,
    ScratchpadWriteFn,
};
use crate::ports::store::ConversationStore;
use crate::ports::tool_observer::{ToolEvent, notify_tool_event};
use crate::ports::tools::ToolExecutor;
use crate::ports::transport::{current_client_label, current_co_location, current_transport_kind};
use crate::sanitize::{sanitize_assistant_text, sanitize_assistant_text_for_stream};
use crate::tools::{
    NoopToolExecutor, categorize_tool_namespaces, summarize_tool_text, summarize_tool_value,
    tool_set_hash,
};
use chrono::{Duration, Local};
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
            scratchpad_goal_read: None,
            scratchpad_write: None,
            scratchpad_list: None,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            host: DEFAULT_HOST_LABEL.to_string(),
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
            scratchpad_goal_read: None,
            scratchpad_write: None,
            scratchpad_list: None,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            host: DEFAULT_HOST_LABEL.to_string(),
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
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        let id = (self.id_generator)();
        let mut conv = Conversation::new(id, title);
        let timestamp = now_timestamp();
        conv.created_at = timestamp.clone();
        conv.updated_at = timestamp;
        self.store.create(conv.clone()).await?;
        Ok(conv)
    }

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        let mut convs = self.store.list().await?;

        if !include_archived {
            convs.retain(|conv| conv.archived_at.is_none());
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

        Ok(convs.iter().map(ConversationSummary::from).collect())
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
        mut on_chunk: ChunkCallback,
        mut on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        // Cooperative cancellation checkpoint (issue #109): bail out
        // before any I/O if the caller has already tripped the token.
        bail_if_cancelled()?;

        let mut conv = self.store.get(conversation_id).await?;
        let is_first_message = conv.messages.is_empty();
        conv.messages.push(Message::new(Role::User, &prompt));
        // Capture the prompt as the active-task anchor for this turn. It is
        // re-injected in `llm_messages_for_turn` when conditions indicate
        // the original message has drifted out of the model's view.
        conv.active_task = Some(prompt.clone());

        // Effective window size for this turn. May shrink further if the
        // provider reports input-token usage above COMPACTION_TOKEN_RATIO.
        let mut target_window = MAX_CONTEXT_MESSAGES;

        // Count of in-turn ContextOverflow recoveries. Bounded so a
        // persistently-oversized request doesn't loop indefinitely.
        let mut overflow_retries: u32 = 0;

        // Per-request system-prompt refinement (installed by the daemon
        // dispatch wrapper from the client's `system_refinement` field; empty
        // for callers that don't route through it). Read once here so the
        // value is stable for every assembly pass in this turn. It is
        // appended to the system prompt for THIS request only — it is never
        // pushed onto `conv.messages` or otherwise persisted, so it stays out
        // of chat history and never affects a later turn.
        let system_refinement = current_system_refinement();

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
        let mut step_stack = StepStack::new();

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

            // The estimator borrows `&self.llm` so the closure is built
            // each iteration; constructing it is cheap (no allocation).
            let estimate = |text: &str| self.llm.estimate_tokens(text);
            let llm_messages = llm_messages_for_turn_with_plan(
                &conv.messages,
                &conv.summaries,
                &tool_defs,
                deferred_ns,
                &conv.context_summary,
                target_window,
                anchor,
                plan.as_deref(),
                tool_rounds_since_anchor,
                &system_refinement,
                current_context_budget(),
                Some(&tool_locality),
                &estimate,
            );
            let mut raw_stream = String::new();
            let mut emitted_visible_len = 0usize;
            let mut visible_chunk_callback = on_chunk;
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
                raw_stream.push_str(&chunk);
                let sanitized = sanitize_assistant_text_for_stream(&raw_stream);

                if sanitized.len() < emitted_visible_len {
                    emitted_visible_len = sanitized.len();
                    return true;
                }

                if sanitized.len() <= emitted_visible_len {
                    return true;
                }

                let visible = sanitized[emitted_visible_len..].to_string();
                emitted_visible_len = sanitized.len();

                if visible.is_empty() {
                    true
                } else {
                    visible_chunk_callback(visible)
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
                    on_chunk = Box::new(|_| true);
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
                    } else {
                        tracing::debug!(
                            input_tokens,
                            max_tokens,
                            window = target_window,
                            "context pressure with window already at minimum"
                        );
                    }
                }
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
                    on_chunk = Box::new(|_| true);
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
                    match with_conversation_id(conversation_id.clone(), exec).await {
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

            // Create a new noop callback for subsequent rounds
            // (the original callback was consumed by stream_completion)
            on_chunk = Box::new(|_| true);
        }

        // If we exhausted all rounds, return what we have
        Err(CoreError::Llm(format!(
            "tool calling loop exceeded maximum of {MAX_TOOL_ROUNDS} rounds"
        )))
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

        async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
            Ok(self.conversations.clone())
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
        let c1 = handler.create_conversation("A".into()).await.unwrap();
        let c2 = handler.create_conversation("B".into()).await.unwrap();
        assert_ne!(c1.id, c2.id);
        assert_eq!(c1.id.as_str(), "conv-1");
        assert_eq!(c2.id.as_str(), "conv-2");
    }

    #[tokio::test]
    async fn create_sets_human_readable_timestamps() {
        let handler = make_handler(vec![]);
        let conv = handler.create_conversation("A".into()).await.unwrap();
        assert!(!conv.created_at.is_empty());
        assert!(!conv.updated_at.is_empty());
        assert_eq!(conv.created_at.len(), 19);
        assert_eq!(conv.updated_at.len(), 19);
        assert_eq!(conv.created_at, conv.updated_at);
    }

    #[tokio::test]
    async fn create_stores_conversation() {
        let handler = make_handler(vec![]);
        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let retrieved = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(retrieved.title, "Test");
    }

    #[tokio::test]
    async fn list_returns_summaries() {
        let handler = make_handler(vec![]);
        handler.create_conversation("A".into()).await.unwrap();
        handler.create_conversation("B".into()).await.unwrap();

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
        let conv = handler.create_conversation("Gone".into()).await.unwrap();
        handler.delete_conversation(&conv.id).await.unwrap();

        let result = handler.get_conversation(&conv.id).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn clear_all_history_removes_all_conversations() {
        let handler = make_handler(vec![]);
        handler.create_conversation("A".into()).await.unwrap();
        handler.create_conversation("B".into()).await.unwrap();

        let deleted = handler.clear_all_history().await.unwrap();
        assert_eq!(deleted, 2);

        let summaries = handler.list_conversations(None, false).await.unwrap();
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn send_prompt_adds_messages_to_history() {
        let handler = make_handler(vec!["Hello", " there"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

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
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

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
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

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
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Loop forever".into(),
                noop_callback(),
                noop_status(),
            )
            .await;
        assert!(matches!(result, Err(CoreError::Llm(_))));
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
    fn user_visible_error_for_overloaded_529() {
        let err = CoreError::RateLimited {
            retry_after: None,
            detail: "overloaded".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
                .contains("builtin_knowledge_base_write/search/delete")
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
            .create_conversation("New Chat".into())
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
            .create_conversation("New Chat".into())
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
        let conv = handler.create_conversation("My Chat".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();
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
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

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
    }

    impl CategorizingLlm {
        fn new(category_payload: String) -> Self {
            Self {
                categorization_calls: Arc::new(AtomicU32::new(0)),
                category_payload: Mutex::new(category_payload),
            }
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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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

        let conv = handler.create_conversation("Test".into()).await.unwrap();

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
        let conv = handler.create_conversation("t".into()).await.unwrap();
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

        let conv = handler.create_conversation("t".into()).await.unwrap();
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

        let conv = handler.create_conversation("t".into()).await.unwrap();
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
        let conv = handler.create_conversation("t".into()).await.unwrap();

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
            let conv = handler.create_conversation("t".into()).await.unwrap();
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
}
