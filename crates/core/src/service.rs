use crate::CoreError;
use crate::domain::{
    Conversation, ConversationId, ConversationSummary, Message, Role, ToolDefinition,
};
use crate::ports::inbound::ConversationService;
use crate::ports::llm::{ChunkCallback, LlmClient};
use crate::ports::store::ConversationStore;
use crate::ports::tools::ToolExecutor;
use chrono::{Duration, Local};

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

/// Per-turn runtime instruction injected for the LLM.
const RUNTIME_SYSTEM_INSTRUCTION: &str = "You are Adele, a desktop assistant named in reference to the Adélie penguin, with optional tool access. \
Your name is Adele. If asked your name or who you are, answer: 'I'm Adele.' \
Follow this priority order and keep responses concise and practical. \
1) Current-turn user instructions override all stored data. \
2) Then prefer project preferences. \
3) Then global preferences. \
4) Then project memory. \
5) Then global memory. \
6) Use lightweight discovery next, and ask only for the smallest missing piece. \
7) If a request is user-specific/project-specific or a reference is unclear, search preferences and memory first (project scope first, then global) before non-memory tools. \
8) If still unclear, ask one brief clarifying question and do not assume. \
9) Do not guess user-specific details (project path, run command, package manager, editor, service name, account, or host). \
10) Before acting on user/project-specific work, make a short internal preflight: known from preferences/memory, verified this turn, and still unknown. \
11) Preferences are key/value datapoints (defaults, paths, IDs, names, commands, hostnames, and other concrete settings). \
12) Memory is prose context (background, rationale, corrections, procedural notes, and explanatory details). \
13) Store memory/preferences judiciously: only durable, reusable, high-confidence information; avoid transient one-off details unless the user asks to remember them. \
14) When both apply, store both: preferences for concrete key/value facts and memory for contextual prose that explains them. \
15) If unsure whether to store, how to scope, or whether it belongs in memory vs preference, ask briefly. \
16) Use namespaced keys: project.<project>.<attribute...> and global.<attribute...>. \
17) Treat project scope as any folder-anchored work context, not only software projects. \
18) For start/open/run <project>, check project.<project>.path, then start_command/run_command/dev_command, then editor/app, then global fallbacks. \
19) For tool-relevant requests (terminal, filesystem, D-Bus, network/web), attempt one best-fit available tool before claiming limitation, after rule 7 when applicable. \
20) Never fabricate tool outputs or claim a tool succeeded when it did not. \
21) If a tool fails, report the exact error briefly and provide the next best step. \
22) If no relevant tool exists, say so clearly and ask for the minimum missing configuration. \
23) When launching GUI apps, use a non-blocking launch pattern (for example nohup plus disown). \
24) Before launching an app, check PATH and also check Flatpak and Snap when available. \
25) Use built-in preference tools (builtin_preferences_remember/search/retrieve/delete). \
26) Use built-in memory tools (builtin_memory_remember/search/retrieve/update/delete).";

fn llm_messages_for_turn(
    conversation_messages: &[Message],
    tool_defs: &[ToolDefinition],
) -> Vec<Message> {
    let tool_note = if tool_defs.is_empty() {
        "No tools are available in this turn.".to_string()
    } else {
        let names = tool_defs
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!("Available tools in this turn: {names}.")
    };

    let mut messages = Vec::with_capacity(conversation_messages.len() + 1);
    messages.push(Message::new(
        Role::System,
        format!("{RUNTIME_SYSTEM_INSTRUCTION}\n\n{tool_note}"),
    ));
    messages.extend_from_slice(conversation_messages);
    messages
}

fn sanitize_assistant_text(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let mut sanitized = output.trim().to_string();
    while sanitized.contains("\n\n\n") {
        sanitized = sanitized.replace("\n\n\n", "\n\n");
    }
    sanitized
}

fn user_visible_llm_error_message(error: &CoreError) -> String {
    let raw = error.to_string();
    let normalized = raw.to_ascii_lowercase();

    if normalized.contains("does not support tools") {
        return format!(
            "This Ollama model does not support tool use. Please switch to a tool-capable model or disable tools for this chat. Details: {raw}"
        );
    }

    if normalized.contains("unable to load model")
        || normalized.contains("model not found")
        || normalized.contains("pull model manifest")
        || normalized.contains("no such file")
    {
        return format!(
            "The selected model could not be loaded or found. Please verify the model name and that it is installed in Ollama. Details: {raw}"
        );
    }

    if normalized.contains("downloading")
        || normalized.contains("currently loading")
        || normalized.contains("is loading")
        || normalized.contains("loading model")
    {
        return format!(
            "The model is still downloading or loading. Please wait a moment and try again. Details: {raw}"
        );
    }

    format!("I hit an LLM backend error and could not complete this request. Details: {raw}")
}

fn sanitize_assistant_text_for_stream(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let partial_len = trailing_tag_prefix_len(&output, "<think>");
    if partial_len > 0 {
        output.truncate(output.len() - partial_len);
    }

    output
}

fn trailing_tag_prefix_len(text: &str, tag: &str) -> usize {
    for len in (1..tag.len()).rev() {
        if text.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

/// Remove the oldest assistant(tool_calls)+tool_result groups from a message
/// list to reduce context size. Keeps the first user message and the most
/// recent tool interaction intact. Returns the number of messages removed.
fn trim_tool_pairs(messages: &mut Vec<Message>) -> usize {
    // Find ranges of (assistant-with-tool-calls, tool_result, ..., tool_result)
    // groups and remove roughly the oldest half.
    let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == Role::Assistant && !messages[i].tool_calls.is_empty() {
            let start = i;
            i += 1;
            while i < messages.len() && messages[i].role == Role::Tool {
                i += 1;
            }
            groups.push(start..i);
        } else {
            i += 1;
        }
    }

    if groups.len() <= 1 {
        // Nothing safe to remove — keep the most recent group
        return 0;
    }

    // Remove the oldest half of groups
    let remove_count = groups.len() / 2;
    let groups_to_remove: Vec<_> = groups[..remove_count].to_vec();

    // Remove in reverse order to keep indices stable
    let mut removed = 0;
    for range in groups_to_remove.into_iter().rev() {
        let len = range.len();
        messages.drain(range);
        removed += len;
    }

    removed
}

/// A no-op tool executor for use when no MCP servers are configured.
pub struct NoopToolExecutor;

impl ToolExecutor for NoopToolExecutor {
    async fn available_tools(&self) -> Vec<crate::domain::ToolDefinition> {
        Vec::new()
    }

    async fn execute_tool(
        &self,
        name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Err(CoreError::ToolExecution(format!(
            "no tool executor configured, cannot execute '{name}'"
        )))
    }
}

/// Core service implementing conversation management.
/// Generic over store, LLM, and tool executor backends for testability.
pub struct ConversationHandler<S, L, T = NoopToolExecutor> {
    store: S,
    llm: L,
    tools: T,
    id_generator: Box<dyn Fn() -> String + Send + Sync>,
}

impl<S, L> ConversationHandler<S, L, NoopToolExecutor> {
    pub fn new(store: S, llm: L, id_generator: Box<dyn Fn() -> String + Send + Sync>) -> Self {
        Self {
            store,
            llm,
            tools: NoopToolExecutor,
            id_generator,
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
            tools,
            id_generator,
        }
    }
}

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
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        let mut convs = self.store.list().await?;

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
    ) -> Result<String, CoreError> {
        let mut conv = self.store.get(conversation_id).await?;
        conv.messages.push(Message::new(Role::User, &prompt));

        let tool_defs = self.tools.available_tools().await;

        for round in 0..MAX_TOOL_ROUNDS {
            let llm_messages = llm_messages_for_turn(&conv.messages, &tool_defs);
            let mut raw_stream = String::new();
            let mut emitted_visible_len = 0usize;
            let mut visible_chunk_callback = on_chunk;
            let filtered_chunk_callback: ChunkCallback = Box::new(move |chunk| {
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

            let response = match self
                .llm
                .stream_completion(llm_messages, &tool_defs, filtered_chunk_callback)
                .await
            {
                Ok(r) => r,
                Err(e) if round > 0 => {
                    // Mid-loop LLM error (e.g. context too long) — trim old
                    // tool call/result pairs and tell the LLM what happened
                    // so it can adjust its approach.
                    tracing::warn!(
                        "LLM call failed on round {}/{}, trimming context: {e}",
                        round + 1,
                        MAX_TOOL_ROUNDS
                    );
                    let removed = trim_tool_pairs(&mut conv.messages);
                    tracing::info!("removed {removed} messages to reduce context");
                    conv.messages.push(Message::new(
                        Role::System,
                        format!(
                            "Your previous tool call could not be processed because \
                             the context became too long. {removed} older messages were \
                             trimmed. The original error was: {e}\n\
                             Please adjust your approach — for example, request less \
                             output or take a different path."
                        ),
                    ));
                    on_chunk = Box::new(|_| true);
                    continue;
                }
                Err(e) => {
                    let friendly = user_visible_llm_error_message(&e);
                    conv.messages.push(Message::new(Role::Assistant, &friendly));
                    conv.updated_at = now_timestamp();
                    self.store.update(conv).await?;
                    return Ok(friendly);
                }
            };

            if !response.has_tool_calls() {
                // Text-only response — we're done
                let visible_text = sanitize_assistant_text(&response.text);
                conv.messages
                    .push(Message::new(Role::Assistant, &visible_text));
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
                tracing::info!("executing tool: {}", tool_call.name);
                let arguments: serde_json::Value =
                    serde_json::from_str(&tool_call.arguments).unwrap_or_default();
                let result = match self.tools.execute_tool(&tool_call.name, arguments).await {
                    Ok(output) => output,
                    Err(e) => format!("Error: {e}"),
                };
                conv.messages
                    .push(Message::tool_result(&tool_call.id, &result));
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
    use crate::domain::{ToolCall, ToolDefinition};
    use crate::ports::llm::LlmResponse;
    use std::collections::HashMap;
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

    impl LlmClient for MockLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
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

        let summaries = handler.list_conversations(None).await.unwrap();
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

        let filtered = handler.list_conversations(Some(7)).await.unwrap();
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

        let summaries = handler.list_conversations(None).await.unwrap();
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn send_prompt_adds_messages_to_history() {
        let handler = make_handler(vec!["Hello", " there"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        let response = handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback())
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
            .send_prompt(&conv.id, "Hi".into(), noop_callback())
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
            .send_prompt(&ConversationId::from("nope"), "hi".into(), noop_callback())
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

    impl LlmClient for ToolCallingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
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
        async fn available_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
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
            .send_prompt(&conv.id, "Read /tmp/test".into(), noop_callback())
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
            .send_prompt(&conv.id, "Do both".into(), noop_callback())
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
            .send_prompt(&conv.id, "Try bad tool".into(), noop_callback())
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
            .send_prompt(&conv.id, "Loop forever".into(), noop_callback())
            .await;
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }

    // --- trim_tool_pairs tests ---

    #[test]
    fn trim_tool_pairs_removes_oldest_half() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            // Group 1
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result_1"),
            // Group 2
            Message::assistant_with_tool_calls(vec![ToolCall::new("c2", "tool_a", "{}")]),
            Message::tool_result("c2", "result_2"),
            // Group 3
            Message::assistant_with_tool_calls(vec![ToolCall::new("c3", "tool_a", "{}")]),
            Message::tool_result("c3", "result_3"),
            // Group 4
            Message::assistant_with_tool_calls(vec![ToolCall::new("c4", "tool_a", "{}")]),
            Message::tool_result("c4", "result_4"),
        ];

        let removed = trim_tool_pairs(&mut messages);
        // 4 groups, remove oldest half (2 groups = 4 messages)
        assert_eq!(removed, 4);
        // Should keep: user + group3 + group4
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].tool_calls[0].id, "c3");
    }

    #[test]
    fn trim_tool_pairs_keeps_single_group() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];

        let removed = trim_tool_pairs(&mut messages);
        assert_eq!(removed, 0);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn trim_tool_pairs_no_groups() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];

        let removed = trim_tool_pairs(&mut messages);
        assert_eq!(removed, 0);
        assert_eq!(messages.len(), 2);
    }

    // --- Context recovery test ---

    /// Mock LLM that fails on a specific call index.
    struct FailingLlm {
        responses: Mutex<Vec<LlmResponse>>,
        fail_on_call: usize,
        call_count: Mutex<usize>,
    }

    impl FailingLlm {
        fn new(responses: Vec<LlmResponse>, fail_on_call: usize) -> Self {
            Self {
                responses: Mutex::new(responses),
                fail_on_call,
                call_count: Mutex::new(0),
            }
        }
    }

    impl LlmClient for FailingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let call_idx = {
                let mut count = self.call_count.lock().unwrap();
                let idx = *count;
                *count += 1;
                idx
            };

            if call_idx == self.fail_on_call {
                return Err(CoreError::Llm("context_length_exceeded".into()));
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
    async fn tool_loop_recovers_from_context_length_error() {
        let tools = vec![ToolDefinition::new(
            "big_tool",
            "Returns lots of data",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests tool call
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "big_tool", "{}")]),
            // Round 1: fails (simulated by FailingLlm, call index 1)
            // Round 2 (retry after trim): LLM succeeds with final text
            LlmResponse::text("I adjusted my approach"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("big_tool".to_string(), "x".repeat(1000));

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 1), // fail on 2nd LLM call
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "Use big tool".into(), noop_callback())
            .await
            .unwrap();
        assert_eq!(result, "I adjusted my approach");

        // Verify the conversation has a system message about trimming
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_system_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(has_system_msg);
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
            .send_prompt(&conv.id, "hello".into(), noop_callback())
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
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 400 Bad Request): {\"error\":\"registry.ollama.ai/library/phi4:14b does not support tools\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("does not support tool use"));
    }

    #[test]
    fn user_visible_error_for_missing_model() {
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 500 Internal Server Error): {\"error\":\"unable to load model\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("could not be loaded or found"));
    }

    #[test]
    fn user_visible_error_for_loading_model() {
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 503 Service Unavailable): {\"error\":\"model is currently loading\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("still downloading or loading"));
    }

    #[tokio::test]
    async fn noop_executor_returns_empty_tools() {
        let executor = NoopToolExecutor;
        assert!(executor.available_tools().await.is_empty());
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

    impl LlmClient for CapturingLlm {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            *self.seen_messages.lock().unwrap() = messages;
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
            .send_prompt(&conv.id, "hello".into(), noop_callback())
            .await
            .unwrap();

        let messages = seen.lock().unwrap();
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0].content.contains(
            "You are Adele, a desktop assistant named in reference to the Adélie penguin"
        ));
        assert!(messages[0].content.contains("Your name is Adele"));
        assert!(messages[0].content.contains("Follow this priority order"));
        assert!(
            messages[0]
                .content
                .contains("Current-turn user instructions override all stored data")
        );
        assert!(
            messages[0]
                .content
                .contains("search preferences and memory first (project scope first, then global) before non-memory tools")
        );
        assert!(
            messages[0]
                .content
                .contains("If still unclear, ask one brief clarifying question and do not assume")
        );
        assert!(
            messages[0]
                .content
                .contains("make a short internal preflight")
        );
        assert!(
            messages[0]
                .content
                .contains("No tools are available in this turn.")
        );
        assert!(messages[0].content.contains("non-blocking launch pattern"));
        assert!(messages[0].content.contains("check PATH"));
        assert!(messages[0].content.contains("check Flatpak and Snap"));
        assert!(
            messages[0]
                .content
                .contains("builtin_preferences_remember/search/retrieve/delete")
        );
        assert!(
            messages[0]
                .content
                .contains("builtin_memory_remember/search/retrieve/update/delete")
        );
        assert!(messages[0].content.contains(
            "Store memory/preferences judiciously: only durable, reusable, high-confidence information"
        ));
        assert!(messages[0].content.contains("Memory is prose context"));
        assert!(
            messages[0]
                .content
                .contains("Preferences are key/value datapoints")
        );
        assert!(messages[0].content.contains("Never fabricate tool outputs"));
    }

    #[test]
    fn runtime_instruction_enforces_memory_first_for_user_specific_requests() {
        let priority_rule = "Current-turn user instructions override all stored data.";
        let memory_first = "If a request is user-specific/project-specific or a reference is unclear, search preferences and memory first (project scope first, then global) before non-memory tools.";
        let ambiguous_reference =
            "If still unclear, ask one brief clarifying question and do not assume.";
        let tool_fallback = "For tool-relevant requests (terminal, filesystem, D-Bus, network/web), attempt one best-fit available tool before claiming limitation, after rule 7 when applicable.";
        let no_guessing = "Do not guess user-specific details (project path, run command, package manager, editor, service name, account, or host).";
        let preference_kv_split = "Preferences are key/value datapoints (defaults, paths, IDs, names, commands, hostnames, and other concrete settings).";
        let memory_prose_split = "Memory is prose context (background, rationale, corrections, procedural notes, and explanatory details).";
        let no_fabrication =
            "Never fabricate tool outputs or claim a tool succeeded when it did not.";

        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(priority_rule));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(memory_first));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(ambiguous_reference));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(no_guessing));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(tool_fallback));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(preference_kv_split));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(memory_prose_split));
        assert!(RUNTIME_SYSTEM_INSTRUCTION.contains(no_fabrication));

        let priority_rule_pos = RUNTIME_SYSTEM_INSTRUCTION.find(priority_rule).unwrap();
        let memory_first_pos = RUNTIME_SYSTEM_INSTRUCTION.find(memory_first).unwrap();
        let ambiguous_reference_pos = RUNTIME_SYSTEM_INSTRUCTION
            .find(ambiguous_reference)
            .unwrap();
        let no_guessing_pos = RUNTIME_SYSTEM_INSTRUCTION.find(no_guessing).unwrap();
        let tool_fallback_pos = RUNTIME_SYSTEM_INSTRUCTION.find(tool_fallback).unwrap();

        assert!(
            priority_rule_pos < memory_first_pos,
            "priority rule must remain before memory/tool decision rules"
        );
        assert!(
            memory_first_pos < tool_fallback_pos,
            "memory-first rule must remain before non-memory tool fallback rule"
        );
        assert!(
            ambiguous_reference_pos < tool_fallback_pos,
            "ambiguity guardrail must remain before non-memory tool fallback rule"
        );
        assert!(
            no_guessing_pos < tool_fallback_pos,
            "no-guessing guardrail must remain before non-memory tool fallback rule"
        );
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
            .send_prompt(&conv.id, "hello".into(), noop_callback())
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
}
