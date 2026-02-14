use ratatui::style::Style;
use tui_textarea::TextArea;

fn new_textarea() -> TextArea<'static> {
    let mut ta = TextArea::default();
    ta.set_cursor_line_style(Style::default());
    ta
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Editing,
    CreatingConversation,
}

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ConversationDetail {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
}

pub struct App {
    pub conversations: Vec<ConversationSummary>,
    pub selected_conversation: Option<usize>,
    pub current_conversation: Option<ConversationDetail>,
    pub textarea: TextArea<'static>,
    pub streaming_buffer: String,
    pub pending_request_id: Option<String>,
    pub mode: InputMode,
    pub status_message: String,
    pub should_quit: bool,
    /// Lines scrolled up from the bottom. 0 = auto-scroll to bottom.
    pub scroll_offset: u16,
}

impl App {
    pub fn new() -> Self {
        Self {
            conversations: Vec::new(),
            selected_conversation: None,
            current_conversation: None,
            textarea: new_textarea(),
            streaming_buffer: String::new(),
            pending_request_id: None,
            mode: InputMode::Normal,
            status_message: "Connected".to_string(),
            should_quit: false,
            scroll_offset: 0,
        }
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    // --- Navigation ---

    pub fn next_conversation(&mut self) {
        if self.conversations.is_empty() {
            return;
        }
        self.selected_conversation = Some(match self.selected_conversation {
            Some(i) => {
                if i >= self.conversations.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        });
    }

    pub fn previous_conversation(&mut self) {
        if self.conversations.is_empty() {
            return;
        }
        self.selected_conversation = Some(match self.selected_conversation {
            Some(i) => {
                if i == 0 {
                    self.conversations.len() - 1
                } else {
                    i - 1
                }
            }
            None => self.conversations.len() - 1,
        });
    }

    // --- Scrolling ---

    pub fn scroll_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines);
    }

    pub fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    // --- Input ---

    /// Returns the textarea content as a single string (lines joined with newlines).
    pub fn textarea_content(&self) -> String {
        self.textarea.lines().join("\n")
    }

    /// Returns (conversation_id, prompt) if valid, None otherwise.
    pub fn submit_prompt(&mut self) -> Option<(String, String)> {
        let content = self.textarea_content();
        if content.is_empty() {
            return None;
        }
        let conv = self.current_conversation.as_mut()?;
        conv.messages.push(ChatMessage {
            role: "user".to_string(),
            content: content.clone(),
        });
        self.textarea = new_textarea();
        self.scroll_offset = 0;
        Some((conv.id.clone(), content))
    }

    /// Returns the title for the new conversation, or None if empty.
    pub fn submit_new_conversation_title(&mut self) -> Option<String> {
        let content = self.textarea_content();
        if content.is_empty() {
            return None;
        }
        self.textarea = new_textarea();
        self.mode = InputMode::Normal;
        Some(content)
    }

    // --- Mode transitions ---

    pub fn enter_editing_mode(&mut self) {
        self.mode = InputMode::Editing;
    }

    pub fn enter_normal_mode(&mut self) {
        self.mode = InputMode::Normal;
    }

    pub fn enter_creating_conversation_mode(&mut self) {
        self.mode = InputMode::CreatingConversation;
        self.textarea = new_textarea();
    }

    // --- Streaming ---

    pub fn start_streaming(&mut self, request_id: String) {
        self.pending_request_id = Some(request_id);
        self.streaming_buffer.clear();
    }

    pub fn receive_chunk(&mut self, request_id: &str, chunk: &str) {
        if self.pending_request_id.as_deref() != Some(request_id) {
            return;
        }
        self.streaming_buffer.push_str(chunk);
        self.scroll_offset = 0;
    }

    pub fn complete_streaming(&mut self, request_id: &str, full_response: &str) {
        if self.pending_request_id.as_deref() != Some(request_id) {
            return;
        }
        if let Some(conv) = self.current_conversation.as_mut() {
            conv.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: full_response.to_string(),
            });
        }
        self.streaming_buffer.clear();
        self.pending_request_id = None;
    }

    pub fn streaming_error(&mut self, request_id: &str, error: &str) {
        if self.pending_request_id.as_deref() != Some(request_id) {
            return;
        }
        self.status_message = format!("Error: {error}");
        self.streaming_buffer.clear();
        self.pending_request_id = None;
    }

    // --- Conversation management ---

    pub fn set_conversations(&mut self, conversations: Vec<ConversationSummary>) {
        self.conversations = conversations;
        // Fix selection if out of bounds
        if let Some(sel) = self.selected_conversation
            && sel >= self.conversations.len()
        {
            self.selected_conversation = if self.conversations.is_empty() {
                None
            } else {
                Some(self.conversations.len() - 1)
            };
        }
    }

    pub fn load_conversation(&mut self, detail: ConversationDetail) {
        self.current_conversation = Some(detail);
    }

    pub fn selected_conversation_id(&self) -> Option<&str> {
        let idx = self.selected_conversation?;
        self.conversations.get(idx).map(|c| c.id.as_str())
    }

    pub fn delete_selected_conversation(&mut self) -> Option<String> {
        let idx = self.selected_conversation?;
        if idx >= self.conversations.len() {
            return None;
        }
        let id = self.conversations[idx].id.clone();
        self.conversations.remove(idx);

        // Clear current conversation if it was the deleted one
        if self
            .current_conversation
            .as_ref()
            .is_some_and(|c| c.id == id)
        {
            self.current_conversation = None;
        }

        // Fix selection
        if self.conversations.is_empty() {
            self.selected_conversation = None;
        } else if idx >= self.conversations.len() {
            self.selected_conversation = Some(self.conversations.len() - 1);
        }

        Some(id)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_conversations() -> Vec<ConversationSummary> {
        vec![
            ConversationSummary {
                id: "1".into(),
                title: "First".into(),
                message_count: 2,
            },
            ConversationSummary {
                id: "2".into(),
                title: "Second".into(),
                message_count: 0,
            },
            ConversationSummary {
                id: "3".into(),
                title: "Third".into(),
                message_count: 5,
            },
        ]
    }

    fn app_with_conversations() -> App {
        let mut app = App::new();
        app.set_conversations(sample_conversations());
        app
    }

    // --- Navigation tests ---

    #[test]
    fn next_on_empty_list_does_nothing() {
        let mut app = App::new();
        app.next_conversation();
        assert_eq!(app.selected_conversation, None);
    }

    #[test]
    fn next_from_none_selects_first() {
        let mut app = app_with_conversations();
        app.next_conversation();
        assert_eq!(app.selected_conversation, Some(0));
    }

    #[test]
    fn next_wraps_around() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(2);
        app.next_conversation();
        assert_eq!(app.selected_conversation, Some(0));
    }

    #[test]
    fn next_advances() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(0);
        app.next_conversation();
        assert_eq!(app.selected_conversation, Some(1));
    }

    #[test]
    fn previous_on_empty_list_does_nothing() {
        let mut app = App::new();
        app.previous_conversation();
        assert_eq!(app.selected_conversation, None);
    }

    #[test]
    fn previous_from_none_selects_last() {
        let mut app = app_with_conversations();
        app.previous_conversation();
        assert_eq!(app.selected_conversation, Some(2));
    }

    #[test]
    fn previous_wraps_around() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(0);
        app.previous_conversation();
        assert_eq!(app.selected_conversation, Some(2));
    }

    #[test]
    fn previous_goes_back() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(2);
        app.previous_conversation();
        assert_eq!(app.selected_conversation, Some(1));
    }

    #[test]
    fn single_item_next_stays() {
        let mut app = App::new();
        app.set_conversations(vec![ConversationSummary {
            id: "1".into(),
            title: "Only".into(),
            message_count: 0,
        }]);
        app.selected_conversation = Some(0);
        app.next_conversation();
        assert_eq!(app.selected_conversation, Some(0));
    }

    // --- Input tests ---

    #[test]
    fn textarea_insert_and_content() {
        let mut app = App::new();
        // Type into textarea using its input method
        app.textarea.insert_char('h');
        app.textarea.insert_char('i');
        assert_eq!(app.textarea_content(), "hi");
        app.textarea.delete_char();
        assert_eq!(app.textarea_content(), "h");
        app.textarea.delete_char();
        assert_eq!(app.textarea_content(), "");
        app.textarea.delete_char(); // no panic on empty
        assert_eq!(app.textarea_content(), "");
    }

    #[test]
    fn submit_prompt_without_conversation_returns_none() {
        let mut app = App::new();
        app.textarea.insert_str("hello");
        assert!(app.submit_prompt().is_none());
        assert_eq!(app.textarea_content(), "hello"); // input preserved
    }

    #[test]
    fn submit_prompt_with_empty_input_returns_none() {
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "1".into(),
            title: "Test".into(),
            messages: vec![],
        });
        assert!(app.submit_prompt().is_none());
    }

    #[test]
    fn submit_prompt_appends_user_message_and_clears_input() {
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "conv1".into(),
            title: "Test".into(),
            messages: vec![],
        });
        app.textarea.insert_str("What is Rust?");

        let result = app.submit_prompt();
        assert_eq!(
            result,
            Some(("conv1".to_string(), "What is Rust?".to_string()))
        );
        assert_eq!(app.textarea_content(), "");

        let msgs = &app.current_conversation.as_ref().unwrap().messages;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "What is Rust?");
    }

    // --- Streaming tests ---

    #[test]
    fn streaming_lifecycle() {
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "c1".into(),
            title: "Test".into(),
            messages: vec![],
        });

        app.start_streaming("req1".into());
        assert_eq!(app.pending_request_id, Some("req1".to_string()));

        app.receive_chunk("req1", "Hello ");
        app.receive_chunk("req1", "world!");
        assert_eq!(app.streaming_buffer, "Hello world!");

        app.complete_streaming("req1", "Hello world!");
        assert_eq!(app.streaming_buffer, "");
        assert_eq!(app.pending_request_id, None);

        let msgs = &app.current_conversation.as_ref().unwrap().messages;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        assert_eq!(msgs[0].content, "Hello world!");
    }

    #[test]
    fn wrong_request_id_ignored() {
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "c1".into(),
            title: "Test".into(),
            messages: vec![],
        });

        app.start_streaming("req1".into());
        app.receive_chunk("wrong_id", "bad data");
        assert_eq!(app.streaming_buffer, "");

        app.complete_streaming("wrong_id", "bad");
        assert!(app.pending_request_id.is_some()); // not cleared
    }

    #[test]
    fn streaming_error_sets_status() {
        let mut app = App::new();
        app.start_streaming("req1".into());
        app.streaming_error("req1", "LLM timeout");
        assert_eq!(app.status_message, "Error: LLM timeout");
        assert_eq!(app.pending_request_id, None);
        assert_eq!(app.streaming_buffer, "");
    }

    // --- Mode transition tests ---

    #[test]
    fn mode_transitions() {
        let mut app = App::new();
        assert_eq!(app.mode, InputMode::Normal);

        app.enter_editing_mode();
        assert_eq!(app.mode, InputMode::Editing);

        app.enter_normal_mode();
        assert_eq!(app.mode, InputMode::Normal);

        app.enter_creating_conversation_mode();
        assert_eq!(app.mode, InputMode::CreatingConversation);
        assert_eq!(app.textarea_content(), ""); // input cleared

        app.textarea.insert_str("leftover");
        app.enter_creating_conversation_mode();
        assert_eq!(app.textarea_content(), ""); // cleared again
    }

    #[test]
    fn submit_new_conversation_title() {
        let mut app = App::new();
        app.enter_creating_conversation_mode();
        app.textarea.insert_str("My Chat");

        let title = app.submit_new_conversation_title();
        assert_eq!(title, Some("My Chat".to_string()));
        assert_eq!(app.textarea_content(), "");
        assert_eq!(app.mode, InputMode::Normal);
    }

    #[test]
    fn submit_empty_conversation_title_returns_none() {
        let mut app = App::new();
        app.enter_creating_conversation_mode();
        assert!(app.submit_new_conversation_title().is_none());
    }

    // --- Conversation management tests ---

    #[test]
    fn set_conversations_fixes_out_of_bounds_selection() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(2);
        app.set_conversations(vec![ConversationSummary {
            id: "1".into(),
            title: "Only".into(),
            message_count: 0,
        }]);
        assert_eq!(app.selected_conversation, Some(0));
    }

    #[test]
    fn set_empty_conversations_clears_selection() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(1);
        app.set_conversations(vec![]);
        assert_eq!(app.selected_conversation, None);
    }

    #[test]
    fn selected_conversation_id() {
        let mut app = app_with_conversations();
        assert_eq!(app.selected_conversation_id(), None);

        app.selected_conversation = Some(1);
        assert_eq!(app.selected_conversation_id(), Some("2"));
    }

    #[test]
    fn delete_selected_conversation() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(1);
        app.current_conversation = Some(ConversationDetail {
            id: "2".into(),
            title: "Second".into(),
            messages: vec![],
        });

        let deleted = app.delete_selected_conversation();
        assert_eq!(deleted, Some("2".to_string()));
        assert_eq!(app.conversations.len(), 2);
        assert!(app.current_conversation.is_none());
        assert_eq!(app.selected_conversation, Some(1)); // stays at 1 (now "Third")
    }

    #[test]
    fn delete_last_item_adjusts_selection() {
        let mut app = app_with_conversations();
        app.selected_conversation = Some(2);

        let deleted = app.delete_selected_conversation();
        assert_eq!(deleted, Some("3".to_string()));
        assert_eq!(app.selected_conversation, Some(1));
    }

    #[test]
    fn delete_only_item_clears_selection() {
        let mut app = App::new();
        app.set_conversations(vec![ConversationSummary {
            id: "1".into(),
            title: "Only".into(),
            message_count: 0,
        }]);
        app.selected_conversation = Some(0);

        let deleted = app.delete_selected_conversation();
        assert_eq!(deleted, Some("1".to_string()));
        assert_eq!(app.selected_conversation, None);
    }

    #[test]
    fn delete_with_no_selection_returns_none() {
        let mut app = app_with_conversations();
        assert!(app.delete_selected_conversation().is_none());
    }

    #[test]
    fn quit_sets_flag() {
        let mut app = App::new();
        assert!(!app.should_quit);
        app.quit();
        assert!(app.should_quit);
    }

    // --- Scroll tests ---

    #[test]
    fn scroll_up_and_down() {
        let mut app = App::new();
        assert_eq!(app.scroll_offset, 0);
        app.scroll_up(5);
        assert_eq!(app.scroll_offset, 5);
        app.scroll_up(3);
        assert_eq!(app.scroll_offset, 8);
        app.scroll_down(3);
        assert_eq!(app.scroll_offset, 5);
        app.scroll_down(100);
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn scroll_to_bottom_resets() {
        let mut app = App::new();
        app.scroll_up(10);
        app.scroll_to_bottom();
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn receive_chunk_resets_scroll() {
        let mut app = App::new();
        app.start_streaming("req1".into());
        app.scroll_up(10);
        app.receive_chunk("req1", "data");
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn submit_prompt_resets_scroll() {
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "c1".into(),
            title: "Test".into(),
            messages: vec![],
        });
        app.scroll_up(10);
        app.textarea.insert_str("hello");
        app.submit_prompt();
        assert_eq!(app.scroll_offset, 0);
    }
}
