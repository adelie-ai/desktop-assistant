use desktop_assistant_api_model as api;

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
    pub archived: bool,
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
    pub model_selection: Option<api::ConversationModelSelectionView>,
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
            role: value.role,
            content: value.content,
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
        }
    }
}
