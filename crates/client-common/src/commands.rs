//! Shared, transport-agnostic command surface.
//!
//! Both the WebSocket and Unix-domain-socket clients speak the same
//! `WsRequest`/`WsFrame` JSON protocol — only the connect step and the
//! on-the-wire framing differ. [`AssistantCommands`] captures that command
//! surface once: an implementor provides [`AssistantCommands::send_command`]
//! (its transport-specific request/response correlation) and inherits every
//! typed command method as a provided default.
//!
//! This is the impl-sharing counterpart to the public
//! [`crate::transport::AssistantClient`] facade, which dispatches across the
//! `TransportClient` enum. `DbusClient` talks a different wire protocol and so
//! implements its typed methods independently rather than via this trait.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use desktop_assistant_api_model as api;

use crate::types::{ConversationDetail, ConversationSummary};

/// Outcome of a single in-flight request, stored by request id in each
/// client's pending map: `Ok` on a `WsFrame::Result`, `Err(message)` on a
/// `WsFrame::Error` or a connection-level failure.
pub type PendingResult = Result<api::CommandResult, String>;

#[async_trait]
pub trait AssistantCommands: Send + Sync {
    /// Serialize `command` as a `WsRequest`, send it over the transport, and
    /// await the correlated `WsFrame` response.
    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult>;

    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let result = self
            .send_command(api::Command::ListConversations {
                max_age_days: None,
                include_archived: false,
            })
            .await?;
        let api::CommandResult::Conversations(items) = result else {
            return Err(anyhow!("unexpected response for list_conversations"));
        };
        Ok(items.into_iter().map(ConversationSummary::from).collect())
    }

    async fn list_conversations_with_archived(&self) -> Result<Vec<ConversationSummary>> {
        let result = self
            .send_command(api::Command::ListConversations {
                max_age_days: None,
                include_archived: true,
            })
            .await?;
        let api::CommandResult::Conversations(items) = result else {
            return Err(anyhow!("unexpected response for list_conversations"));
        };
        Ok(items.into_iter().map(ConversationSummary::from).collect())
    }

    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        let result = self
            .send_command(api::Command::GetConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Conversation(conversation) = result else {
            return Err(anyhow!("unexpected response for get_conversation"));
        };
        Ok(ConversationDetail::from(conversation))
    }

    async fn create_conversation(&self, title: &str) -> Result<String> {
        let result = self
            .send_command(api::Command::CreateConversation {
                title: title.to_string(),
            })
            .await?;
        let api::CommandResult::ConversationId { id } = result else {
            return Err(anyhow!("unexpected response for create_conversation"));
        };
        Ok(id)
    }

    async fn delete_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for delete_conversation"));
        };
        Ok(())
    }

    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::RenameConversation {
                id: id.to_string(),
                title: title.to_string(),
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for rename_conversation"));
        };
        Ok(())
    }

    async fn archive_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::ArchiveConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for archive_conversation"));
        };
        Ok(())
    }

    async fn unarchive_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::UnarchiveConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for unarchive_conversation"));
        };
        Ok(())
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        let result = self
            .send_command(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
                override_selection: None,
            })
            .await?;
        // Post-#114 the daemon returns `SendMessageAck { task_id }` when its
        // handler is wired with a `BackgroundTaskRegistry`; older / test
        // daemons may still return the legacy bare `Ack`. Both are valid
        // wire-level acks here — the task id is surfaced via streaming events.
        match result {
            api::CommandResult::SendMessageAck { task_id } => Ok(task_id),
            api::CommandResult::Ack => Ok(String::new()),
            other => Err(anyhow!("unexpected response for send_prompt: {other:?}")),
        }
    }

    // --- Knowledge management (issue #73) -------------------------------

    async fn list_knowledge_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            })
            .await?;
        let api::CommandResult::KnowledgeEntries(items) = result else {
            return Err(anyhow!("unexpected response for list_knowledge_entries"));
        };
        Ok(items)
    }

    async fn get_knowledge_entry(&self, id: &str) -> Result<Option<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::GetKnowledgeEntry { id: id.to_string() })
            .await?;
        let api::CommandResult::KnowledgeEntry(entry) = result else {
            return Err(anyhow!("unexpected response for get_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn search_knowledge_entries(
        &self,
        query: &str,
        tag_filter: Option<Vec<String>>,
        limit: u32,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::SearchKnowledgeEntries {
                query: query.to_string(),
                tag_filter,
                limit,
            })
            .await?;
        let api::CommandResult::KnowledgeEntries(items) = result else {
            return Err(anyhow!("unexpected response for search_knowledge_entries"));
        };
        Ok(items)
    }

    async fn create_knowledge_entry(
        &self,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let result = self
            .send_command(api::Command::CreateKnowledgeEntry {
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        let api::CommandResult::KnowledgeEntryWritten(entry) = result else {
            return Err(anyhow!("unexpected response for create_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn update_knowledge_entry(
        &self,
        id: &str,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let result = self
            .send_command(api::Command::UpdateKnowledgeEntry {
                id: id.to_string(),
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        let api::CommandResult::KnowledgeEntryWritten(entry) = result else {
            return Err(anyhow!("unexpected response for update_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn delete_knowledge_entry(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteKnowledgeEntry { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for delete_knowledge_entry"));
        };
        Ok(())
    }
}
