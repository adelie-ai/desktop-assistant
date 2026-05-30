use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, Message, MessageSummary, Role, ToolCall,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::inbound::ConversationModelSelection;
use desktop_assistant_core::ports::store::ConversationStore;
use sqlx::PgPool;

pub struct PgConversationStore {
    pool: PgPool,
}

impl PgConversationStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Set (or clear) the stored model selection for a conversation.
    ///
    /// Passing `None` clears the column (NULL). Issue #11: used by the core
    /// service after an override-driven send and by the dangling-selection
    /// fallback path.
    pub async fn set_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let json = match selection {
            Some(sel) => Some(
                serde_json::to_value(sel)
                    .map_err(|e| CoreError::Storage(format!("selection json: {e}")))?,
            ),
            None => None,
        };
        let result = sqlx::query(
            "UPDATE conversations SET last_model_selection = $3 \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(json)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            // Either the conversation id is unknown or it belongs to a
            // different user. We return `ConversationNotFound` in both
            // cases so cross-user probes can't distinguish "doesn't
            // exist" from "not yours" (#105: don't leak existence).
            return Err(CoreError::ConversationNotFound(conversation_id.0.clone()));
        }
        Ok(())
    }

    /// Read the stored model selection for a conversation. Returns `None`
    /// when the conversation exists but has no stored selection; returns
    /// `ConversationNotFound` when the id is unknown OR belongs to a
    /// different user.
    pub async fn get_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(Option<serde_json::Value>,)> = sqlx::query_as(
            "SELECT last_model_selection FROM conversations \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let row = row.ok_or_else(|| CoreError::ConversationNotFound(conversation_id.0.clone()))?;
        let Some(json) = row.0 else {
            return Ok(None);
        };
        let sel: ConversationModelSelection = serde_json::from_value(json).map_err(|e| {
            CoreError::Storage(format!("last_model_selection JSON in DB is malformed: {e}"))
        })?;
        Ok(Some(sel))
    }
}

fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

fn str_to_role(s: &str) -> Role {
    match s {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

impl ConversationStore for PgConversationStore {
    async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        sqlx::query(
            "INSERT INTO conversations \
                (id, user_id, title, created_at, updated_at, context_summary, \
                 compacted_through, archived_at, active_task) \
             VALUES ($1, $2, $3, $4::timestamptz, $5::timestamptz, $6, $7, $8, $9)",
        )
        .bind(&conv.id.0)
        .bind(user_id.as_str())
        .bind(&conv.title)
        .bind(parse_timestamp(&conv.created_at))
        .bind(parse_timestamp(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i32)
        .bind(conv.archived_at.as_deref().map(parse_timestamp))
        .bind(conv.active_task.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            insert_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let user_id = current_user_id();
        let row: Option<ConvRow> = sqlx::query_as(
            "SELECT id, title, created_at, updated_at, context_summary, \
                    compacted_through, archived_at, active_task \
             FROM conversations WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let row = row.ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;

        let msg_rows: Vec<MsgRow> = sqlx::query_as(
            "SELECT ordinal, role, content, tool_calls, tool_call_id, summary_id \
             FROM messages \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY ordinal",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let messages = msg_rows.into_iter().map(msg_from_row).collect();

        let summary_rows: Vec<SummaryRow> = sqlx::query_as(
            "SELECT id, summary \
             FROM message_summaries \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY start_ordinal",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let summaries = summary_rows
            .into_iter()
            .map(|r| MessageSummary {
                id: r.id,
                summary: r.summary,
            })
            .collect();

        Ok(Conversation {
            id: ConversationId(row.id),
            title: row.title,
            created_at: format_timestamp(row.created_at),
            updated_at: format_timestamp(row.updated_at),
            messages,
            context_summary: row.context_summary,
            compacted_through: row.compacted_through as usize,
            summaries,
            archived_at: row.archived_at.map(format_timestamp),
            active_task: row.active_task,
        })
    }

    async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
        let user_id = current_user_id();
        let rows: Vec<ConvRow> = sqlx::query_as(
            "SELECT id, title, created_at, updated_at, context_summary, \
                    compacted_through, archived_at, active_task \
             FROM conversations \
             WHERE user_id = $1 \
             ORDER BY updated_at DESC",
        )
        .bind(user_id.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let msg_rows: Vec<MsgRow> = sqlx::query_as(
                "SELECT ordinal, role, content, tool_calls, tool_call_id, summary_id \
                 FROM messages \
                 WHERE user_id = $1 AND conversation_id = $2 \
                 ORDER BY ordinal",
            )
            .bind(user_id.as_str())
            .bind(&row.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            let messages = msg_rows.into_iter().map(msg_from_row).collect();

            let summary_rows: Vec<SummaryRow> = sqlx::query_as(
                "SELECT id, summary \
                 FROM message_summaries \
                 WHERE user_id = $1 AND conversation_id = $2 \
                 ORDER BY start_ordinal",
            )
            .bind(user_id.as_str())
            .bind(&row.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            let summaries = summary_rows
                .into_iter()
                .map(|r| MessageSummary {
                    id: r.id,
                    summary: r.summary,
                })
                .collect();

            conversations.push(Conversation {
                id: ConversationId(row.id),
                title: row.title,
                created_at: format_timestamp(row.created_at),
                updated_at: format_timestamp(row.updated_at),
                messages,
                context_summary: row.context_summary,
                compacted_through: row.compacted_through as usize,
                summaries,
                archived_at: row.archived_at.map(format_timestamp),
                active_task: row.active_task,
            });
        }

        Ok(conversations)
    }

    async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let result = sqlx::query(
            "UPDATE conversations \
             SET title = $3, updated_at = $4::timestamptz, \
                 context_summary = $5, compacted_through = $6, active_task = $7 \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&conv.id.0)
        .bind(&conv.title)
        .bind(parse_timestamp(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i32)
        .bind(conv.active_task.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(conv.id.0.clone()));
        }

        // Replace all messages: delete existing, re-insert. Scoped by
        // user_id as defense-in-depth — the UPDATE above already
        // proved the conversation belongs to this user.
        sqlx::query("DELETE FROM messages WHERE user_id = $1 AND conversation_id = $2")
            .bind(user_id.as_str())
            .bind(&conv.id.0)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            insert_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let result = sqlx::query("DELETE FROM conversations WHERE user_id = $1 AND id = $2")
            .bind(user_id.as_str())
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(id.0.clone()));
        }
        Ok(())
    }

    async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let result = sqlx::query(
            "UPDATE conversations SET archived_at = NOW() \
             WHERE user_id = $1 AND id = $2 AND archived_at IS NULL",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            // Either not found, already archived, or owned by a
            // different user — `SELECT 1 …` distinguishes. The
            // existence probe is itself user-scoped so a cross-user
            // lookup still returns "not found" without leaking.
            let exists: Option<(i64,)> =
                sqlx::query_as("SELECT 1 FROM conversations WHERE user_id = $1 AND id = $2")
                    .bind(user_id.as_str())
                    .bind(&id.0)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| CoreError::Storage(e.to_string()))?;
            if exists.is_none() {
                return Err(CoreError::ConversationNotFound(id.0.clone()));
            }
        }
        Ok(())
    }

    async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let result = sqlx::query(
            "UPDATE conversations SET archived_at = NULL \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(id.0.clone()));
        }
        Ok(())
    }

    async fn create_summary(
        &self,
        conversation_id: &ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        let user_id = current_user_id();
        let id = uuid::Uuid::now_v7().to_string();
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        sqlx::query(
            "INSERT INTO message_summaries \
                (id, user_id, conversation_id, summary, start_ordinal, end_ordinal) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(&id)
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(&summary)
        .bind(start_ordinal as i32)
        .bind(end_ordinal as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        sqlx::query(
            "UPDATE messages SET summary_id = $3 \
             WHERE user_id = $1 AND conversation_id = $2 AND ordinal BETWEEN $4 AND $5",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(&id)
        .bind(start_ordinal as i32)
        .bind(end_ordinal as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(id)
    }

    async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // ON DELETE SET NULL on messages.summary_id handles clearing the references.
        sqlx::query("DELETE FROM message_summaries WHERE user_id = $1 AND id = $2")
            .bind(user_id.as_str())
            .bind(summary_id)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}

async fn insert_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    user_id: &str,
    conversation_id: &str,
    ordinal: usize,
    msg: &Message,
) -> Result<(), CoreError> {
    let tool_calls_json = if msg.tool_calls.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&msg.tool_calls).unwrap_or_default())
    };

    sqlx::query(
        "INSERT INTO messages \
            (id, user_id, conversation_id, ordinal, role, content, \
             tool_calls, tool_call_id, summary_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(user_id)
    .bind(conversation_id)
    .bind(ordinal as i32)
    .bind(role_to_str(&msg.role))
    .bind(&msg.content)
    .bind(tool_calls_json)
    .bind(&msg.tool_call_id)
    .bind(&msg.summary_id)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(())
}

fn msg_from_row(r: MsgRow) -> Message {
    let mut msg = Message::new(str_to_role(&r.role), &r.content);
    if let Some(tc_json) = r.tool_calls
        && let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCall>>(tc_json)
    {
        msg.tool_calls = tool_calls;
    }
    msg.tool_call_id = r.tool_call_id;
    msg.summary_id = r.summary_id;
    msg
}

fn parse_timestamp(s: &str) -> chrono::DateTime<chrono::Utc> {
    // Try parsing the local format "YYYY-MM-DD HH:MM:SS" as UTC
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .map(|naive| naive.and_utc())
        .unwrap_or_else(|_| chrono::Utc::now())
}

fn format_timestamp(dt: chrono::DateTime<chrono::Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

#[derive(sqlx::FromRow)]
struct ConvRow {
    id: String,
    title: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    context_summary: String,
    compacted_through: i32,
    archived_at: Option<chrono::DateTime<chrono::Utc>>,
    active_task: Option<String>,
}

#[derive(sqlx::FromRow)]
struct MsgRow {
    #[allow(dead_code)]
    ordinal: i32,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
    summary_id: Option<String>,
}

#[derive(sqlx::FromRow)]
struct SummaryRow {
    id: String,
    summary: String,
}
