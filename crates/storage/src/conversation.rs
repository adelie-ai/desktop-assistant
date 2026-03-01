use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, Message, Role, ToolCall,
};
use desktop_assistant_core::ports::store::ConversationStore;
use sqlx::PgPool;

pub struct PgConversationStore {
    pool: PgPool,
}

impl PgConversationStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
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
        let mut tx = self.pool.begin().await.map_err(|e| CoreError::Storage(e.to_string()))?;

        sqlx::query(
            "INSERT INTO conversations (id, title, created_at, updated_at, context_summary, compacted_through)
             VALUES ($1, $2, $3::timestamptz, $4::timestamptz, $5, $6)"
        )
        .bind(&conv.id.0)
        .bind(&conv.title)
        .bind(parse_timestamp(&conv.created_at))
        .bind(parse_timestamp(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            insert_message(&mut tx, &conv.id.0, ordinal, msg).await?;
        }

        tx.commit().await.map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let row: Option<ConvRow> = sqlx::query_as(
            "SELECT id, title, created_at, updated_at, context_summary, compacted_through
             FROM conversations WHERE id = $1"
        )
        .bind(&id.0)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let row = row.ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;

        let msg_rows: Vec<MsgRow> = sqlx::query_as(
            "SELECT ordinal, role, content, tool_calls, tool_call_id
             FROM messages WHERE conversation_id = $1 ORDER BY ordinal"
        )
        .bind(&id.0)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let messages = msg_rows.into_iter().map(|r| {
            let mut msg = Message::new(str_to_role(&r.role), &r.content);
            if let Some(tc_json) = r.tool_calls {
                if let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCall>>(tc_json) {
                    msg.tool_calls = tool_calls;
                }
            }
            msg.tool_call_id = r.tool_call_id;
            msg
        }).collect();

        Ok(Conversation {
            id: ConversationId(row.id),
            title: row.title,
            created_at: format_timestamp(row.created_at),
            updated_at: format_timestamp(row.updated_at),
            messages,
            context_summary: row.context_summary,
            compacted_through: row.compacted_through as usize,
        })
    }

    async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
        let rows: Vec<ConvRow> = sqlx::query_as(
            "SELECT id, title, created_at, updated_at, context_summary, compacted_through
             FROM conversations ORDER BY updated_at DESC"
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let msg_rows: Vec<MsgRow> = sqlx::query_as(
                "SELECT ordinal, role, content, tool_calls, tool_call_id
                 FROM messages WHERE conversation_id = $1 ORDER BY ordinal"
            )
            .bind(&row.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            let messages = msg_rows.into_iter().map(|r| {
                let mut msg = Message::new(str_to_role(&r.role), &r.content);
                if let Some(tc_json) = r.tool_calls {
                    if let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCall>>(tc_json) {
                        msg.tool_calls = tool_calls;
                    }
                }
                msg.tool_call_id = r.tool_call_id;
                msg
            }).collect();

            conversations.push(Conversation {
                id: ConversationId(row.id),
                title: row.title,
                created_at: format_timestamp(row.created_at),
                updated_at: format_timestamp(row.updated_at),
                messages,
                context_summary: row.context_summary,
                compacted_through: row.compacted_through as usize,
            });
        }

        Ok(conversations)
    }

    async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
        let mut tx = self.pool.begin().await.map_err(|e| CoreError::Storage(e.to_string()))?;

        let result = sqlx::query(
            "UPDATE conversations SET title = $2, updated_at = $3::timestamptz,
                    context_summary = $4, compacted_through = $5
             WHERE id = $1"
        )
        .bind(&conv.id.0)
        .bind(&conv.title)
        .bind(parse_timestamp(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(conv.id.0.clone()));
        }

        // Replace all messages: delete existing, re-insert
        sqlx::query("DELETE FROM messages WHERE conversation_id = $1")
            .bind(&conv.id.0)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            insert_message(&mut tx, &conv.id.0, ordinal, msg).await?;
        }

        tx.commit().await.map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        let result = sqlx::query("DELETE FROM conversations WHERE id = $1")
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(id.0.clone()));
        }
        Ok(())
    }
}

async fn insert_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
        "INSERT INTO messages (conversation_id, ordinal, role, content, tool_calls, tool_call_id)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(conversation_id)
    .bind(ordinal as i32)
    .bind(role_to_str(&msg.role))
    .bind(&msg.content)
    .bind(tool_calls_json)
    .bind(&msg.tool_call_id)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(())
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
}

#[derive(sqlx::FromRow)]
struct MsgRow {
    #[allow(dead_code)]
    ordinal: i32,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
}
