//! SQLite adapter for [`ConversationStore`] (increment 1).
//!
//! Mirrors `desktop-assistant-storage`'s `PgConversationStore`: every query is
//! `(user_id, …)` scoped via [`current_user_id`], cross-user reads behave like
//! the row does not exist (no existence leak, #105), and `update` does a
//! structural diff so unchanged message rows keep their ids.
//!
//! Postgres-ism translations specific to conversations:
//! - `TIMESTAMPTZ` timestamps are stored as canonical `"YYYY-MM-DD HH:MM:SS"`
//!   (UTC) TEXT, which is exactly the domain's own string shape — so no
//!   chrono round-trip is needed and lexicographic order == chronological.
//! - `tags TEXT[]` becomes a JSON-array TEXT column (`json1`); no in-scope
//!   method filters on individual tags, so a join table would be overkill.
//! - `tool_calls` / `last_model_selection` / `personality_override` JSONB
//!   become TEXT holding JSON, read/written as `serde_json::Value`.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, Message, MessageSummary, Role, ToolCall,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::inbound::ConversationModelSelection;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::prompts::PersonalityOverride;
use sqlx::SqlitePool;

/// SQLite adapter for the `conversations` / `messages` / `message_summaries`
/// tables.
pub struct SqliteConversationStore {
    pool: SqlitePool,
}

impl SqliteConversationStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Set (or clear) the stored model selection for a conversation.
    ///
    /// Passing `None` clears the column. Returns [`CoreError::ConversationNotFound`]
    /// when the id is unknown OR belongs to another user (#105: don't leak
    /// existence).
    pub async fn set_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        let json = optional_json(selection, "selection")?;
        self.set_json_column(
            "UPDATE conversations SET last_model_selection = ? WHERE user_id = ? AND id = ?",
            conversation_id,
            json,
        )
        .await
    }

    /// Read the stored model selection for a conversation. `None` when the
    /// conversation exists but has no selection; [`CoreError::ConversationNotFound`]
    /// when the id is unknown or cross-user.
    pub async fn get_conversation_model_selection(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        self.get_json_column(
            "SELECT last_model_selection FROM conversations WHERE user_id = ? AND id = ?",
            conversation_id,
            "last_model_selection",
        )
        .await
    }

    /// Set (or clear) the stored personality override for a conversation (#227).
    pub async fn set_conversation_personality(
        &self,
        conversation_id: &ConversationId,
        personality: Option<&PersonalityOverride>,
    ) -> Result<(), CoreError> {
        let json = optional_json(personality, "personality")?;
        self.set_json_column(
            "UPDATE conversations SET personality_override = ? WHERE user_id = ? AND id = ?",
            conversation_id,
            json,
        )
        .await
    }

    /// Read the stored personality override for a conversation (#227).
    pub async fn get_conversation_personality(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<PersonalityOverride>, CoreError> {
        self.get_json_column(
            "SELECT personality_override FROM conversations WHERE user_id = ? AND id = ?",
            conversation_id,
            "personality_override",
        )
        .await
    }

    /// Read the conversation's tags (e.g. `"voice"`). Empty when the id is
    /// unknown or belongs to another user — fail-closed so an unroutable turn
    /// just misses tag-based routing rather than erroring (voice#126).
    pub async fn get_conversation_tags(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<String>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(String,)> =
            sqlx::query_as("SELECT tags FROM conversations WHERE user_id = ? AND id = ?")
                .bind(user_id.as_str())
                .bind(&conversation_id.0)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(|(tags,)| parse_tags(&tags)).unwrap_or_default())
    }

    /// Shared body of the JSONB-column setters. `sql` is a trusted `'static`
    /// literal supplied at the call site (never user input), so there is no
    /// injection surface.
    async fn set_json_column(
        &self,
        sql: &'static str,
        conversation_id: &ConversationId,
        json: Option<serde_json::Value>,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let result = sqlx::query(sql)
            .bind(json)
            .bind(user_id.as_str())
            .bind(&conversation_id.0)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(conversation_id.0.clone()));
        }
        Ok(())
    }

    /// Shared body of the JSONB-column getters. `sql` is a trusted `'static`
    /// literal supplied at the call site.
    async fn get_json_column<T: serde::de::DeserializeOwned>(
        &self,
        sql: &'static str,
        conversation_id: &ConversationId,
        what: &str,
    ) -> Result<Option<T>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(Option<serde_json::Value>,)> = sqlx::query_as(sql)
            .bind(user_id.as_str())
            .bind(&conversation_id.0)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let row = row.ok_or_else(|| CoreError::ConversationNotFound(conversation_id.0.clone()))?;
        let Some(json) = row.0 else {
            return Ok(None);
        };
        let value = serde_json::from_value(json)
            .map_err(|e| CoreError::Storage(format!("{what} JSON in DB is malformed: {e}")))?;
        Ok(Some(value))
    }
}

impl ConversationStore for SqliteConversationStore {
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
                 compacted_through, archived_at, active_task, tags) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&conv.id.0)
        .bind(user_id.as_str())
        .bind(&conv.title)
        .bind(canonical_ts(&conv.created_at))
        .bind(canonical_ts(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i64)
        .bind(conv.archived_at.as_deref().map(canonical_ts))
        .bind(conv.active_task.as_deref())
        .bind(tags_to_json(&conv.tags))
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
                    compacted_through, archived_at, active_task, tags \
             FROM conversations WHERE user_id = ? AND id = ?",
        )
        .bind(user_id.as_str())
        .bind(&id.0)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let row = row.ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;

        let msg_rows: Vec<MsgRow> = sqlx::query_as(
            "SELECT id, ordinal, role, content, tool_calls, tool_call_id, summary_id, \
                    idempotency_key \
             FROM messages \
             WHERE user_id = ? AND conversation_id = ? \
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
             WHERE user_id = ? AND conversation_id = ? \
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
            created_at: row.created_at,
            updated_at: row.updated_at,
            messages,
            context_summary: row.context_summary,
            compacted_through: row.compacted_through.max(0) as usize,
            summaries,
            archived_at: row.archived_at,
            active_task: row.active_task,
            tags: parse_tags(&row.tags),
        })
    }

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        // DS-6 (#295): a single aggregate query — a `LEFT JOIN messages` +
        // `GROUP BY` computes `message_count` without loading any body. Grouping
        // by the primary key alone is sufficient in SQLite (the other selected
        // columns are functionally dependent on it).
        let user_id = current_user_id();
        let rows: Vec<ConvListRow> = sqlx::query_as(
            "SELECT c.id, c.title, c.created_at, c.updated_at, \
                    c.archived_at, c.tags, COUNT(m.id) AS message_count \
             FROM conversations c \
             LEFT JOIN messages m \
                    ON m.user_id = c.user_id AND m.conversation_id = c.id \
             WHERE c.user_id = ? \
             GROUP BY c.id \
             ORDER BY c.updated_at DESC",
        )
        .bind(user_id.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| ConversationSummary {
                id: ConversationId(row.id),
                title: row.title,
                created_at: row.created_at,
                updated_at: row.updated_at,
                message_count: row.message_count.max(0) as usize,
                archived: row.archived_at.is_some(),
                tags: parse_tags(&row.tags),
            })
            .collect())
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
             SET title = ?, updated_at = ?, \
                 context_summary = ?, compacted_through = ?, active_task = ? \
             WHERE user_id = ? AND id = ?",
        )
        .bind(&conv.title)
        .bind(canonical_ts(&conv.updated_at))
        .bind(&conv.context_summary)
        .bind(conv.compacted_through as i64)
        .bind(conv.active_task.as_deref())
        .bind(user_id.as_str())
        .bind(&conv.id.0)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(conv.id.0.clone()));
        }

        // Structural diff-and-write (DS-5): load existing rows inside this
        // transaction and write only what changed, keeping row ids stable per
        // (conversation, ordinal) slot.
        let existing: Vec<ExistingMsgRow> = sqlx::query_as(
            "SELECT ordinal, role, content, tool_calls, tool_call_id, summary_id \
             FROM messages \
             WHERE user_id = ? AND conversation_id = ? \
             ORDER BY ordinal",
        )
        .bind(user_id.as_str())
        .bind(&conv.id.0)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            match existing.get(ordinal) {
                Some(row) if row.matches(msg) => {}
                Some(_) => {
                    update_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
                }
                None => {
                    insert_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
                }
            }
        }

        // Tail truncation: drop persisted rows past the new length.
        if existing.len() > conv.messages.len() {
            sqlx::query(
                "DELETE FROM messages \
                 WHERE user_id = ? AND conversation_id = ? AND ordinal >= ?",
            )
            .bind(user_id.as_str())
            .bind(&conv.id.0)
            .bind(conv.messages.len() as i64)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // Child rows (messages, message_summaries, idempotency_keys) are removed
        // by ON DELETE CASCADE — enabled per-connection via PRAGMA foreign_keys.
        let result = sqlx::query("DELETE FROM conversations WHERE user_id = ? AND id = ?")
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
            "UPDATE conversations SET archived_at = ? \
             WHERE user_id = ? AND id = ? AND archived_at IS NULL",
        )
        .bind(now_ts())
        .bind(user_id.as_str())
        .bind(&id.0)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            // Either not found, already archived, or owned by a different user.
            // A user-scoped existence probe distinguishes "not yours / gone"
            // (error) from "already archived" (ok) without leaking existence.
            let exists: Option<(i64,)> =
                sqlx::query_as("SELECT 1 FROM conversations WHERE user_id = ? AND id = ?")
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
        let result =
            sqlx::query("UPDATE conversations SET archived_at = NULL WHERE user_id = ? AND id = ?")
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
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(&summary)
        .bind(start_ordinal as i64)
        .bind(end_ordinal as i64)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        sqlx::query(
            "UPDATE messages SET summary_id = ? \
             WHERE user_id = ? AND conversation_id = ? AND ordinal BETWEEN ? AND ?",
        )
        .bind(&id)
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(start_ordinal as i64)
        .bind(end_ordinal as i64)
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
        // ON DELETE SET NULL on messages.summary_id clears the references.
        sqlx::query("DELETE FROM message_summaries WHERE user_id = ? AND id = ?")
            .bind(user_id.as_str())
            .bind(summary_id)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}

/// Serialize an optional value to a JSON `Value` for a nullable JSON column.
fn optional_json<T: serde::Serialize>(
    value: Option<&T>,
    what: &str,
) -> Result<Option<serde_json::Value>, CoreError> {
    match value {
        Some(v) => {
            Ok(Some(serde_json::to_value(v).map_err(|e| {
                CoreError::Storage(format!("{what} json: {e}"))
            })?))
        }
        None => Ok(None),
    }
}

/// JSON representation of a message's tool calls. Empty tool calls map to SQL
/// `NULL` so the unchanged-row comparison in [`ExistingMsgRow::matches`] stays
/// consistent with what is written.
fn tool_calls_json(msg: &Message) -> Option<serde_json::Value> {
    if msg.tool_calls.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&msg.tool_calls).unwrap_or_default())
    }
}

/// Encode conversation tags as a JSON-array string for the `tags` TEXT column.
fn tags_to_json(tags: &[String]) -> String {
    serde_json::to_string(tags).unwrap_or_else(|_| "[]".to_string())
}

/// Decode the `tags` JSON-array column. Fail-closed to empty on malformed JSON
/// (tags are advisory routing hints, never a correctness input) with a warning.
fn parse_tags(json: &str) -> Vec<String> {
    match serde_json::from_str::<Vec<String>>(json) {
        Ok(tags) => tags,
        Err(e) => {
            tracing::warn!(error = %e, raw = %json, "malformed tags JSON in DB; treating as empty");
            Vec::new()
        }
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
        "assistant" => Role::Assistant,
        "system" => Role::System,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

/// The current UTC time as a canonical `"YYYY-MM-DD HH:MM:SS"` string — the
/// domain's own timestamp shape, stored verbatim in TEXT columns.
fn now_ts() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Normalize a domain timestamp string for storage. A value already in the
/// canonical `"YYYY-MM-DD HH:MM:SS"` shape is stored verbatim (so ordering is
/// preserved); an empty or unparseable value falls back to `now()` with a
/// warning, mirroring `PgConversationStore`'s `parse_timestamp`.
fn canonical_ts(s: &str) -> String {
    match chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        Ok(_) => s.to_string(),
        Err(e) => {
            tracing::warn!(timestamp = %s, error = %e, "unparseable timestamp; storing now()");
            now_ts()
        }
    }
}

async fn insert_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    user_id: &str,
    conversation_id: &str,
    ordinal: usize,
    msg: &Message,
) -> Result<(), CoreError> {
    sqlx::query(
        "INSERT INTO messages \
            (id, user_id, conversation_id, ordinal, role, content, \
             tool_calls, tool_call_id, summary_id, idempotency_key) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    // Persist the message's own monotonic UUIDv7 (assigned at creation) so the
    // id is a single source of truth from creation through storage (#1).
    .bind(&msg.id)
    .bind(user_id)
    .bind(conversation_id)
    .bind(ordinal as i64)
    .bind(role_to_str(&msg.role))
    .bind(&msg.content)
    .bind(tool_calls_json(msg))
    .bind(&msg.tool_call_id)
    .bind(&msg.summary_id)
    // #570 Phase 1b: carried on USER rows only; NULL otherwise.
    .bind(&msg.idempotency_key)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(())
}

async fn update_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    user_id: &str,
    conversation_id: &str,
    ordinal: usize,
    msg: &Message,
) -> Result<(), CoreError> {
    sqlx::query(
        "UPDATE messages \
         SET role = ?, content = ?, tool_calls = ?, tool_call_id = ?, summary_id = ? \
         WHERE user_id = ? AND conversation_id = ? AND ordinal = ?",
    )
    .bind(role_to_str(&msg.role))
    .bind(&msg.content)
    .bind(tool_calls_json(msg))
    .bind(&msg.tool_call_id)
    .bind(&msg.summary_id)
    .bind(user_id)
    .bind(conversation_id)
    .bind(ordinal as i64)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(())
}

fn msg_from_row(r: MsgRow) -> Message {
    let mut msg = Message::new(str_to_role(&r.role), &r.content);
    // Carry the persisted UUIDv7 identity (not the fresh one `Message::new`
    // minted) so the id is stable across loads and usable as a client cursor.
    msg.id = r.id;
    if let Some(tc_json) = r.tool_calls
        && let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCall>>(tc_json)
    {
        msg.tool_calls = tool_calls;
    }
    msg.tool_call_id = r.tool_call_id;
    msg.summary_id = r.summary_id;
    // Surface the persisted key (#570 Phase 1b) so a reconnecting client dedups
    // by exact match rather than a content compare.
    msg.idempotency_key = r.idempotency_key;
    msg
}

#[derive(sqlx::FromRow)]
struct ConvRow {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    context_summary: String,
    compacted_through: i64,
    archived_at: Option<String>,
    active_task: Option<String>,
    tags: String,
}

/// Light projection for [`ConversationStore::list`] (DS-6 #295).
#[derive(sqlx::FromRow)]
struct ConvListRow {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    archived_at: Option<String>,
    tags: String,
    message_count: i64,
}

#[derive(sqlx::FromRow)]
struct MsgRow {
    id: String,
    #[allow(dead_code)]
    ordinal: i64,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
    summary_id: Option<String>,
    /// The client idempotency key (#570 Phase 1b); carried onto the domain
    /// `Message` on load. Populated on USER rows only, else NULL.
    idempotency_key: Option<String>,
}

#[derive(sqlx::FromRow)]
struct SummaryRow {
    id: String,
    summary: String,
}

/// An existing persisted message row, loaded inside `update`'s transaction to
/// drive the structural diff (see `PgConversationStore` for the rationale).
#[derive(sqlx::FromRow)]
struct ExistingMsgRow {
    #[allow(dead_code)]
    ordinal: i64,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
    summary_id: Option<String>,
}

impl ExistingMsgRow {
    /// Whether this persisted row is structurally identical to `msg` (a
    /// re-insert would produce the same data). `tool_calls` is compared as a
    /// `serde_json::Value` so key-order normalization can't cause a false
    /// mismatch; empty tool calls are stored as SQL `NULL` on both sides.
    fn matches(&self, msg: &Message) -> bool {
        self.role == role_to_str(&msg.role)
            && self.content == msg.content
            && self.tool_call_id == msg.tool_call_id
            && self.summary_id == msg.summary_id
            && self.tool_calls == tool_calls_json(msg)
    }
}
