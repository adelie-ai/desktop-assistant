use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, Message, MessageSummary, Role, ToolCall,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::inbound::ConversationModelSelection;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::prompts::PersonalityOverride;
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

    /// Set (or clear) the stored personality override for a conversation (#227).
    ///
    /// Passing `None` (or an empty/all-`None` override — handled by the caller)
    /// clears the column (NULL). Mirrors
    /// [`Self::set_conversation_model_selection`]: JSONB column, user-scoped,
    /// `ConversationNotFound` on a missing/cross-user row (#105: don't leak
    /// existence).
    pub async fn set_conversation_personality(
        &self,
        conversation_id: &ConversationId,
        personality: Option<&PersonalityOverride>,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let json = match personality {
            Some(p) => Some(
                serde_json::to_value(p)
                    .map_err(|e| CoreError::Storage(format!("personality json: {e}")))?,
            ),
            None => None,
        };
        let result = sqlx::query(
            "UPDATE conversations SET personality_override = $3 \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id.0)
        .bind(json)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            return Err(CoreError::ConversationNotFound(conversation_id.0.clone()));
        }
        Ok(())
    }

    /// Read the stored personality override for a conversation (#227). Returns
    /// `None` when the conversation exists but has no stored override; returns
    /// `ConversationNotFound` when the id is unknown OR belongs to a different
    /// user. Mirrors [`Self::get_conversation_model_selection`].
    pub async fn get_conversation_personality(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Option<PersonalityOverride>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(Option<serde_json::Value>,)> = sqlx::query_as(
            "SELECT personality_override FROM conversations \
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
        let ovr: PersonalityOverride = serde_json::from_value(json).map_err(|e| {
            CoreError::Storage(format!("personality_override JSON in DB is malformed: {e}"))
        })?;
        Ok(Some(ovr))
    }

    /// Read the conversation's tags (e.g. `"voice"`). Empty when the id is
    /// unknown or belongs to another user — fail-closed: an unroutable turn
    /// simply doesn't get tag-based routing rather than erroring. Mirrors
    /// [`Self::get_conversation_personality`] (voice#126).
    pub async fn get_conversation_tags(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<String>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(Vec<String>,)> =
            sqlx::query_as("SELECT tags FROM conversations WHERE user_id = $1 AND id = $2")
                .bind(user_id.as_str())
                .bind(&conversation_id.0)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(|(tags,)| tags).unwrap_or_default())
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
                 compacted_through, archived_at, active_task, tags) \
             VALUES ($1, $2, $3, $4::timestamptz, $5::timestamptz, $6, $7, $8, $9, $10)",
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
        .bind(&conv.tags)
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
             FROM conversations WHERE user_id = $1 AND id = $2",
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
            tags: row.tags,
        })
    }

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        // DS-6 (#295): a single aggregate query. The previous implementation
        // ran 1 + 2N queries (a per-conversation message fetch and a
        // per-conversation summary fetch) and loaded every message body just
        // to produce a list. `list` only needs metadata plus a count, so we
        // LEFT JOIN messages and GROUP BY to compute `message_count` in one
        // round trip — no bodies leave the database.
        let user_id = current_user_id();
        let rows: Vec<ConvListRow> = sqlx::query_as(
            "SELECT c.id, c.title, c.created_at, c.updated_at, \
                    c.archived_at, c.tags, COUNT(m.id) AS message_count \
             FROM conversations c \
             LEFT JOIN messages m \
                    ON m.user_id = c.user_id AND m.conversation_id = c.id \
             WHERE c.user_id = $1 \
             GROUP BY c.id, c.title, c.created_at, c.updated_at, c.archived_at, c.tags \
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
                created_at: format_timestamp(row.created_at),
                updated_at: format_timestamp(row.updated_at),
                message_count: row.message_count.max(0) as usize,
                archived: row.archived_at.is_some(),
                tags: row.tags,
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

        // Structural diff-and-write (DS-5): load the existing rows in this
        // same transaction (which already proved ownership) and write only
        // what actually changed, keeping row ids stable per (conversation,
        // ordinal) slot. The dominant cases — append-only turns and
        // compaction stamping `summary_id` — touch no prior rows, so their
        // tsvectors are not regenerated. Every statement stays scoped by
        // user_id as defense-in-depth.
        let existing: Vec<ExistingMsgRow> = sqlx::query_as(
            "SELECT ordinal, role, content, tool_calls, tool_call_id, summary_id \
             FROM messages \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY ordinal",
        )
        .bind(user_id.as_str())
        .bind(&conv.id.0)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (ordinal, msg) in conv.messages.iter().enumerate() {
            match existing.get(ordinal) {
                // Row at this slot is structurally identical — no write at
                // all, so its id and tsvector are untouched.
                Some(row) if row.matches(msg) => {}
                // Row exists but differs (e.g. summary_id stamped, or
                // content shifted up by a mid-history removal): UPDATE in
                // place, preserving the row id.
                Some(_) => {
                    update_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
                }
                // No row at this slot — a genuinely new (appended) message.
                None => {
                    insert_message(&mut tx, user_id.as_str(), &conv.id.0, ordinal, msg).await?;
                }
            }
        }

        // Tail truncation: drop any persisted rows past the new length in a
        // single ranged DELETE.
        if existing.len() > conv.messages.len() {
            sqlx::query(
                "DELETE FROM messages \
                 WHERE user_id = $1 AND conversation_id = $2 AND ordinal >= $3",
            )
            .bind(user_id.as_str())
            .bind(&conv.id.0)
            .bind(conv.messages.len() as i32)
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
            //
            // The literal `1` is Postgres `int4`, so it must decode into an
            // `i32`: decoding into `i64` errored ("Rust type i64 … not
            // compatible with SQL type INT4") only when a row actually
            // matched — i.e. re-archiving your OWN already-archived
            // conversation returned a Storage error instead of Ok. The
            // not-found path never decoded a row, which is why the bug hid.
            let exists: Option<(i32,)> =
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

/// JSONB representation of a message's tool calls. Empty tool calls map to
/// SQL `NULL` (the historical `insert_message` behavior), so the unchanged-row
/// comparison in `ExistingMsgRow::matches` stays consistent.
fn tool_calls_json(msg: &Message) -> Option<serde_json::Value> {
    if msg.tool_calls.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&msg.tool_calls).unwrap_or_default())
    }
}

async fn insert_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    user_id: &str,
    conversation_id: &str,
    ordinal: usize,
    msg: &Message,
) -> Result<(), CoreError> {
    let tool_calls_json = tool_calls_json(msg);

    sqlx::query(
        "INSERT INTO messages \
            (id, user_id, conversation_id, ordinal, role, content, \
             tool_calls, tool_call_id, summary_id, idempotency_key) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
    )
    // Persist the message's own monotonic UUIDv7 (assigned at creation) rather
    // than minting a fresh one here, so the id is a single source of truth from
    // creation through storage to the client cursor (#1).
    .bind(&msg.id)
    .bind(user_id)
    .bind(conversation_id)
    .bind(ordinal as i32)
    .bind(role_to_str(&msg.role))
    .bind(&msg.content)
    .bind(tool_calls_json)
    .bind(&msg.tool_call_id)
    .bind(&msg.summary_id)
    // #570 Phase 1b: carried on USER rows only (the persist site stamps it);
    // NULL for assistant/tool rows and keyless sends. `update_message` does not
    // touch this column, so the key is stable once written.
    .bind(&msg.idempotency_key)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(())
}

/// In-place UPDATE of the message occupying `(conversation_id, ordinal)`,
/// preserving its row id (and so its primary-key identity). Only called when
/// the slot's content differs from what's persisted, so the regenerated
/// tsvector cost is paid only for genuinely changed rows.
async fn update_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    user_id: &str,
    conversation_id: &str,
    ordinal: usize,
    msg: &Message,
) -> Result<(), CoreError> {
    let tool_calls_json = tool_calls_json(msg);

    sqlx::query(
        "UPDATE messages \
         SET role = $4, content = $5, tool_calls = $6, \
             tool_call_id = $7, summary_id = $8 \
         WHERE user_id = $1 AND conversation_id = $2 AND ordinal = $3",
    )
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

fn parse_timestamp(s: &str) -> chrono::DateTime<chrono::Utc> {
    // Try parsing the local format "YYYY-MM-DD HH:MM:SS" as UTC.
    match chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        Ok(naive) => naive.and_utc(),
        Err(e) => {
            // DS-9: a malformed timestamp used to silently become `now()`,
            // which scrambles ordering invisibly. Surface it so corrupt data
            // is diagnosable rather than masquerading as a fresh row.
            tracing::warn!(
                timestamp = %s,
                error = %e,
                "failed to parse stored timestamp; falling back to now()"
            );
            chrono::Utc::now()
        }
    }
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
    tags: Vec<String>,
}

/// Light projection for [`ConversationStore::list`] (DS-6 #295): conversation
/// metadata plus an aggregate `message_count`, with no message bodies.
#[derive(sqlx::FromRow)]
struct ConvListRow {
    id: String,
    title: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    archived_at: Option<chrono::DateTime<chrono::Utc>>,
    tags: Vec<String>,
    message_count: i64,
}

#[derive(sqlx::FromRow)]
struct MsgRow {
    /// The message's stable UUIDv7 id (migration 005); carried onto the domain
    /// `Message` so clients get a sortable identity + cursor (#1).
    id: String,
    #[allow(dead_code)]
    ordinal: i32,
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
/// drive the structural diff. Carries exactly the columns `insert_message`
/// writes (minus the id, which is what we are preserving) so a row can be
/// compared structurally against an in-memory `Message`.
#[derive(sqlx::FromRow)]
struct ExistingMsgRow {
    #[allow(dead_code)]
    ordinal: i32,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
    summary_id: Option<String>,
}

impl ExistingMsgRow {
    /// Whether this persisted row is structurally identical to `msg`, i.e. a
    /// re-insert would produce the same data. `tool_calls` is compared as a
    /// `serde_json::Value` so JSONB key-order normalization cannot cause a
    /// false mismatch; empty tool calls are stored as SQL `NULL` on both sides
    /// (see `tool_calls_json`).
    fn matches(&self, msg: &Message) -> bool {
        self.role == role_to_str(&msg.role)
            && self.content == msg.content
            && self.tool_call_id == msg.tool_call_id
            && self.summary_id == msg.summary_id
            && self.tool_calls == tool_calls_json(msg)
    }
}
