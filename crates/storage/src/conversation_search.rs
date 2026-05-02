use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::Role;
use desktop_assistant_core::ports::conversation_search::{ConversationSearchStore, MessageHit};
use sqlx::PgPool;

pub struct PgConversationSearchStore {
    pool: PgPool,
}

impl PgConversationSearchStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// Internal SQL row shape. We materialise role as the raw text Postgres
/// stores (matches `messages.role` in the legacy schema) and parse it
/// into [`Role`] when constructing the public hit.
#[derive(sqlx::FromRow)]
struct MessageHitRow {
    conversation_id: String,
    conversation_title: String,
    ordinal: i32,
    role: String,
    content: String,
    snippet: String,
    rank: f32,
    updated_at: chrono::DateTime<chrono::Utc>,
}

fn parse_role(s: &str) -> Role {
    // The DB column is unconstrained text; we map known values and fall
    // back to `User` for anything unexpected so a corrupt row doesn't
    // crash search. The role filter on inbound queries is enum-checked
    // by the caller before reaching SQL, so this lossy fallback is
    // confined to read-side display.
    match s {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

fn role_to_db(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

impl ConversationSearchStore for PgConversationSearchStore {
    async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        role_filter: Option<Role>,
    ) -> Result<Vec<MessageHit>, CoreError> {
        let role_db = role_filter.map(role_to_db);
        let limit_i64 = limit as i64;

        // Hits where the message itself matches OR the conversation
        // title/summary matches; rank by message tsv when present, else
        // by conversation tsv. Snippet runs over message content so the
        // tool result has something concrete to show even when the
        // match was on title/summary (an empty `ts_headline` is fine).
        let rows: Vec<MessageHitRow> = sqlx::query_as(
            "WITH q AS (
                 SELECT plainto_tsquery('english', $1) AS query
             )
             SELECT m.conversation_id      AS conversation_id,
                    c.title                AS conversation_title,
                    m.ordinal              AS ordinal,
                    m.role                 AS role,
                    m.content              AS content,
                    ts_headline(
                        'english',
                        m.content,
                        (SELECT query FROM q),
                        'StartSel=<mark>,StopSel=</mark>,MaxFragments=1,MaxWords=20,MinWords=5'
                    )                      AS snippet,
                    GREATEST(
                        COALESCE(ts_rank_cd(m.tsv, (SELECT query FROM q)), 0.0),
                        COALESCE(ts_rank_cd(c.tsv, (SELECT query FROM q)), 0.0)
                    )::REAL                AS rank,
                    c.updated_at           AS updated_at
             FROM messages m
             JOIN conversations c ON c.id = m.conversation_id
             WHERE
                 (m.tsv @@ (SELECT query FROM q)
                  OR c.tsv @@ (SELECT query FROM q))
                 AND ($2::text IS NULL OR m.role = $2)
             ORDER BY rank DESC, c.updated_at DESC, m.ordinal ASC
             LIMIT $3",
        )
        .bind(query)
        .bind(role_db)
        .bind(limit_i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| MessageHit {
                conversation_id: r.conversation_id,
                conversation_title: r.conversation_title,
                ordinal: r.ordinal,
                role: parse_role(&r.role),
                content: r.content,
                snippet: r.snippet,
                rank: r.rank,
                updated_at: r.updated_at.to_rfc3339(),
            })
            .collect())
    }
}
