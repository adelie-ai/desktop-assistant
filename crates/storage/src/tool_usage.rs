//! Postgres adapter for the tool-usage cost aggregate (#599).
//!
//! Aggregated entirely SQL-side. A long conversation must not cost a full
//! history load to draw one chart, and the numbers are derived from rows that
//! already exist (`messages.tool_calls` joined to their `Role::Tool` results),
//! so there is nothing to keep in sync and it works retroactively.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::planning::COMPACTION_POINTER_PREFIX;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::tool_usage::{ToolUsage, ToolUsageStore};
use sqlx::PgPool;

pub struct PgToolUsageStore {
    pool: PgPool,
}

impl PgToolUsageStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(sqlx::FromRow)]
struct ToolUsageRow {
    tool_name: String,
    call_count: i64,
    result_bytes: i64,
    max_result_bytes: i64,
    evicted_results: i64,
    first_ordinal: i32,
    last_ordinal: i32,
    first_used_at: Option<chrono::DateTime<chrono::Utc>>,
    last_used_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ToolUsageRow {
    fn into_usage(self) -> ToolUsage {
        ToolUsage {
            // Namespace is resolved by the caller from the live tool registry,
            // not stored per call — a name's namespace is a property of the
            // registry today, and baking a stale one into history would make the
            // grouping lie after a server is renamed.
            namespace: None,
            tool_name: self.tool_name,
            call_count: self.call_count.max(0) as u32,
            result_bytes: self.result_bytes.max(0) as u64,
            max_result_bytes: self.max_result_bytes.max(0) as u64,
            evicted_results: self.evicted_results.max(0) as u32,
            first_ordinal: self.first_ordinal,
            last_ordinal: self.last_ordinal,
            first_used_at: self.first_used_at.map(|t| t.to_rfc3339()),
            last_used_at: self.last_used_at.map(|t| t.to_rfc3339()),
        }
    }
}

impl ToolUsageStore for PgToolUsageStore {
    async fn tool_usage(&self, conversation_id: &str) -> Result<Vec<ToolUsage>, CoreError> {
        let user_id = current_user_id();
        // `evicted` is matched on the pointer's stable PREFIX, which is a
        // deliberate constant shared with the compaction code rather than a
        // literal duplicated here — so a reworded pointer can't silently stop
        // being recognised and quietly zero the eviction column.
        let evicted_like = format!("{COMPACTION_POINTER_PREFIX}%");
        let rows: Vec<ToolUsageRow> = sqlx::query_as(
            // Calls and results are aggregated SEPARATELY and then joined, so a
            // call can never be counted twice. Aggregating over a single joined
            // set would let `COUNT(*)` count RESULT rows rather than calls — any
            // call matched by more than one result row (a duplicated
            // `tool_call_id`) would silently inflate the frequency axis, which is
            // the number this whole view is read for.
            "WITH calls AS ( \
                 SELECT tc->>'name' AS tool_name, \
                        tc->>'id'   AS call_id, \
                        m.ordinal   AS ordinal, \
                        m.id        AS message_id \
                 FROM messages m \
                 CROSS JOIN LATERAL jsonb_array_elements(m.tool_calls) AS tc \
                 WHERE m.user_id = $1 AND m.conversation_id = $2 \
                   AND m.tool_calls IS NOT NULL \
                   AND jsonb_typeof(m.tool_calls) = 'array' \
             ), \
             call_agg AS ( \
                 SELECT tool_name, \
                        COUNT(*)     AS call_count, \
                        MIN(ordinal) AS first_ordinal, \
                        MAX(ordinal) AS last_ordinal, \
                        MIN(uuidv7_ts(message_id)) AS first_used_at, \
                        MAX(uuidv7_ts(message_id)) AS last_used_at \
                 FROM calls WHERE tool_name IS NOT NULL GROUP BY tool_name \
             ), \
             result_agg AS ( \
                 SELECT c.tool_name AS tool_name, \
                        SUM(CASE WHEN r.content LIKE $3 THEN 0 \
                                 ELSE octet_length(r.content) END) AS result_bytes, \
                        MAX(CASE WHEN r.content LIKE $3 THEN 0 \
                                 ELSE octet_length(r.content) END) AS max_result_bytes, \
                        COUNT(*) FILTER (WHERE r.content LIKE $3) AS evicted_results \
                 FROM calls c \
                 JOIN messages r \
                   ON r.user_id = $1 AND r.conversation_id = $2 \
                  AND r.tool_call_id = c.call_id \
                 WHERE c.tool_name IS NOT NULL \
                 GROUP BY c.tool_name \
             ) \
             SELECT ca.tool_name                             AS tool_name, \
                    ca.call_count::bigint                    AS call_count, \
                    COALESCE(ra.result_bytes, 0)::bigint     AS result_bytes, \
                    COALESCE(ra.max_result_bytes, 0)::bigint AS max_result_bytes, \
                    COALESCE(ra.evicted_results, 0)::bigint  AS evicted_results, \
                    ca.first_ordinal::int                    AS first_ordinal, \
                    ca.last_ordinal::int                     AS last_ordinal, \
                    ca.first_used_at                         AS first_used_at, \
                    ca.last_used_at                          AS last_used_at \
             FROM call_agg ca \
             LEFT JOIN result_agg ra ON ra.tool_name = ca.tool_name \
             ORDER BY ca.call_count DESC, ca.tool_name ASC",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(&evicted_like)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(ToolUsageRow::into_usage).collect())
    }
}
