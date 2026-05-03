use desktop_assistant_core::CoreError;
use sqlx::postgres::PgRow;
use sqlx::{Column, PgPool, Row, TypeInfo};

/// Execute a SQL query and return results as JSON.
///
/// **Read queries** (SELECT / WITH / TABLE / VALUES / EXPLAIN) run inside a
/// READ ONLY transaction with an automatic LIMIT appended when absent.
///
/// **Write queries** (CREATE / INSERT / UPDATE / DELETE / DROP / ALTER / …)
/// run in a normal transaction that is committed on success.  The transaction's
/// `search_path` is set to `scratch, public` so that unqualified table
/// references in DDL/DML resolve to the `scratch` schema while all public
/// tables remain readable.  The `scratch` schema is created lazily if it does
/// not yet exist.
///
/// Returns:
/// - Row-returning queries: `{ "columns": [...], "rows": [[...], ...], "row_count": N }`
/// - Non-row-returning writes: `{ "rows_affected": N }`
pub async fn execute_database_query(
    pool: &PgPool,
    sql: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let sql_trimmed = sql.trim().trim_end_matches(';');
    let upper = sql_trimmed.to_uppercase();

    // Classify on the *first non-comment* keyword (#40). A naive
    // `split_whitespace().next()` returns `/*` for a query that opens
    // with a block comment, which falls through to the write path —
    // bypassing the READ ONLY transaction reads run under. Strip
    // leading SQL comments first so `/* */ SELECT *` correctly
    // routes through `execute_read`.
    let stripped = strip_leading_sql_comments(&upper);
    let first_keyword = stripped.split_whitespace().next().unwrap_or("");

    let is_read = matches!(
        first_keyword,
        "SELECT" | "WITH" | "TABLE" | "VALUES" | "EXPLAIN"
    );

    if is_read {
        execute_read(pool, sql_trimmed, &upper, limit).await
    } else {
        execute_write(pool, sql_trimmed, &upper).await
    }
}

/// Strip leading SQL comments (`--` line comments and `/* … */` block
/// comments, including nested blocks per Postgres) plus the
/// whitespace between them. Returns a substring of `sql` starting at
/// the first character that is neither a comment nor whitespace.
///
/// On a malformed leading block comment (no closing `*/`), returns an
/// empty string — the caller treats that as "no recognisable
/// keyword", which routes to the write path where Postgres rejects
/// the malformed statement at parse time. Same outcome as a
/// nonsensical query without the comment.
fn strip_leading_sql_comments(sql: &str) -> &str {
    let bytes = sql.as_bytes();
    let mut i = 0;
    loop {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i + 1 >= bytes.len() {
            break;
        }
        if bytes[i] == b'-' && bytes[i + 1] == b'-' {
            // Line comment runs to end of line (LF or CR/LF) or end of input.
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            // Skip the newline so the next iteration sees the post-comment text.
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            // Block comment, with nesting (Postgres extension to ANSI SQL).
            let mut depth: usize = 1;
            i += 2;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                // Unterminated block comment — treat as if the whole
                // remainder is still inside a comment so the caller
                // sees no keyword and routes to the write path, where
                // Postgres will reject the malformed statement.
                return "";
            }
            continue;
        }
        break;
    }
    &sql[i..]
}

/// Read path — READ ONLY transaction, auto-LIMIT, always rolled back.
async fn execute_read(
    pool: &PgPool,
    sql: &str,
    upper: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let has_limit = upper.contains(" LIMIT ");

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // When the user query lacks a LIMIT clause, wrap it in a subquery with a
    // parameterised limit to avoid string-formatting user SQL.
    let rows: Vec<PgRow> = if has_limit {
        sqlx::query(sql)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?
    } else {
        let wrapped = format!("SELECT * FROM ({sql}) AS _limited LIMIT $1");
        sqlx::query(&wrapped)
            .bind(limit as i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?
    };

    tx.rollback()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    rows_to_json(&rows)
}

/// Write path — ensures `scratch` schema exists, sets search_path to
/// `scratch, public`, executes the statement, and commits.
async fn execute_write(
    pool: &PgPool,
    sql: &str,
    upper: &str,
) -> Result<serde_json::Value, CoreError> {
    // Ensure the scratch schema exists (idempotent).
    sqlx::query("CREATE SCHEMA IF NOT EXISTS scratch")
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // Unqualified writes go to scratch; public tables are still readable.
    sqlx::query("SET LOCAL search_path TO scratch, public")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // If the statement contains RETURNING it will produce rows.
    let has_returning = upper.contains("RETURNING");

    if has_returning {
        let rows: Vec<PgRow> = sqlx::query(sql)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        rows_to_json(&rows)
    } else {
        let result = sqlx::query(sql)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(serde_json::json!({
            "rows_affected": result.rows_affected()
        }))
    }
}

/// Convert a slice of `PgRow` into the standard JSON result envelope.
fn rows_to_json(rows: &[PgRow]) -> Result<serde_json::Value, CoreError> {
    let columns: Vec<String> = if let Some(first) = rows.first() {
        first
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    } else {
        return Ok(serde_json::json!({
            "columns": [],
            "rows": [],
            "row_count": 0
        }));
    };

    let mut json_rows: Vec<Vec<serde_json::Value>> = Vec::with_capacity(rows.len());

    for row in rows {
        let mut json_row = Vec::with_capacity(columns.len());
        for (i, col) in row.columns().iter().enumerate() {
            let type_name = col.type_info().name();
            json_row.push(pg_value_to_json(row, i, type_name));
        }
        json_rows.push(json_row);
    }

    let row_count = json_rows.len();
    Ok(serde_json::json!({
        "columns": columns,
        "rows": json_rows,
        "row_count": row_count
    }))
}

/// Convert a single column value from a PgRow into a serde_json::Value.
fn pg_value_to_json(row: &PgRow, index: usize, type_name: &str) -> serde_json::Value {
    match type_name {
        "TEXT" | "VARCHAR" | "CHAR" | "BPCHAR" | "NAME" => {
            match row.try_get::<Option<String>, _>(index) {
                Ok(Some(v)) => serde_json::Value::String(v),
                Ok(None) => serde_json::Value::Null,
                Err(_) => serde_json::Value::Null,
            }
        }
        "UUID" => match row.try_get::<Option<uuid::Uuid>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v.to_string()),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT2" => match row.try_get::<Option<i16>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT4" => match row.try_get::<Option<i32>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "INT8" => match row.try_get::<Option<i64>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "FLOAT4" => match row.try_get::<Option<f32>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "FLOAT8" | "NUMERIC" => match row.try_get::<Option<f64>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "BOOL" => match row.try_get::<Option<bool>, _>(index) {
            Ok(Some(v)) => serde_json::json!(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "TIMESTAMPTZ" | "TIMESTAMP" => {
            match row.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>(index) {
                Ok(Some(v)) => serde_json::Value::String(v.to_rfc3339()),
                Ok(None) => serde_json::Value::Null,
                Err(_) => match row.try_get::<Option<chrono::NaiveDateTime>, _>(index) {
                    Ok(Some(v)) => serde_json::Value::String(v.to_string()),
                    _ => serde_json::Value::Null,
                },
            }
        }
        "DATE" => match row.try_get::<Option<chrono::NaiveDate>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v.to_string()),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "JSON" | "JSONB" => match row.try_get::<Option<serde_json::Value>, _>(index) {
            Ok(Some(v)) => v,
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
        "TEXT[]" | "_TEXT" | "VARCHAR[]" | "_VARCHAR" => {
            match row.try_get::<Option<Vec<String>>, _>(index) {
                Ok(Some(v)) => serde_json::json!(v),
                Ok(None) => serde_json::Value::Null,
                Err(_) => serde_json::Value::Null,
            }
        }
        _ => match row.try_get::<Option<String>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::String(format!("<unsupported type: {type_name}>")),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(sql: &str) -> bool {
        // Mirror what `execute_database_query` does to pick the path,
        // without needing a live Postgres pool. Returns `true` for
        // reads, `false` for writes.
        let trimmed = sql.trim().trim_end_matches(';');
        let upper = trimmed.to_uppercase();
        let stripped = strip_leading_sql_comments(&upper);
        let first_keyword = stripped.split_whitespace().next().unwrap_or("");
        matches!(
            first_keyword,
            "SELECT" | "WITH" | "TABLE" | "VALUES" | "EXPLAIN"
        )
    }

    #[test]
    fn plain_select_routes_to_read() {
        assert!(classify("SELECT * FROM conversations"));
        assert!(classify("WITH x AS (SELECT 1) SELECT * FROM x"));
        assert!(classify("EXPLAIN SELECT 1"));
    }

    #[test]
    fn plain_write_routes_to_write() {
        assert!(!classify("DELETE FROM scratch.foo"));
        assert!(!classify("INSERT INTO scratch.foo VALUES (1)"));
        assert!(!classify("UPDATE scratch.foo SET bar = 1"));
        assert!(!classify("CREATE TABLE scratch.foo (id INT)"));
    }

    #[test]
    fn leading_block_comment_does_not_promote_write_to_read() {
        // The original bypass: `/* */ DELETE` previously had
        // `first_keyword = "/*"` which doesn't match read keywords,
        // so it routed to the *write* path — but as an unwanted side
        // effect a leading comment in front of a SELECT also routed
        // to write (commits). After #40, comment-prefixed reads are
        // recognised as reads, and comment-prefixed writes still
        // route to write (so legitimate writes keep working).
        assert!(classify("/* comment */ SELECT * FROM conversations"));
        assert!(!classify("/* comment */ DELETE FROM public.foo"));
    }

    #[test]
    fn line_comment_is_stripped() {
        assert!(classify("-- hi\nSELECT 1"));
        assert!(classify("--  multiple    spaces \nSELECT 1"));
        assert!(!classify("-- hi\nDELETE FROM scratch.foo"));
    }

    #[test]
    fn nested_block_comments_are_handled() {
        // Postgres allows `/* outer /* inner */ still outer */`. A
        // naive `find("*/")` strip would terminate after the inner
        // close and mis-classify the outer text.
        assert!(classify("/* outer /* nested */ still outer */ SELECT 1"));
        assert!(classify(
            "/* /* /* deep */ */ */ WITH x AS (SELECT 1) SELECT * FROM x"
        ));
    }

    #[test]
    fn mixed_comment_kinds_strip_correctly() {
        assert!(classify("-- first\n/* block */\n-- another\nSELECT 1"));
        assert!(!classify("/* */ -- line\n /* */ DELETE FROM scratch.foo"));
    }

    #[test]
    fn unterminated_block_comment_routes_to_write() {
        // No `*/` — every char is consumed as comment, no keyword,
        // routes to the write path where Postgres will reject the
        // malformed statement at parse time.
        assert!(!classify("/* never closes SELECT 1"));
    }

    #[test]
    fn empty_or_whitespace_only_routes_to_write() {
        assert!(!classify(""));
        assert!(!classify("   "));
        assert!(!classify("\n\t\n"));
        assert!(!classify("-- only a comment"));
        assert!(!classify("/* only */"));
    }

    #[test]
    fn strip_does_not_modify_keyword_after_skipping() {
        // The strip should land *exactly* on the first non-comment
        // character so the upstream `to_uppercase()` + keyword match
        // still sees the canonical keyword.
        let stripped = strip_leading_sql_comments("/* x */SELECT 1");
        assert_eq!(stripped, "SELECT 1");
        let stripped = strip_leading_sql_comments("--c\n--d\nSELECT 1");
        assert_eq!(stripped, "SELECT 1");
    }
}
