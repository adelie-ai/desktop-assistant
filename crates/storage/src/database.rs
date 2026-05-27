use desktop_assistant_core::CoreError;
use sqlx::postgres::PgRow;
use sqlx::{Column, PgPool, Row, TypeInfo};

/// Output of `prepare_select_for_user` — the rewritten SELECT plus the
/// caller's user_id ready to bind as `$1` when grafting was needed.
///
/// **Stub for #141.** This is the failing-tests commit; the
/// implementation lands in the follow-up commit. Returning an
/// `unimplemented` error from the stub lets the failing tests compile
/// (so the spec is reviewable on its own) without accidentally passing
/// against pre-implementation code.
#[allow(dead_code)]
pub(crate) struct PreparedSelect {
    pub sql: String,
    pub bound_user_id: Option<String>,
}

/// Parse `sql` as a single SELECT, validate it, and (if it references
/// any personal-data tables) graft a `user_id = $N` predicate scoped
/// to `user_id`. See `database_query_user_id_scoping.rs` for the
/// behavioural contract.
///
/// **Stub for #141.**
#[allow(dead_code)]
pub(crate) fn prepare_select_for_user(
    _sql: &str,
    _user_id: &str,
) -> Result<PreparedSelect, CoreError> {
    Err(CoreError::ToolExecution(
        "prepare_select_for_user: not yet implemented (#141)".to_string(),
    ))
}

/// Parse `sql` as a single non-SELECT statement and verify it does not
/// reference any personal-data table (qualified or otherwise).
///
/// **Stub for #141.**
#[allow(dead_code)]
pub(crate) fn validate_write_statement(_sql: &str) -> Result<(), CoreError> {
    Err(CoreError::ToolExecution(
        "validate_write_statement: not yet implemented (#141)".to_string(),
    ))
}

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

    // -------------------------------------------------------------------
    // #141: parser-level validation/rewriting. These tests don't need a
    // live DB — they exercise the AST-based rules that gate the
    // statement before it ever reaches the pool.
    // -------------------------------------------------------------------

    /// Helper that mirrors what `execute_database_query` does internally
    /// for the read path: parse, validate, rewrite for user_id. We test
    /// just the rewriter so test failures point straight at the rule
    /// that broke, not at the DB round-trip.
    fn rewrite_select(sql: &str, user_id: &str) -> Result<String, CoreError> {
        super::prepare_select_for_user(sql, user_id).map(|p| p.sql)
    }

    /// Helper for write-path validation — no DB required.
    fn validate_write(sql: &str) -> Result<(), CoreError> {
        super::validate_write_statement(sql).map(|_| ())
    }

    #[test]
    fn rewrite_grafts_user_id_into_bare_select() {
        let rewritten =
            rewrite_select("SELECT id FROM conversations", "alice").expect("rewrite");
        // The rewriter must inject a parameterised user_id filter
        // qualified by the `conversations` alias so it survives joins
        // against tables that also happen to have a `user_id` column.
        let lower = rewritten.to_ascii_lowercase();
        assert!(
            lower.contains("user_id ="),
            "rewritten SQL must include user_id filter, got: {rewritten}"
        );
        assert!(
            lower.contains("$1") || lower.contains("'alice'"),
            "rewritten SQL must bind/quote the caller user_id, got: {rewritten}"
        );
    }

    #[test]
    fn rewrite_ands_into_existing_where() {
        let rewritten =
            rewrite_select("SELECT id FROM conversations WHERE id = 'x'", "alice")
                .expect("rewrite");
        let lower = rewritten.to_ascii_lowercase();
        // Both predicates must survive — the original (id = 'x') and
        // the grafted (user_id = …).
        assert!(lower.contains("id = 'x'"), "original predicate dropped: {rewritten}");
        assert!(lower.contains("user_id ="), "user_id predicate missing: {rewritten}");
        // And there must be an explicit AND joining them, not an OR
        // or a comma — OR would weaken the guard, comma would mean
        // "SELECT a, b FROM …" which makes no sense in WHERE.
        assert!(lower.contains(" and "), "predicates must be AND'd, got: {rewritten}");
    }

    #[test]
    fn rewrite_skips_tables_without_user_id_column() {
        // System catalogs and `tool_definitions` (the system-wide tool
        // registry from #105's allowlist) have no user_id column, so
        // the rewriter must NOT graft anything onto them.
        let rewritten = rewrite_select(
            "SELECT table_name FROM information_schema.tables",
            "alice",
        )
        .expect("rewrite");
        assert!(
            !rewritten.to_ascii_lowercase().contains("user_id"),
            "must not graft user_id onto information_schema, got: {rewritten}"
        );

        let rewritten = rewrite_select("SELECT name FROM tool_definitions", "alice")
            .expect("rewrite");
        assert!(
            !rewritten.to_ascii_lowercase().contains("user_id"),
            "must not graft user_id onto tool_definitions, got: {rewritten}"
        );
    }

    #[test]
    fn rewrite_rejects_compound_select() {
        // Two statements is always wrong — we don't want statement-
        // stuffing slipping past a too-permissive first-keyword check.
        let err = rewrite_select("SELECT 1; SELECT 2", "alice").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.to_ascii_lowercase().contains("single") || msg.contains("compound"),
            "rejection message must explain the compound-statement rule, got: {msg}"
        );
    }

    #[test]
    fn rewrite_rejects_non_select_statement() {
        // The read path is reserved for SELECT/WITH only.
        let err = rewrite_select("DELETE FROM conversations", "alice").unwrap_err();
        let msg = format!("{err:?}").to_ascii_lowercase();
        // The rejection must name DELETE specifically OR explain the
        // "SELECT-only" rule — a generic "not implemented" doesn't
        // count.
        assert!(
            msg.contains("delete") || msg.contains("only select") || msg.contains("not allowed"),
            "rejection message must name the offending statement type or the SELECT-only \
             rule, got: {msg}"
        );
    }

    #[test]
    fn validate_write_rejects_personal_data_targets() {
        // The write path runs in the scratch namespace; touching a
        // personal-data table from there — qualified or otherwise —
        // is a hostile move and must be refused.
        for sql in [
            "DROP TABLE public.conversations",
            "DROP TABLE conversations",
            "UPDATE public.conversations SET title = 'x'",
            "DELETE FROM messages WHERE 1=1",
            "INSERT INTO knowledge_base (id, content) VALUES ('x', 'y')",
            "TRUNCATE public.messages",
            "ALTER TABLE conversations DROP COLUMN title",
        ] {
            let err = validate_write(sql).unwrap_err_or_else(|_| {
                panic!("validate_write must reject {sql:?}");
            });
            let msg = format!("{err:?}").to_ascii_lowercase();
            assert!(
                msg.contains("personal-data") || msg.contains("not allowed"),
                "rejection message must explain the personal-data rule for {sql:?}, got: {msg}"
            );
        }
    }

    #[test]
    fn validate_write_accepts_scratch_namespace_ddl() {
        // Unqualified DDL — what the LLM uses for staging tables. Must
        // pass through to the existing scratch search_path machinery.
        for sql in [
            "CREATE TABLE staging_foo (id INT)",
            "DROP TABLE staging_foo",
            "CREATE SCHEMA my_scratch",
            "CREATE TABLE scratch.intermediate (x INT)",
        ] {
            validate_write(sql).unwrap_or_else(|e| {
                panic!("validate_write must accept {sql:?}, got: {e:?}");
            });
        }
    }

    #[test]
    fn validate_write_rejects_compound_statement() {
        // `CREATE TABLE foo (); DROP TABLE public.conversations` must
        // not slip in via the write path either.
        let err = validate_write("CREATE TABLE foo (); DROP TABLE public.conversations")
            .unwrap_err_or_else(|_| panic!("compound write must be rejected"));
        let msg = format!("{err:?}").to_ascii_lowercase();
        assert!(
            msg.contains("single") || msg.contains("compound"),
            "rejection must explain the compound-statement rule, got: {msg}"
        );
    }

    /// Small `Result::unwrap_err`-style helper that produces a clearer
    /// failure message when the result is unexpectedly `Ok`. The
    /// closure runs only on the `Ok` path.
    trait UnwrapErrOrElse<T, E> {
        fn unwrap_err_or_else<F: FnOnce(&T)>(self, f: F) -> E;
    }
    impl<T, E> UnwrapErrOrElse<T, E> for Result<T, E> {
        fn unwrap_err_or_else<F: FnOnce(&T)>(self, f: F) -> E {
            match self {
                Ok(v) => {
                    f(&v);
                    panic!("expected Err, got Ok");
                }
                Err(e) => e,
            }
        }
    }
}
