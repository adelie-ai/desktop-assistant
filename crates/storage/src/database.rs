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
    let first_keyword = upper.split_whitespace().next().unwrap_or("");

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

/// Read path — READ ONLY transaction, auto-LIMIT, always rolled back.
async fn execute_read(
    pool: &PgPool,
    sql: &str,
    upper: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let query = if upper.contains(" LIMIT ") {
        sql.to_string()
    } else {
        format!("{sql} LIMIT {limit}")
    };

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let rows: Vec<PgRow> = sqlx::query(&query)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

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
