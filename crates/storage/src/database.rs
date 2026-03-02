use desktop_assistant_core::CoreError;
use sqlx::postgres::PgRow;
use sqlx::{Column, PgPool, Row, TypeInfo};

/// Execute a read-only SQL query and return results as JSON.
///
/// The query is wrapped in a READ ONLY transaction (PostgreSQL enforces no writes).
/// A LIMIT clause is appended if one is not already present.
/// The transaction is always rolled back (read-only, nothing to commit).
///
/// Returns `{ "columns": [...], "rows": [[...], ...], "row_count": N }`.
pub async fn execute_readonly_query(
    pool: &PgPool,
    sql: &str,
    limit: usize,
) -> Result<serde_json::Value, CoreError> {
    let sql_trimmed = sql.trim().trim_end_matches(';');

    // Reject obviously non-read queries at the application level as an extra guard.
    let upper = sql_trimmed.to_uppercase();
    let first_keyword = upper.split_whitespace().next().unwrap_or("");
    if !matches!(
        first_keyword,
        "SELECT" | "WITH" | "TABLE" | "VALUES" | "EXPLAIN"
    ) {
        return Err(CoreError::ToolExecution(format!(
            "only SELECT/WITH/TABLE/VALUES/EXPLAIN queries are allowed, got: {first_keyword}"
        )));
    }

    // Append LIMIT if not already present.
    let has_limit = upper.contains(" LIMIT ");
    let query = if has_limit {
        sql_trimmed.to_string()
    } else {
        format!("{sql_trimmed} LIMIT {limit}")
    };

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // Set the transaction to READ ONLY so PostgreSQL rejects any writes.
    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    let rows: Vec<PgRow> = sqlx::query(&query)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::ToolExecution(format!("query error: {e}")))?;

    // Always rollback — read-only transaction has nothing to commit.
    tx.rollback()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

    // Extract column names from the first row (or return empty if no rows).
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

    for row in &rows {
        let mut json_row = Vec::with_capacity(columns.len());
        for (i, col) in row.columns().iter().enumerate() {
            let type_name = col.type_info().name();
            let value = pg_value_to_json(row, i, type_name);
            json_row.push(value);
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
    // Check for NULL first — works for any type.
    // We try the most common type (String) and if it's null, return null.
    // This is a workaround because sqlx doesn't expose a generic is_null check.
    match type_name {
        "TEXT" | "VARCHAR" | "CHAR" | "BPCHAR" | "NAME" => match row.try_get::<Option<String>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        },
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
                Err(_) => {
                    // Fallback: try as NaiveDateTime
                    match row.try_get::<Option<chrono::NaiveDateTime>, _>(index) {
                        Ok(Some(v)) => serde_json::Value::String(v.to_string()),
                        _ => serde_json::Value::Null,
                    }
                }
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
        // Fallback: try to get as String
        _ => match row.try_get::<Option<String>, _>(index) {
            Ok(Some(v)) => serde_json::Value::String(v),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::String(format!("<unsupported type: {type_name}>")),
        },
    }
}
