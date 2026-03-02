use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;

/// Boxed async closure for executing read-only SQL queries against the database.
/// Takes a SQL string and a row limit; returns a JSON value with columns, rows, and row_count.
pub type DbQueryFn = Arc<
    dyn Fn(
            String,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, CoreError>> + Send>>
        + Send
        + Sync,
>;
