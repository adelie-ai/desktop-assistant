//! One place that bounds a scan on the server (#1167).
//!
//! A ceiling the caller keeps is not a ceiling the database keeps.
//! `tokio::time::timeout` abandons the client future, and `sqlx` sends no
//! cancel when that future drops, so the backend goes on working for a caller
//! that has already stopped waiting. Every read behind the `[Recall]` block
//! runs before a turn's first round, so an unbounded one accumulates abandoned
//! scans at the rate turns arrive - each still holding its share of the
//! connection pool and the server's CPU.
//!
//! **The bound has to be stated inside a transaction.** `set_config(name,
//! value, true)` is `SET LOCAL`: outside a transaction block it applies to the
//! statement that calls it and to nothing after, so a bound set on a bare
//! connection reads as a fix in the diff and changes nothing at run time. That
//! is the whole reason this is one function rather than four copies of two
//! statements.

use std::time::Duration;

use desktop_assistant_core::CoreError;
use sqlx::{PgPool, Postgres, Transaction};

/// Begin a transaction the database stops working on after `ceiling`.
///
/// The caller runs its scan on the returned transaction and commits it. A read
/// writes nothing, so the commit is only what releases the snapshot; the bound
/// goes with the transaction either way.
///
/// A scan that outruns the ceiling comes back as `CoreError::Storage` carrying
/// PostgreSQL's own "canceling statement due to statement timeout". What each
/// caller does with that is the caller's own decision - the recall arms already
/// differ on it, and this function deliberately does not settle it for them.
///
/// **A zero `ceiling` means no timeout at all in PostgreSQL**, so passing one
/// disables the bound rather than tightening it. Nothing here rejects that: the
/// value comes from a constant beside its own call site, and each of those is
/// pinned by a test that says the constant is positive.
pub async fn begin_bounded(
    pool: &PgPool,
    ceiling: Duration,
) -> Result<Transaction<'static, Postgres>, CoreError> {
    let mut scan = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    sqlx::query("SELECT set_config('statement_timeout', $1, true)")
        .bind(ceiling.as_millis().to_string())
        .execute(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(scan)
}
