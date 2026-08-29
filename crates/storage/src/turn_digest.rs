//! Postgres adapter for the episodic turn index (#1349).
//!
//! One digest per turn, scoped to the person rather than to the conversation,
//! so a past turn is reachable by relevance from anywhere in the account. The
//! digest itself is built by `desktop_assistant_core::turn_capture`, whose
//! header states what it may and may not hold; this module is only its home.
//!
//! Follows the patterns the other personal-data adapters set:
//! - `user_id`-scoped queries throughout, with `current_user_id()` read from
//!   the task-local - nothing here takes a `UserId` parameter.
//! - Cross-user reads answer empty rather than erroring. A cross-user WRITE
//!   does not reach a guard at all: migration 062's foreign key references
//!   `conversations (user_id, id)`, so a digest naming a conversation somebody
//!   else owns is refused by the database. Two further layers sit behind that
//!   and are deliberately kept - the conflict target carries `user_id`, so it
//!   cannot cross tenants even if the reference were relaxed, and the upsert's
//!   `WHERE turn_digests.user_id = EXCLUDED.user_id` guard fails closed the way
//!   `PgKnowledgeBaseStore::write` does (#809). Neither is load-bearing while
//!   the composite reference stands.
//! - Disposition is honoured on the read rather than at each call site.
//!   `obsolete` is left out unless a caller asks for it, exactly as
//!   `knowledge_search` does, and every other value comes back carrying
//!   `TurnDigest::marked_text`'s marker.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::Disposition;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::turn_digest::{NewTurnDigest, TurnDigest, TurnDigestStore};
use pgvector::Vector;
use sqlx::PgPool;

/// Every column [`DigestRow`] reads.
///
/// The list exists so a test can hold every query to it. A column added to the
/// row and missed by one query is invisible to any test that builds its
/// fixtures in memory: the row type and the reading code agree with each
/// other, and only the SQL disagrees, so nothing fails until a statement
/// reaches Postgres.
///
/// Test-only: the integration suites link the lib without `cfg(test)`, so a
/// const only the guard reads is dead code in that build.
#[cfg(test)]
const DIGEST_COLUMNS: &[&str] = &[
    "id",
    "conversation_id",
    "opening_message_id",
    "content",
    "after_outside_read",
    "disposition",
    "disposition_reason",
    "superseded_by",
    "created_at",
    "updated_at",
];

/// Batch upsert, keyed on the turn rather than on the row id.
///
/// Three things stop a write reaching another tenant's row, and the outermost
/// is the one that fires: migration 062's composite foreign key refuses a
/// digest that names a conversation somebody else owns, so the collision this
/// upsert would have to resolve is not constructible. The conflict target
/// carrying `user_id`, and the `WHERE turn_digests.user_id = EXCLUDED.user_id`
/// guard below, are kept behind it rather than removed with the hazard.
///
/// The vector is CLEARED on every update and written back by the statement
/// after it. Clearing is what keeps a vector honest: an upsert replaces the
/// content, so a vector left in place would describe text that is no longer
/// there while its stamp still named the current model - putting it beyond
/// both the stale sweep and the backfill, which act only on a missing or
/// superseded stamp. A cleared row is simply re-embedded.
const WRITE_UPSERT_SQL: &str = "\
    INSERT INTO turn_digests \
        (id, user_id, conversation_id, opening_message_id, content, after_outside_read) \
    SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::bool[]) \
    ON CONFLICT (user_id, conversation_id, opening_message_id) DO UPDATE \
       SET content = EXCLUDED.content, \
           after_outside_read = EXCLUDED.after_outside_read, \
           embedding = NULL, \
           embedding_model = NULL, \
           updated_at = now() \
     WHERE turn_digests.user_id = EXCLUDED.user_id \
    RETURNING id, conversation_id, opening_message_id, content, after_outside_read, \
              disposition, disposition_reason, superseded_by, created_at, updated_at";

/// The person's own episodes, newest first, across every conversation they
/// own.
///
/// `$3` is `include_dispositioned`. `obsolete` is the one disposition an
/// ordinary read leaves out - it was true and no longer applies - and every
/// other value is admitted and carries its marker, so a refuted episode stays
/// findable when the query is about its subject.
const RECENT_SQL: &str = "\
    SELECT id, conversation_id, opening_message_id, content, after_outside_read, \
           disposition, disposition_reason, superseded_by, created_at, updated_at \
      FROM turn_digests \
     WHERE user_id = $1 \
       AND deleted_at IS NULL \
       AND (disposition <> 'obsolete' OR $3) \
     ORDER BY created_at DESC, id DESC \
     LIMIT $2";

/// One digest by its row id, for this person only.
const GET_SQL: &str = "\
    SELECT id, conversation_id, opening_message_id, content, after_outside_read, \
           disposition, disposition_reason, superseded_by, created_at, updated_at \
      FROM turn_digests \
     WHERE user_id = $1 AND id = $2 AND deleted_at IS NULL";

/// Record what a digest is judged to be.
const SET_DISPOSITION_SQL: &str = "\
    UPDATE turn_digests \
       SET disposition = $3, \
           disposition_reason = $4, \
           superseded_by = $5, \
           updated_at = now() \
     WHERE user_id = $1 AND id = $2 AND deleted_at IS NULL";

/// Every query in this adapter that answers [`DigestRow`], for the test that
/// holds them all to [`DIGEST_COLUMNS`].
#[cfg(test)]
const DIGEST_ROW_QUERIES: &[(&str, &str)] = &[
    ("WRITE_UPSERT_SQL", WRITE_UPSERT_SQL),
    ("RECENT_SQL", RECENT_SQL),
    ("GET_SQL", GET_SQL),
];

/// Stable code for the one refusal this adapter raises: a disposition that
/// resolves through a successor was given without one.
pub const DISPOSITION_NEEDS_A_SUCCESSOR_CODE: &str = "turn_digest_disposition_needs_successor";

/// Postgres adapter for the episodic turn index.
pub struct PgTurnDigestStore {
    pool: PgPool,
}

impl PgTurnDigestStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(sqlx::FromRow)]
struct DigestRow {
    id: String,
    conversation_id: String,
    opening_message_id: String,
    content: String,
    after_outside_read: bool,
    disposition: String,
    disposition_reason: Option<String>,
    superseded_by: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl DigestRow {
    fn into_digest(self) -> TurnDigest {
        TurnDigest {
            id: self.id,
            conversation_id: self.conversation_id,
            opening_message_id: self.opening_message_id,
            content: self.content,
            after_outside_read: self.after_outside_read,
            // The database CHECK constraint means a stored spelling is always
            // one of the six. An unrecognized one degrades to the default
            // rather than panicking, the same way `knowledge` reads it.
            disposition: Disposition::parse(&self.disposition).unwrap_or_default(),
            disposition_reason: self.disposition_reason,
            superseded_by: self.superseded_by,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }
}

#[async_trait::async_trait]
impl TurnDigestStore for PgTurnDigestStore {
    async fn write(
        &self,
        conversation_id: &str,
        digests: &[NewTurnDigest],
    ) -> Result<Vec<TurnDigest>, CoreError> {
        if digests.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        let ids: Vec<String> = (0..digests.len())
            .map(|_| uuid::Uuid::now_v7().to_string())
            .collect();
        let user_ids: Vec<String> = vec![user_id.as_str().to_string(); digests.len()];
        let conv_ids: Vec<String> = vec![conversation_id.to_string(); digests.len()];
        let openings: Vec<String> = digests
            .iter()
            .map(|d| d.opening_message_id.clone())
            .collect();
        let contents: Vec<String> = digests.iter().map(|d| d.content.clone()).collect();
        let stamps: Vec<bool> = digests.iter().map(|d| d.after_outside_read).collect();

        // Both statements run in one transaction, so no reader ever sees a
        // digest carrying the previous content's vector.
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let rows: Vec<DigestRow> = sqlx::query_as(WRITE_UPSERT_SQL)
            .bind(&ids)
            .bind(&user_ids)
            .bind(&conv_ids)
            .bind(&openings)
            .bind(&contents)
            .bind(&stamps)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // One statement per embedded digest. A single statement cannot carry
        // them all: every digest's `vector[]` has its own chunk count, and a
        // Postgres array of arrays must be rectangular. A real write carries
        // one digest.
        //
        // A digest finds its stored row by `opening_message_id`, which is its
        // identity in the table. A row the cross-tenant guard refused is
        // absent from `rows`, so it is skipped rather than written to.
        for digest in digests {
            let Some(embedding) = digest.embedding.as_ref() else {
                continue;
            };
            if embedding.chunks.is_empty() {
                continue;
            }
            let Some(row) = rows
                .iter()
                .find(|r| r.opening_message_id == digest.opening_message_id)
            else {
                continue;
            };
            let vectors: Vec<Vector> = embedding.chunks.iter().cloned().map(Vector::from).collect();
            sqlx::query(
                "UPDATE turn_digests SET embedding = $1::vector[], embedding_model = $2 \
                 WHERE id = $3 AND user_id = $4",
            )
            .bind(&vectors)
            .bind(&embedding.model)
            .bind(&row.id)
            .bind(user_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(DigestRow::into_digest).collect())
    }

    async fn recent(
        &self,
        limit: usize,
        include_dispositioned: bool,
    ) -> Result<Vec<TurnDigest>, CoreError> {
        let user_id = current_user_id();
        let rows: Vec<DigestRow> = sqlx::query_as(RECENT_SQL)
            .bind(user_id.as_str())
            .bind(limit as i64)
            .bind(include_dispositioned)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(DigestRow::into_digest).collect())
    }

    async fn get(&self, id: &str) -> Result<Option<TurnDigest>, CoreError> {
        let user_id = current_user_id();
        let row: Option<DigestRow> = sqlx::query_as(GET_SQL)
            .bind(user_id.as_str())
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(DigestRow::into_digest))
    }

    async fn set_disposition(
        &self,
        id: &str,
        disposition: Disposition,
        reason: Option<&str>,
        superseded_by: Option<&str>,
    ) -> Result<bool, CoreError> {
        // The database CHECK is the backstop; refusing here names the missing
        // field instead of surfacing a constraint violation that names neither
        // the row nor what it needed.
        if matches!(
            disposition,
            Disposition::Superseded | Disposition::Redundant
        ) && superseded_by.is_none()
        {
            return Err(CoreError::InvalidInput {
                code: DISPOSITION_NEEDS_A_SUCCESSOR_CODE,
                description: format!(
                    "disposition `{}` resolves through its successor, so it must name one",
                    disposition.as_str()
                ),
                message: "That disposition means the episode was replaced, so it has to \
                          say what replaced it."
                    .to_string(),
            });
        }
        let user_id = current_user_id();
        let result = sqlx::query(SET_DISPOSITION_SQL)
            .bind(user_id.as_str())
            .bind(id)
            .bind(disposition.as_str())
            .bind(reason)
            .bind(superseded_by)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{DIGEST_COLUMNS, DIGEST_ROW_QUERIES};

    /// The columns `sql` actually projects, as whole names.
    ///
    /// Reads the `RETURNING` list where there is one and the `SELECT` list
    /// otherwise, then splits on commas and strips any `alias.` prefix and
    /// ` AS name` suffix. Whole names, not substrings: `contains("id")` is
    /// satisfied by `conversation_id`, and `contains("disposition")` by
    /// `disposition_reason`, so a substring check passes on a projection that
    /// is missing the very column it claims to be checking.
    ///
    /// The LAST `SELECT`, not the first, because a CTE's inner select comes
    /// first in the text and the row is built from the outer one. No query
    /// here is a CTE yet; the recall read that #1350 adds will be, since every
    /// sibling adapter writes a vector or hybrid query that way. Reading the
    /// inner list would then pass a query whose outer list drops a column,
    /// which is exactly the runtime failure this check exists to catch, and it
    /// would pass silently.
    ///
    /// Other shapes fail loudly through the `extra` assertion rather than
    /// silently: lowercase ` as `, `coalesce(a, b) AS x`,
    /// `extract(epoch FROM x)` and `SELECT DISTINCT` all produce a name that
    /// is not a column and are named as such. Only the CTE case was silent.
    fn projected_columns(sql: &str) -> BTreeSet<String> {
        let projection = match sql.rfind("RETURNING ") {
            Some(at) => &sql[at + "RETURNING ".len()..],
            None => {
                let from = sql.rfind("SELECT ").expect("a SELECT or a RETURNING") + "SELECT ".len();
                let to = sql[from..].find(" FROM ").expect("a FROM") + from;
                &sql[from..to]
            }
        };
        projection
            .split(',')
            .map(|column| {
                let column = column.trim();
                let column = column.rsplit(" AS ").next().unwrap_or(column);
                let column = column.rsplit('.').next().unwrap_or(column);
                column.trim().to_string()
            })
            .filter(|column| !column.is_empty())
            .collect()
    }

    /// Every query answering a `DigestRow` projects every column the row
    /// reads, and no others.
    ///
    /// A column added to the row and missed by one statement compiles, and
    /// fails only when that statement reaches Postgres - which is how
    /// `after_outside_read` reached `main` missing from four queries with the
    /// ordinary gate green (#1277).
    #[test]
    fn every_digest_query_selects_every_column_the_row_reads() {
        let expected: BTreeSet<String> = DIGEST_COLUMNS.iter().map(|c| (*c).to_string()).collect();
        for (name, sql) in DIGEST_ROW_QUERIES {
            let projected = projected_columns(sql);
            let missing: Vec<&String> = expected.difference(&projected).collect();
            assert!(
                missing.is_empty(),
                "{name} does not project {missing:?}, which DigestRow reads"
            );
            let extra: Vec<&String> = projected.difference(&expected).collect();
            assert!(
                extra.is_empty(),
                "{name} projects {extra:?}, which DigestRow does not read"
            );
        }
    }

    /// The reader above is what makes the check whole-name rather than
    /// substring, so it is held to the two collisions that actually exist in
    /// this table: `id` inside `conversation_id`, and `disposition` inside
    /// `disposition_reason`.
    #[test]
    fn the_projection_reader_returns_whole_column_names() {
        let projected = projected_columns(
            "SELECT conversation_id, disposition_reason, td.content, now() AS updated_at \
             FROM turn_digests WHERE user_id = $1",
        );
        assert!(!projected.contains("id"), "{projected:?}");
        assert!(!projected.contains("disposition"), "{projected:?}");
        assert!(projected.contains("conversation_id"), "{projected:?}");
        assert!(projected.contains("disposition_reason"), "{projected:?}");
        assert!(projected.contains("content"), "an alias prefix is stripped");
        assert!(projected.contains("updated_at"), "an AS alias is the name");
    }

    /// A CTE's OUTER select list is what builds the row, so that is the list
    /// read.
    ///
    /// The inner list here carries a column the outer one drops. Reading the
    /// inner one would report the query as complete while sqlx fails on it at
    /// runtime - the silent pass this whole check exists to prevent, and the
    /// shape #1350's recall read will have.
    #[test]
    fn the_projection_reader_reads_a_ctes_outer_select_list() {
        let projected = projected_columns(
            "WITH d AS (SELECT id, content, disposition FROM turn_digests WHERE user_id = $1) \
             SELECT id, content FROM d",
        );
        assert_eq!(
            projected,
            ["content".to_string(), "id".to_string()]
                .into_iter()
                .collect::<BTreeSet<String>>(),
            "the outer list is the row, so `disposition` must not be counted as projected"
        );
    }
}
