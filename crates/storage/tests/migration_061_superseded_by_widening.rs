//! Migration 061 converges a store that already applied migration 056's old,
//! tighter `knowledge_base_superseded_by_chk` onto the corrected forward
//! implication a fresh 056 now installs directly (#1345).
//!
//! A fresh database never exercises 061's `DROP CONSTRAINT` / `ADD
//! CONSTRAINT` pair for real, because 056 already installs the corrected
//! definition and 061 finds nothing to converge. This suite puts a database
//! back into the shape 061 exists for: the old, tighter constraint actually
//! on disk, with 061 unrecorded so it replays.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test migration_061_superseded_by_widening
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;

/// Put a freshly migrated database back into the shape a store that ran the
/// pre-#1345 migration 056 is in: the old biconditional constraint on disk,
/// and migration 061 unrecorded so the next [`run_migrations`] call replays
/// it.
async fn install_old_biconditional_constraint(pool: &PgPool) {
    sqlx::query(
        "ALTER TABLE knowledge_base DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk",
    )
    .execute(pool)
    .await
    .expect("drop the corrected constraint");

    sqlx::query(
        "ALTER TABLE knowledge_base \
             ADD CONSTRAINT knowledge_base_superseded_by_chk \
             CHECK ((disposition IN ('superseded', 'redundant')) = (superseded_by IS NOT NULL))",
    )
    .execute(pool)
    .await
    .expect("install the old, tighter constraint");

    sqlx::query(
        "DELETE FROM schema_migrations WHERE name = '061_superseded_by_constraint_widening.sql'",
    )
    .execute(pool)
    .await
    .expect("unrecord migration 061 so it replays");
}

#[tokio::test]
async fn a_store_holding_the_old_constraint_converges_on_the_new_one() {
    let Some(fx) = support::DbFixture::try_new("mig061_converges").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    install_old_biconditional_constraint(&fx.pool).await;

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition) \
         VALUES ('kb-successor', 'u1', 'the corrected fact', 'active')",
    )
    .execute(&fx.pool)
    .await
    .expect("seed a successor row");

    // Under the old constraint this insert is refused: 'refuted' is not
    // 'superseded' or 'redundant', so the biconditional's right-to-left half
    // forbids naming a successor at all.
    let refused_before = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-refuted-before', 'u1', 'x', 'refuted', 'kb-successor')",
    )
    .execute(&fx.pool)
    .await;
    assert!(
        refused_before.is_err(),
        "the setup must actually install the old, tighter constraint"
    );

    run_migrations(&fx.pool)
        .await
        .expect("migration 061 converges the store onto the corrected constraint");

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-refuted-after', 'u1', 'x', 'refuted', 'kb-successor')",
    )
    .execute(&fx.pool)
    .await
    .expect("after convergence, a refuted row naming its successor must be permitted");

    // The half that is still load-bearing must survive convergence too.
    let still_refuses_orphan = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-orphan', 'u1', 'x', 'superseded', NULL)",
    )
    .execute(&fx.pool)
    .await;
    assert!(
        still_refuses_orphan.is_err(),
        "a superseded row naming no successor must still be refused after convergence"
    );

    let recorded: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM schema_migrations \
         WHERE name = '061_superseded_by_constraint_widening.sql'",
    )
    .fetch_one(&fx.pool)
    .await
    .expect("read the ledger");
    assert_eq!(recorded, 1, "migration 061 must be recorded as applied");

    fx.cleanup().await;
}

/// The two migration files' own text, embedded the same way `pool.rs`'s
/// `migration!` macro embeds them (a relative `include_str!` from this file's
/// directory, one level up into `migrations/`) -- the only difference is this
/// reads the text for comparison rather than to run it.
const MIGRATION_056: &str = include_str!("../migrations/056_kb_disposition.sql");
const MIGRATION_061: &str = include_str!("../migrations/061_superseded_by_constraint_widening.sql");

/// Pull the `CHECK (...)` expression that follows
/// `ADD CONSTRAINT knowledge_base_superseded_by_chk` out of a migration
/// file's raw SQL text, with whitespace normalized to single spaces so
/// formatting differences between the two files (indentation, line breaks)
/// do not register as a difference in the expression itself.
///
/// The anchor is the literal `ADD CONSTRAINT knowledge_base_superseded_by_chk`
/// text, not just the constraint name, because the name alone also appears in
/// each file's catalog guard (`conname = 'knowledge_base_superseded_by_chk'`)
/// and in the pre-flight diagnostic's exception message; anchoring on the
/// full `ADD CONSTRAINT` phrase reaches the one place each file actually
/// declares the expression.
fn superseded_by_check_expression(migration_sql: &str) -> String {
    const ANCHOR: &str = "ADD CONSTRAINT knowledge_base_superseded_by_chk";
    const CHECK_KEYWORD: &str = "CHECK (";

    let after_anchor = migration_sql
        .split_once(ANCHOR)
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("{ANCHOR:?} not found in migration text"));
    let after_check = after_anchor
        .split_once(CHECK_KEYWORD)
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("{CHECK_KEYWORD:?} not found after {ANCHOR:?}"));

    // Balance parens from just inside the CHECK's own opening paren (already
    // consumed by the split above) so a nested paren in the expression, such
    // as the `IN (...)` list, does not end the scan early.
    let mut depth = 1i32;
    let mut end = None;
    for (i, c) in after_check.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.expect("unbalanced parens scanning the CHECK expression");

    after_check[..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// #1345: 061 unconditionally re-adds `knowledge_base_superseded_by_chk`
/// after 056, so on a fresh store 061's own `CHECK` text is what actually
/// ends up on disk -- 056's copy runs first and is immediately overwritten.
/// Nothing before this test held the two declarations to agreement: editing
/// 056's expression without also editing 061's is a silent no-op on a fresh
/// database, caught by no other test in this suite.
#[test]
fn migrations_056_and_061_declare_the_same_superseded_by_check() {
    let expr_056 = superseded_by_check_expression(MIGRATION_056);
    let expr_061 = superseded_by_check_expression(MIGRATION_061);
    assert_eq!(
        expr_056, expr_061,
        "056 and 061 must declare the identical knowledge_base_superseded_by_chk \
         CHECK expression -- 061 always re-adds the constraint after 056, so a \
         mismatch means 056's copy is dead text and only 061's takes effect"
    );
}

/// Pull the `WHERE ...;` predicate out of a migration file's pre-flight
/// diagnostic -- the `SELECT count(*) INTO offending_count FROM
/// knowledge_base WHERE ...` block that runs just before the `ADD
/// CONSTRAINT` the file installs. Whitespace is normalized the same way
/// [`superseded_by_check_expression`] normalizes the CHECK expression, so an
/// indentation difference between files never registers as a difference in
/// the predicate itself.
///
/// The anchor is `INTO offending_count`, the diagnostic's own output
/// variable, because it appears exactly once per file and sits immediately
/// before the `FROM knowledge_base WHERE ...` this function reads.
fn preflight_predicate(migration_sql: &str) -> String {
    const ANCHOR: &str = "INTO offending_count";
    const WHERE_KEYWORD: &str = "WHERE";

    let after_anchor = migration_sql
        .split_once(ANCHOR)
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("{ANCHOR:?} not found in migration text"));
    let after_where = after_anchor
        .split_once(WHERE_KEYWORD)
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("{WHERE_KEYWORD:?} not found after {ANCHOR:?}"));
    let end = after_where
        .find(';')
        .unwrap_or_else(|| panic!("no terminating ';' found after {WHERE_KEYWORD:?}"));

    after_where[..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// #1345: each migration's pre-flight diagnostic must refuse exactly the
/// rows its own `CHECK` would refuse. `migrations_056_and_061_declare_the_same_superseded_by_check`
/// above only pins the two files' CHECK expressions to each other -- nothing
/// before this test held either migration's pre-flight predicate to its own
/// CHECK, so a future change to the disposition vocabulary could edit both
/// CHECKs and leave a pre-flight behind: the drift test still passes (the
/// CHECKs still agree with each other), and a store holding a row of the new
/// shape aborts at `ADD CONSTRAINT` with Postgres's own undiagnosed error --
/// exactly what the pre-flight exists to prevent.
///
/// Proved behaviourally rather than by comparing text: for every disposition,
/// this seeds a row carrying that disposition with no successor and checks
/// that the migration's own pre-flight predicate flags the row if and only if
/// the migration's own CHECK expression would refuse it. Run once per
/// migration file, because each declares its own copy of both and could
/// drift from either independently.
#[tokio::test]
async fn the_preflight_predicate_agrees_with_the_check_for_every_disposition() {
    use desktop_assistant_core::domain::knowledge::Disposition;

    for (migration_name, migration_sql) in [("056", MIGRATION_056), ("061", MIGRATION_061)] {
        let Some(fx) =
            support::DbFixture::try_new(&format!("mig{migration_name}_preflight_agrees")).await
        else {
            eprintln!("skip: TEST_DATABASE_URL not set");
            return;
        };

        let check_expr = superseded_by_check_expression(migration_sql);
        let predicate = preflight_predicate(migration_sql);

        for disposition in Disposition::ALL {
            // Install this migration's own CHECK text, so the oracle below
            // reflects that file's declaration specifically, rather than
            // whichever migration's copy happens to survive last on a fresh
            // database.
            sqlx::query(
                "ALTER TABLE knowledge_base \
                     DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk",
            )
            .execute(&fx.pool)
            .await
            .expect("drop the constraint to install this migration's own text");
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "ALTER TABLE knowledge_base \
                     ADD CONSTRAINT knowledge_base_superseded_by_chk CHECK ({check_expr})"
            )))
            .execute(&fx.pool)
            .await
            .expect("install this migration's own CHECK expression");

            let id = format!("chk-{migration_name}-{}", disposition.as_str());
            let insert = sqlx::query(
                "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
                 VALUES ($1, 'u1', 'x', $2, NULL)",
            )
            .bind(&id)
            .bind(disposition.as_str())
            .execute(&fx.pool)
            .await;
            let check_refuses = insert.is_err();
            if insert.is_ok() {
                sqlx::query("DELETE FROM knowledge_base WHERE id = $1")
                    .bind(&id)
                    .execute(&fx.pool)
                    .await
                    .expect("clean up the row the CHECK permitted");
            }

            // Drop the constraint so a row shaped exactly the way the CHECK
            // would refuse can still be written, for the pre-flight
            // predicate to examine on its own terms.
            sqlx::query(
                "ALTER TABLE knowledge_base \
                     DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk",
            )
            .execute(&fx.pool)
            .await
            .expect("drop the constraint to seed a possibly-violating row");

            let pre_id = format!("pre-{migration_name}-{}", disposition.as_str());
            sqlx::query(
                "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
                 VALUES ($1, 'u1', 'x', $2, NULL)",
            )
            .bind(&pre_id)
            .bind(disposition.as_str())
            .execute(&fx.pool)
            .await
            .expect("seed the row for the pre-flight predicate to examine");

            let flagged: (i64,) = sqlx::query_as(sqlx::AssertSqlSafe(format!(
                "SELECT count(*) FROM knowledge_base WHERE id = $1 AND ({predicate})"
            )))
            .bind(&pre_id)
            .fetch_one(&fx.pool)
            .await
            .expect("run the extracted pre-flight predicate");
            let preflight_flags = flagged.0 > 0;

            sqlx::query("DELETE FROM knowledge_base WHERE id = $1")
                .bind(&pre_id)
                .execute(&fx.pool)
                .await
                .expect("clean up the seeded row");

            assert_eq!(
                check_refuses, preflight_flags,
                "migration {migration_name}, disposition {disposition:?} with no \
                 successor: the pre-flight predicate and the CHECK constraint must \
                 agree on whether this row is refused"
            );
        }

        fx.cleanup().await;
    }
}
