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
