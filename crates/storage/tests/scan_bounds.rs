//! The server-side ceiling every full scan behind the `[Recall]` block carries
//! (#1167).
//!
//! A ceiling the caller keeps is not a ceiling the database keeps. A
//! `tokio::time::timeout` abandons the client future, and `sqlx` sends no
//! cancel when that future drops, so the backend goes on scanning for a caller
//! that has already given up. Recall runs before every turn, so those
//! abandoned scans accumulate at the rate turns arrive.
//!
//! Two properties, and the second is the one that cannot be seen by reading the
//! code:
//!
//! 1. The bound is stated to the database, **inside a transaction**. Outside
//!    one, `set_config(..., true)` is scoped to the statement that calls it and
//!    silently does nothing - the shape that looks fixed and changes nothing.
//! 2. The bound actually cancels a scan that outruns it, in each of the two
//!    degraded full-text reads.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test scan_bounds
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::time::Duration;

use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgSkillIndexStore, begin_bounded};
use sqlx::PgPool;

/// A synthetic tenant, never a real identity.
const USER: &str = "scan-bounds-user";

/// A bound no real scan of a seeded fixture can meet, so a scan that is
/// genuinely held to it is cancelled rather than merely quick.
///
/// One millisecond is the smallest bound PostgreSQL can be given:
/// `statement_timeout` is counted in whole milliseconds and a zero means no
/// timeout at all.
const UNMEETABLE: Duration = Duration::from_millis(1);

/// How many rows the fixture seeds.
///
/// Measured rather than guessed: over this corpus the full-text read takes
/// more than a millisecond every time, so the cancellation below is the bound
/// firing and not a flake waiting to happen.
const CORPUS: i32 = 2_000;

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("scanbounds1167").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set; scan_bounds pass-skipped");
    }
    fx
}

/// Seed a corpus every full-text query in this suite matches, in one statement.
async fn seed_knowledge(pool: &PgPool) {
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, tags, metadata)
         SELECT 'kb-' || i, $1, 'wobble corpus row ' || i || ' with filler words here',
                '{}'::text[], '{}'::jsonb
         FROM generate_series(1, $2) i",
    )
    .bind(USER)
    .bind(CORPUS)
    .execute(pool)
    .await
    .expect("seed the knowledge corpus");
}

/// The same corpus in the skill catalog: approved, locally authored, and so
/// offerable - the only rows that store's degraded read considers at all.
async fn seed_skills(pool: &PgPool) {
    sqlx::query(
        "INSERT INTO skill_index
             (name, owner_user_id, description, disk_path, content_hash, trust_tier,
              body, approved_at)
         SELECT 'skill-' || i, NULL, 'wobble corpus row ' || i || ' with filler words here',
                '/nowhere/skill-' || i, 'hash-' || i, 'local', '', NOW()
         FROM generate_series(1, $1) i",
    )
    .bind(CORPUS)
    .execute(pool)
    .await
    .expect("seed the skill catalog");
}

/// Whether an error is the database saying it stopped the statement itself.
fn is_a_statement_timeout(error: &desktop_assistant_core::CoreError) -> bool {
    error.to_string().contains("statement timeout")
}

/// Acceptance (#1167): the bound is stated to the database, and it is stated
/// inside the transaction that runs the scan.
///
/// Narrower than "the scan is bounded", and named for exactly what it checks:
/// this reads the setting back from the same transaction. Set outside one,
/// `set_config(..., true)` applies to the calling statement alone and the next
/// statement runs unbounded - which is the failure mode that looks fixed in a
/// diff and changes nothing at run time.
#[tokio::test]
async fn a_bounded_scan_states_its_ceiling_to_the_database_inside_the_transaction() {
    let Some(fx) = fixture().await else {
        return;
    };

    let mut scan = begin_bounded(&fx.pool, Duration::from_secs(4))
        .await
        .expect("a bounded transaction begins");
    let inside: String = sqlx::query_scalar("SHOW statement_timeout")
        .fetch_one(&mut *scan)
        .await
        .expect("the setting reads back");
    scan.commit().await.expect("nothing was written");

    assert_eq!(
        inside, "4s",
        "the scan's own transaction must carry the ceiling, or the database never applies it"
    );
    fx.cleanup().await;
}

/// Acceptance (#1167): the knowledge base's degraded full-text read is bounded
/// server-side - a scan that outruns the ceiling is cancelled by the database,
/// not merely abandoned by the caller.
#[tokio::test]
async fn knowledge_search_text_any_term_is_cancelled_by_the_database_past_its_ceiling() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_knowledge(&fx.pool).await;
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

    let answer = with_user_id(UserId::new(USER), async {
        store
            .search_text_any_term_within("wobble corpus", 10, UNMEETABLE)
            .await
    })
    .await;

    let error = answer.expect_err("a scan past its ceiling must be stopped, not answered");
    assert!(
        is_a_statement_timeout(&error),
        "the scan must be stopped by the database's own statement timeout; got {error}"
    );
    fx.cleanup().await;
}

/// Acceptance (#1167): the same for the skill catalog's degraded full-text
/// read. The gap was identical in both stores, so the coverage is too.
#[tokio::test]
async fn skill_search_text_any_term_is_cancelled_by_the_database_past_its_ceiling() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_skills(&fx.pool).await;
    let store = PgSkillIndexStore::new(fx.pool.clone());

    let answer = with_user_id(UserId::new(USER), async {
        store
            .search_text_any_term_within("wobble corpus", 10, UNMEETABLE)
            .await
    })
    .await;

    let error = answer.expect_err("a scan past its ceiling must be stopped, not answered");
    assert!(
        is_a_statement_timeout(&error),
        "the scan must be stopped by the database's own statement timeout; got {error}"
    );
    fx.cleanup().await;
}

/// The deployment's own ceiling still answers a scan it can meet, so the bound
/// costs nothing on the ordinary path.
///
/// Both stores, because a bound that refused every read would pass the two
/// tests above and break the feature.
#[tokio::test]
async fn a_scan_inside_its_ceiling_still_answers() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_knowledge(&fx.pool).await;
    seed_skills(&fx.pool).await;
    let entries = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let skills = PgSkillIndexStore::new(fx.pool.clone());

    let (found_entries, found_skills) = with_user_id(UserId::new(USER), async {
        (
            entries
                .search_text_any_term("wobble corpus", 5)
                .await
                .expect("the knowledge read answers inside its ceiling"),
            skills
                .search_text_any_term("wobble corpus", 5)
                .await
                .expect("the skill read answers inside its ceiling"),
        )
    })
    .await;

    assert_eq!(found_entries.len(), 5, "the knowledge read is still a read");
    assert_eq!(found_skills.len(), 5, "the skill read is still a read");
    fx.cleanup().await;
}
