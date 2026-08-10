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
//! 2. The bound actually cancels a scan that outruns it, in each of the reads.
//!
//! **Every test here drives the public method**, with the store's own ceiling
//! overridden by `with_scan_ceiling`. Reaching past the public method to a
//! bounded variant would leave the delegation untested: the public method could
//! stop applying the bound and every test here would still pass, which is
//! precisely the stated-but-unheld property this change exists to remove.
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

use desktop_assistant_core::domain::{Conversation, Message, Role};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{
    PgConversationStore, PgKnowledgeBaseStore, PgScratchpadStore, PgSkillIndexStore, begin_bounded,
};
use sqlx::PgPool;

/// A synthetic tenant, never a real identity.
const USER: &str = "scan-bounds-user";
const CONVERSATION: &str = "scan-bounds-conv";

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

/// The same corpus on one conversation's pad, behind the conversation row its
/// foreign key needs.
async fn seed_notes(pool: &PgPool) {
    let mut conversation = Conversation::new(CONVERSATION, "scan bounds");
    conversation.created_at = "2026-08-10 00:00:00".to_string();
    conversation.updated_at = "2026-08-10 00:00:00".to_string();
    conversation
        .messages
        .push(Message::new(Role::User, "hello"));
    with_user_id(UserId::new(USER), async {
        PgConversationStore::new(pool.clone())
            .create(conversation)
            .await
            .expect("create the conversation");
    })
    .await;
    sqlx::query(
        "INSERT INTO scratchpads
             (id, user_id, conversation_id, owner_todo, note_key, content, note_type)
         SELECT 'sp-' || i, $1, $2, '', 'note-' || i,
                'wobble corpus row ' || i || ' with filler words here', 'note'
         FROM generate_series(1, $3) i",
    )
    .bind(USER)
    .bind(CONVERSATION)
    .bind(CORPUS)
    .execute(pool)
    .await
    .expect("seed the pad");
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
/// server-side - a scan that outruns the store's ceiling is cancelled by the
/// database, not merely abandoned by the caller.
///
/// Driven through the public method, so the delegation is what is under test.
#[tokio::test]
async fn knowledge_search_text_any_term_is_cancelled_by_the_database_past_its_ceiling() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_knowledge(&fx.pool).await;
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default())
        .with_scan_ceiling(UNMEETABLE);

    let answer = with_user_id(UserId::new(USER), async {
        store.search_text_any_term("wobble corpus", 10).await
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
    let store = PgSkillIndexStore::new(fx.pool.clone()).with_scan_ceiling(UNMEETABLE);

    let answer = with_user_id(UserId::new(USER), async {
        store.search_text_any_term("wobble corpus", 10).await
    })
    .await;

    let error = answer.expect_err("a scan past its ceiling must be stopped, not answered");
    assert!(
        is_a_statement_timeout(&error),
        "the scan must be stopped by the database's own statement timeout; got {error}"
    );
    fx.cleanup().await;
}

/// The pad's degraded read carries the same bound as its measured counterpart.
///
/// One pad holds far fewer rows than a knowledge base, so this is cheaper
/// insurance than the two above - but one of two sibling reads being bounded is
/// the shape a later reader misreads, and a read that states no ceiling still
/// leaves the backend working for a caller that has given up.
#[tokio::test]
async fn scratchpad_search_text_any_term_is_cancelled_by_the_database_past_its_ceiling() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_notes(&fx.pool).await;
    let pad = PgScratchpadStore::new(fx.pool.clone()).with_scan_ceiling(UNMEETABLE);

    let answer = with_user_id(UserId::new(USER), async {
        pad.search_text_any_term(CONVERSATION, "wobble corpus", 10)
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

/// The search tool's own scan carries the bound too, on the same public path.
///
/// It is the full scan this change introduced: stating the store's spread means
/// reading every comparable row in scope, and the model can run the tool
/// several times inside one turn.
///
/// Only this scan, not the recall scan beside it. Over a corpus this size the
/// recall scan's own vector pass finishes inside a millisecond, so a test of it
/// here would prove the fixture was small rather than that the bound reaches
/// the statement. Its bound is unchanged by this work and
/// `the_recall_scan_states_a_statement_timeout` pins the constant.
#[tokio::test]
async fn the_hybrid_search_scan_is_cancelled_by_the_database_past_its_ceiling() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_knowledge(&fx.pool).await;
    sqlx::query(
        "UPDATE knowledge_base
         SET embedding = ARRAY['[1,0,0]'::vector], embedding_model = 'scan-bounds-model'",
    )
    .execute(&fx.pool)
    .await
    .expect("stamp the corpus with vectors");
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default())
        .with_scan_ceiling(UNMEETABLE);

    let answer = with_user_id(UserId::new(USER), async {
        store
            .search(
                "wobble corpus",
                vec![1.0, 0.0, 0.0],
                "scan-bounds-model",
                None,
                None,
                10,
            )
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
/// All three stores, because a bound that refused every read would pass every
/// test above and break the feature.
#[tokio::test]
async fn a_scan_inside_its_ceiling_still_answers() {
    let Some(fx) = fixture().await else {
        return;
    };
    seed_knowledge(&fx.pool).await;
    seed_skills(&fx.pool).await;
    seed_notes(&fx.pool).await;
    let entries = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let skills = PgSkillIndexStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    let (found_entries, found_skills, found_notes) = with_user_id(UserId::new(USER), async {
        (
            entries
                .search_text_any_term("wobble corpus", 5)
                .await
                .expect("the knowledge read answers inside its ceiling"),
            skills
                .search_text_any_term("wobble corpus", 5)
                .await
                .expect("the skill read answers inside its ceiling"),
            pad.search_text_any_term(CONVERSATION, "wobble corpus", 5)
                .await
                .expect("the pad read answers inside its ceiling"),
        )
    })
    .await;

    assert_eq!(found_entries.len(), 5, "the knowledge read is still a read");
    assert_eq!(found_skills.len(), 5, "the skill read is still a read");
    assert_eq!(found_notes.len(), 5, "the pad read is still a read");
    fx.cleanup().await;
}
