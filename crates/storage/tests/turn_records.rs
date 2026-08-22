//! The turn-record store (issue #1252): what a turn said, kept for a bounded
//! window and readable only by the person whose turn it was.
//!
//! These tables hold the largest personal-data surface in the schema - the
//! whole assembled prompt, the whole reply, and every tool result - so the two
//! properties this suite exists for are tenancy and retention. A record nobody
//! else can read and that ages out is a debugging aid; without either it is a
//! surveillance archive.
//!
//! Run with `just test-db --test turn_records`. When `TEST_DATABASE_URL` is
//! unset every test pass-skips, so a green `just check` proves nothing here.

mod support;

use chrono::{Duration, Utc};
use desktop_assistant_core::domain::{Message, Role, ToolCall};
use desktop_assistant_core::ports::llm::TokenUsage;
use desktop_assistant_core::ports::turn_record::{
    RoundRecord, RoundToolResults, TurnRecord, TurnRecorder,
};
use desktop_assistant_storage::turn_records::{PgTurnRecordStore, sweep_expired_turn_records};
use desktop_assistant_storage::{UserId, with_user_id};

const ALICE: &str = "turnrec-alice";
const BOB: &str = "turnrec-bob";
const CONVERSATION: &str = "conv-turnrec-1";

/// The system prompt as assembled, which is the thing nothing else stores.
const SYSTEM_PROMPT: &str = "You are Adele. TURNREC-SYSTEM-BLOCK";

/// A block the person never typed, injected by recall.
const RECALL_BLOCK: &str = "[Recall] TURNREC-INJECTED-MEMORY";

const USER_PROMPT: &str = "TURNREC-PROMPT-what-did-you-see";
const REPLY: &str = "TURNREC-REPLY-here-is-what-I-concluded";
const TOOL_OUTPUT: &str = "TURNREC-RESULT-the-file-this-tool-read";

async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("turnrec1252").await
}

fn correlation(suffix: &str) -> String {
    format!("11111111-2222-4333-8444-00000000{suffix}")
}

fn turn(correlation_id: &str) -> TurnRecord {
    TurnRecord {
        correlation_id: correlation_id.to_string(),
        conversation_id: CONVERSATION.to_string(),
        connection_id: Some("conn-primary".to_string()),
        provider: Some("example-connector".to_string()),
        model: Some("example-model-v1".to_string()),
        tool_policy: "standard".to_string(),
    }
}

fn request() -> Vec<Message> {
    vec![
        Message::new(Role::System, SYSTEM_PROMPT),
        Message::new(Role::System, RECALL_BLOCK),
        Message::new(Role::User, USER_PROMPT),
    ]
}

fn round(correlation_id: &str, index: u32) -> RoundRecord {
    RoundRecord {
        correlation_id: correlation_id.to_string(),
        conversation_id: CONVERSATION.to_string(),
        round: index,
        request: request(),
        response_text: REPLY.to_string(),
        response_tool_calls: vec![ToolCall::new("call-1", "write_note", r#"{"a":1}"#)],
        usage: Some(TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(10),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        }),
        error: None,
    }
}

fn results(correlation_id: &str, index: u32) -> RoundToolResults {
    RoundToolResults {
        correlation_id: correlation_id.to_string(),
        conversation_id: CONVERSATION.to_string(),
        round: index,
        results: vec![Message::tool_result("call-1", TOOL_OUTPUT)],
    }
}

/// Write one whole turn - the turn row, one round and its results - as `user`.
async fn write_turn(store: &PgTurnRecordStore, user: &str, correlation_id: &str) {
    with_user_id(UserId::new(user), async {
        store
            .record_turn(turn(correlation_id))
            .await
            .expect("record the turn");
        store
            .record_round(round(correlation_id, 1))
            .await
            .expect("record the round");
        store
            .record_round_results(results(correlation_id, 1))
            .await
            .expect("record the round's tool results");
    })
    .await;
}

#[tokio::test]
async fn a_turn_and_its_rounds_round_trip() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; a_turn_and_its_rounds_round_trip");
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("01");
    write_turn(&store, ALICE, &id).await;

    let read = with_user_id(UserId::new(ALICE), async { store.read_turn(&id).await })
        .await
        .expect("read the turn back")
        .expect("the turn was written, so it reads back");

    assert_eq!(read.turn, turn(&id));
    assert_eq!(read.rounds.len(), 1);
    let stored = &read.rounds[0];
    assert_eq!(
        stored.request,
        request(),
        "the request reads back as it was sent, system prompt and injected \
         block included"
    );
    assert_eq!(stored.response_text, REPLY);
    assert_eq!(stored.response_tool_calls.len(), 1);
    assert_eq!(
        stored.tool_results,
        results(&id, 1).results,
        "and each tool result reads back as the turn stored it"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn recording_the_same_round_twice_leaves_one_record() {
    // Retries and redeliveries are ordinary. A second write of the same round
    // must leave the store where one write left it.
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; recording_the_same_round_twice_leaves_one_record"
        );
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("02");
    write_turn(&store, ALICE, &id).await;
    write_turn(&store, ALICE, &id).await;

    let read = with_user_id(UserId::new(ALICE), async { store.read_turn(&id).await })
        .await
        .expect("read the turn back")
        .expect("the turn was written");
    assert_eq!(read.rounds.len(), 1, "one round, not two");
    assert_eq!(
        read.rounds[0].tool_results.len(),
        1,
        "and one set of tool results"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn replaying_a_round_does_not_erase_the_results_already_recorded() {
    // `record_round` deliberately leaves `tool_results` out of its update
    // list, because the results arrive after it. Nothing else holds that:
    // writing a whole turn twice re-writes the results afterwards, so a
    // regression that named the column would stay green.
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; replaying_a_round_does_not_erase_the_results_already_recorded"
        );
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("07");
    write_turn(&store, ALICE, &id).await;

    with_user_id(UserId::new(ALICE), async {
        store
            .record_round(round(&id, 1))
            .await
            .expect("record the round a second time, with no results after it");
    })
    .await;

    let read = with_user_id(UserId::new(ALICE), async { store.read_turn(&id).await })
        .await
        .expect("read the turn back")
        .expect("the turn was written");
    assert_eq!(
        read.rounds[0].tool_results,
        results(&id, 1).results,
        "a replayed round keeps the results recorded after the first write"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_round_naming_another_conversation_is_refused_rather_than_stored() {
    // The correlation id is the CLIENT's to choose. Reusing one across two
    // conversations would otherwise file the second conversation's rounds
    // under a turn row naming the first, and the record would contradict
    // itself with nothing to say which half was right.
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; a_round_naming_another_conversation_is_refused_rather_than_stored"
        );
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("08");
    write_turn(&store, ALICE, &id).await;

    const ELSEWHERE: &str = "conv-turnrec-somewhere-else";

    // Round 1 is the case that actually happens, because every turn starts
    // there - and it is the one a foreign key alone cannot catch: the round
    // already exists, so the upsert becomes an UPDATE, and an update that
    // leaves the referencing columns alone re-checks no key.
    let mut colliding = round(&id, 1);
    colliding.conversation_id = ELSEWHERE.to_string();
    colliding.response_text = "REPLY-FROM-THE-OTHER-CONVERSATION".to_string();
    let refused = with_user_id(UserId::new(ALICE), async {
        store.record_round(colliding).await
    })
    .await;
    assert!(
        refused.is_err(),
        "a round overwriting one of another conversation must be refused"
    );

    // Round 2 does not exist yet, so this is the foreign key's half.
    let mut fresh = round(&id, 2);
    fresh.conversation_id = ELSEWHERE.to_string();
    let refused = with_user_id(UserId::new(ALICE), async {
        store.record_round(fresh).await
    })
    .await;
    assert!(
        refused.is_err(),
        "and a new round naming another conversation must be refused too"
    );

    let read = with_user_id(UserId::new(ALICE), async { store.read_turn(&id).await })
        .await
        .expect("read the turn back")
        .expect("the turn was written");
    assert_eq!(read.rounds.len(), 1, "nothing of either reaches the store");
    assert_eq!(
        read.rounds[0].response_text, REPLY,
        "and the round that was already there is untouched"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn one_users_turn_records_are_invisible_to_another() {
    let Some(fx) = fixture().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; one_users_turn_records_are_invisible_to_another"
        );
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("03");
    write_turn(&store, ALICE, &id).await;

    let seen_by_bob = with_user_id(UserId::new(BOB), async { store.read_turn(&id).await })
        .await
        .expect("the read succeeds");
    assert!(
        seen_by_bob.is_none(),
        "a turn record holds one person's prompts and replies whole; another \
         tenant naming its correlation id must read nothing"
    );

    // The rounds carry the content, and the turn read returns early when there
    // is no turn row - so the check above never reaches the rounds statement.
    // Give Bob a turn under the same correlation id and he must see one round,
    // his own, rather than two.
    with_user_id(UserId::new(BOB), async {
        store.record_turn(turn(&id)).await.expect("record the turn");
        let mut his = round(&id, 1);
        his.response_text = "BOB-REPLY".to_string();
        store.record_round(his).await.expect("record the round");
    })
    .await;
    let his = with_user_id(UserId::new(BOB), async { store.read_turn(&id).await })
        .await
        .expect("the read succeeds")
        .expect("Bob has a turn of his own now");
    assert_eq!(his.rounds.len(), 1, "one round, his");
    assert_eq!(
        his.rounds[0].response_text, "BOB-REPLY",
        "the rounds statement is scoped by user too, so Alice's round is not \
         returned beside his"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn retention_drops_records_past_the_window() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; retention_drops_records_past_the_window");
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let old = correlation("04");
    let fresh = correlation("05");
    write_turn(&store, ALICE, &old).await;
    write_turn(&store, ALICE, &fresh).await;
    age_turn(&fx.pool, &old, 9).await;

    let removed = sweep_expired_turn_records(&fx.pool, 7)
        .await
        .expect("the sweep runs");
    assert_eq!(removed, 1, "one turn was past the window");

    let gone = with_user_id(UserId::new(ALICE), async { store.read_turn(&old).await })
        .await
        .expect("the read succeeds");
    assert!(gone.is_none(), "a record past the window is gone whole");

    let kept = with_user_id(UserId::new(ALICE), async { store.read_turn(&fresh).await })
        .await
        .expect("the read succeeds")
        .expect("a record inside the window is untouched");
    assert_eq!(
        kept.rounds.len(),
        1,
        "and it keeps its rounds, so the sweep is not a partial delete"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_swept_turn_takes_its_rounds_with_it() {
    // The rounds carry the content. A sweep that dropped the turn row and left
    // them would report a retention window it does not keep.
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; a_swept_turn_takes_its_rounds_with_it");
        return;
    };
    let store = PgTurnRecordStore::new(fx.pool.clone());
    let id = correlation("06");
    write_turn(&store, ALICE, &id).await;
    age_turn(&fx.pool, &id, 30).await;

    sweep_expired_turn_records(&fx.pool, 7)
        .await
        .expect("the sweep runs");

    let orphans: i64 =
        sqlx::query_scalar("SELECT count(*) FROM turn_round_records WHERE correlation_id = $1")
            .bind(&id)
            .fetch_one(&fx.pool)
            .await
            .expect("count the rounds left behind");
    assert_eq!(orphans, 0, "the rounds went with their turn");

    fx.cleanup().await;
}

/// Move one turn's start time `days` into the past, so the sweep sees it as
/// aged. Written straight to the table because nothing in the write path can
/// backdate a record, which is the point.
async fn age_turn(pool: &sqlx::PgPool, correlation_id: &str, days: i64) {
    let when = Utc::now() - Duration::days(days);
    sqlx::query("UPDATE turn_records SET started_at = $2 WHERE correlation_id = $1")
        .bind(correlation_id)
        .bind(when)
        .execute(pool)
        .await
        .expect("backdate the turn");
}
