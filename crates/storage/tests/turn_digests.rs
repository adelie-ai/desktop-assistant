//! Integration tests for the episodic turn index (#1349).
//!
//! Exercises `PgTurnDigestStore` and `backfill_turn_digests` end-to-end
//! against a real Postgres with the migrations applied. Each acceptance
//! criterion from the issue is one named test below, and each pairs its permit
//! with its refusal: a scope test that only proved a read succeeds would pass
//! against a store with no scoping at all.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`). Run
//! them against an ephemeral Postgres with `just test-db`.

mod support;

use desktop_assistant_core::domain::{
    Conversation, ConversationId, Disposition, Message, RESERVED_SUBAGENT_TAG, Role,
};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::turn_digest::{NewTurnDigest, TurnDigestStore};
use desktop_assistant_core::turn_capture::FAILED_TURN_NOTICE_PREFIXES;
use desktop_assistant_storage::{
    PgConversationStore, PgTurnDigestStore, UserId, backfill_turn_digests, with_user_id,
};
use sqlx::PgPool;

use support::DbFixture;

/// A conversation row, written directly so a test can choose its tags and its
/// owner without going through a request scope.
async fn seed_conversation(pool: &PgPool, user_id: &str, id: &str, tags: &[&str]) {
    let tags: Vec<String> = tags.iter().map(|t| (*t).to_string()).collect();
    sqlx::query("INSERT INTO conversations (id, title, user_id, tags) VALUES ($1, 'test', $2, $3)")
        .bind(id)
        .bind(user_id)
        .bind(&tags)
        .execute(pool)
        .await
        .expect("seed conversation");
}

/// One message row.
#[allow(clippy::too_many_arguments)]
async fn seed_message(
    pool: &PgPool,
    user_id: &str,
    conversation_id: &str,
    id: &str,
    ordinal: i32,
    role: &str,
    content: &str,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<&str>,
) {
    sqlx::query(
        "INSERT INTO messages \
             (id, conversation_id, user_id, ordinal, role, content, tool_calls, tool_call_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(user_id)
    .bind(ordinal)
    .bind(role)
    .bind(content)
    .bind(tool_calls)
    .bind(tool_call_id)
    .execute(pool)
    .await
    .expect("seed message");
}

/// A digest for `opening_message_id`, unembedded.
fn digest(opening_message_id: &str, content: &str) -> NewTurnDigest {
    NewTurnDigest::new(opening_message_id, content)
}

/// How many digests exist for `conversation_id`, read straight off the table
/// so the assertion does not depend on the read path it is checking.
async fn digest_count(pool: &PgPool, conversation_id: &str) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM turn_digests WHERE conversation_id = $1")
        .bind(conversation_id)
        .fetch_one(pool)
        .await
        .expect("count digests")
}

/// Every digest row's `(opening_message_id, content)`, ordered.
async fn contents(pool: &PgPool, conversation_id: &str) -> Vec<(String, String)> {
    sqlx::query_as(
        "SELECT opening_message_id, content FROM turn_digests \
         WHERE conversation_id = $1 ORDER BY opening_message_id",
    )
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .expect("read contents")
}

/// Every digest row's `(opening_message_id, after_outside_read)`, ordered.
async fn stamps(pool: &PgPool, conversation_id: &str) -> Vec<(String, bool)> {
    sqlx::query_as(
        "SELECT opening_message_id, after_outside_read FROM turn_digests \
         WHERE conversation_id = $1 ORDER BY opening_message_id",
    )
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .expect("read stamps")
}

/// Acceptance 1 (#1349): the index is scoped to the person, so a turn from one
/// conversation is reachable while the person is working in another. This is
/// the whole point of the move - a conversation-scoped digest can never answer
/// "when did I last deal with this".
#[tokio::test]
async fn a_turn_digest_is_readable_from_a_different_conversation_by_the_same_user() {
    let Some(fx) = DbFixture::try_new("td_crossconv").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    with_user_id(UserId::new("alice"), async {
        seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;
        seed_conversation(&fx.pool, "alice", "conv-b", &[]).await;

        store
            .write("conv-a", &[digest("m-a", "Asked: where did I leave the kustomization")])
            .await
            .expect("write in conversation a");
        store
            .write("conv-b", &[digest("m-b", "Asked: what is the registry port")])
            .await
            .expect("write in conversation b");

        let index = store.recent(50, false).await.expect("read the index");
        let openings: Vec<&str> = index
            .iter()
            .map(|d| d.opening_message_id.as_str())
            .collect();
        assert!(
            openings.contains(&"m-a") && openings.contains(&"m-b"),
            "the index must answer with turns from every conversation this person owns, got {openings:?}"
        );

        // Each row still says which conversation it came from, so a reader can
        // follow it back into the transcript it belongs to.
        let from_a = index
            .iter()
            .find(|d| d.opening_message_id == "m-a")
            .expect("the digest written in conversation a");
        assert_eq!(from_a.conversation_id, "conv-a");
        assert!(
            from_a.content.contains("kustomization"),
            "{}",
            from_a.content
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance 2 (#1349): the index is one person's, and never another's. The
/// permit is paired with the refusal, and the write path is checked as well as
/// the reads.
///
/// What the write does has changed twice, so it is stated exactly. Bob's write
/// names a conversation alice owns, and migration 062's composite foreign key
/// refuses it at the database - it is an error, not a decline. Two earlier
/// shapes are worth naming so nobody reads this test as covering them: with a
/// key of (conversation_id, opening_message_id) bob's row would have collided
/// with alice's and been silently dropped by the `EXCLUDED.user_id` guard, and
/// with `user_id` in the key but a single-column reference bob would have got
/// his own row carrying alice's `conversation_id`. Neither is reachable now,
/// and `a_write_naming_another_users_conversation_is_refused_by_the_database`
/// holds the refusal on its own.
#[tokio::test]
async fn a_turn_digest_is_never_readable_by_a_different_user() {
    let Some(fx) = DbFixture::try_new("td_tenant").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;

    let alices = with_user_id(UserId::new("alice"), async {
        store
            .write("conv-a", &[digest("m-a", "Asked: alice's own words")])
            .await
            .expect("alice writes")
    })
    .await;
    assert_eq!(alices.len(), 1);
    let alices_id = alices[0].id.clone();

    with_user_id(UserId::new("bob"), async {
        // The refusal, three ways.
        let index = store.recent(50, false).await.expect("bob reads the index");
        assert!(
            index.is_empty(),
            "bob's index must not hold alice's turns, got {} row(s)",
            index.len()
        );
        let by_id = store.get(&alices_id).await.expect("bob fetches by id");
        assert!(by_id.is_none(), "bob must not fetch alice's digest by id");
        let marked = store
            .set_disposition(&alices_id, Disposition::Trivial, Some("not mine"), None)
            .await
            .expect("bob marks");
        assert!(
            !marked,
            "bob must not be able to disposition alice's digest"
        );

        // And the write path: bob cannot store a digest against a conversation
        // he does not own. The database refuses it, so this is an error rather
        // than an empty answer.
        let refused = store
            .write("conv-a", &[digest("m-a", "bob's overwrite")])
            .await;
        assert!(
            refused.is_err(),
            "a digest naming another person's conversation must be refused: {refused:?}"
        );
    })
    .await;

    with_user_id(UserId::new("alice"), async {
        // The permit: alice still reads exactly her own row, unchanged.
        let index = store
            .recent(50, false)
            .await
            .expect("alice reads the index");
        assert_eq!(index.len(), 1, "{index:?}");
        assert_eq!(index[0].content, "Asked: alice's own words");
        assert!(
            store
                .get(&alices_id)
                .await
                .expect("alice fetches")
                .is_some(),
            "alice must still reach her own digest by id"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A write naming another person's conversation is refused by the DATABASE,
/// not merely unreachable through the handler (#1349).
///
/// The foreign key is composite - `(user_id, conversation_id)` references
/// `conversations (user_id, id)` - so the pair itself has to exist. A
/// single-column reference asks only whether the conversation exists, which
/// makes a cross-tenant digest storable and leaves the refusal to whatever the
/// application remembers to check.
///
/// Paired with the permit: the owner's own write still lands, and still lands
/// after the refused one, so the refusal cannot be the store simply failing.
#[tokio::test]
async fn a_write_naming_another_users_conversation_is_refused_by_the_database() {
    let Some(fx) = DbFixture::try_new("td_fk").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;

    let refused = with_user_id(UserId::new("bob"), async {
        store
            .write("conv-a", &[digest("m-a", "bob's squatted row")])
            .await
    })
    .await;
    assert!(
        refused.is_err(),
        "the database must refuse a digest naming another person's conversation, \
         got {refused:?}"
    );
    assert_eq!(
        digest_count(&fx.pool, "conv-a").await,
        0,
        "and must store nothing"
    );

    // The permit: the owner's own write lands.
    let written = with_user_id(UserId::new("alice"), async {
        store
            .write("conv-a", &[digest("m-a", "alice's own words")])
            .await
            .expect("alice writes")
    })
    .await;
    assert_eq!(written.len(), 1, "{written:?}");
    assert_eq!(written[0].content, "alice's own words");

    fx.cleanup().await;
}

/// The digest key carries the tenant.
///
/// This is a schema assertion rather than a behavioural one, and deliberately
/// so: the composite foreign key above makes a cross-tenant collision
/// unconstructible through the store, so nothing observable is left to drive.
/// The key still carries `user_id` as the layer behind that reference, and
/// this is what stops the layer being dropped as dead weight by someone who
/// cannot see what it is for.
#[tokio::test]
async fn the_digest_key_carries_the_tenant() {
    let Some(fx) = DbFixture::try_new("td_key").await else {
        return;
    };

    let columns: Vec<String> = sqlx::query_scalar(
        "SELECT a.attname \
           FROM pg_constraint c \
           JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE \
           JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum \
          WHERE c.conrelid = 'turn_digests'::regclass AND c.contype = 'u' \
          ORDER BY k.ord",
    )
    .fetch_all(&fx.pool)
    .await
    .expect("read the unique constraint's columns");

    assert_eq!(
        columns,
        vec![
            "user_id".to_string(),
            "conversation_id".to_string(),
            "opening_message_id".to_string()
        ],
        "the digest's identity must carry the tenant"
    );

    fx.cleanup().await;
}

/// Acceptance 3 (#1349): once the store is user-scoped a digest's lifecycle is
/// no longer the conversation's, so the cascade has to be proven rather than
/// assumed - and proven on the path the application actually uses, which is
/// `ConversationStore::delete` and not a hand-written DELETE.
#[tokio::test]
async fn deleting_a_conversation_deletes_its_turn_digests() {
    let Some(fx) = DbFixture::try_new("td_cascade").await else {
        return;
    };
    let conversations = PgConversationStore::new(fx.pool.clone());
    let store = PgTurnDigestStore::new(fx.pool.clone());

    with_user_id(UserId::new("alice"), async {
        for id in ["conv-a", "conv-b"] {
            let mut conv = Conversation::new(id, "test");
            conv.created_at = "2026-01-01 00:00:00".to_string();
            conv.updated_at = "2026-01-01 00:00:00".to_string();
            conv.messages.push(Message::new(Role::User, "hello"));
            conversations.create(conv).await.expect("create");
            store
                .write(id, &[digest(&format!("m-{id}"), "Asked: something")])
                .await
                .expect("write a digest");
        }
        assert_eq!(digest_count(&fx.pool, "conv-a").await, 1);
        assert_eq!(digest_count(&fx.pool, "conv-b").await, 1);

        conversations
            .delete(&ConversationId::from("conv-a"))
            .await
            .expect("delete through the application's own path");

        assert_eq!(
            digest_count(&fx.pool, "conv-a").await,
            0,
            "deleting a conversation must delete its digests, or deletion is a \
             promise the product does not keep"
        );
        // The refusal: the cascade reaches this conversation's digests and no
        // others.
        assert_eq!(
            digest_count(&fx.pool, "conv-b").await,
            1,
            "another conversation's digests must survive"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance 6 (#1349): the backfill re-derives `after_outside_read` from the
/// tool traffic the transcript stored, rather than defaulting it. Both
/// directions are checked in one conversation, because a test that only proved
/// the stamped case would pass against a backfill that stamped everything.
#[tokio::test]
async fn the_backfill_rederives_the_provenance_stamp_from_stored_tool_traffic() {
    let Some(fx) = DbFixture::try_new("td_backfill_prov").await else {
        return;
    };
    seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;

    // Turn one reads a local file: trusted, so no stamp.
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-01",
        0,
        "user",
        "check the deploy notes",
        None,
        None,
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-02",
        1,
        "assistant",
        "",
        Some(serde_json::json!([{"id": "c1", "name": "read_file", "arguments": "{}"}])),
        None,
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-03",
        2,
        "tool",
        "the deploy notes",
        None,
        Some("c1"),
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-04",
        3,
        "assistant",
        "it is fine",
        None,
        None,
    )
    .await;

    // Turn two fetches a page: outside content, so the stamp.
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-05",
        4,
        "user",
        "what does that page say",
        None,
        None,
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-06",
        5,
        "assistant",
        "",
        Some(serde_json::json!([{"id": "c2", "name": "web_fetch", "arguments": "{}"}])),
        None,
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-07",
        6,
        "tool",
        "<html>the page</html>",
        None,
        Some("c2"),
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-08",
        7,
        "assistant",
        "it says something",
        None,
        None,
    )
    .await;

    let outcome = backfill_turn_digests(&fx.pool, false)
        .await
        .expect("backfill");
    assert_eq!(outcome.digests_written, 2, "{outcome:?}");
    assert_eq!(
        outcome.provenance_underivable, 0,
        "every tool result in this transcript names a call the turn holds"
    );

    let stamped = stamps(&fx.pool, "conv-a").await;
    assert_eq!(
        stamped,
        vec![("m-01".to_string(), false), ("m-05".to_string(), true)],
        "the turn that read a local file carries no stamp and the turn that \
         fetched a page does"
    );

    fx.cleanup().await;
}

/// A turn that failed has no answer half worth keeping, and a backfill is not
/// told which exit a stored turn took. It recognises the harness's own failure
/// text instead (#1351), so a historical outage is not re-filed as a reply the
/// index can offer.
///
/// Paired with the permit, because a backfill that dropped every answer half
/// would pass a test that only checked the failed turn.
#[tokio::test]
async fn the_backfill_does_not_file_a_failed_turns_error_text_as_an_answer() {
    let Some(fx) = DbFixture::try_new("td_backfill_failed").await else {
        return;
    };
    seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;

    // A turn the assistant answered.
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-01",
        0,
        "user",
        "how do I hold the deploy key",
        None,
        None,
    )
    .await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-02",
        1,
        "assistant",
        "use the sealed secret for that",
        None,
        None,
    )
    .await;

    // A turn the provider failed, whose closing text is the failure the user
    // read. The transcript keeps it; the index must not offer it as an answer.
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-03",
        2,
        "user",
        "always use the sealed secret",
        None,
        None,
    )
    .await;
    let failure = format!(
        "{}upstream-endpoint-unreachable",
        FAILED_TURN_NOTICE_PREFIXES[6]
    );
    seed_message(
        &fx.pool,
        "alice",
        "conv-a",
        "m-04",
        3,
        "assistant",
        &failure,
        None,
        None,
    )
    .await;

    backfill_turn_digests(&fx.pool, false)
        .await
        .expect("backfill");

    let bodies = contents(&fx.pool, "conv-a").await;
    let answered = &bodies
        .iter()
        .find(|(opening, _)| opening == "m-01")
        .expect("the answered turn's digest")
        .1;
    let failed = &bodies
        .iter()
        .find(|(opening, _)| opening == "m-03")
        .expect("the failed turn's digest")
        .1;

    assert!(
        answered.contains("Answered: use the sealed secret for that"),
        "an answered turn keeps what it answered: {answered}"
    );
    assert!(
        !failed.contains("upstream-endpoint-unreachable"),
        "a provider failure is not a recallable answer: {failed}"
    );
    assert!(
        failed.contains("always use the sealed secret"),
        "and the question is the one thing capture exists to keep: {failed}"
    );

    fx.cleanup().await;
}

/// Acceptance 7 (#1349): the pass writes one row per turn and re-running it
/// changes nothing (AGENTS.md 8.4). Without this a redelivery or a second
/// daemon would leave a second copy of somebody's conversation.
#[tokio::test]
async fn the_backfill_is_idempotent_and_writes_one_row_per_turn() {
    let Some(fx) = DbFixture::try_new("td_backfill_idem").await else {
        return;
    };
    seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;
    for (ordinal, (id, role, content)) in [
        ("m-01", "user", "first"),
        ("m-02", "assistant", "one"),
        ("m-03", "user", "second"),
        ("m-04", "assistant", "two"),
        ("m-05", "user", "third"),
    ]
    .into_iter()
    .enumerate()
    {
        seed_message(
            &fx.pool,
            "alice",
            "conv-a",
            id,
            ordinal as i32,
            role,
            content,
            None,
            None,
        )
        .await;
    }

    let first = backfill_turn_digests(&fx.pool, false)
        .await
        .expect("first pass");
    assert_eq!(first.digests_written, 3, "three prompts, three turns");
    assert_eq!(digest_count(&fx.pool, "conv-a").await, 3);

    let second = backfill_turn_digests(&fx.pool, false)
        .await
        .expect("second pass");
    assert_eq!(
        second.digests_written, 0,
        "a second pass must find nothing left to write"
    );
    assert_eq!(
        second.conversations_scanned, 0,
        "and must not even re-read the conversation: a pass that re-claims what \
         it already wrote never converges"
    );
    assert_eq!(
        digest_count(&fx.pool, "conv-a").await,
        3,
        "a re-run must not leave a second copy of any turn"
    );

    fx.cleanup().await;
}

/// The backfill leaves a subagent's conversation out of the shared store, on
/// the same reserved tag the live capture reads. Paired with the permit,
/// because a test that only proved the exclusion would pass against a backfill
/// that wrote nothing at all.
///
/// The live write path's own version of this is
/// `a_subagent_conversation_writes_no_digest_to_the_shared_store` in
/// `desktop_assistant_core::turn_capture`.
#[tokio::test]
async fn the_backfill_writes_no_digest_for_a_subagent_conversation() {
    let Some(fx) = DbFixture::try_new("td_backfill_sub").await else {
        return;
    };
    seed_conversation(&fx.pool, "alice", "conv-sub", &[RESERVED_SUBAGENT_TAG]).await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-sub",
        "m-sub",
        0,
        "user",
        "read that page and report back",
        None,
        None,
    )
    .await;
    seed_conversation(&fx.pool, "alice", "conv-own", &[]).await;
    seed_message(
        &fx.pool,
        "alice",
        "conv-own",
        "m-own",
        0,
        "user",
        "what did I decide",
        None,
        None,
    )
    .await;

    let outcome = backfill_turn_digests(&fx.pool, false)
        .await
        .expect("backfill");

    assert_eq!(
        digest_count(&fx.pool, "conv-sub").await,
        0,
        "a subagent conversation's turns are mechanism, not the person's record"
    );
    assert_eq!(
        digest_count(&fx.pool, "conv-own").await,
        1,
        "the person's own conversation is still backfilled: {outcome:?}"
    );

    fx.cleanup().await;
}

/// Acceptance 8 (#1349): a digest carries a disposition and the store acts on
/// it. `obsolete` is left out of an ordinary read and admitted when a caller
/// asks; every other value stays findable and comes back carrying its marker,
/// which is the asymmetry the vocabulary exists for.
#[tokio::test]
async fn a_digest_carries_a_disposition_and_honours_it() {
    let Some(fx) = DbFixture::try_new("td_disposition").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    with_user_id(UserId::new("alice"), async {
        seed_conversation(&fx.pool, "alice", "conv-a", &[]).await;
        let written = store
            .write(
                "conv-a",
                &[digest("m-a", "Asked: which registry do we push to")],
            )
            .await
            .expect("write");
        let id = written[0].id.clone();
        assert_eq!(
            written[0].disposition,
            Disposition::Active,
            "a new digest is a live record, judged nothing else"
        );

        // The permit: an active digest is in the ordinary read.
        let index = store.recent(50, false).await.expect("read");
        assert_eq!(index.len(), 1, "{index:?}");
        assert!(
            index[0].marked_text().starts_with("Asked:"),
            "an active digest carries no marker: {}",
            index[0].marked_text()
        );

        // The refusal: `obsolete` was true and no longer applies, so it is not
        // offered unless it is asked for.
        assert!(
            store
                .set_disposition(
                    &id,
                    Disposition::Obsolete,
                    Some("that registry is gone"),
                    None
                )
                .await
                .expect("mark obsolete")
        );
        assert!(
            store.recent(50, false).await.expect("read").is_empty(),
            "an obsolete digest must not be offered by an ordinary read"
        );
        let asked = store.recent(50, true).await.expect("read with history");
        assert_eq!(asked.len(), 1, "and must still be reachable when asked for");
        assert_eq!(asked[0].disposition, Disposition::Obsolete);
        assert!(
            asked[0].marked_text().starts_with("no longer applies: "),
            "a dispositioned digest is never shown unmarked: {}",
            asked[0].marked_text()
        );

        // `refuted` is the asymmetric one: still findable, never a current
        // claim.
        assert!(
            store
                .set_disposition(
                    &id,
                    Disposition::Refuted,
                    Some("it was never that one"),
                    None
                )
                .await
                .expect("mark refuted")
        );
        let index = store.recent(50, false).await.expect("read");
        assert_eq!(
            index.len(),
            1,
            "a refuted digest stays findable when the query is about its subject"
        );
        assert!(
            index[0]
                .marked_text()
                .starts_with("recorded, later refuted: "),
            "{}",
            index[0].marked_text()
        );
        assert_eq!(
            index[0].disposition_reason.as_deref(),
            Some("it was never that one")
        );

        // A disposition that resolves through a successor cannot be carried
        // without one.
        let refused = store
            .set_disposition(&id, Disposition::Superseded, Some("replaced"), None)
            .await;
        assert!(
            refused.is_err(),
            "`superseded` asserts a link, so it must name one: {refused:?}"
        );
    })
    .await;

    fx.cleanup().await;
}
