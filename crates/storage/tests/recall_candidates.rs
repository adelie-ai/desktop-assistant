//! The reads behind the `[Recall]` block (issues #1100 and #1101).
//!
//! `PgKnowledgeBaseStore::nearest_by_embedding`, `tag_registry::nearest_tags`
//! and `PgScratchpadStore::nearest_by_embedding` are what a turn asks before
//! the model's first move. All are new query surfaces over personal data, so
//! the suite pins the properties the block's correctness rests on.
//!
//! 1. **One user's rows and no other's.** Row-level security is a non-FORCE
//!    backstop the table owner bypasses, so the `WHERE user_id` predicate in
//!    each query is the only real guard. A read that leaked another tenant's
//!    memory would put it in front of the model as this user's own.
//! 2. **Nearest first, with a usable distance.** The block sets a relevance
//!    floor over the distance these queries return. An unordered result, or one
//!    whose distance is not a cosine distance, would make the floor meaningless.
//! 3. **Only rows embedded by the query's own model.** A stored vector from
//!    another model has another dimension, and the vector operator answers that
//!    with an error rather than a miss - which would fail the read instead of
//!    degrading it.
//! 4. **The degraded arm answers.** `search_text_any_term` is what runs when no
//!    embedding is available. It has to match on *any* of a whole user
//!    sentence's terms, because a fallback that answers nothing is not a
//!    fallback.
//! 5. **One conversation's pad and no other's** (#1101). The scratchpad is
//!    per-conversation by design, so both scratchpad reads carry a
//!    `conversation_id` predicate beside the `user_id` one.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test recall_candidates
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::{Conversation, ConversationId, KnowledgeEntry, Message, Role};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::scratchpad::{
    NewScratchpadNote, NoteEmbedding, ScratchpadStore,
};
use desktop_assistant_core::ports::scratchpad_scope::{SubagentScope, with_subagent_scope};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::tag_registry::nearest_tags;
use desktop_assistant_storage::{
    PgConversationStore, PgKnowledgeBaseStore, PgPool, PgScratchpadStore, UserId, with_user_id,
};
use pgvector::Vector;

const USER: &str = "recall-user";
const OTHER_USER: &str = "recall-other-user";

/// The model every seeded vector is stamped with, and the one every read below
/// passes.
const MODEL: &str = "recall-test-model";

/// A model of another dimension entirely, used to prove the scope predicate
/// keeps incompatible vectors out of the comparison.
const OTHER_MODEL: &str = "recall-other-model";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("recall1100").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// A three-dimensional unit vector pointing along one axis. Cosine distance
/// between two of these is 1.0; between a vector and itself it is 0.0, which
/// makes every expectation below readable without a tolerance argument.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 3];
    v[i] = 1.0;
    v
}

/// Write an entry as `user`, then stamp an embedding on it. Writes never embed
/// inline - the background backfill does - so the test stands in for it.
async fn seed_entry(pool: &PgPool, user: &str, id: &str, content: &str, chunk: Vec<f32>) {
    seed_entry_with_model(pool, user, id, content, chunk, MODEL).await;
}

async fn seed_entry_with_model(
    pool: &PgPool,
    user: &str,
    id: &str,
    content: &str,
    chunk: Vec<f32>,
    model: &str,
) {
    let store = PgKnowledgeBaseStore::new(pool.clone());
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["topic".to_string()]))
            .await
            .expect("write succeeds");
    })
    .await;

    let vectors: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = $3, embeddings_updated_at = NOW() \
         WHERE id = $2",
    )
    .bind(&vectors)
    .bind(id)
    .bind(model)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

/// Insert a registry row directly so the test owns its vector and its model
/// stamp, rather than going through the dedup path that would compute both.
async fn seed_tag(pool: &PgPool, user: &str, name: &str, chunk: Vec<f32>, model: &str) {
    let vector = Vector::from(chunk);
    sqlx::query(
        "INSERT INTO tag_registry \
            (user_id, name, description, examples, distinguish_from, embedding, embedding_model) \
         VALUES ($1, $2, 'seeded', '[]'::jsonb, '{}', $3, $4)",
    )
    .bind(user)
    .bind(name)
    .bind(&vector)
    .bind(model)
    .execute(pool)
    .await
    .expect("seed tag");
}

// -- the knowledge arm -------------------------------------------------------

#[tokio::test]
async fn nearest_entries_come_back_nearest_first_with_their_distance() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-near", "the near one", axis(0)).await;
    seed_entry(&fx.pool, USER, "kb-far", "the far one", axis(1)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.id, "kb-near", "nearest first");
        assert!(
            hits[0].1 < 1e-6,
            "a vector against itself is at distance 0, got {}",
            hits[0].1
        );
        assert_eq!(hits[1].0.id, "kb-far");
        assert!(
            (hits[1].1 - 1.0).abs() < 1e-6,
            "orthogonal vectors are at cosine distance 1, got {}",
            hits[1].1
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_never_cross_the_user_boundary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // The other tenant's entry is a perfect match for the query vector, so
    // anything but an explicit scope would rank it first.
    seed_entry(&fx.pool, OTHER_USER, "kb-theirs", "their secret", axis(0)).await;
    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(1)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let ids: Vec<&str> = hits.iter().map(|(e, _)| e.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["kb-mine"],
            "recall must never offer another tenant's memory as this user's own"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_ignore_a_row_embedded_by_another_model() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // Four dimensions against the query's three: comparing them raises rather
    // than missing, so the scope predicate must keep the row out entirely.
    seed_entry_with_model(
        &fx.pool,
        USER,
        "kb-other-model",
        "embedded by something else",
        vec![1.0, 0.0, 0.0, 0.0],
        OTHER_MODEL,
    )
    .await;
    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(0)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("a row from another model must not fail the read");

        let ids: Vec<&str> = hits.iter().map(|(e, _)| e.id.as_str()).collect();
        assert_eq!(ids, vec!["kb-mine"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_stop_at_the_scan_limit() {
    // The limit is what makes the block's "and N more" a bounded count rather
    // than an unbounded read.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    for i in 0..5 {
        seed_entry(&fx.pool, USER, &format!("kb-{i}"), "a note", axis(0)).await;
    }

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 3)
            .await
            .expect("the read succeeds");
        assert_eq!(hits.len(), 3);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_skip_a_retired_entry() {
    // A soft-deleted entry is hidden from every other read path, and offering
    // it back as a recall candidate would resurrect it in the model's view.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-retired", "a retired note", axis(0)).await;
    sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind("kb-retired")
        .execute(&fx.pool)
        .await
        .expect("retire the entry");

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");
        assert!(hits.is_empty(), "a retired entry is not a candidate");
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_answer_nothing_without_an_embedding() {
    // No embedding means no vector arm. The caller has a full-text path to fall
    // back to; what it must not get is an error from a zero-dimension vector.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(0)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(Vec::new(), MODEL, 10)
            .await
            .expect("an absent embedding is not an error");
        assert!(hits.is_empty());
    })
    .await;

    fx.cleanup().await;
}

// -- the tag arm -------------------------------------------------------------

#[tokio::test]
async fn nearest_tags_come_back_nearest_first_with_their_distance() {
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:near", axis(0), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:far", axis(1), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:near", "topic:far"]);
        assert!(hits[0].1 < 1e-6, "got {}", hits[0].1);
        assert!((hits[1].1 - 1.0).abs() < 1e-6, "got {}", hits[1].1);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_never_cross_the_user_boundary() {
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, OTHER_USER, "topic:theirs", axis(0), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:mine", axis(1), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["topic:mine"],
            "another tenant's vocabulary is not this user's"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_skip_a_deprecated_tag() {
    // A deprecated tag points at its replacement. Offering it as vocabulary
    // would send the model's next search at a name no row carries any more.
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:current", axis(1), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:retired", axis(0), MODEL).await;
    sqlx::query("UPDATE tag_registry SET deprecated_for_tag = $1 WHERE user_id = $2 AND name = $3")
        .bind("topic:current")
        .bind(USER)
        .bind("topic:retired")
        .execute(&fx.pool)
        .await
        .expect("deprecate the tag");

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:current"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_ignore_a_row_embedded_by_another_model() {
    let Some(fx) = fixture().await else { return };

    seed_tag(
        &fx.pool,
        USER,
        "topic:other-model",
        vec![1.0, 0.0, 0.0, 0.0],
        OTHER_MODEL,
    )
    .await;
    seed_tag(&fx.pool, USER, "topic:mine", axis(0), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("a row from another model must not fail the read");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:mine"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_answer_nothing_without_an_embedding() {
    // The registry carries no full-text index, so the tag arm has nothing to
    // degrade to. It goes quiet rather than raising.
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:mine", axis(0), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, Vec::new(), MODEL, 10)
            .await
            .expect("an absent embedding is not an error");
        assert!(hits.is_empty());
    })
    .await;

    fx.cleanup().await;
}

// -- the degraded (no embedding) arm ----------------------------------------

#[tokio::test]
async fn any_term_search_matches_an_entry_that_carries_one_of_the_prompts_words() {
    // The defect this query exists to avoid: `plainto_tsquery` ANDs every
    // lexeme, so a whole sentence asks one entry to contain all of it. The
    // entry below says nothing about "live", and must still be found.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(
                "kb-registry",
                "The registry is on the storage host.",
                vec!["infra".to_string()],
            ))
            .await
            .expect("write succeeds");

        let all_terms = store
            .search_text("where does the registry live?", None, 10)
            .await
            .expect("the read succeeds");
        assert!(
            all_terms.is_empty(),
            "precondition: the AND-joined search finds nothing for a sentence"
        );

        let any_term = store
            .search_text_any_term("where does the registry live?", 10)
            .await
            .expect("the read succeeds");
        let ids: Vec<&str> = any_term.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["kb-registry"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_ranks_the_entry_that_carries_more_of_the_terms_first() {
    // Widening the match set must not put the weakest hit at the top: the block
    // renders the first eight and drops the rest.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        // Both entries match, so the ordering is what is under test: `kb-weak`
        // carries one of the query's terms and `kb-strong` carries both.
        for (id, content) in [
            ("kb-weak", "A note about where the lab equipment lives."),
            (
                "kb-strong",
                "The registry lives on the storage host in the lab.",
            ),
        ] {
            store
                .write(KnowledgeEntry::new(id, content, vec![]))
                .await
                .expect("write succeeds");
        }

        let hits = store
            .search_text_any_term("where does the registry live?", 10)
            .await
            .expect("the read succeeds");
        let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["kb-strong", "kb-weak"],
            "both match, and the one carrying more of the terms ranks first"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_never_crosses_the_user_boundary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(OTHER_USER), async {
        store
            .write(KnowledgeEntry::new(
                "kb-theirs",
                "The registry is on their storage host.",
                vec![],
            ))
            .await
            .expect("write succeeds");
    })
    .await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .search_text_any_term("where does the registry live?", 10)
            .await
            .expect("the read succeeds");
        assert!(
            hits.is_empty(),
            "the degraded arm is scoped exactly like the semantic one"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_skips_a_retired_entry() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(
                "kb-retired",
                "The registry is on the storage host.",
                vec![],
            ))
            .await
            .expect("write succeeds");
    })
    .await;
    sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind("kb-retired")
        .execute(&fx.pool)
        .await
        .expect("retire the entry");

    with_user_id(UserId::new(USER), async {
        let hits = store
            .search_text_any_term("where does the registry live?", 10)
            .await
            .expect("the read succeeds");
        assert!(hits.is_empty());
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_answers_nothing_for_a_prompt_with_no_terms() {
    // "the a of" reduces to no lexemes at all, which builds a NULL query. It
    // must match no row rather than every row.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new("kb-mine", "Some content here.", vec![]))
            .await
            .expect("write succeeds");

        for prompt in ["the a of", "   ", ""] {
            let hits = store
                .search_text_any_term(prompt, 10)
                .await
                .expect("the read succeeds");
            assert!(
                hits.is_empty(),
                "prompt {prompt:?} matched {} rows",
                hits.len()
            );
        }
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_treats_tsquery_operators_as_text() {
    // A user prompt is text. Left as syntax, `!` and `&` and `<->` would either
    // raise or silently change what was asked.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(
                "kb-mine",
                "A note about the registry.",
                vec![],
            ))
            .await
            .expect("write succeeds");

        let hits = store
            .search_text_any_term("registry & !nonsense <-> 'quoted' | more", 10)
            .await
            .expect("operators in a prompt must not fail the read");
        let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["kb-mine"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn any_term_search_stops_at_its_limit() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        for i in 0..5 {
            store
                .write(KnowledgeEntry::new(
                    format!("kb-{i}"),
                    "A note about the registry.",
                    vec![],
                ))
                .await
                .expect("write succeeds");
        }

        let hits = store
            .search_text_any_term("registry", 3)
            .await
            .expect("the read succeeds");
        assert_eq!(hits.len(), 3);
    })
    .await;

    fx.cleanup().await;
}

// -- the scratchpad arm (#1101) ----------------------------------------------

/// A conversation row, so the pad's foreign key resolves.
fn make_conversation(id: &str) -> Conversation {
    let mut conv = Conversation::new(id, "recall scratchpad test");
    conv.created_at = "2026-08-06 00:00:00".to_string();
    conv.updated_at = "2026-08-06 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

/// Write one note already carrying `chunk` under `model`, exactly as an
/// inline-embedded write does.
async fn seed_note(
    pad: &PgScratchpadStore,
    conversation_id: &str,
    key: &str,
    content: &str,
    chunk: Vec<f32>,
    model: &str,
) {
    seed_typed_note(pad, conversation_id, key, content, "note", chunk, model).await;
}

/// The same, for a note of a `note_type` other than the free-form default.
async fn seed_typed_note(
    pad: &PgScratchpadStore,
    conversation_id: &str,
    key: &str,
    content: &str,
    note_type: &str,
    chunk: Vec<f32>,
    model: &str,
) {
    let mut note = NewScratchpadNote::new(key, content);
    note.note_type = note_type.to_string();
    note.embedding = Some(NoteEmbedding {
        chunks: vec![chunk],
        model: model.to_string(),
    });
    pad.write(conversation_id, &[note])
        .await
        .expect("write embedded note");
}

/// Acceptance (#1101): the pad is per-conversation by design. A read that
/// reached across conversations would put another task's working notes in front
/// of the model as this task's own.
#[tokio::test]
async fn recall_block_scratchpad_arm_stays_within_the_current_conversation() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs
            .create(make_conversation("c1"))
            .await
            .expect("conv c1");
        convs
            .create(make_conversation("c2"))
            .await
            .expect("conv c2");
        // The other conversation's note is a perfect match for the query
        // vector, so anything but an explicit scope would rank it first.
        seed_note(
            &pad,
            "c2",
            "theirs",
            "another task's finding",
            axis(0),
            MODEL,
        )
        .await;
        seed_note(&pad, "c1", "mine", "this task's finding", axis(1), MODEL).await;

        let hits = pad
            .nearest_by_embedding("c1", axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let keys: Vec<&str> = hits.iter().map(|(n, _)| n.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["mine"],
            "recall must never offer another conversation's pad"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_notes_never_cross_the_user_boundary() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(OTHER_USER), async {
        convs
            .create(make_conversation("shared"))
            .await
            .expect("their conv");
        seed_note(&pad, "shared", "theirs", "their secret", axis(0), MODEL).await;
    })
    .await;
    with_user_id(UserId::new(USER), async {
        convs
            .create(make_conversation("mine"))
            .await
            .expect("my conv");
        seed_note(&pad, "mine", "mine", "my own note", axis(1), MODEL).await;

        let hits = pad
            .nearest_by_embedding("shared", axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        assert!(
            hits.is_empty(),
            "another tenant's pad must be invisible even by its own conversation id"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_notes_come_back_nearest_first_with_their_distance() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        seed_note(&pad, "c1", "near", "the near one", axis(0), MODEL).await;
        seed_note(&pad, "c1", "far", "the far one", axis(1), MODEL).await;

        let hits = pad
            .nearest_by_embedding("c1", axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.key, "near", "nearest first");
        assert!(
            hits[0].1 < 1e-6,
            "a vector against itself is at distance 0, got {}",
            hits[0].1
        );
        assert!(
            (hits[1].1 - 1.0).abs() < 1e-6,
            "orthogonal vectors are at cosine distance 1, got {}",
            hits[1].1
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_notes_ignore_a_note_embedded_by_another_model() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        // Four dimensions against the query's three: comparing them raises
        // rather than missing, so the scope predicate must keep the row out.
        seed_note(
            &pad,
            "c1",
            "other-model",
            "embedded by something else",
            vec![1.0, 0.0, 0.0, 0.0],
            OTHER_MODEL,
        )
        .await;
        seed_note(&pad, "c1", "mine", "embedded by this one", axis(0), MODEL).await;

        let hits = pad
            .nearest_by_embedding("c1", axis(0), MODEL, 10)
            .await
            .expect("a row of another dimension must be skipped, not raise");

        let keys: Vec<&str> = hits.iter().map(|(n, _)| n.key.as_str()).collect();
        assert_eq!(keys, vec!["mine"]);
    })
    .await;

    fx.cleanup().await;
}

/// The block drops a pinned note, because `[Pinned]` already carries its whole
/// content. That decision is the core's, so the flag has to survive the read.
#[tokio::test]
async fn nearest_notes_report_whether_a_note_is_pinned() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        seed_note(&pad, "c1", "pinned-one", "in view already", axis(0), MODEL).await;
        seed_note(&pad, "c1", "loose-one", "nowhere else", axis(0), MODEL).await;
        pad.set_pinned("c1", &["pinned-one".to_string()], true)
            .await
            .expect("pin one note");

        let hits = pad
            .nearest_by_embedding("c1", axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let pinned: Vec<(&str, bool)> = hits
            .iter()
            .map(|(n, _)| (n.key.as_str(), n.pinned))
            .collect();
        assert!(pinned.contains(&("pinned-one", true)), "{pinned:?}");
        assert!(pinned.contains(&("loose-one", false)), "{pinned:?}");
    })
    .await;

    fx.cleanup().await;
}

/// The degraded arm has to match on *any* of a whole user sentence's terms.
/// `plainto_tsquery` joins every lexeme with AND, which answers almost nothing
/// for a sentence - a fallback that answers nothing is not a fallback (#1100).
#[tokio::test]
async fn the_degraded_scratchpad_arm_matches_any_term_of_a_whole_sentence() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        pad.write(
            "c1",
            &[NewScratchpadNote::new(
                "deploy-window",
                "the deploy runs on Fridays",
            )],
        )
        .await
        .expect("write note");

        // Precondition: the AND-joined query the ordinary search builds finds
        // nothing, because the note never says "when" or "next".
        let anded = pad
            .search("c1", "when is the next deploy?", Vec::new(), "", None, 10)
            .await
            .expect("the ordinary search succeeds");
        assert!(
            anded.is_empty(),
            "precondition: an AND-joined sentence must not match this note"
        );

        let hits = pad
            .search_text_any_term("c1", "when is the next deploy?", 10)
            .await
            .expect("the read succeeds");

        let keys: Vec<&str> = hits.iter().map(|n| n.key.as_str()).collect();
        assert_eq!(keys, vec!["deploy-window"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn the_degraded_scratchpad_arm_stays_within_the_current_conversation() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs
            .create(make_conversation("c1"))
            .await
            .expect("conv c1");
        convs
            .create(make_conversation("c2"))
            .await
            .expect("conv c2");
        pad.write(
            "c2",
            &[NewScratchpadNote::new(
                "theirs",
                "the deploy runs on Fridays",
            )],
        )
        .await
        .expect("write note");
        pad.write(
            "c1",
            &[NewScratchpadNote::new("mine", "the deploy runs on Fridays")],
        )
        .await
        .expect("write note");

        let hits = pad
            .search_text_any_term("c1", "when is the next deploy?", 10)
            .await
            .expect("the read succeeds");

        let keys: Vec<&str> = hits.iter().map(|n| n.key.as_str()).collect();
        assert_eq!(keys, vec!["mine"]);
    })
    .await;

    fx.cleanup().await;
}

/// A prompt of nothing but stop words reduces to no lexemes at all. The query
/// is then NULL, which must match no row rather than every row.
#[tokio::test]
async fn the_degraded_scratchpad_arm_matches_nothing_for_a_prompt_of_stop_words() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        pad.write("c1", &[NewScratchpadNote::new("mine", "a real finding")])
            .await
            .expect("write note");

        let hits = pad
            .search_text_any_term("c1", "the and of", 10)
            .await
            .expect("the read succeeds");

        assert!(hits.is_empty(), "a query with no lexemes matches nothing");
    })
    .await;

    fx.cleanup().await;
}

/// The arm must read the same set the `[Scratchpad]` index advertises: the
/// free-form notes. The `goal` note, `outcome:<step>` notes and `todo`-typed
/// steps are rendered by `[Current task]` and `[Plan]` on every round, and the
/// goal note is by construction the pad row nearest a prompt about the current
/// task - so without the carve-out the arm's first line restates the task the
/// prompt already carries.
#[tokio::test]
async fn nearest_notes_read_only_the_free_form_pad() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        // Every excluded kind sits at distance 0 from the query, so only the
        // carve-out can keep it out.
        seed_note(&pad, "c1", "goal", "ship the deploy", axis(0), MODEL).await;
        seed_note(
            &pad,
            "c1",
            "outcome:1.2",
            "the step's finding",
            axis(0),
            MODEL,
        )
        .await;
        seed_typed_note(&pad, "c1", "1", "a plan step", "todo", axis(0), MODEL).await;
        // ...and the one free-form note sits furthest away.
        seed_note(&pad, "c1", "finding", "the pool leaks", axis(1), MODEL).await;

        let hits = pad
            .nearest_by_embedding("c1", axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let keys: Vec<&str> = hits.iter().map(|(n, _)| n.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["finding"],
            "only free-form notes; the rest are already rendered by other blocks"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn the_degraded_scratchpad_arm_reads_only_the_free_form_pad() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        let mut step = NewScratchpadNote::new("1", "the deploy runs on Fridays");
        step.note_type = "todo".to_string();
        pad.write(
            "c1",
            &[
                NewScratchpadNote::new("goal", "the deploy runs on Fridays"),
                NewScratchpadNote::new("outcome:1", "the deploy runs on Fridays"),
                step,
                NewScratchpadNote::new("finding", "the deploy runs on Fridays"),
            ],
        )
        .await
        .expect("write notes");

        let hits = pad
            .search_text_any_term("c1", "when is the next deploy?", 10)
            .await
            .expect("the read succeeds");

        let keys: Vec<&str> = hits.iter().map(|n| n.key.as_str()).collect();
        assert_eq!(keys, vec!["finding"]);
    })
    .await;

    fx.cleanup().await;
}

/// A subagent turn reads a spawn-time snapshot of the pad: its own subtree at
/// any id, PLUS pre-marker rows from its ancestors - never a concurrent
/// sibling's in-flight notes. Both new reads carry that predicate, and a later
/// edit that dropped it from one of them would leave every other suite green.
///
/// Every note here carries the same vector and the same words, so only the
/// snapshot predicate can keep the sibling's note out of either arm.
#[tokio::test]
async fn the_scratchpad_arm_honours_the_subagent_read_snapshot() {
    let Some(fx) = fixture().await else { return };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    fn sub_scope(owner: &str, marker: &str, ancestors: &[&str]) -> SubagentScope {
        SubagentScope {
            session_conversation_id: ConversationId::from("c1"),
            owner_todo: owner.to_string(),
            visible_before: marker.to_string(),
            ancestors: ancestors.iter().map(|s| s.to_string()).collect(),
        }
    }

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        // Root context and a concurrent sibling, both before the spawn marker.
        seed_note(
            &pad,
            "c1",
            "ctx",
            "the deploy pipeline is red",
            axis(0),
            MODEL,
        )
        .await;
        with_subagent_scope(sub_scope("1.2", "", &[]), async {
            seed_note(
                &pad,
                "c1",
                "sib",
                "the deploy pipeline is red",
                axis(0),
                MODEL,
            )
            .await;
        })
        .await;

        let marker = uuid::Uuid::now_v7().to_string();

        with_subagent_scope(sub_scope("1.1", "", &[]), async {
            seed_note(
                &pad,
                "c1",
                "own",
                "the deploy pipeline is red",
                axis(0),
                MODEL,
            )
            .await;
        })
        .await;

        let (nearest, lexical) = with_subagent_scope(sub_scope("1.1", &marker, &[""]), async {
            let nearest = pad
                .nearest_by_embedding("c1", axis(0), MODEL, 50)
                .await
                .expect("the vector read succeeds");
            let lexical = pad
                .search_text_any_term("c1", "is the deploy pipeline red?", 50)
                .await
                .expect("the lexical read succeeds");
            (nearest, lexical)
        })
        .await;

        for keys in [
            nearest
                .iter()
                .map(|(n, _)| n.key.as_str())
                .collect::<Vec<_>>(),
            lexical.iter().map(|n| n.key.as_str()).collect::<Vec<_>>(),
        ] {
            assert!(
                keys.contains(&"ctx"),
                "ancestor pre-marker context: {keys:?}"
            );
            assert!(keys.contains(&"own"), "own namespace at any id: {keys:?}");
            assert!(
                !keys.contains(&"sib"),
                "a concurrent sibling's notes must stay invisible: {keys:?}"
            );
        }
    })
    .await;

    fx.cleanup().await;
}
