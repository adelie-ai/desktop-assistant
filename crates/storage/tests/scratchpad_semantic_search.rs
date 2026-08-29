//! The scratchpad is searched semantically as well as lexically (issue #717).
//!
//! The scratchpad is the agent's own working memory: it writes a distilled
//! finding, then looks for it again later. Lexical search finds that note only
//! when the agent happens to reuse the same words, and the whole point of the
//! pad is that the agent re-summarizes as it goes, so the wording moves.
//!
//! These tests hold the search to the same contract `knowledge_base` already
//! meets -- a vector arm and a full-text arm fused by reciprocal rank -- plus
//! the two rules that keep a model change survivable:
//!
//! * The **vector arm is model-scoped**, so a row embedded by a superseded
//!   model is invisible rather than fatal (pgvector answers a mismatched
//!   comparison with an error, not a miss).
//! * The **full-text arm is never model-scoped**, so a model change costs
//!   recall quality and not all recall.
//!
//! And the two scope rules the pad has always had, now re-proved across the new
//! query shape: `user_id` + `conversation_id`, and the `owner_todo` subtree.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test scratchpad_semantic_search
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use desktop_assistant_core::domain::{
    Conversation, ConversationId, LEGACY_TURN_NOTE_TYPE, Message, Role,
};
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::scratchpad::{
    NewScratchpadNote, NoteEmbedding, ScratchpadStore, embed_notes,
};
use desktop_assistant_core::ports::scratchpad_scope::{SubagentScope, with_subagent_scope};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::embedding_backfill::{
    BackfillEmbedFn, backfill_scratchpad_embeddings, invalidate_stale_embeddings,
};
use desktop_assistant_storage::{PgConversationStore, PgScratchpadStore, UserId, with_user_id};
use sqlx::PgPool;

const USER: &str = "alice";
const OTHER_USER: &str = "bob";
const DIGEST: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_DIGEST: &str = "2222222222222222222222222222222222222222222222222222222222222222";

/// The model this daemon is currently embedding with. Three dimensions.
fn current_model() -> String {
    format!("nomic-embed-text@{DIGEST}")
}

/// A genuinely different model. Its vectors are four-dimensional, so any
/// comparison against a current-model query raises rather than missing.
fn superseded_model() -> String {
    format!("mxbai-embed-large@{OTHER_DIGEST}")
}

async fn fixture(prefix: &str) -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new(prefix).await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set; {prefix} pass-skipped");
    }
    fx
}

fn make_conversation(id: &str) -> Conversation {
    let mut conv = Conversation::new(id, "scratchpad semantic test");
    conv.created_at = "2026-08-06 00:00:00".to_string();
    conv.updated_at = "2026-08-06 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

/// An `EmbedFn` that answers every text with `vector`, recording what it saw.
fn constant_embed(vector: Vec<f32>) -> EmbedFn {
    Arc::new(move |texts: Vec<String>| {
        let vector = vector.clone();
        Box::pin(async move { Ok(vec![vector; texts.len()]) })
    })
}

/// A `BackfillEmbedFn` that answers every text with `vector`, recording what it
/// saw so a test can assert the backfill embeds the same text the inline write
/// does.
fn recording_backfill_embed(seen: Arc<Mutex<Vec<String>>>, vector: Vec<f32>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        let seen = Arc::clone(&seen);
        let vector = vector.clone();
        Box::pin(async move {
            let n = texts.len();
            seen.lock().expect("record texts").extend(texts);
            Ok(vec![vector; n])
        })
    })
}

/// Write one note already carrying `vector` under `model`, exactly as an
/// inline-embedded write does.
async fn write_embedded(
    pad: &PgScratchpadStore,
    conversation_id: &str,
    key: &str,
    content: &str,
    vector: Vec<f32>,
    model: &str,
) {
    let mut note = NewScratchpadNote::new(key, content);
    note.embedding = Some(NoteEmbedding {
        chunks: vec![vector],
        model: model.to_string(),
    });
    pad.write(conversation_id, &[note])
        .await
        .expect("write embedded note");
}

async fn note_state(pool: &PgPool, key: &str) -> (bool, Option<String>) {
    sqlx::query_as::<_, (bool, Option<String>)>(
        "SELECT embedding IS NOT NULL, embedding_model FROM scratchpads WHERE note_key = $1",
    )
    .bind(key)
    .fetch_one(pool)
    .await
    .expect("probe note")
}

fn key_set(notes: &[desktop_assistant_core::domain::ScratchpadNote]) -> HashSet<String> {
    notes.iter().map(|n| n.key.clone()).collect()
}

/// Acceptance: a note is found by a semantically similar query that shares no
/// significant words with it. This is the whole feature -- the agent
/// re-summarizes as it works, so the words it searches with are not the words
/// it wrote.
#[tokio::test]
async fn a_note_is_found_by_a_semantically_similar_query_sharing_no_words() {
    let Some(fx) = fixture("sp717a").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        write_embedded(
            &pad,
            "c1",
            "n1",
            "Remember to drink water at your desk",
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;
        write_embedded(
            &pad,
            "c1",
            "n2",
            "Apples and oranges are in the bowl",
            vec![0.0, 1.0, 0.0],
            &current_model(),
        )
        .await;

        // Precondition: the query has no lexical overlap at all, so the
        // full-text arm alone finds nothing.
        let lexical = pad
            .search("c1", "stay hydrated", Vec::new(), "", None, 10)
            .await
            .expect("lexical search");
        assert!(
            lexical.is_empty(),
            "precondition: the query must share no significant words with the note"
        );

        let hits = pad
            .search(
                "c1",
                "stay hydrated",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("hybrid search");

        assert_eq!(
            hits.first().map(|n| n.key.as_str()),
            Some("n1"),
            "the semantically nearest note must rank first, got {:?}",
            hits.iter().map(|n| &n.key).collect::<Vec<_>>()
        );
    })
    .await;

    fx.cleanup().await;
}

/// A per-turn capture written before the episodic turn index existed (#1349)
/// stays on the pad but is never offered by the `[Recall]` pad arm.
///
/// The digests of those same turns land in `turn_digests` once the transcript
/// backfill runs. If the pad arm still offered the legacy rows, one turn could
/// arrive twice in one `[Recall]` block through two arms under two different
/// keys, and the already-in-view dedup keys on identity, so it would not see
/// that the two are one turn.
///
/// Both reads behind the block are checked, and each is paired with its permit:
/// a test that only proved the exclusion would pass against an arm that
/// returned nothing at all.
#[tokio::test]
async fn a_legacy_turn_capture_stays_on_the_pad_but_is_not_offered_by_the_recall_arm() {
    let Some(fx) = fixture("sp1349legacy").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");

        // An ordinary note and a legacy turn capture, identical in every way
        // the two reads rank on: same vector, same words.
        let words = "Asked: deploy with the kustomization";
        write_embedded(
            &pad,
            "c1",
            "n1",
            words,
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;

        let mut capture = NewScratchpadNote::new("turn:m-01", words);
        capture.note_type = LEGACY_TURN_NOTE_TYPE.to_string();
        capture.embedding = Some(NoteEmbedding {
            chunks: vec![vec![1.0, 0.0, 0.0]],
            model: current_model(),
        });
        pad.write("c1", &[capture]).await.expect("write capture");

        // It is still on the pad: this is a durable record, and until the
        // transcript backfill has run it is the only copy of that capture.
        let listed = pad.list("c1", None, 50).await.expect("list");
        assert!(
            key_set(&listed).contains("turn:m-01"),
            "the legacy row must not be destroyed: {:?}",
            key_set(&listed)
        );

        // The vector arm: the ordinary note comes back, the capture does not.
        let nearest = pad
            .nearest_by_embedding("c1", vec![1.0, 0.0, 0.0], &current_model(), 10)
            .await
            .expect("nearest");
        let offered: HashSet<String> = nearest.notes.iter().map(|(n, _)| n.key.clone()).collect();
        assert!(
            offered.contains("n1"),
            "the permit: an ordinary note is still offered: {offered:?}"
        );
        assert!(
            !offered.contains("turn:m-01"),
            "a legacy turn capture must not be offered beside its own digest: {offered:?}"
        );

        // The full-text arm, on the same terms.
        let lexical = pad
            .search_text_any_term("c1", "kustomization deploy", 10)
            .await
            .expect("text search");
        let offered = key_set(&lexical);
        assert!(
            offered.contains("n1"),
            "the permit: an ordinary note is still offered lexically: {offered:?}"
        );
        assert!(
            !offered.contains("turn:m-01"),
            "a legacy turn capture must not be offered lexically either: {offered:?}"
        );
    })
    .await;

    fx.cleanup().await;
}

/// The vector arm over-fetches `limit * 2` candidates to feed the fusion, and
/// that truncation must keep the NEAREST candidates, not an arbitrary subset.
///
/// Truncating without an order is silent: the arm still returns rows, the
/// fusion still ranks them, and the caller gets a plausible page that simply
/// omits the best matches. Semantic recall is the half of this search the whole
/// feature exists for, so a candidate set chosen by physical row order defeats
/// it without erroring.
///
/// Twelve notes at monotonically increasing distance, a query with no lexical
/// match so only the vector arm contributes, and a limit of three -- so six
/// candidates are fetched from twelve and the three nearest must survive both
/// truncations.
#[tokio::test]
async fn the_vector_arm_truncates_to_the_nearest_candidates_not_an_arbitrary_subset() {
    let Some(fx) = fixture("sp717l").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        // Note `i` sits at cosine distance increasing in `i` from the query
        // vector [1, 0]: 1 - 1/sqrt(1 + (0.05i)^2). So `n00` is nearest and
        // `n11` furthest, with no ties.
        for i in 0..12 {
            write_embedded(
                &pad,
                "c1",
                &format!("n{i:02}"),
                "a distilled finding with no query words in it",
                vec![1.0, i as f32 * 0.05],
                &current_model(),
            )
            .await;
        }

        let hits = pad
            .search(
                "c1",
                // Matches no token, so `text_ranked` is empty and every
                // returned row came through the vector arm.
                "zzqqxx",
                vec![1.0, 0.0],
                &current_model(),
                None,
                3,
            )
            .await
            .expect("hybrid search");

        let keys: Vec<String> = hits.iter().map(|n| n.key.clone()).collect();
        assert_eq!(
            keys,
            vec!["n00".to_string(), "n01".to_string(), "n02".to_string()],
            "the three nearest notes must survive truncation, in distance order; \
             a different set means the vector arm truncated an unordered candidate list"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance: the full-text arm still returns exact-token matches, and it is
/// NOT model-scoped. A note whose vector was produced by a superseded model is
/// invisible to the vector arm, and must still be findable lexically --
/// otherwise a model change makes content unfindable rather than merely
/// degrading recall.
#[tokio::test]
async fn the_fts_arm_returns_exact_token_matches_and_is_not_model_scoped() {
    let Some(fx) = fixture("sp717b").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        // Four dimensions under the superseded model: unusable by, and
        // therefore invisible to, a three-dimensional current-model query.
        write_embedded(
            &pad,
            "c1",
            "n1",
            "the deploy pipeline is red",
            vec![1.0, 0.0, 0.0, 0.0],
            &superseded_model(),
        )
        .await;

        let hits = pad
            .search(
                "c1",
                "deploy",
                vec![0.0, 0.0, 1.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("hybrid search must not raise on a superseded-model row");

        assert_eq!(
            key_set(&hits),
            HashSet::from(["n1".to_string()]),
            "an exact token match must survive a model change through the full-text arm"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance: a note written during a turn is semantically findable within
/// that same turn. The background backfill runs on a 300-second cadence, and
/// the case that matters is the agent looking for what it wrote moments ago --
/// exactly the window a background backfill leaves open.
#[tokio::test]
async fn a_note_written_during_a_turn_is_semantically_findable_within_the_same_turn() {
    let Some(fx) = fixture("sp717c").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());
    let embed = constant_embed(vec![1.0, 0.0, 0.0]);

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");

        let mut notes = vec![NewScratchpadNote::new(
            "n1",
            "Remember to drink water at your desk",
        )];
        embed_notes(&embed, &current_model(), &mut notes).await;
        pad.write("c1", &notes).await.expect("write");

        // No backfill pass in between -- this is the same turn.
        let hits = pad
            .search(
                "c1",
                "stay hydrated",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("hybrid search");

        assert_eq!(
            key_set(&hits),
            HashSet::from(["n1".to_string()]),
            "an inline-embedded note must be semantically findable immediately"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance (second half of the wedged-backend criterion; the first half,
/// that the embed call itself is bounded, is
/// `a_wedged_embedding_backend_does_not_block_the_write` in
/// `core::ports::scratchpad`): a note that landed without a vector still writes
/// and is still found lexically.
#[tokio::test]
async fn a_note_written_without_a_vector_is_still_found_lexically() {
    let Some(fx) = fixture("sp717d").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());
    let failing: EmbedFn = Arc::new(|_| {
        Box::pin(async {
            Err(desktop_assistant_core::CoreError::Storage(
                "embedding backend unavailable".to_string(),
            ))
        })
    });

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");

        let mut notes = vec![NewScratchpadNote::new("n1", "the deploy pipeline is red")];
        embed_notes(&failing, &current_model(), &mut notes).await;
        assert!(
            notes[0].embedding.is_none(),
            "precondition: the backend must have failed to produce a vector"
        );

        let saved = pad
            .write("c1", &notes)
            .await
            .expect("the write must land even with no vector");
        assert_eq!(saved.len(), 1);

        let hits = pad
            .search(
                "c1",
                "deploy",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("hybrid search");
        assert_eq!(key_set(&hits), HashSet::from(["n1".to_string()]));

        let (has_embedding, model) = note_state(&fx.pool, "n1").await;
        assert!(
            !has_embedding && model.is_none(),
            "the row must be left for the backfill to pick up"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance: notes stamped with a superseded model are excluded from the
/// vector arm rather than raising a dimension error. A table legitimately holds
/// two models' vectors during any reindex and for the whole of a live backend
/// swap.
#[tokio::test]
async fn notes_stamped_with_a_superseded_model_are_excluded_from_the_vector_arm() {
    let Some(fx) = fixture("sp717e").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        write_embedded(
            &pad,
            "c1",
            "old",
            "orbital mechanics of a transfer burn",
            vec![1.0, 0.0, 0.0, 0.0],
            &superseded_model(),
        )
        .await;
        write_embedded(
            &pad,
            "c1",
            "new",
            "the quarterly budget spreadsheet",
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;

        // A query with no lexical match at all, so only the vector arm can
        // contribute. Without the model predicate this comparison raises.
        let hits = pad
            .search(
                "c1",
                "zzqqxx",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("a superseded-model row must be a miss, not an error");

        assert_eq!(
            key_set(&hits),
            HashSet::from(["new".to_string()]),
            "only the current-model row may reach the vector arm"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A purely cosmetic model rename with an unchanged digest is the same model
/// (#655), so its rows stay visible to the vector arm. Hiding them until the
/// sweep restamps them would blank semantic recall over the whole pad for no
/// reason.
#[tokio::test]
async fn a_renamed_model_with_an_unchanged_digest_stays_visible_to_the_vector_arm() {
    let Some(fx) = fixture("sp717j").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        write_embedded(
            &pad,
            "c1",
            "n1",
            "Remember to drink water at your desk",
            vec![1.0, 0.0, 0.0],
            &format!("nomic-embed-text:latest@{DIGEST}"),
        )
        .await;

        let hits = pad
            .search(
                "c1",
                "stay hydrated",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("hybrid search");

        assert_eq!(
            key_set(&hits),
            HashSet::from(["n1".to_string()]),
            "the same digest means the same model; a rename must not hide the vector"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Rewriting a note's content must clear the vector that described the old
/// content. Left in place, that vector would keep a current model stamp and so
/// sit beyond both the stale sweep and the backfill -- permanently describing
/// text the note no longer holds.
#[tokio::test]
async fn rewriting_a_notes_content_clears_the_vector_of_the_old_content() {
    let Some(fx) = fixture("sp717k").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        write_embedded(
            &pad,
            "c1",
            "n1",
            "the deploy pipeline is red",
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;
        let (has_embedding, _) = note_state(&fx.pool, "n1").await;
        assert!(has_embedding, "precondition: the note starts embedded");

        // An unembedded rewrite, as a caller with no embedding backend makes.
        pad.write(
            "c1",
            &[NewScratchpadNote::new("n1", "the deploy pipeline is green")],
        )
        .await
        .expect("rewrite");

        let (has_embedding, model) = note_state(&fx.pool, "n1").await;
        assert!(
            !has_embedding,
            "a vector describing the previous content must not survive the rewrite"
        );
        assert!(
            model.is_none(),
            "the stamp must clear with the vector, or the backfill never re-embeds the note"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance: the stale sweep clears superseded scratchpad vectors, and the
/// backfill re-embeds them -- so the pad converges instead of going permanently
/// unembedded after a model change.
#[tokio::test]
async fn the_stale_sweep_clears_scratchpad_vectors_and_the_backfill_re_embeds_them() {
    let Some(fx) = fixture("sp717f").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        write_embedded(
            &pad,
            "c1",
            "n1",
            "the deploy pipeline is red",
            vec![1.0, 0.0, 0.0, 0.0],
            &superseded_model(),
        )
        .await;
    })
    .await;

    invalidate_stale_embeddings(&fx.pool, &current_model())
        .await
        .expect("sweep");

    let (has_embedding, model) = note_state(&fx.pool, "n1").await;
    assert!(
        !has_embedding,
        "a note embedded under a superseded model must be invalidated, or the next \
         search compares mismatched dimensions and errors"
    );
    assert!(
        model.is_none(),
        "the stale stamp must clear with the vector so the backfill re-embeds it"
    );

    let seen = Arc::new(Mutex::new(Vec::new()));
    let embedded = backfill_scratchpad_embeddings(
        &fx.pool,
        &recording_backfill_embed(Arc::clone(&seen), vec![0.5, 0.5, 0.5]),
        &current_model(),
    )
    .await
    .expect("backfill");

    assert_eq!(embedded, 1, "the invalidated note must be re-embedded");
    let (has_embedding, model) = note_state(&fx.pool, "n1").await;
    assert!(has_embedding, "the backfill must restore the vector");
    assert_eq!(model.as_deref(), Some(current_model().as_str()));

    fx.cleanup().await;
}

/// The backfill must embed the same text the inline write does, or a note
/// embedded by one path is not comparable with a query matched against the
/// other. Both use `note_key + content`, matching the table's `tsv`.
#[tokio::test]
async fn the_backfill_embeds_the_same_text_the_inline_write_does() {
    let Some(fx) = fixture("sp717g").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv");
        pad.write(
            "c1",
            &[NewScratchpadNote::new("deploy", "ship it on Friday")],
        )
        .await
        .expect("write unembedded note");
    })
    .await;

    let seen = Arc::new(Mutex::new(Vec::new()));
    backfill_scratchpad_embeddings(
        &fx.pool,
        &recording_backfill_embed(Arc::clone(&seen), vec![1.0, 0.0, 0.0]),
        &current_model(),
    )
    .await
    .expect("backfill");

    let texts = seen.lock().expect("read recorded texts").clone();
    assert_eq!(
        texts,
        vec!["deploy ship it on Friday".to_string()],
        "embed_notes embeds `key content`; the backfill must match it"
    );

    fx.cleanup().await;
}

/// Acceptance: the search stays `user_id` and `conversation_id` scoped. Row
/// level security is a non-FORCE backstop the table owner bypasses, so the
/// `WHERE user_id` clause is the only real guard -- a missing predicate here is
/// a cross-tenant leak, not a style problem.
#[tokio::test]
async fn search_stays_user_and_conversation_scoped() {
    let Some(fx) = fixture("sp717h").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        convs.create(make_conversation("c1")).await.expect("conv 1");
        convs.create(make_conversation("c2")).await.expect("conv 2");
        // Identical vector and identical content in both conversations, so only
        // the conversation predicate can tell them apart.
        write_embedded(
            &pad,
            "c1",
            "n1",
            "the deploy pipeline is red",
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;
        write_embedded(
            &pad,
            "c2",
            "n2",
            "the deploy pipeline is red",
            vec![1.0, 0.0, 0.0],
            &current_model(),
        )
        .await;

        let hits = pad
            .search(
                "c1",
                "deploy",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("own search");
        assert_eq!(
            key_set(&hits),
            HashSet::from(["n1".to_string()]),
            "a search must not reach into another conversation's pad"
        );
    })
    .await;

    with_user_id(UserId::new(OTHER_USER), async {
        let hits = pad
            .search(
                "c1",
                "deploy",
                vec![1.0, 0.0, 0.0],
                &current_model(),
                None,
                10,
            )
            .await
            .expect("cross-tenant search");
        assert!(
            hits.is_empty(),
            "a cross-tenant query must return nothing, through both arms"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance: `owner_todo` scoping still holds, so a subagent's notes stay
/// confined to its own subtree. Every note here carries the same vector, so the
/// vector arm would return all three if the snapshot predicate were missing
/// from it.
#[tokio::test]
async fn owner_todo_scoping_confines_a_subagent_to_its_own_subtree() {
    let Some(fx) = fixture("sp717i").await else {
        return;
    };
    let convs = PgConversationStore::new(fx.pool.clone());
    let pad = PgScratchpadStore::new(fx.pool.clone());
    let vector = vec![1.0, 0.0, 0.0];
    let model = current_model();

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
        write_embedded(
            &pad,
            "c1",
            "ctx",
            "the deploy pipeline is red",
            vector.clone(),
            &model,
        )
        .await;
        with_subagent_scope(sub_scope("1.2", "", &[]), async {
            write_embedded(
                &pad,
                "c1",
                "sib",
                "the deploy pipeline is red",
                vector.clone(),
                &model,
            )
            .await;
        })
        .await;

        let marker = uuid::Uuid::now_v7().to_string();

        with_subagent_scope(sub_scope("1.1", "", &[]), async {
            write_embedded(
                &pad,
                "c1",
                "own",
                "the deploy pipeline is red",
                vector.clone(),
                &model,
            )
            .await;
        })
        .await;

        let seen = with_subagent_scope(sub_scope("1.1", &marker, &[""]), async {
            pad.search("c1", "deploy", vector.clone(), &model, None, 50)
                .await
                .expect("scoped search")
        })
        .await;

        let keys = key_set(&seen);
        assert!(
            keys.contains("ctx"),
            "ancestor pre-marker context is visible"
        );
        assert!(keys.contains("own"), "own namespace is visible at any id");
        assert!(
            !keys.contains("sib"),
            "a concurrent sibling's notes must stay invisible through both arms"
        );
    })
    .await;

    fx.cleanup().await;
}
