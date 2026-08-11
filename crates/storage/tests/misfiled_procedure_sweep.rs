//! The mis-filed-procedure sweep (#1175): the routines already written as facts.
//!
//! The store holds entries that are really procedures. They read as neither a
//! fact nor a method, they compete with real facts for the recall block's
//! attention budget, and nothing can follow them. This suite pins the
//! properties the sweep's safety rests on, and every one of them lives in a
//! `WHERE` clause, a conflict target or a write, so none is reachable without a
//! real database.
//!
//! - `the_sweep_proposes_an_unapproved_skill_and_leaves_the_entry_untouched`
//! - `an_entry_is_read_once_and_not_again_until_its_text_changes`
//! - `an_edited_entry_is_read_again`
//! - `the_sweep_does_not_mix_two_tenants_entries_or_their_proposals`
//! - `a_proposal_never_overwrites_a_skill_the_person_approved`
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test misfiled_procedure_sweep
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::{Arc, Mutex};

use desktop_assistant_core::domain::{
    IndexedSkill, KnowledgeEntry, Locality, SkillApproval, SkillKind, SkillScope, TrustTier,
};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use desktop_assistant_storage::dreaming::{DreamingLlmFn, run_misfiled_sweep_phase};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgSkillIndexStore, UserId, with_user_id};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

const ALICE: &str = "sweep-alice";
const BOB: &str = "sweep-bob";

/// A routine somebody wrote down as if it were a fact.
const A_MISFILED_PROCEDURE: &str =
    "To publish a crate: bump the version, tag the commit, then push the tag.";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("misfiled1175").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// The ids a sweep prompt showed, read back the way the model must read them:
/// the prompt heads each entry with `## <id>`.
fn entries_in_prompt(prompt: &str) -> Vec<String> {
    prompt
        .lines()
        .filter_map(|line| line.strip_prefix("## "))
        .map(|id| id.trim().to_string())
        .collect()
}

/// A dreaming LLM that calls every entry it is shown a method, and records the
/// prompts it was asked so a test can see which entries were read.
fn llm_calling_everything_a_method(prompts: Arc<Mutex<Vec<String>>>) -> DreamingLlmFn {
    Box::new(move |_system, user| {
        let ids = entries_in_prompt(&user);
        prompts.lock().expect("not poisoned").push(user);
        let skills: Vec<String> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                format!(
                    r#"{{"from_entry":{},"name":"publish-a-crate-{i}",
                       "description":"Cut a release and push it.",
                       "steps":[{{"goal":"bump","outcome":"the manifest moved"}},
                                {{"goal":"tag","outcome":"the tag points at the bump"}},
                                {{"goal":"push","outcome":"the registry has it"}}]}}"#,
                    serde_json::to_string(id).expect("an id serializes"),
                )
            })
            .collect();
        let response = format!(r#"{{"skills":[{}]}}"#, skills.join(","));
        Box::pin(async move { Ok(response) })
    })
}

/// A dreaming LLM that finds no method anywhere.
fn llm_finding_no_method(prompts: Arc<Mutex<Vec<String>>>) -> DreamingLlmFn {
    Box::new(move |_system, user| {
        prompts.lock().expect("not poisoned").push(user);
        Box::pin(async move { Ok(r#"{"skills":[]}"#.to_string()) })
    })
}

fn recorded(prompts: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    prompts.lock().expect("not poisoned").clone()
}

async fn write_entry(pool: &PgPool, user: &str, id: &str, content: &str) {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(
                id,
                content,
                vec!["instruction".to_string()],
            ))
            .await
            .unwrap_or_else(|e| panic!("write {id}: {e}"));
    })
    .await;
}

async fn read_entry(pool: &PgPool, user: &str, id: &str) -> Option<KnowledgeEntry> {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(user), async {
        store.get(id).await.expect("the read succeeds")
    })
    .await
}

async fn skills_of(pool: &PgPool, user: &str) -> Vec<IndexedSkill> {
    let store = PgSkillIndexStore::new(pool.clone());
    with_user_id(UserId::new(user), async {
        store.list(None).await.expect("the catalog lists")
    })
    .await
}

/// Acceptance (#1175): a procedure stored as a knowledge entry is proposed as
/// an unapproved skill, and the entry is left exactly as it was.
///
/// The proposing half is the point: the entry is the person's own writing, and
/// a background pass that rewrote it would be editing somebody's notes on a
/// guess, unattended and overnight.
#[tokio::test]
async fn the_sweep_proposes_an_unapproved_skill_and_leaves_the_entry_untouched() {
    let Some(fx) = fixture().await else { return };
    write_entry(&fx.pool, ALICE, "kb-publish", A_MISFILED_PROCEDURE).await;
    let before = read_entry(&fx.pool, ALICE, "kb-publish")
        .await
        .expect("the entry is there");

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let stats = run_misfiled_sweep_phase(
        &fx.pool,
        &llm_calling_everything_a_method(Arc::clone(&prompts)),
        &CancellationToken::new(),
    )
    .await
    .expect("the sweep runs");

    assert_eq!(stats.judged, 1);
    assert_eq!(stats.proposed, 1);

    let proposed = skills_of(&fx.pool, ALICE).await;
    assert_eq!(proposed.len(), 1, "one proposal: {proposed:?}");
    let skill = &proposed[0];
    assert!(!skill.is_approved(), "a proposal is not a decision");
    assert_eq!(
        skill.metadata["from_entry"], "kb-publish",
        "the proposal names the entry it says should not have been one"
    );
    assert_eq!(skill.source.as_deref(), Some("misfiled-knowledge"));

    let after = read_entry(&fx.pool, ALICE, "kb-publish")
        .await
        .expect("the entry is still there");
    assert_eq!(after.content, before.content, "the entry is not rewritten");
    assert_eq!(after.tags, before.tags);
    assert_eq!(after.updated_at, before.updated_at, "and not even touched");

    fx.cleanup().await;
}

/// An entry is read once. Without that the pass would spend a model call per
/// entry per cycle to re-derive an answer it already had, for as long as the
/// store existed.
#[tokio::test]
async fn an_entry_is_read_once_and_not_again_until_its_text_changes() {
    let Some(fx) = fixture().await else { return };
    write_entry(
        &fx.pool,
        ALICE,
        "kb-a-fact",
        "The deploy target is the lab cluster.",
    )
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..2 {
        run_misfiled_sweep_phase(
            &fx.pool,
            &llm_finding_no_method(Arc::clone(&prompts)),
            &CancellationToken::new(),
        )
        .await
        .expect("the sweep runs");
    }

    let calls = recorded(&prompts);
    assert_eq!(
        calls.len(),
        1,
        "an entry that read as a fact is not re-read: {calls:?}"
    );

    fx.cleanup().await;
}

/// An entry whose text changes is read again: the answer was about the words
/// that have now been replaced.
#[tokio::test]
async fn an_edited_entry_is_read_again() {
    let Some(fx) = fixture().await else { return };
    write_entry(&fx.pool, ALICE, "kb-edited", "A plain fact.").await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_misfiled_sweep_phase(
        &fx.pool,
        &llm_finding_no_method(Arc::clone(&prompts)),
        &CancellationToken::new(),
    )
    .await
    .expect("the first sweep runs");

    write_entry(&fx.pool, ALICE, "kb-edited", A_MISFILED_PROCEDURE).await;

    run_misfiled_sweep_phase(
        &fx.pool,
        &llm_finding_no_method(Arc::clone(&prompts)),
        &CancellationToken::new(),
    )
    .await
    .expect("the second sweep runs");

    assert_eq!(
        recorded(&prompts).len(),
        2,
        "the edit puts the entry back in the worklist"
    );

    fx.cleanup().await;
}

/// The sweep is a cross-user background pass, so each user's entries are read
/// under that user's own scope and land in that user's own catalog.
#[tokio::test]
async fn the_sweep_does_not_mix_two_tenants_entries_or_their_proposals() {
    let Some(fx) = fixture().await else { return };
    write_entry(&fx.pool, ALICE, "kb-alice", A_MISFILED_PROCEDURE).await;
    write_entry(&fx.pool, BOB, "kb-bob", A_MISFILED_PROCEDURE).await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    run_misfiled_sweep_phase(
        &fx.pool,
        &llm_calling_everything_a_method(Arc::clone(&prompts)),
        &CancellationToken::new(),
    )
    .await
    .expect("the sweep runs");

    for call in recorded(&prompts) {
        let ids = entries_in_prompt(&call);
        assert!(
            !(ids.contains(&"kb-alice".to_string()) && ids.contains(&"kb-bob".to_string())),
            "one call never shows two tenants' entries: {ids:?}"
        );
    }

    let alice = skills_of(&fx.pool, ALICE).await;
    assert_eq!(alice.len(), 1);
    assert_eq!(alice[0].metadata["from_entry"], "kb-alice");
    assert_eq!(alice[0].owner_user_id.as_deref(), Some(ALICE));

    let bob = skills_of(&fx.pool, BOB).await;
    assert_eq!(bob.len(), 1);
    assert_eq!(bob[0].metadata["from_entry"], "kb-bob");

    fx.cleanup().await;
}

/// A proposal never overwrites a skill the person owns.
///
/// `write_authored` replaces the body, relabels the provenance and marks the
/// row absent from disk, so a proposal that landed on an approved skill would
/// destroy work a person did. The guard is the same one the promotion tool
/// applies.
///
/// **The seed is an approved own-draft on purpose.** `is_own_draft` refuses on
/// three independent clauses - not approved, absent from disk, and a source in
/// the drafting set - so a fixture that fails more than one of them passes with
/// the approval clause deleted, and this test's name would then be describing
/// something it does not check. It is also the shape this ticket creates: a
/// draft the sweep itself wrote, still off disk, that a person has since
/// approved through `SetSkillApproval`.
#[tokio::test]
async fn a_proposal_never_overwrites_a_skill_the_person_approved() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    write_entry(&fx.pool, ALICE, "kb-publish", A_MISFILED_PROCEDURE).await;

    // The name the fake model proposes, already held by a skill this person
    // approved.
    let mine = IndexedSkill {
        name: "publish-a-crate-0".to_string(),
        description: "The one I already wrote.".to_string(),
        kind: SkillKind::Workflow,
        disk_path: "/skills/publish-a-crate-0/SKILL.md".to_string(),
        owner_user_id: Some(ALICE.to_string()),
        locality: Locality::Daemon,
        content_hash: "hash-mine".to_string(),
        trust_tier: TrustTier::Local,
        source: Some("self-authored".to_string()),
        tags: vec![],
        attachments: vec![],
        body: "# mine\n\nMy own steps.\n".to_string(),
        metadata: serde_json::json!({}),
        present_on_disk: false,
        last_seen_at: None,
        approved_at: None,
        approved_by: None,
    };
    with_user_id(UserId::new(ALICE), async {
        store.upsert(&mine, chrono::Utc::now()).await.expect("seed");
        store
            .set_approval(
                &SkillScope::Owner(ALICE.to_string()),
                &["publish-a-crate-0".to_string()],
                Some(SkillApproval {
                    at: chrono::Utc::now(),
                    by: Some(ALICE.to_string()),
                }),
            )
            .await
            .expect("approve");
    })
    .await;

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let stats = run_misfiled_sweep_phase(
        &fx.pool,
        &llm_calling_everything_a_method(Arc::clone(&prompts)),
        &CancellationToken::new(),
    )
    .await
    .expect("the sweep runs");

    assert_eq!(stats.judged, 1, "the entry was still read");
    assert_eq!(stats.proposed, 0, "and its proposal was refused a home");

    let held = skills_of(&fx.pool, ALICE).await;
    let kept = held
        .iter()
        .find(|s| s.name == "publish-a-crate-0")
        .expect("the person's own skill is still there");
    assert_eq!(kept.description, "The one I already wrote.");
    assert!(kept.is_approved(), "and still approved");
    assert!(kept.present_on_disk, "and still on disk");

    fx.cleanup().await;
}
