//! Contract tests for [`SqliteSkillIndexStore`] (#594, #639).
//!
//! Catalog semantics come from the shared `SkillIndexStore` contract in
//! `core::ports::skill_index::conformance`, run here against a fresh in-memory
//! database -- one test per case, so a failure names the broken guarantee and
//! not just this adapter. The tests below the contract block cover what is
//! genuinely local to SQLite: FTS5 search behavior (the Postgres adapter ranks
//! hybrid vector + full-text instead) and the text-JSON column encoding.
#![cfg(feature = "sqlite")]

use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, SkillScope, TrustTier};
use desktop_assistant_core::ports::skill_index::{SkillIndexStore, conformance};
use desktop_assistant_core::skill_catalog::reconcile_scan;
use desktop_assistant_storage_sqlite::{SqliteSkillIndexStore, create_memory_pool};

async fn store() -> SqliteSkillIndexStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteSkillIndexStore::new(pool)
}

/// One test per contract case, each against a fresh database.
macro_rules! conformance_tests {
    ($($case:ident),+ $(,)?) => {
        $(
            #[tokio::test]
            async fn $case() {
                conformance::$case(&store().await).await;
            }
        )+
    };
}

conformance_tests!(
    removed_skill_survives_reconcile,
    empty_scan_preserves_the_catalog,
    unseen_skill_keeps_its_last_seen_at,
    rescan_restores_presence_when_skill_returns,
    reconcile_leaves_other_scopes_untouched,
    absent_skills_are_still_searchable,
    reconcile_is_idempotent,
    upsert_ignores_caller_supplied_presence,
    get_is_scope_addressed,
    set_presence_tolerates_unknown_and_empty,
);

fn skill(name: &str, description: &str, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: description.to_string(),
        kind: if body.contains("## Steps") {
            SkillKind::Workflow
        } else {
            SkillKind::Skill
        },
        disk_path: format!("/usr/share/adelie/skills/{name}/SKILL.md"),
        owner_user_id: None,
        locality: Locality::Daemon,
        content_hash: "h".to_string(),
        trust_tier: TrustTier::Local,
        source: Some("system".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec!["scripts/run.sh".to_string()],
        body: body.to_string(),
        metadata: serde_json::json!({"author": "test"}),
        present_on_disk: true,
        last_seen_at: None,
    }
}

async fn seed(store: &SqliteSkillIndexStore, skills: Vec<IndexedSkill>) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        skills,
        conformance::first_scan_at(),
    )
    .await
    .expect("seed scan");
}

#[tokio::test]
async fn json_columns_and_kind_round_trip() {
    // tags/attachments/metadata are TEXT here (JSONB on Postgres), so the
    // encode/decode path is this adapter's own and worth pinning.
    let s = store().await;
    seed(
        &s,
        vec![
            skill("invoice-run", "generate monthly invoices", "prose"),
            skill("deploy-blog", "publish the blog", "## Steps\n1. go"),
        ],
    )
    .await;

    let got = s
        .get("invoice-run", None)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.description, "generate monthly invoices");
    assert_eq!(got.tags, vec!["ops"]);
    assert_eq!(got.attachments, vec!["scripts/run.sh"]);
    assert_eq!(got.source.as_deref(), Some("system"));
    assert_eq!(got.metadata["author"], "test");

    let workflow = s.get("deploy-blog", None).await.unwrap().unwrap();
    assert_eq!(workflow.kind, SkillKind::Workflow);

    assert_eq!(s.list(None).await.unwrap().len(), 2);
}

#[tokio::test]
async fn fts_search_finds_by_keyword_ignoring_embedding() {
    let s = store().await;
    seed(
        &s,
        vec![
            skill("invoice-run", "generate monthly invoices", "billing prose"),
            skill("deploy-blog", "publish the blog", "static site"),
        ],
    )
    .await;

    // A non-empty embedding is ignored on SQLite; FTS matches by keyword, and
    // porter stemming means "invoice" matches "invoices".
    let hits = s
        .search("invoice", vec![0.1, 0.2, 0.3], 10)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name, "invoice-run");

    // Matches on the body too, and respects the limit.
    let body_hit = s.search("billing", vec![], 10).await.unwrap();
    assert_eq!(body_hit.len(), 1);
    assert_eq!(body_hit[0].name, "invoice-run");
}

#[tokio::test]
async fn search_returns_empty_for_no_match_or_blank_query() {
    let s = store().await;
    seed(&s, vec![skill("a", "alpha", "body")]).await;
    assert!(
        s.search("zzzznotaword", vec![], 10)
            .await
            .unwrap()
            .is_empty()
    );
    // A query with no usable tokens must not issue an invalid MATCH.
    assert!(s.search("   ", vec![], 10).await.unwrap().is_empty());
    assert!(s.search("!!! @@@", vec![], 10).await.unwrap().is_empty());
}

#[tokio::test]
async fn upsert_keeps_the_fts_index_in_sync() {
    // The upsert path updates in place, so FTS sync rides on the AFTER UPDATE
    // trigger rather than insert/delete. A stale index would keep answering for
    // the old description.
    let s = store().await;
    seed(&s, vec![skill("runbook", "old description", "body")]).await;
    seed(&s, vec![skill("runbook", "renewed description", "body")]).await;

    assert!(
        s.search("renewed", vec![], 10)
            .await
            .unwrap()
            .iter()
            .any(|r| r.name == "runbook"),
        "the updated description is searchable"
    );
    assert!(
        s.search("old", vec![], 10).await.unwrap().is_empty(),
        "and the superseded one is gone from the index"
    );
}
