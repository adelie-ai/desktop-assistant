//! Contract tests for [`SqliteSkillIndexStore`] (#594) — the FTS5-backed SQLite
//! mirror of the Postgres skill index. Verifies reindex upsert/prune, full-text
//! search (ignoring the query embedding), owner-scoped `get`, and `list`.
#![cfg(feature = "sqlite")]

use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use desktop_assistant_storage_sqlite::{SqliteSkillIndexStore, create_memory_pool};

async fn store() -> SqliteSkillIndexStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteSkillIndexStore::new(pool)
}

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
    }
}

#[tokio::test]
async fn reindex_get_and_list_roundtrip() {
    let s = store().await;
    s.reindex_global(vec![
        skill("invoice-run", "generate monthly invoices", "prose"),
        skill("deploy-blog", "publish the blog", "## Steps\n1. go"),
    ])
    .await
    .expect("reindex");

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
async fn reindex_prunes_removed_skills() {
    let s = store().await;
    s.reindex_global(vec![skill("a", "first", "x"), skill("b", "second", "y")])
        .await
        .unwrap();
    s.reindex_global(vec![skill("a", "first", "x")])
        .await
        .unwrap();
    assert!(s.get("a", None).await.unwrap().is_some());
    assert!(s.get("b", None).await.unwrap().is_none());
}

#[tokio::test]
async fn empty_reindex_clears_catalog() {
    let s = store().await;
    s.reindex_global(vec![skill("a", "x", "y")]).await.unwrap();
    s.reindex_global(vec![]).await.unwrap();
    assert!(s.list(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn fts_search_finds_by_keyword_ignoring_embedding() {
    let s = store().await;
    s.reindex_global(vec![
        skill("invoice-run", "generate monthly invoices", "billing prose"),
        skill("deploy-blog", "publish the blog", "static site"),
    ])
    .await
    .unwrap();

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
    s.reindex_global(vec![skill("a", "alpha", "body")])
        .await
        .unwrap();
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
async fn get_is_owner_scoped() {
    let s = store().await;
    s.reindex_global(vec![skill("deploy", "global", "x")])
        .await
        .unwrap();
    // The global skill is addressed by owner = None; a user-scoped get misses.
    assert!(s.get("deploy", None).await.unwrap().is_some());
    assert!(s.get("deploy", Some("nobody")).await.unwrap().is_none());
}

fn owned(name: &str, owner: &str, description: &str) -> IndexedSkill {
    let mut s = skill(name, description, "prose");
    s.owner_user_id = Some(owner.to_string());
    s.locality = Locality::Client;
    s
}

#[tokio::test]
async fn reindex_for_owner_replaces_only_that_owner() {
    let s = store().await;
    s.reindex_global(vec![skill("shared", "global", "x")])
        .await
        .unwrap();
    s.reindex_for_owner("alice", vec![owned("old", "alice", "a1")])
        .await
        .unwrap();
    s.reindex_for_owner("bob", vec![owned("bob-only", "bob", "b1")])
        .await
        .unwrap();

    // Rescan alice: her old row is replaced; global and bob's are untouched.
    s.reindex_for_owner("alice", vec![owned("new", "alice", "a2")])
        .await
        .unwrap();

    assert!(s.get("old", Some("alice")).await.unwrap().is_none());
    assert!(s.get("new", Some("alice")).await.unwrap().is_some());
    assert!(s.get("shared", None).await.unwrap().is_some());
    assert!(s.get("bob-only", Some("bob")).await.unwrap().is_some());
}
