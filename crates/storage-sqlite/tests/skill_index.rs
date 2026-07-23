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
        present_on_disk: true,
        last_seen_at: None,
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
async fn skill_removed_from_disk_survives_reindex() {
    // The catalog is cumulative: the database is the authoritative copy, not a
    // shadow of the last scan. A skill vanishing from disk keeps its body (the
    // procedure is still good) and is marked not-present (its attachments and
    // disk_path no longer resolve).
    let s = store().await;
    s.reindex_global(vec![skill("a", "first", "x"), skill("b", "second", "y")])
        .await
        .unwrap();
    s.reindex_global(vec![skill("a", "first", "x")])
        .await
        .unwrap();

    let survivor = s
        .get("b", None)
        .await
        .expect("get b")
        .expect("a skill absent from disk is retained, not deleted");
    assert_eq!(survivor.body, "y", "its body is still readable from the DB");
    assert!(
        !survivor.present_on_disk,
        "but it is flagged as no longer present on disk"
    );
    assert!(
        s.get("a", None)
            .await
            .unwrap()
            .expect("a is still indexed")
            .present_on_disk,
        "the skill the scan did see stays present"
    );
}

#[tokio::test]
async fn empty_scan_preserves_the_catalog() {
    // The unhappy path that motivated this: a root that is momentarily
    // unreadable must never be able to empty the catalog.
    let s = store().await;
    s.reindex_global(vec![skill("a", "x", "y")]).await.unwrap();
    s.reindex_global(vec![]).await.unwrap();

    let rows = s.list(None).await.unwrap();
    assert_eq!(rows.len(), 1, "an empty scan deletes nothing");
    assert!(!rows[0].present_on_disk, "everything is marked absent");
}

#[tokio::test]
async fn rescan_restores_presence_when_skill_returns() {
    let s = store().await;
    s.reindex_global(vec![skill("a", "x", "y")]).await.unwrap();
    s.reindex_global(vec![]).await.unwrap();
    s.reindex_global(vec![skill("a", "x", "y")]).await.unwrap();

    assert!(
        s.get("a", None).await.unwrap().unwrap().present_on_disk,
        "a returning skill is present again"
    );
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
async fn reindex_for_owner_leaves_other_scopes_untouched() {
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

    // Rescan alice with a different skill: hers accumulate, and no other scope
    // is touched -- including its presence flag.
    s.reindex_for_owner("alice", vec![owned("new", "alice", "a2")])
        .await
        .unwrap();

    let old = s
        .get("old", Some("alice"))
        .await
        .unwrap()
        .expect("alice's earlier skill is retained");
    assert!(!old.present_on_disk, "but flagged absent from her scan");
    assert!(s.get("new", Some("alice")).await.unwrap().is_some());

    let global = s.get("shared", None).await.unwrap().expect("global intact");
    assert!(
        global.present_on_disk,
        "an owner scan must not mark global skills absent"
    );
    let bob = s
        .get("bob-only", Some("bob"))
        .await
        .unwrap()
        .expect("bob intact");
    assert!(
        bob.present_on_disk,
        "nor another owner's -- presence is per-scope"
    );
}
