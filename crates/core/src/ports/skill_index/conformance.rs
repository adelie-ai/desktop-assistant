//! Executable contract for [`SkillIndexStore`] (#639).
//!
//! Behavior must not depend on which store is configured. A trait pins
//! signatures, not semantics, so the guarantee is enforced here instead: every
//! implementation runs the same cases. The Postgres adapter, the SQLite adapter
//! and the in-memory reference implementation each invoke them from their own
//! test suite, one test per case, so a failure names the broken guarantee
//! rather than a line number.
//!
//! Each case assumes an **empty store** and cleans up nothing -- give it a fresh
//! one (a per-test schema, a fresh in-memory pool). Cases take `&dyn
//! SkillIndexStore` so a caller can pass any store by reference.
//!
//! What is deliberately *not* covered: ranking (Postgres searches hybrid
//! vector + full-text, SQLite full-text only) and derived storage-only data
//! (embedding retention across an unchanged-hash rescan). Those are adapter
//! properties with adapter-specific tests; everything about the catalog's
//! *contract* is here.

use chrono::{DateTime, TimeZone, Utc};

use super::SkillIndexStore;
use crate::domain::{IndexedSkill, Locality, SkillKind, SkillScope, TrustTier};
use crate::skill_catalog::reconcile_scan;

/// A fixed instant for the first pass; cases that need a later one use
/// [`later`]. Deterministic so `last_seen_at` assertions are exact.
pub fn first_scan_at() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
        .single()
        .expect("a valid, unambiguous UTC instant")
}

/// A second fixed instant, one hour after [`first_scan_at`].
pub fn later() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 13, 0, 0)
        .single()
        .expect("a valid, unambiguous UTC instant")
}

/// Build a scan-shaped skill. `owner` selects the scope: `None` is global.
pub fn sample_skill(name: &str, owner: Option<&str>, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: format!("{name} description"),
        kind: SkillKind::Skill,
        disk_path: format!("/skills/{name}/SKILL.md"),
        owner_user_id: owner.map(str::to_string),
        locality: if owner.is_some() {
            Locality::Client
        } else {
            Locality::Daemon
        },
        content_hash: format!("hash-{name}"),
        trust_tier: TrustTier::Local,
        source: Some("conformance".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec!["scripts/run.sh".to_string()],
        body: body.to_string(),
        metadata: serde_json::json!({"author": "conformance"}),
        present_on_disk: true,
        last_seen_at: None,
    }
}

async fn fetch(store: &dyn SkillIndexStore, name: &str, owner: Option<&str>) -> IndexedSkill {
    store
        .get(name, owner)
        .await
        .expect("get must not error")
        .unwrap_or_else(|| panic!("expected {name} to be in the catalog"))
}

/// A skill the scan no longer sees is retained with its content intact, and
/// flagged absent rather than deleted.
pub async fn removed_skill_survives_reconcile(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![
            sample_skill("stays", None, "first body"),
            sample_skill("vanishes", None, "second body"),
        ],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("stays", None, "first body")],
        later(),
    )
    .await
    .expect("second scan");

    let gone = fetch(store, "vanishes", None).await;
    assert_eq!(
        gone.body, "second body",
        "the procedure is still readable from the catalog"
    );
    assert_eq!(
        gone.attachments,
        vec!["scripts/run.sh".to_string()],
        "and its metadata is intact -- only its availability changed"
    );
    assert!(
        !gone.present_on_disk,
        "a skill absent from the scan is flagged, not deleted"
    );
    assert!(
        fetch(store, "stays", None).await.present_on_disk,
        "the skill the scan did see stays present"
    );
}

/// The unhappy path that motivates the whole design: a scope whose roots are
/// momentarily unreadable scans as empty, and that must delete nothing.
pub async fn empty_scan_preserves_the_catalog(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    let outcome = reconcile_scan(store, &SkillScope::Global, vec![], later())
        .await
        .expect("empty scan");

    assert_eq!(outcome.upserted, 0);
    assert_eq!(outcome.marked_absent, 1);

    let rows = store
        .list_scope(&SkillScope::Global)
        .await
        .expect("list_scope");
    assert_eq!(rows.len(), 1, "an empty scan deletes nothing");
    assert!(!rows[0].present_on_disk, "everything is marked absent");
}

/// Marking a skill absent must not disturb when it was last seen -- that is the
/// record of when the procedure was last known good on disk.
pub async fn unseen_skill_keeps_its_last_seen_at(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    let seen_at = fetch(store, "alpha", None).await.last_seen_at;
    assert_eq!(
        seen_at,
        Some(first_scan_at()),
        "upsert stamps the scan instant"
    );

    reconcile_scan(store, &SkillScope::Global, vec![], later())
        .await
        .expect("empty scan");

    let after = fetch(store, "alpha", None).await;
    assert!(!after.present_on_disk);
    assert_eq!(
        after.last_seen_at, seen_at,
        "the absent row still records when it was last on disk"
    );
}

/// A skill that comes back is present again, and its last-seen advances.
pub async fn rescan_restores_presence_when_skill_returns(store: &dyn SkillIndexStore) {
    let scan = || vec![sample_skill("alpha", None, "body")];
    reconcile_scan(store, &SkillScope::Global, scan(), first_scan_at())
        .await
        .expect("first scan");
    reconcile_scan(store, &SkillScope::Global, vec![], first_scan_at())
        .await
        .expect("empty scan");

    let outcome = reconcile_scan(store, &SkillScope::Global, scan(), later())
        .await
        .expect("third scan");
    assert_eq!(outcome.restored, 1, "the return is reported");

    let back = fetch(store, "alpha", None).await;
    assert!(back.present_on_disk, "a returning skill is present again");
    assert_eq!(back.last_seen_at, Some(later()), "and freshly seen");
}

/// Presence is per-scope: reconciling one owner must not touch global skills or
/// another owner's, present or absent.
pub async fn reconcile_leaves_other_scopes_untouched(store: &dyn SkillIndexStore) {
    let alice = SkillScope::Owner("alice".to_string());
    let bob = SkillScope::Owner("bob".to_string());

    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("shared", None, "global body")],
        first_scan_at(),
    )
    .await
    .expect("global scan");
    reconcile_scan(
        store,
        &alice,
        vec![sample_skill("alice-old", Some("alice"), "a1")],
        first_scan_at(),
    )
    .await
    .expect("alice scan");
    reconcile_scan(
        store,
        &bob,
        vec![sample_skill("bob-only", Some("bob"), "b1")],
        first_scan_at(),
    )
    .await
    .expect("bob scan");

    // Alice rescans with a different skill: hers accumulate, nobody else moves.
    reconcile_scan(
        store,
        &alice,
        vec![sample_skill("alice-new", Some("alice"), "a2")],
        later(),
    )
    .await
    .expect("alice rescan");

    let old = fetch(store, "alice-old", Some("alice")).await;
    assert!(
        !old.present_on_disk,
        "alice's earlier skill is retained and flagged"
    );
    assert!(
        fetch(store, "alice-new", Some("alice"))
            .await
            .present_on_disk
    );
    assert!(
        fetch(store, "shared", None).await.present_on_disk,
        "an owner scan must not mark global skills absent"
    );
    assert!(
        fetch(store, "bob-only", Some("bob")).await.present_on_disk,
        "nor another owner's"
    );
}

/// An absent skill stays discoverable. Hiding it would quietly recreate the
/// deletion behavior the cumulative catalog exists to remove; the flag is how a
/// caller learns the scripts are gone.
pub async fn absent_skills_are_still_searchable(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("invoices", None, "reconcile the ledger")],
        first_scan_at(),
    )
    .await
    .expect("first scan");
    reconcile_scan(store, &SkillScope::Global, vec![], later())
        .await
        .expect("empty scan");

    let hits = store
        .search("invoices", Vec::new(), 10)
        .await
        .expect("search");
    let hit = hits
        .iter()
        .find(|s| s.name == "invoices")
        .expect("an absent skill is still returned by search");
    assert!(!hit.present_on_disk, "flagged so the caller can tell");
}

/// Re-running the same scan changes nothing after the first pass.
pub async fn reconcile_is_idempotent(store: &dyn SkillIndexStore) {
    let scan = || {
        vec![
            sample_skill("alpha", None, "a"),
            sample_skill("beta", None, "b"),
        ]
    };
    reconcile_scan(store, &SkillScope::Global, scan(), first_scan_at())
        .await
        .expect("first pass");
    let before = store.list_scope(&SkillScope::Global).await.expect("list");

    let outcome = reconcile_scan(store, &SkillScope::Global, scan(), first_scan_at())
        .await
        .expect("second pass");

    assert_eq!(outcome.marked_absent, 0, "nothing goes absent on a repeat");
    assert_eq!(outcome.restored, 0, "and nothing is 'restored'");

    let after = store.list_scope(&SkillScope::Global).await.expect("list");
    assert_eq!(before.len(), after.len(), "no rows added or removed");
    for row in &after {
        assert!(row.present_on_disk);
        assert_eq!(row.last_seen_at, Some(first_scan_at()));
    }
}

/// Presence is index state. A caller handing over content it just read off disk
/// cannot also declare that content missing -- otherwise a buggy or hostile
/// scanner could mark the catalog absent while writing to it.
pub async fn upsert_ignores_caller_supplied_presence(store: &dyn SkillIndexStore) {
    let mut lying = sample_skill("alpha", None, "body");
    lying.present_on_disk = false;
    lying.last_seen_at = None;

    store
        .upsert(&lying, first_scan_at())
        .await
        .expect("upsert must not error");

    let stored = fetch(store, "alpha", None).await;
    assert!(
        stored.present_on_disk,
        "the store records what the scan proves, not what the caller claims"
    );
    assert_eq!(
        stored.last_seen_at,
        Some(first_scan_at()),
        "and stamps the scan instant it was given"
    );
}

/// `get` addresses one scope: the global skill and a user's skill of the same
/// name are different rows, and neither answers for the other.
pub async fn get_is_scope_addressed(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("deploy", None, "the global one")],
        first_scan_at(),
    )
    .await
    .expect("global scan");
    reconcile_scan(
        store,
        &SkillScope::Owner("alice".to_string()),
        vec![sample_skill("deploy", Some("alice"), "alice's own")],
        first_scan_at(),
    )
    .await
    .expect("alice scan");

    assert_eq!(fetch(store, "deploy", None).await.body, "the global one");
    assert_eq!(
        fetch(store, "deploy", Some("alice")).await.body,
        "alice's own"
    );
    assert!(
        store
            .get("deploy", Some("nobody"))
            .await
            .expect("get")
            .is_none(),
        "an unknown owner matches nothing"
    );
}

/// `set_presence` tolerates names that aren't in the scope, so a concurrent
/// removal can't fail a reconcile, and an empty name list is a no-op.
pub async fn set_presence_tolerates_unknown_and_empty(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    store
        .set_presence(&SkillScope::Global, &[], false)
        .await
        .expect("an empty name list is a no-op, not an error");
    store
        .set_presence(&SkillScope::Global, &["ghost".to_string()], false)
        .await
        .expect("an unknown name is ignored, not an error");

    assert!(
        fetch(store, "alpha", None).await.present_on_disk,
        "neither call touched an unrelated row"
    );
}
