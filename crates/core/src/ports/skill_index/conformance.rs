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
use crate::domain::{IndexedSkill, Locality, SkillApproval, SkillKind, SkillScope, TrustTier};
use crate::ports::auth::{UserId, with_user_id};
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
        // A scan stamps approval when it inserts (see `reconcile_scan`); the
        // scan-shaped sample carries none of its own.
        approved_at: None,
        approved_by: None,
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
/// another owner's, present or absent. The owner-scoped reads below run as
/// that owner via [`with_user_id`] -- `get` resolves "the caller's own" from
/// the caller's real identity (#911), not from the `owner` argument, so a
/// read of alice's or bob's row has to happen inside their own scope.
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

    let (old, new_present) = with_user_id(UserId::new("alice"), async {
        let old = fetch(store, "alice-old", Some("alice")).await;
        let new_present = fetch(store, "alice-new", Some("alice"))
            .await
            .present_on_disk;
        (old, new_present)
    })
    .await;
    assert!(
        !old.present_on_disk,
        "alice's earlier skill is retained and flagged"
    );
    assert!(new_present);

    assert!(
        fetch(store, "shared", None).await.present_on_disk,
        "an owner scan must not mark global skills absent"
    );

    let bob_present = with_user_id(UserId::new("bob"), async {
        fetch(store, "bob-only", Some("bob")).await.present_on_disk
    })
    .await;
    assert!(bob_present, "nor another owner's");
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
        .search("invoices", Vec::new(), "test-model", 10)
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
/// name are different rows, and neither answers for the other. Reading the
/// owner-scoped row happens as alice, via [`with_user_id`] -- `get` resolves
/// "the caller's own" from the caller's real identity, never from the
/// `owner` argument's string value (#911), so exercising it as anyone else
/// would not prove this case.
///
/// The `bob names alice` case below is the one that actually distinguishes a
/// compliant implementation from a pre-#911 one. Passing `Some("alice")`
/// while installed as alice (the case above) proves nothing on its own: an
/// implementation that still trusts the literal argument passes it
/// identically to one that correctly consults the caller's real identity,
/// because here the two happen to agree. Installing a *different* identity
/// (bob, who has no "deploy" of his own) and naming a real, seeded owner
/// (alice) as the argument is what forces disagreement: a compliant store
/// resolves bob's own (empty) scope and returns nothing, while a store that
/// still binds the argument literally returns alice's row -- exactly the
/// leak #911 fixed. Any adapter's `get` must fail this specific case before
/// its fix, or the case is not exercising the boundary at all.
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

    let alice_own = with_user_id(UserId::new("alice"), fetch(store, "deploy", Some("alice"))).await;
    assert_eq!(alice_own.body, "alice's own");

    // The discriminating case (see the doc comment above): bob has no
    // "deploy" of his own, but names alice -- who genuinely has one -- as
    // the `owner` argument.
    let bob_naming_alice = with_user_id(UserId::new("bob"), async {
        store.get("deploy", Some("alice")).await.expect("get")
    })
    .await;
    assert!(
        bob_naming_alice.is_none(),
        "an owner argument naming a different, real, seeded owner must not surface that \
         owner's row -- the caller's real identity decides scope, not the argument"
    );

    assert!(
        store
            .get("deploy", Some("nobody"))
            .await
            .expect("get")
            .is_none(),
        "a caller with no matching scope gets nothing back"
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

// --- the approval axis (#1155) ----------------------------------------------

/// Build a skill the assistant authored from a completed plan: no file on
/// disk, no attachments, and locally authored provenance.
pub fn authored_skill(name: &str, owner: Option<&str>, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: format!("{name} description"),
        kind: SkillKind::Workflow,
        disk_path: String::new(),
        owner_user_id: owner.map(str::to_string),
        locality: Locality::Daemon,
        content_hash: format!("hash-{name}"),
        trust_tier: TrustTier::Local,
        source: Some("self-authored".to_string()),
        tags: Vec::new(),
        attachments: Vec::new(),
        body: body.to_string(),
        metadata: serde_json::json!({}),
        present_on_disk: true,
        last_seen_at: None,
        approved_at: Some(first_scan_at()),
        approved_by: Some("a-liar".to_string()),
    }
}

/// A skill Adele writes for herself is unapproved, and stays unapproved until
/// somebody approves it -- on an axis distinct from `trust_tier`.
///
/// This is the whole point of the second column. The authored skill below is
/// `TrustTier::Local`, the most trusted provenance the catalog has, because
/// that is exactly what it is: authored locally. Provenance says nothing about
/// consent, so it must not be followable yet.
pub async fn authored_skill_is_unapproved_until_approved(store: &dyn SkillIndexStore) {
    let skill = authored_skill("promoted", None, "## Steps\n1. do it\n");
    store
        .write_authored(&skill, first_scan_at())
        .await
        .expect("write_authored must not error");

    let stored = fetch(store, "promoted", None).await;
    assert_eq!(
        stored.trust_tier,
        TrustTier::Local,
        "authored locally is the provenance, and it is the most trusted one"
    );
    assert!(
        !stored.is_approved(),
        "and it must still not be followable: the argument's approval is not honoured"
    );
    assert_eq!(stored.approved_by, None, "nor its claimed approver");
    assert!(
        !stored.present_on_disk,
        "nothing was read off disk, so the row does not claim to be there"
    );

    store
        .set_approval(
            &SkillScope::Global,
            &["promoted".to_string()],
            Some(SkillApproval {
                at: later(),
                by: Some("the-user".to_string()),
            }),
        )
        .await
        .expect("set_approval must not error");

    let approved = fetch(store, "promoted", None).await;
    assert!(approved.is_approved(), "approval is recorded");
    assert_eq!(approved.approved_at, Some(later()));
    assert_eq!(approved.approved_by.as_deref(), Some("the-user"));
    assert_eq!(
        approved.body, "## Steps\n1. do it\n",
        "and approving changed nothing else"
    );
}

/// Approval is preserved across a rescan. A scan re-reads a file; it does not
/// re-decide whether a person consented to it.
pub async fn upsert_preserves_approval_across_a_rescan(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "first body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    store
        .set_approval(&SkillScope::Global, &["alpha".to_string()], None)
        .await
        .expect("withdraw approval");
    assert!(!fetch(store, "alpha", None).await.is_approved());

    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "second body")],
        later(),
    )
    .await
    .expect("second scan");

    let after = fetch(store, "alpha", None).await;
    assert_eq!(after.body, "second body", "the rescan updated the content");
    assert!(
        !after.is_approved(),
        "but it did not silently re-approve a skill a person had unapproved"
    );
}

/// A scan records approval on the rows it creates: putting a file in a skill
/// root is a deliberate human act, so a skill that arrives that way is
/// approved, and only that path may say so.
pub async fn a_scanned_skill_arrives_approved(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    let stored = fetch(store, "alpha", None).await;
    assert!(
        stored.is_approved(),
        "a file a person placed in a skill root is approved by that act"
    );
    assert_eq!(stored.approved_at, Some(first_scan_at()));
}

/// Amending an approved skill through the authored path drops its approval:
/// the approval was of the old body, and nobody has seen the new one.
pub async fn amending_an_approved_skill_withdraws_its_approval(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "the reviewed body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");
    assert!(fetch(store, "alpha", None).await.is_approved());

    let mut revised = authored_skill("alpha", None, "## Steps\n1. something new\n");
    revised.content_hash = "hash-revised".to_string();
    store
        .write_authored(&revised, later())
        .await
        .expect("amend");

    let after = fetch(store, "alpha", None).await;
    assert_eq!(after.body, "## Steps\n1. something new\n");
    assert!(
        !after.is_approved(),
        "new content must not wear the old content's approval"
    );
}

/// `set_approval` tolerates names that aren't in the scope and an empty name
/// list, matching `set_presence`.
pub async fn set_approval_tolerates_unknown_and_empty(store: &dyn SkillIndexStore) {
    reconcile_scan(
        store,
        &SkillScope::Global,
        vec![sample_skill("alpha", None, "body")],
        first_scan_at(),
    )
    .await
    .expect("first scan");

    store
        .set_approval(&SkillScope::Global, &[], None)
        .await
        .expect("an empty name list is a no-op, not an error");
    store
        .set_approval(&SkillScope::Global, &["ghost".to_string()], None)
        .await
        .expect("an unknown name is ignored, not an error");

    assert!(
        fetch(store, "alpha", None).await.is_approved(),
        "neither call touched an unrelated row"
    );
}
