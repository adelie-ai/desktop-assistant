//! Reconciling an on-disk skill scan against the catalog (#639).
//!
//! The catalog is **cumulative**: it is the authoritative copy of a skill, not a
//! shadow of the last scan. Skills accrete, so Adele gets better over time, and
//! a skill vanishing from disk is never a reason to forget the procedure. What
//! absence changes is what still *works* -- the body reads fine, but the skill's
//! `disk_path` and attachments no longer resolve, so its bundled scripts cannot
//! be run. That is what [`IndexedSkill::present_on_disk`] records.
//!
//! Removal is therefore an explicit act, never inferred from a scan. A root that
//! is momentarily unreadable, a home directory that belongs to a client which
//! happens to be offline, a partial scan that only reached two of three roots --
//! none of those may delete anything.
//!
//! **Why this lives in `core` rather than in each adapter.** Reconciling is
//! policy. When it was expressed as `reindex_global` / `reindex_for_owner` on
//! the store port, each adapter re-implemented that policy in its own SQL and
//! the two drifted apart: Postgres pruned by name-list, SQLite deleted the
//! scope wholesale, and identical inputs produced different catalogs depending
//! only on which store was configured. Here there is one implementation and the
//! port keeps only primitives, so there is nothing left to diverge on.
//!
//! **Why it does not need a transaction.** It used to: a half-finished pass
//! could delete skills. Nothing is deleted now, so the worst a partial pass can
//! leave behind is a stale presence flag, which the next scan corrects. The pass
//! is idempotent -- running it twice over the same scan changes nothing after
//! the first -- so retries and reboots are safe by construction.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};

use crate::CoreError;
use crate::domain::{IndexedSkill, SkillScope};
use crate::ports::skill_index::SkillIndexStore;

/// What a reconcile pass did, for logging and for the caller's own reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileOutcome {
    /// Skills the scan saw and wrote (inserted or updated).
    pub upserted: usize,
    /// Previously-present skills the scan did not see, now flagged absent.
    pub marked_absent: usize,
    /// Skills that were flagged absent and are back on disk.
    pub restored: usize,
}

/// Reconcile one scan of one scope against the catalog.
///
/// Upserts everything `scanned` contains, stamping `now` as the moment each was
/// last seen, then flags every other skill *in that scope* as no longer present
/// on disk. Nothing is deleted, and no other scope is touched: a global scan
/// cannot affect a user's skills, and one user's scan cannot affect another's.
///
/// `scope` is authoritative over the scanned skills' own `owner_user_id` -- each
/// is stamped to match before it is written, so a caller cannot accidentally
/// write into a scope it is not reconciling (which would leave the skill outside
/// the presence sweep that just ran).
///
/// `now` is injected rather than read from the clock so tests are deterministic,
/// following [`crate::clock`]'s convention.
pub async fn reconcile_scan(
    store: &dyn SkillIndexStore,
    scope: &SkillScope,
    scanned: Vec<IndexedSkill>,
    now: DateTime<Utc>,
) -> Result<ReconcileOutcome, CoreError> {
    // Snapshot first: "what did this scan not see?" has to be answered against
    // the catalog as it stood before the pass wrote anything.
    let known = store.list_scope(scope).await?;

    // name -> was it flagged absent before this pass?
    let was_absent: HashMap<&str, bool> = known
        .iter()
        .map(|k| (k.name.as_str(), !k.present_on_disk))
        .collect();

    let mut outcome = ReconcileOutcome::default();
    let mut seen: HashSet<String> = HashSet::with_capacity(scanned.len());

    for mut skill in scanned {
        skill.owner_user_id = scope.owner().map(str::to_string);
        // Count a return once, even if two roots both offer this name (the
        // scanner resolves such collisions, but the count must not depend on
        // that).
        if !seen.contains(&skill.name) && was_absent.get(skill.name.as_str()) == Some(&true) {
            outcome.restored += 1;
        }
        store.upsert(&skill, now).await?;
        seen.insert(skill.name);
        outcome.upserted += 1;
    }

    let absent: Vec<String> = known
        .iter()
        .filter(|k| !seen.contains(&k.name) && k.present_on_disk)
        .map(|k| k.name.clone())
        .collect();
    outcome.marked_absent = absent.len();
    if !absent.is_empty() {
        store.set_presence(scope, &absent, false).await?;
    }

    Ok(outcome)
}
