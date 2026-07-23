//! Skill index store port — outbound trait for the disk-sourced skill catalog
//! (#573).
//!
//! The catalog is **host-global**: global skills scanned from system roots have
//! no owner (`owner_user_id = None`), following the `tool_definitions`
//! precedent rather than the per-user RLS tables. User-scoped skills (a client
//! registers from a home directory) carry an owner and land via a later slice;
//! this port already models the owner so the two coexist behind one interface.
//!
//! Embeddings are a storage concern filled by a background backfill, not part of
//! this trait: [`SkillIndexStore::search`] takes a pre-computed query embedding
//! (empty ⇒ full-text only) exactly as the knowledge-base search does.

#[cfg(any(test, feature = "test-support"))]
pub mod conformance;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::CoreError;
use crate::domain::{IndexedSkill, SkillScope};

/// Boxed-closure boundary for skill search, wired by the daemon over a
/// [`SkillIndexStore`]. Args: `(query, query_embedding, limit)`.
pub type SkillSearchFn = Arc<
    dyn Fn(
            String,
            Vec<f32>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<IndexedSkill>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed-closure boundary for fetching one skill. Args: `(name, owner)`, where
/// `owner = None` addresses the global skill.
pub type SkillGetFn = Arc<
    dyn Fn(
            String,
            Option<String>,
        ) -> Pin<Box<dyn Future<Output = Result<Option<IndexedSkill>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Outbound port for persisting and searching the skill catalog.
///
/// Implementations exist for Postgres (hybrid vector + full-text) and SQLite
/// (full-text only until sqlite-vec lands). Both are held as
/// `Arc<dyn SkillIndexStore>`, so the trait is dyn-compatible (`#[async_trait]`).
///
/// **Primitives only.** Reconciling a scan against the catalog is policy --
/// what accretes, what is marked absent, what is never deleted -- and it lives
/// once in [`crate::skill_catalog::reconcile_scan`] rather than in each
/// adapter's SQL. That split is not cosmetic: when the two adapters each
/// implemented a `reindex_*` verb, they diverged (one pruned by name-list, the
/// other deleted the scope wholesale), and identical inputs produced different
/// catalogs depending only on which store was configured. Nothing here decides
/// what to keep; these methods just do as they are told.
///
/// The contract every implementation owes is executable, not prose:
/// [`conformance`] runs it against any store, and each adapter's test suite
/// invokes it.
#[async_trait::async_trait]
pub trait SkillIndexStore: Send + Sync {
    /// Insert or update one skill, keyed on `(name, owner_user_id)`.
    ///
    /// `seen_at` is the instant the scan that produced this skill ran: the store
    /// records it as `last_seen_at` and marks the row present. Presence is index
    /// state, so [`IndexedSkill::present_on_disk`] and
    /// [`IndexedSkill::last_seen_at`] on the *argument* are ignored -- a caller
    /// cannot write a row that claims to be absent while handing over content it
    /// just read off disk.
    ///
    /// An implementation that keeps derived data alongside a row (an embedding,
    /// say) must preserve it when `content_hash` is unchanged and invalidate it
    /// when it changes; nothing else about that data is part of this contract.
    async fn upsert(&self, skill: &IndexedSkill, seen_at: DateTime<Utc>) -> Result<(), CoreError>;

    /// Every skill in `scope`, present or absent, unfiltered by the calling
    /// user.
    ///
    /// Deliberately not scoped to the caller like [`Self::list`]: the reconcile
    /// pass runs at startup with no request context and has to see the whole
    /// partition it is about to update, or it would mark rows absent that it
    /// simply could not read. Callers are trusted, host-level code.
    async fn list_scope(&self, scope: &SkillScope) -> Result<Vec<IndexedSkill>, CoreError>;

    /// Set [`IndexedSkill::present_on_disk`] to `present` for the named skills
    /// within `scope`, leaving every other column -- `last_seen_at` included --
    /// untouched.
    ///
    /// Names not in the scope are ignored rather than being an error, so a
    /// concurrent removal cannot fail a reconcile. An empty `names` is a no-op.
    async fn set_presence(
        &self,
        scope: &SkillScope,
        names: &[String],
        present: bool,
    ) -> Result<(), CoreError>;

    /// Search the catalog, returning up to `limit` skills best matching
    /// `query`.
    ///
    /// `query_embedding` is a pre-computed vector for semantic ranking; an empty
    /// vector selects full-text-only search (used when the embedding backend is
    /// unavailable, and always on the SQLite adapter). Results are scoped to the
    /// caller: global skills plus the current user's own.
    async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError>;

    /// Fetch a single skill by name within a scope: `owner = None` addresses the
    /// global skill of that name, `owner = Some(user)` the user's own. Returns
    /// `Ok(None)` when absent.
    async fn get(&self, name: &str, owner: Option<&str>)
    -> Result<Option<IndexedSkill>, CoreError>;

    /// Enumerate catalog entries (global plus the current user's own), newest
    /// first, capped at `limit` when supplied. For browse/audit surfaces.
    async fn list(&self, limit: Option<u32>) -> Result<Vec<IndexedSkill>, CoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Minimal in-memory reference implementation of the port.
    ///
    /// It exists so the contract in [`conformance`] is proven against something
    /// with no database at all: the cases below run on every `cargo test`, even
    /// where `TEST_DATABASE_URL` is unset and the Postgres suite pass-skips.
    /// Search is a case-insensitive substring match -- the DB adapters add
    /// ranking, which is explicitly not part of the contract.
    #[derive(Default)]
    struct InMemorySkillIndex {
        rows: Mutex<Vec<IndexedSkill>>,
    }

    #[async_trait::async_trait]
    impl SkillIndexStore for InMemorySkillIndex {
        async fn upsert(
            &self,
            skill: &IndexedSkill,
            seen_at: DateTime<Utc>,
        ) -> Result<(), CoreError> {
            let mut stored = skill.clone();
            // Presence is index state: what the caller supplied is discarded.
            stored.present_on_disk = true;
            stored.last_seen_at = Some(seen_at);

            let mut rows = self.rows.lock().expect("lock");
            match rows
                .iter_mut()
                .find(|r| r.name == stored.name && r.owner_user_id == stored.owner_user_id)
            {
                Some(existing) => *existing = stored,
                None => rows.push(stored),
            }
            Ok(())
        }

        async fn list_scope(&self, scope: &SkillScope) -> Result<Vec<IndexedSkill>, CoreError> {
            let rows = self.rows.lock().expect("lock");
            Ok(rows
                .iter()
                .filter(|r| r.owner_user_id.as_deref() == scope.owner())
                .cloned()
                .collect())
        }

        async fn set_presence(
            &self,
            scope: &SkillScope,
            names: &[String],
            present: bool,
        ) -> Result<(), CoreError> {
            let mut rows = self.rows.lock().expect("lock");
            for row in rows.iter_mut() {
                if row.owner_user_id.as_deref() == scope.owner() && names.contains(&row.name) {
                    row.present_on_disk = present;
                }
            }
            Ok(())
        }

        async fn search(
            &self,
            query: &str,
            _query_embedding: Vec<f32>,
            limit: usize,
        ) -> Result<Vec<IndexedSkill>, CoreError> {
            let needle = query.to_lowercase();
            let rows = self.rows.lock().expect("lock");
            Ok(rows
                .iter()
                .filter(|r| {
                    r.name.to_lowercase().contains(&needle)
                        || r.description.to_lowercase().contains(&needle)
                })
                .take(limit)
                .cloned()
                .collect())
        }

        async fn get(
            &self,
            name: &str,
            owner: Option<&str>,
        ) -> Result<Option<IndexedSkill>, CoreError> {
            let rows = self.rows.lock().expect("lock");
            Ok(rows
                .iter()
                .find(|r| r.name == name && r.owner_user_id.as_deref() == owner)
                .cloned())
        }

        async fn list(&self, limit: Option<u32>) -> Result<Vec<IndexedSkill>, CoreError> {
            let rows = self.rows.lock().expect("lock");
            let take = limit.map(|l| l as usize).unwrap_or(rows.len());
            Ok(rows.iter().take(take).cloned().collect())
        }
    }

    /// One test per contract case, so a failure names the broken guarantee.
    macro_rules! conformance_tests {
        ($($case:ident),+ $(,)?) => {
            $(
                #[tokio::test]
                async fn $case() {
                    conformance::$case(&InMemorySkillIndex::default()).await;
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
}
