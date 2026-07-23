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

use crate::CoreError;
use crate::domain::IndexedSkill;

/// Outbound port for persisting and searching the on-disk skill catalog.
///
/// Implementations exist for Postgres (hybrid vector + full-text) and SQLite
/// (full-text only until sqlite-vec lands). Both are held as
/// `Arc<dyn SkillIndexStore>`, so the trait is dyn-compatible (`#[async_trait]`).
#[async_trait::async_trait]
pub trait SkillIndexStore: Send + Sync {
    /// Atomically replace the entire set of **global** (owner-less) skills with
    /// the supplied scan output, in one transaction.
    ///
    /// User-scoped rows (those with an `owner_user_id`) are left untouched, so a
    /// startup rescan of the system roots never disturbs a client's registered
    /// skills. New rows are written with a NULL embedding for the backfill to
    /// fill later.
    async fn reindex_global(&self, skills: Vec<IndexedSkill>) -> Result<(), CoreError>;

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
    use crate::domain::{Locality, SkillKind, TrustTier};
    use std::sync::Mutex;

    /// Minimal in-memory reference implementation: proves the trait is usable
    /// and pins the owner-scoping / reindex-replace semantics the real adapters
    /// must honor. Search is a case-insensitive substring match (the DB adapters
    /// add ranking; the contract here is "returns matches, respects scope").
    #[derive(Default)]
    struct InMemorySkillIndex {
        rows: Mutex<Vec<IndexedSkill>>,
    }

    fn skill(name: &str, owner: Option<&str>, description: &str) -> IndexedSkill {
        IndexedSkill {
            name: name.to_string(),
            description: description.to_string(),
            kind: SkillKind::Skill,
            disk_path: format!("/skills/{name}/SKILL.md"),
            owner_user_id: owner.map(str::to_string),
            locality: if owner.is_some() {
                Locality::Client
            } else {
                Locality::Daemon
            },
            content_hash: "hash".to_string(),
            trust_tier: TrustTier::Local,
            source: None,
            tags: vec![],
            attachments: vec![],
            body: String::new(),
            metadata: serde_json::Value::Null,
        }
    }

    #[async_trait::async_trait]
    impl SkillIndexStore for InMemorySkillIndex {
        async fn reindex_global(&self, skills: Vec<IndexedSkill>) -> Result<(), CoreError> {
            let mut rows = self.rows.lock().expect("lock");
            rows.retain(|r| r.owner_user_id.is_some()); // keep user-scoped rows
            rows.extend(skills);
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

    #[tokio::test]
    async fn reindex_global_replaces_global_but_keeps_user_rows() {
        let store = InMemorySkillIndex::default();
        // Seed a user-scoped row plus an initial global scan.
        store
            .reindex_global(vec![skill("old-global", None, "first")])
            .await
            .unwrap();
        store
            .rows
            .lock()
            .unwrap()
            .push(skill("mine", Some("user-a"), "user skill"));

        // A rescan replaces the global set but must not drop the user row.
        store
            .reindex_global(vec![skill("new-global", None, "second")])
            .await
            .unwrap();

        assert!(store.get("old-global", None).await.unwrap().is_none());
        assert!(store.get("new-global", None).await.unwrap().is_some());
        assert!(store.get("mine", Some("user-a")).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn get_is_owner_scoped() {
        let store = InMemorySkillIndex::default();
        store.rows.lock().unwrap().extend([
            skill("deploy", None, "global"),
            skill("deploy", Some("u1"), "mine"),
        ]);

        let global = store.get("deploy", None).await.unwrap().unwrap();
        assert_eq!(global.description, "global");
        let owned = store.get("deploy", Some("u1")).await.unwrap().unwrap();
        assert_eq!(owned.description, "mine");
        assert!(store.get("deploy", Some("other")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn search_matches_name_or_description_and_respects_limit() {
        let store = InMemorySkillIndex::default();
        store
            .reindex_global(vec![
                skill("invoice-run", None, "generate monthly invoices"),
                skill("deploy-blog", None, "publish the blog"),
                skill("backup-verify", None, "check the invoice archive"),
            ])
            .await
            .unwrap();

        let hits = store.search("invoice", vec![], 10).await.unwrap();
        assert_eq!(hits.len(), 2, "matches name and description");

        let capped = store.search("e", vec![], 1).await.unwrap();
        assert_eq!(capped.len(), 1, "honors the limit");
    }

    #[tokio::test]
    async fn list_caps_at_limit() {
        let store = InMemorySkillIndex::default();
        store
            .reindex_global(vec![
                skill("a", None, "x"),
                skill("b", None, "y"),
                skill("c", None, "z"),
            ])
            .await
            .unwrap();
        assert_eq!(store.list(Some(2)).await.unwrap().len(), 2);
        assert_eq!(store.list(None).await.unwrap().len(), 3);
    }
}
