//! Postgres adapter for the knowledge use log (#698).
//!
//! Two tables, created by `044_knowledge_use_log.sql`. `knowledge_use_stats`
//! holds one row per entry - the counters, the first-seen stamp, the recent-use
//! window, and the entry's standing offer. `knowledge_use_marks` holds one
//! standing mark per source per entry. Both are bounded per entry by design;
//! the module doc on [`desktop_assistant_core::domain::knowledge_use`] says why.
//!
//! ## Every write is guarded by ownership, not only scoped by it
//!
//! `knowledge_base.id` is a global primary key, so an id another user owns is
//! an id this user can name. Each write therefore selects the entry out of
//! `knowledge_base` under the caller's `user_id` and inserts from that select,
//! rather than inserting the id it was handed. An id the caller does not own,
//! or one that has been retired, matches nothing and is silently not recorded -
//! the same answer `get_many` gives for the same id.
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so these predicates are the guard.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::ports::knowledge_use::{KnowledgeUseLog, MarkRequest, OfferScope};
use sqlx::PgPool;

/// The knowledge use log, backed by Postgres.
pub struct PgKnowledgeUseLog {
    #[allow(dead_code)]
    pool: PgPool,
}

impl PgKnowledgeUseLog {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

impl KnowledgeUseLog for PgKnowledgeUseLog {
    async fn record_offered(
        &self,
        scope: OfferScope,
        entry_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let _ = (scope, entry_ids);
        unimplemented!()
    }

    async fn record_opened(
        &self,
        conversation_id: String,
        entry_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let _ = (conversation_id, entry_ids);
        unimplemented!()
    }

    async fn record_mark(&self, request: MarkRequest) -> Result<Vec<String>, CoreError> {
        let _ = request;
        unimplemented!()
    }

    async fn records(&self, entry_ids: Vec<String>) -> Result<Vec<KnowledgeUseRecord>, CoreError> {
        let _ = entry_ids;
        unimplemented!()
    }
}
