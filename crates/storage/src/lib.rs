//! SQLite-backed persistence for conversations, knowledge, and assistant state.

pub mod background_tasks;
pub mod context_breakdown;
pub mod context_plans;
pub mod context_window_observations;
pub mod conversation;
pub mod conversation_search;
pub mod database;
pub mod dreaming;
pub mod embedded_tables;
pub mod embedding_backfill;
pub mod error_classifications;
pub mod idempotency_keys;
pub mod kb_metadata;
pub mod knowledge;
pub mod knowledge_delete;
pub mod knowledge_search;
pub mod knowledge_use;
pub mod migrate_json;
pub mod negative_memory;
pub mod pool;
pub mod scan_bound;
pub mod scratchpad;
pub mod skill_index;
pub mod skill_use;
pub mod tag_registry;
pub mod tool_registry;
pub mod tool_usage;
pub mod turn_records;
pub mod turn_state;

pub use desktop_assistant_auth_jwt::{DEFAULT_USER_ID, UserId};
/// Re-export the request-scoped user-id task-local API so storage call
/// sites can resolve `current_user_id()` without depending directly on
/// `desktop_assistant_core::ports::auth`. The actual storage adapters
/// in this crate use this helper at SQL composition time (issue #105).
pub use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};
/// The one tag normalizer, re-exported at the path storage callers already
/// use. It lives in `core` because the knowledge-base write tool has to apply
/// the same rule to match a caller's tag description against the tag it
/// describes, and that tool cannot depend on a storage adapter.
pub use desktop_assistant_core::tag_normalize;

pub use background_tasks::PgBackgroundTaskStore;
pub use context_plans::{PgContextPlanStore, sweep_expired_context_plans};
pub use context_window_observations::PgLearnedWindowStore;
pub use conversation::PgConversationStore;
pub use conversation_search::PgConversationSearchStore;
pub use database::{
    TOOL_QUERY_ROLE, WRITE_SANDBOX_SCHEMA, execute_database_query, personal_data_tables,
};
pub use error_classifications::PgErrorClassificationStore;
pub use idempotency_keys::PgIdempotencyKeyStore;
pub use knowledge::{NearestEntries, PgKnowledgeBaseStore, RECALL_SCAN_STATEMENT_TIMEOUT};
pub use knowledge_use::{PgKnowledgeUseLog, USE_LOG_READ_STATEMENT_TIMEOUT};
pub use migrate_json::{
    is_conversations_table_empty, is_knowledge_base_table_empty, migrate_conversations,
    migrate_knowledge,
};
pub use negative_memory::PgNegativeMemoryStore;
pub use pool::{create_pool, run_migrations};
pub use scan_bound::begin_bounded;
pub use scratchpad::{NearestNotes, PgScratchpadStore};
pub use skill_index::{
    NearestSkill, NearestSkills, PgSkillIndexStore, SKILL_RECALL_SCAN_STATEMENT_TIMEOUT,
};
pub use skill_use::PgSkillUseLog;
/// Re-exported so daemon-side consumers can name the pool type (e.g. the
/// knowledge-maintenance service) without taking a direct `sqlx` dependency.
pub use sqlx::PgPool;
pub use tool_registry::{PROVIDER_BOOST_WEIGHT, PgToolRegistryStore, ToolRegisterBatch};
pub use turn_records::{PgTurnRecordStore, sweep_expired_turn_records};
pub use turn_state::PgTurnStateStore;
