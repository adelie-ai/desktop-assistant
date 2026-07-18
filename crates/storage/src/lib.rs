//! SQLite-backed persistence for conversations, knowledge, and assistant state.

pub mod background_tasks;
pub mod context_window_observations;
pub mod conversation;
pub mod conversation_search;
pub mod database;
pub mod dreaming;
pub mod embedding_backfill;
pub mod error_classifications;
pub mod idempotency_keys;
pub mod kb_metadata;
pub mod knowledge;
pub mod migrate_json;
pub mod pool;
pub mod scratchpad;
pub mod tag_normalize;
pub mod tag_registry;
pub mod tool_registry;
pub mod turn_state;

pub use desktop_assistant_auth_jwt::{DEFAULT_USER_ID, UserId};
/// Re-export the request-scoped user-id task-local API so storage call
/// sites can resolve `current_user_id()` without depending directly on
/// `desktop_assistant_core::ports::auth`. The actual storage adapters
/// in this crate use this helper at SQL composition time (issue #105).
pub use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};

pub use background_tasks::PgBackgroundTaskStore;
pub use context_window_observations::PgLearnedWindowStore;
pub use conversation::PgConversationStore;
pub use conversation_search::PgConversationSearchStore;
pub use database::{TOOL_QUERY_ROLE, execute_database_query, personal_data_tables};
pub use error_classifications::PgErrorClassificationStore;
pub use idempotency_keys::PgIdempotencyKeyStore;
pub use knowledge::PgKnowledgeBaseStore;
pub use migrate_json::{
    is_conversations_table_empty, is_knowledge_base_table_empty, migrate_conversations,
    migrate_knowledge,
};
pub use pool::{create_pool, run_migrations};
pub use scratchpad::PgScratchpadStore;
/// Re-exported so daemon-side consumers can name the pool type (e.g. the
/// knowledge-maintenance service) without taking a direct `sqlx` dependency.
pub use sqlx::PgPool;
pub use tool_registry::PgToolRegistryStore;
pub use turn_state::PgTurnStateStore;
