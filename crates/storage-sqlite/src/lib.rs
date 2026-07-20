//! Embeddable SQLite persistence for the desktop-assistant storage ports.
//!
//! This is the sibling adapter to `desktop-assistant-storage` (Postgres): it
//! implements the same `desktop-assistant-core` ports against an embedded
//! SQLite database, so a single downloadable binary can persist its state with
//! **no external PostgreSQL**. See `DESIGN.md` for the port surface, the
//! Postgres-ism translations, and the increment roadmap.
//!
//! ## Scope (increment 1)
//!
//! Only the **relational** (non-vector, non-FTS) stores:
//!
//! - [`SqliteConversationStore`] — [`desktop_assistant_core::ports::store::ConversationStore`]
//! - [`SqliteTurnStateStore`] — [`desktop_assistant_core::ports::store::TurnStateStore`]
//! - [`SqliteBackgroundTaskStore`] — [`desktop_assistant_core::ports::store::BackgroundTaskStore`]
//! - [`SqliteErrorClassificationStore`] — [`desktop_assistant_core::ports::store::ErrorClassificationStore`]
//! - [`SqliteLearnedWindowStore`] — [`desktop_assistant_core::ports::store::LearnedWindowStore`]
//! - [`SqliteIdempotencyKeyStore`] — [`desktop_assistant_core::ports::store::IdempotencyKeyStore`]
//!
//! Vector search (sqlite-vec), FTS5, dreaming, `db_query`, and daemon wiring
//! are deferred to later increments (see `DESIGN.md`).
//!
//! ## Feature gate
//!
//! Everything here is behind the **off-by-default** `sqlite` feature so the
//! standard workspace build and the daemon are byte-unchanged and never pull
//! the sqlite C library. Build/test the real adapter with `--features sqlite`.

// TODO(sqlite inc2): add the vector stores (KnowledgeBaseStore,
// ToolRegistryStore) on sqlite-vec and the FTS5 stores (ScratchpadStore search,
// ConversationSearchStore). See DESIGN.md for the fixed-dimension-vs-per-model
// `vector[]` risk that gates the vector half.
// TODO(sqlite inc3): port the dreaming/consolidation passes and the
// `execute_database_query` (db_query) tool.
#[cfg(feature = "sqlite")]
mod background_tasks;
#[cfg(feature = "sqlite")]
mod context_window_observations;
#[cfg(feature = "sqlite")]
mod conversation;
#[cfg(feature = "sqlite")]
mod error_classifications;
#[cfg(feature = "sqlite")]
mod idempotency_keys;
#[cfg(feature = "sqlite")]
mod pool;
#[cfg(feature = "sqlite")]
mod turn_state;

#[cfg(feature = "sqlite")]
pub use background_tasks::SqliteBackgroundTaskStore;
#[cfg(feature = "sqlite")]
pub use context_window_observations::SqliteLearnedWindowStore;
#[cfg(feature = "sqlite")]
pub use conversation::SqliteConversationStore;
#[cfg(feature = "sqlite")]
pub use error_classifications::SqliteErrorClassificationStore;
#[cfg(feature = "sqlite")]
pub use idempotency_keys::SqliteIdempotencyKeyStore;
#[cfg(feature = "sqlite")]
pub use pool::{create_memory_pool, create_pool, run_migrations};
#[cfg(feature = "sqlite")]
pub use turn_state::SqliteTurnStateStore;

/// Re-exported so daemon-side consumers can name the pool type without taking a
/// direct `sqlx` dependency (mirrors the Postgres adapter's `PgPool` re-export).
#[cfg(feature = "sqlite")]
pub use sqlx::SqlitePool;

/// Re-export the multi-tenant identity helpers so call sites (and tests) can
/// scope by user without depending directly on `auth-jwt` / `core::ports::auth`.
#[cfg(feature = "sqlite")]
pub use desktop_assistant_auth_jwt::{DEFAULT_USER_ID, UserId};
#[cfg(feature = "sqlite")]
pub use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};
