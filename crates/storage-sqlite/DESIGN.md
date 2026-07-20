# storage-sqlite — design

An embeddable SQLite adapter for the desktop-assistant storage ports, the
sibling of `desktop-assistant-storage` (Postgres). Its purpose is the
single-binary endgame: a downloadable Adele that persists its state with **no
external PostgreSQL**. The daemon already infers "no database" purely from a
missing URL and disables the knowledge base / tool-search / scratchpad while
falling back to a JSON conversation store; SQLite lets the same feature set run
with real persistence and zero services.

This crate is built in increments. **Increment 1 (this crate today) covers the
relational backbone only** — the non-vector, non-FTS stores — behind an
off-by-default `sqlite` feature, and is not yet wired into daemon startup.

## Port surface

Hexagonal: the port traits live in `desktop-assistant-core::ports::store` and
`::auth`. This adapter implements them; nothing in `core` changes.

### Implemented (increment 1)

| Port trait | Adapter | Backing tables |
|---|---|---|
| `ConversationStore` | `SqliteConversationStore` | `conversations`, `messages`, `message_summaries` |
| `TurnStateStore` | `SqliteTurnStateStore` | `turns` |
| `BackgroundTaskStore` | `SqliteBackgroundTaskStore` | `background_tasks` |
| `ErrorClassificationStore` | `SqliteErrorClassificationStore` | `error_classifications` |
| `LearnedWindowStore` | `SqliteLearnedWindowStore` | `context_window_observations` |
| `IdempotencyKeyStore` | `SqliteIdempotencyKeyStore` | `idempotency_keys` |

`ConversationStore` is the RPIT (`impl Future`) trait style; the other five are
`#[async_trait]` (held as `Arc<dyn …>` by the application layer). Both styles
port unchanged. `SqliteConversationStore` also carries the inherent JSON-column
accessors the daemon uses on the concrete type (`get`/`set_conversation_model_selection`,
`get`/`set_conversation_personality`, `get_conversation_tags`) so it is a true
drop-in for increment 1b.

### Deferred (later increments — see roadmap)

`KnowledgeBaseStore`, `ToolRegistryStore` (vector search), `ScratchpadStore`,
`ConversationSearchStore` (FTS), the dreaming/consolidation passes, and
`execute_database_query`. The tables those need are **not** created here; the
schema is the relational subset only.

## Postgres → SQLite translation decisions

SQLite has no server, no real date type, no array type, and one integer width.
The mapping below is what the adapter and the `001_relational_schema.sql` DDL
implement. It is deliberately faithful — the SQL shapes mirror the Postgres
adapter statement-for-statement except where a SQLite dialect difference forces
a change.

| Postgres | SQLite | Notes |
|---|---|---|
| `$N` placeholders | `?` | Positional, bound in statement order. |
| `TIMESTAMPTZ` | `TEXT` (`"YYYY-MM-DD HH:MM:SS"`, UTC) | Stored as the domain's own canonical string, so no chrono encode/decode round trip is needed and **lexicographic order == chronological order** (so `ORDER BY updated_at DESC` is correct). `canonical_ts` normalizes on write; an empty/unparseable value falls back to `now()` with a warning, mirroring `PgConversationStore::parse_timestamp`. |
| `NOW()` / `now()` | a bound canonical timestamp, or `CURRENT_TIMESTAMP` | Value columns read back into the domain (e.g. `conversations.archived_at`) get a bound canonical string so their format matches the rest; bookkeeping columns never read back (`turns.updated_at`, etc.) use `CURRENT_TIMESTAMP`. |
| `JSONB` | `TEXT` + `json1`, read/written as `serde_json::Value` | `tool_calls`, `turns.state_json`, `background_tasks.kind_json`, `conversations.last_model_selection` / `personality_override`. sqlx's `json` feature maps `serde_json::Value` ⇄ TEXT. |
| `TEXT[]` (`conversations.tags`) | `TEXT` holding a JSON array | The simplest correct model: no in-scope method filters on individual tags, so a tag join table (or `json_each`) would be premature. `tags_to_json` / `parse_tags` convert at the boundary; malformed JSON fails closed to empty (tags are advisory routing hints). |
| `= ANY($2)` / `IN (…)` | `IN ('a','b')` | Only the fixed status-set filters use this; all are literal constants. |
| `ON CONFLICT … DO UPDATE SET … [WHERE]` | same, `EXCLUDED` → `excluded` | Supported by SQLite incl. the conditional `WHERE` used for the learned-window down-only ratchet. |
| `IS DISTINCT FROM` | `IS NOT` | SQLite's `IS` / `IS NOT` are null-safe equality/inequality — exactly `IS [NOT] DISTINCT FROM`. Used in `record_overflow`'s ratchet so a NULL configured window compares correctly. |
| `GREATEST(a, b)` | `MAX(a, b)` (scalar) | Success high-water in `record_success`. |
| `strpos(hay, needle) > 0` | `instr(hay, needle) > 0` | Error-classification substring match; `lower(...)` on both sides keeps it case-insensitive, and the signature is never interpreted as a `LIKE` pattern. |
| `UPDATE … RETURNING` | same | SQLite ≥ 3.35. Used for the classification lookup (match + hit-count bump + read in one round trip). |
| `BIGSERIAL` | `INTEGER PRIMARY KEY` (rowid) | `error_classifications.id`; never surfaced to the domain. |
| `BIGINT` | `INTEGER` | SQLite integers are 64-bit; epoch-millis (`started_at`/`ended_at`) and token counts bind/read as `i64`. |
| `ON DELETE CASCADE` / `SET NULL` | same, **but** requires `PRAGMA foreign_keys = ON` | SQLite leaves FK enforcement off per connection. The pool sets `foreign_keys(true)` on every connection, so deleting a conversation cascades to its messages / summaries / idempotency rows, and deleting a summary SET-NULLs `messages.summary_id` (reversible expand). |
| `sqlx` dynamic `&str` SQL | `&'static str` literals (or `AssertSqlSafe`) | sqlx 0.9's `SqlSafeStr` bound rejects `&String`. The two JSON-column helpers take a `'static` literal from the call site; the background-task list uses two literal statements rather than an interpolated status filter. No SQL is built from user input. |

### Multi-tenancy

Preserved exactly. Personal-data stores scope every query to `current_user_id()`
(the `core::ports::auth` task-local); cross-user reads behave like the row does
not exist (#105 opacity). `error_classifications` and
`context_window_observations` are global (connector/model knowledge, not
personal data) and deliberately carry no `user_id`. The `scan_non_terminal`
sweeps intentionally bypass the per-user scope (system-task callers at startup).

### Migrations

A hand-registered `include_str!` list applied idempotently at pool init
(`run_migrations`), mirroring the Postgres adapter's style — not sqlx-cli
migrations. Increment 1 ships one consolidated relational-backbone script
(`001_relational_schema.sql`, all `IF NOT EXISTS`); later increments append
ordinally-numbered scripts. A unit-test guard (`every_migration_is_registered`)
fails the build if a `.sql` file is added but not wired in.

## Feature gate

The `sqlite` feature is **off by default**. Only `sqlx` (which drags in the
sqlite C library via `libsqlite3-sys`) is gated behind it; all other deps are
pure-Rust and already in the workspace graph. Consequences:

- `cargo build` / `cargo build --workspace` / the daemon build compile this
  crate as an empty shell — no `sqlx-sqlite`, byte-unchanged default build.
- Build and test the real adapter with `--features sqlite`.

Tests build pools via the crate's own `create_memory_pool`, so the test target
needs no `sqlx` dev-dependency (which would otherwise compile the sqlite C
library into a default `cargo test --workspace`).

## Testing

Contract tests run against an **in-memory** SQLite database (a pool pinned to a
single persistent connection, since an in-memory DB lives only as long as its
connection) — no external service, no container, fully deterministic. Every
acceptance criterion is a named test: CRUD + the aggregate list projection,
message append/truncate, reversible summaries via `ON DELETE SET NULL`,
archive/unarchive, duplicate-create rejection, non-terminal scans, cross-user
isolation, idempotent upsert, the error-classification longest-match /
case-insensitive / connector-scoped rules, and the learned-window down-only
ratchet + success high-water mark and their independence.

## Increment roadmap

1. **inc1 — relational backbone (this crate today).** The six stores above,
   feature-gated, tested in isolation, not wired.
2. **inc1b — daemon wiring + `sqlite://` URL selector.** Add a `Sqlite` variant
   to `AnyConversationStore` in `crates/daemon/src/main.rs` and select the
   backend from the database-URL scheme (`sqlite://…` vs `postgres://…`) in
   `resolve_database_config`.
3. **inc2 — vector + FTS hybrid via sqlite-vec + FTS5.** `KnowledgeBaseStore`
   and `ToolRegistryStore` (vector search + hybrid RRF) on `sqlite-vec`, plus
   the FTS5 half (`ScratchpadStore` search, `ConversationSearchStore`).
4. **inc3 — dreaming + db_query.** Port `crates/storage/src/dreaming/*` (raw
   `&PgPool` today, not behind a port) and the LLM-facing `execute_database_query`
   tool (Postgres RLS + `sqlparser` `PostgreSqlDialect` today).

### Key risk (inc2)

`sqlite-vec` ANN indexes are **fixed-dimension per virtual table**, but the
schema models embeddings as a **dimensionless, per-model `vector[]` chunk array**
(migration 007 made `knowledge_base.embedding` a `vector[]`, and the embedding
model — hence its dimension — is user-configurable). Reconciling a
fixed-dimension index with dimensionless per-model chunk arrays is the central
design problem of inc2 and must be solved before the single binary can drop
Postgres entirely. Candidate approaches to weigh: one vec table per dimension,
a dimension-tagged table, or storing raw vectors + brute-force cosine for the
small local corpora a single-user desktop install actually has.
