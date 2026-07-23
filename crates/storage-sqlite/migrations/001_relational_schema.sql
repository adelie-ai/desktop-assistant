-- storage-sqlite relational backbone (increment 1).
--
-- Mirrors the relational (non-vector, non-FTS) subset of the Postgres schema
-- in `crates/storage/migrations/*` against SQLite. Applied idempotently at
-- pool init by `run_migrations` (every statement is `IF NOT EXISTS`).
--
-- Postgres-ism translations (see DESIGN.md for the full table):
--   * TIMESTAMPTZ  -> TEXT holding a canonical "YYYY-MM-DD HH:MM:SS" (UTC)
--     string. Lexicographic order == chronological order, so `ORDER BY` works.
--   * JSONB        -> TEXT holding JSON (read/written via serde_json + json1).
--   * TEXT[] tags  -> TEXT holding a JSON array (the simplest correct model;
--     no scoped store needs to filter on individual tags this increment).
--   * BIGSERIAL    -> INTEGER PRIMARY KEY (rowid alias).
--   * BIGINT       -> INTEGER (SQLite integers are 64-bit).
--   * NOW()        -> a bound canonical timestamp, or CURRENT_TIMESTAMP for
--     bookkeeping columns that are never read back into the domain.
--
-- Foreign-key actions (ON DELETE CASCADE / SET NULL) require per-connection
-- `PRAGMA foreign_keys = ON`, which the pool sets on every connection.

-- ---------------------------------------------------------------------------
-- conversations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL DEFAULT 'default',
    title                TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    context_summary      TEXT NOT NULL DEFAULT '',
    compacted_through    INTEGER NOT NULL DEFAULT 0,
    archived_at          TEXT,
    active_task          TEXT,
    -- Postgres `tags TEXT[] NOT NULL DEFAULT '{}'` -> JSON array text.
    tags                 TEXT NOT NULL DEFAULT '[]',
    -- Postgres JSONB -> TEXT holding JSON. NULL == unset.
    last_model_selection TEXT,
    personality_override TEXT
);

CREATE INDEX IF NOT EXISTS conversations_user_id_updated_at_idx
    ON conversations (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_user_id_archived_at_idx
    ON conversations (user_id, archived_at);

-- ---------------------------------------------------------------------------
-- message_summaries (child of conversations; referenced by messages, so it is
-- created before `messages`)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_summaries (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    start_ordinal   INTEGER NOT NULL,
    end_ordinal     INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS message_summaries_user_id_conv_idx
    ON message_summaries (user_id, conversation_id);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_calls      TEXT,   -- JSON (Postgres JSONB); NULL when there are none
    tool_call_id    TEXT,
    -- ON DELETE SET NULL: deleting a summary auto-expands its messages.
    summary_id      TEXT REFERENCES message_summaries(id) ON DELETE SET NULL,
    -- #570 Phase 1b: client idempotency key, carried on USER rows only so a
    -- reload/reconnect surfaces it; NULL for assistant/tool rows and keyless
    -- sends.
    idempotency_key TEXT,
    UNIQUE (conversation_id, ordinal)
);

CREATE INDEX IF NOT EXISTS messages_user_id_conv_ordinal_idx
    ON messages (user_id, conversation_id, ordinal);

-- ---------------------------------------------------------------------------
-- turns (issue #107) — DB-persisted turn state machine
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    status          TEXT NOT NULL,
    state_json      TEXT NOT NULL DEFAULT '{}',
    last_error      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS turns_user_id_conversation_id_idx
    ON turns (user_id, conversation_id);
CREATE INDEX IF NOT EXISTS turns_user_id_status_idx
    ON turns (user_id, status);
CREATE INDEX IF NOT EXISTS turns_non_terminal_status_idx
    ON turns (status)
    WHERE status NOT IN ('complete', 'failed');

-- ---------------------------------------------------------------------------
-- background_tasks (issue #115)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS background_tasks (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    kind_json       TEXT NOT NULL,   -- JSON (Postgres JSONB)
    task_status     TEXT NOT NULL,
    parent_task_id  TEXT,
    title           TEXT NOT NULL,
    last_error      TEXT,
    progress_hint   TEXT,
    started_at      INTEGER NOT NULL,   -- unix epoch millis (Postgres BIGINT)
    ended_at        INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS background_tasks_user_id_started_at_idx
    ON background_tasks (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS background_tasks_non_terminal_status_idx
    ON background_tasks (task_status)
    WHERE task_status NOT IN ('completed', 'failed', 'cancelled');
CREATE INDEX IF NOT EXISTS background_tasks_parent_task_id_idx
    ON background_tasks (parent_task_id)
    WHERE parent_task_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- error_classifications (epic #178) — GLOBAL, no user_id
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS error_classifications (
    id              INTEGER PRIMARY KEY,   -- rowid alias (Postgres BIGSERIAL)
    connector       TEXT NOT NULL,
    signature       TEXT NOT NULL,
    cause           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'learned',
    hit_count       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    last_matched_at TEXT,
    UNIQUE (connector, signature)
);

CREATE INDEX IF NOT EXISTS idx_error_classifications_connector
    ON error_classifications (connector);

-- ---------------------------------------------------------------------------
-- idempotency_keys (issue #204)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    response        TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    PRIMARY KEY (user_id, conversation_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idempotency_keys_created_at_idx
    ON idempotency_keys (created_at);

-- ---------------------------------------------------------------------------
-- context_window_observations (issues #343 / #425) — GLOBAL, no user_id.
-- observed_limit / configured_window are nullable (a row may hold only a
-- success high-water mark), max_success_input is the #425 addition.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS context_window_observations (
    connector         TEXT NOT NULL,
    model             TEXT NOT NULL,
    observed_limit    INTEGER,
    configured_window INTEGER,
    max_success_input INTEGER,
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    PRIMARY KEY (connector, model)
);
