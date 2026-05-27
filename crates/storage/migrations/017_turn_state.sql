-- Issue #107: conversation turn state machine for client-side execution.
--
-- The turn becomes a DB-persisted state machine so it can suspend on a
-- client-local tool call and resume when the chat client posts the
-- result back. See `docs/architecture-evolution.md` Phase 2 + rule #8
-- for the bigger picture.
--
-- Why a single JSON column for the state body
--   The transition machinery (pending tool calls, partial history, retry
--   counters, ...) evolves with the in-memory state machine. A per-shape
--   migration on every internal-state tweak is friction; a versioned
--   JSON column keeps the schema stable while the in-memory shape
--   matures. The `(user_id, status)` and `(user_id, conversation_id)`
--   indexes cover every hot query path identified so far (cold-restart
--   sweep + "what's pending for this conversation").
--
-- Why we don't FK to `conversations(id)`
--   A long-running turn might survive the conversation being deleted
--   (e.g. user races the cleanup); the turn row should NOT prevent
--   conversation deletion. We keep the conversation_id as plain text
--   and let the application layer reconcile on observation. Note also
--   that #105's scoping rules already keep cross-user reads opaque, so
--   a stale conversation_id never leaks.
--
-- Backfill / reversibility
--   No backfill — this is a brand-new table. The existing migration
--   tooling (`pool::run_migrations`) is forward-only; reversing
--   requires `DROP TABLE turns` plus dropping indexes manually.

CREATE TABLE IF NOT EXISTS turns (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    -- One of: 'pending_llm', 'pending_tool_dispatch',
    -- 'pending_client_tool', 'complete', 'failed'.
    -- Mirrors `core::ports::store::TurnStatus::as_key()`.
    status          TEXT NOT NULL,
    state_json      JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Set on `failed`; nullable in every other state. Populated
    -- verbatim from the in-memory state machine's reason string
    -- ("cancelled", "daemon_restarted", "<llm error>", etc.).
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-user, per-conversation scan: the application layer reads the
-- turn row for a SendMessage's task_id directly by id (no index
-- needed for that), but a future "what's pending for this conversation"
-- view will scan via (user_id, conversation_id).
CREATE INDEX IF NOT EXISTS turns_user_id_conversation_id_idx
    ON turns (user_id, conversation_id);

-- Per-user, per-status scan: the cold-restart sweep walks rows in
-- non-terminal status across all users, but reads of a single user's
-- pending turns (UI "show in-flight" view, future) go through this.
CREATE INDEX IF NOT EXISTS turns_user_id_status_idx
    ON turns (user_id, status);

-- Global non-terminal sweep: dedicated narrow index for the startup
-- hook in `client_tools::sweep_non_terminal_turns_on_startup`. Without
-- this, the sweep would force a sequential scan whose cost grows with
-- the total turn history. A partial index keyed only on non-terminal
-- statuses keeps the index small (most turns are 'complete').
CREATE INDEX IF NOT EXISTS turns_non_terminal_status_idx
    ON turns (status)
    WHERE status NOT IN ('complete', 'failed');
