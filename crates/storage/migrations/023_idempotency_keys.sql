-- Issue #204: SendMessage idempotency keys — crash-safe completed-dedup.
--
-- Lets a client safely retry a `SendMessage` whose connection dropped (or
-- whose daemon restarted) mid-turn without re-running the LLM and
-- re-executing the turn's tool actions. The orchestrator records the
-- committed reply keyed by (user_id, conversation_id, idempotency_key); a
-- retry that finds a completed row replays the stored reply instead of
-- dispatching a fresh turn.
--
-- Only *completed* turns are recorded — a turn that never finished records
-- nothing, so a retry re-runs it (the action did not complete). The common
-- "connection dropped after the turn already finished" case is the one made
-- recoverable here; in-flight re-attach (a duplicate key while the original
-- is still running in the same process) is handled by the application
-- layer's in-memory registry, not this table.
--
-- Multi-tenant (#102/#105)
--   `user_id` is part of the primary key so every lookup scopes to one user
--   and a key presented by one user can never collide with or read another
--   user's stored reply — the same opacity rule as the other personal-data
--   tables.
--
-- Cascade
--   `conversation_id` is a real FK with ON DELETE CASCADE (same shape as
--   `scratchpads` / `messages`): deleting a conversation drops its
--   idempotency rows.
--
-- Backfill / reversibility
--   Brand-new table, no backfill. Forward-only; reversing is DROP TABLE.

CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL,
    -- The internal per-request id of the turn that produced `response`,
    -- stored for audit/debugging only (retries reply under their own id).
    request_id      TEXT NOT NULL,
    -- The committed assistant reply replayed to a retry. Present only for
    -- completed turns (rows are written on completion).
    response        TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conversation_id, idempotency_key)
);

-- Supports a future age-based cleanup job (idempotency rows have no reason to
-- live forever); unused until that job exists.
CREATE INDEX IF NOT EXISTS idempotency_keys_created_at_idx
    ON idempotency_keys (created_at);
