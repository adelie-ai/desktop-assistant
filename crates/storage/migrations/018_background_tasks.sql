-- Issue #115: durable background tasks (resume across daemon restart).
--
-- Adds a `background_tasks` table that mirrors the in-memory
-- `BackgroundTaskRegistry` rows so a daemon restart can sweep abandoned
-- tasks instead of silently losing them. See the issue body and
-- `docs/architecture-evolution.md` Phase 2 for the wider motivation.
--
-- Why a parallel table rather than extending `turns`
--   The `turns` table (migration 017) tracks one row per *conversation
--   turn* — a single LLM-call lifecycle that may suspend on a client
--   tool call. A background task is a longer-lived unit: a Standalone
--   agent may run several turns over its lifetime, a Subagent has its
--   own turn(s), and a Conversation task is 1:1 with a turn. The two
--   concepts have different status enums (TurnStatus vs TaskStatus),
--   different parent/child relationships (TurnStatus parents nothing;
--   TaskKind::Subagent links to a parent task), and different terminal
--   semantics (a task can be `Cancelled`; a turn can only `Complete`
--   or `Fail`). Cramming both into a single row conflates concerns
--   that we already have to keep separate in the in-memory registry.
--
-- Why `kind_json` is a JSON column rather than a discriminator + columns
--   `TaskKind` is a tagged enum (Conversation / Subagent / Standalone)
--   whose payload differs per-variant. A normalized schema would need
--   per-variant columns plus null-checks; storing the variant as JSON
--   matches the same pattern `turns.state_json` uses for similar
--   reasons. The `(user_id, task_status)` index covers every hot query
--   path identified so far (cold-restart sweep + per-user list).
--
-- Parent linkage
--   `parent_task_id` references the same table, nullable. We do NOT
--   declare a FOREIGN KEY: the parent might be cleaned up before the
--   child in pathological cases (cancelled parent, child still
--   draining). The application layer is responsible for not following
--   stale parent ids; tests cover the case.
--
-- Backfill / reversibility
--   Brand-new table; nothing to backfill. The `pool::run_migrations`
--   harness is forward-only; reversing requires `DROP TABLE
--   background_tasks` plus dropping indexes.

CREATE TABLE IF NOT EXISTS background_tasks (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    -- JSON payload mirroring `api_model::TaskKind` (tagged snake_case).
    kind_json       JSONB NOT NULL,
    -- One of: 'pending', 'running', 'completed', 'failed', 'cancelled'.
    -- Mirrors `api_model::TaskStatus` rename_all = "snake_case".
    task_status     TEXT NOT NULL,
    -- Self-reference for `TaskKind::Subagent` parents. No FK on purpose;
    -- see header.
    parent_task_id  TEXT,
    title           TEXT NOT NULL,
    last_error      TEXT,
    progress_hint   TEXT,
    -- Unix epoch millis the task transitioned to `Running`. Mirrors
    -- `TaskView.started_at` for restart-survival.
    started_at      BIGINT NOT NULL,
    -- Unix epoch millis the task reached a terminal state. NULL while
    -- non-terminal.
    ended_at        BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-user listing — covers `registry.list(user_id, …)` and the per-user
-- subset of the cold-restart sweep.
CREATE INDEX IF NOT EXISTS background_tasks_user_id_started_at_idx
    ON background_tasks (user_id, started_at DESC);

-- Cold-restart sweep: scan every non-terminal row across all users.
-- A partial index keeps the index small (most rows end up terminal).
CREATE INDEX IF NOT EXISTS background_tasks_non_terminal_status_idx
    ON background_tasks (task_status)
    WHERE task_status NOT IN ('completed', 'failed', 'cancelled');

-- Parent → child traversal (cheap when fully indexed). The list of a
-- task's children is rebuilt at restart by joining against this index.
CREATE INDEX IF NOT EXISTS background_tasks_parent_task_id_idx
    ON background_tasks (parent_task_id)
    WHERE parent_task_id IS NOT NULL;
