-- Issue #287: persist the subagent namespace + spawn snapshot marker on every
-- background task so a wait=false subagent's owner_todo/spawn_marker survive a
-- daemon restart (single source of truth on the row; the registry is a
-- passthrough). Set once at spawn (slice 5); update_task never touches them.
--
-- Additive + idempotent (the runner re-executes every migration on every
-- startup with no version table). owner_todo backfills legacy rows as root '';
-- spawn_marker is nullable (legacy / non-subagent tasks never read a pad
-- snapshot). The RLS 029 background_tasks_user_isolation policy is row-level and
-- covers the new columns unchanged.

ALTER TABLE background_tasks ADD COLUMN IF NOT EXISTS owner_todo TEXT NOT NULL DEFAULT '';
ALTER TABLE background_tasks ADD COLUMN IF NOT EXISTS spawn_marker TEXT;
