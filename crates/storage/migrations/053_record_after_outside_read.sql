-- Issue #1247: a durable record keeps the words a turn wrote, and states
-- whether the turn that wrote them had already read content from outside the
-- trust boundary.
--
-- Two write paths destroyed that text before the row was written: a plan step's
-- goal and its outcome, and a negative memory's account of what went wrong.
-- Each stored a placeholder in place of the words, so no later reader could
-- recover them - not the model, not the person, not an audit. The rule they
-- served is real, but it belongs at the render, where the two audiences part:
-- the model reads a placeholder, and the person reads what was written.
--
-- Hence a flag rather than a loss. It states one fact about the WRITING turn.
-- Every model-facing render decides from that fact plus the READING turn's own
-- tool policy, so the level a person chose is what decides what the model sees.
--
-- FALSE is the honest default for every row already stored. A row written
-- before this migration either carries the model's own wording, because its
-- turn read nothing from outside, or carries the placeholder wording itself,
-- which the render path recognises by its text.
--
-- Row level security needs nothing here: migration 029 already enabled it on
-- `scratchpads` and migration 049 on `negative_memory`, and both policies are
-- on `user_id`, which these columns do not change.
--
-- The migration runner (pool.rs) applies each migration at most once, tracked
-- in the `schema_migrations` ledger, but every statement here MUST still be
-- idempotent: a database migrated before that ledger existed replays the whole
-- set once on its first boot under it.

ALTER TABLE scratchpads
    ADD COLUMN IF NOT EXISTS after_outside_read BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE negative_memory
    ADD COLUMN IF NOT EXISTS after_outside_read BOOLEAN NOT NULL DEFAULT FALSE;
