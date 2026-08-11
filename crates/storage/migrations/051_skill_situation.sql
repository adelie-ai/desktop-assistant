-- #1175: the situations a skill has been followed in.
--
-- #1125 gave a knowledge entry a situation record so a recurring situation
-- could reach it again, and #1154 shipped the skill arm without one. That is
-- the weaker half of the arm's own argument: nobody retrieves how to ride a
-- bicycle by searching their memory for it, and "deploy this" is a weak query
-- and a strong situation. A procedure is if anything more situational than a
-- fact.
--
-- The same shape as `knowledge_situation` (migration 047), and the same three
-- rules, which are stated there in full:
--
-- * The match is presence, not frequency. `times` and `last_seen_at` exist for
--   eviction and for a later reader; no ranking rule reads them.
-- * Accumulation never touches `skill_index`. A skill rewritten to learn where
--   it is useful would move its content hash, drop its embedding, and put
--   itself back in the backfill queue.
-- * The fan of a value is a `count(*)` under one index, which is what lets the
--   read weight each cue value by how much it separates one skill from another.
--
-- Keyed on the skill's NAME, scoped to the reading user, for the reason
-- migration 048 gives: `skill_index` is host-global and unique on
-- `(name, owner_key)`, but a reader sees the global skills plus their own, so
-- within one user's view a name resolves to exactly one skill - and the name is
-- what `builtin_skill_get` takes, so it is the only identity the model can act
-- on. The record is per-user because the use is: one person's situations say
-- nothing about another's.
--
-- No foreign key, deliberately, on the same reasoning as `skill_use_stats`.
-- The catalog is cumulative (#639): a skill whose files leave disk is marked
-- absent, never deleted, so there is no reap for a cascade to follow.
--
-- The one write path is a taken-up offer. A scan reads a file at daemon start
-- and the dream cycle authors a skill in a background pass, and neither happens
-- in anybody's situation, so there is no counterpart to the knowledge log's
-- write-path `record_situation`.
--
-- Personal data. The table carries `user_id`, every query scopes by it, and it
-- enables its own row-level security below - migration 029's policy list is
-- static and does not reach a table created later. The name is also registered
-- in `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query
-- tool grafts a `user_id` predicate onto any LLM-supplied SQL that names it.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS skill_situation (
    user_id       TEXT NOT NULL,
    -- The catalog name, which is the handle the skill is fetched by.
    skill_name    TEXT NOT NULL,
    -- The dimension: 'host', 'time_of_day' or 'weekday' today. Stored as text
    -- rather than an enum type so a later dimension is a new value and not a
    -- schema change; the reader skips a name it does not know.
    field         TEXT NOT NULL,
    -- The value seen for that dimension.
    value         TEXT NOT NULL,
    -- How many observations have landed on this pair. Recorded, and read by
    -- nothing that ranks.
    times         BIGINT NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, skill_name, field, value)
);

-- The fan read: how many of this user's skills carry one (field, value). The
-- cue asks this once per field per turn, so it must not be a sequential scan.
CREATE INDEX IF NOT EXISTS skill_situation_fan_idx
    ON skill_situation (user_id, field, value);

-- The eviction read: the least recently seen value of one field on one skill.
CREATE INDEX IF NOT EXISTS skill_situation_evict_idx
    ON skill_situation (user_id, skill_name, field, last_seen_at DESC);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so this one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS skill_situation_user_isolation ON skill_situation;
ALTER TABLE skill_situation ENABLE ROW LEVEL SECURITY;
CREATE POLICY skill_situation_user_isolation ON skill_situation
    USING (user_id = current_setting('app.user_id', true));
