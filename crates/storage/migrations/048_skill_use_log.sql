-- #1154: the skill use log - which skills the `[Recall]` block put in front of
-- the model, and which of those it went on to open.
--
-- The block offers skills nobody searched for, so the two facts that judge such
-- an offer have to be recorded somewhere: a skill surfaced on twenty prompts
-- and opened on none is a skill the catalog would be better without, and a
-- skill opened again and again has earned its place at the top of the arm.
-- Ranking reads the second (`domain::activation`), and a person reads the
-- first.
--
-- Why not the tables migration 044 created. Same shape, different key.
-- `knowledge_use_stats` and `knowledge_offers` carry a foreign key to
-- `knowledge_base(id)`, and that key is what frees an entry's use rows when the
-- entry is reaped. A skill has no row in that table, so recording one there
-- would mean dropping the key the knowledge log depends on.
--
-- Keyed on the skill's NAME, scoped to the reading user. `skill_index` is
-- host-global and its uniqueness is the composite `(name, owner_key)`, but a
-- reader sees the global skills plus their own, so within one user's view a
-- name resolves to exactly one skill - and the name is what
-- `builtin_skill_get` takes, so it is the only identity the model can act on.
-- The use is per-user because one person's opens say nothing about another's.
--
-- No foreign key, deliberately. The skill catalog is cumulative (#639): a skill
-- whose files leave disk is marked absent, never deleted, so there is no reap
-- for a cascade to follow. A stats row therefore outlives nothing, and a row
-- naming a skill the catalog no longer holds cannot arise from a delete.
--
-- No marks table. The knowledge log records a third act - "this helped" or
-- "this was wrong" - and no tool sets one on a skill. A table with no writer is
-- a table nobody maintains; the act arrives with the tool that performs it.
--
-- Both tables are personal data. Each carries `user_id`, every query scopes by
-- it, both enable their own RLS policy below (migration 029's list is static
-- and does not reach a table created later), and both are registered in
-- `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query tool
-- grafts a `user_id` predicate onto any LLM-supplied SQL that names them.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS skill_use_stats (
    user_id         TEXT NOT NULL,
    -- The catalog name, which is the handle the skill is fetched by.
    skill_name      TEXT NOT NULL,
    -- How many times the skill appeared in a [Recall] block. Mainly a
    -- denominator: surfaced is not useful.
    offered_count   BIGINT NOT NULL DEFAULT 0,
    -- How many times the model read the skill's body after it was offered.
    opened_count    BIGINT NOT NULL DEFAULT 0,
    -- When the skill first entered the log. With the counters, this is what the
    -- tail approximation reads for the uses that fell out of the window.
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_offered_at TIMESTAMPTZ,
    -- The most recent use timestamps, newest first. The writer caps the length,
    -- so this array is bounded however long the skill lives.
    recent_uses     TIMESTAMPTZ[] NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, skill_name)
);

-- The standing skill offers: which procedures are in front of the model right
-- now, and where.
--
-- An open counts only against a standing offer, and counting it deletes the
-- row - so a second read of the same skill in the same turn is one open, and a
-- read of a skill nothing offered is not an open at all.
--
-- Bounded by how it is written. A [Recall] block renders once per turn and
-- deletes this conversation's rows before inserting its own, so a conversation
-- holds one turn's offers. Nothing else offers a skill today; the writer caps
-- the set anyway, on the same rule the knowledge log's writer follows.
CREATE TABLE IF NOT EXISTS skill_offers (
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    skill_name      TEXT NOT NULL,
    offered_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, conversation_id, skill_name)
);

-- The two writes that are not keyed on the skill: the per-turn clear, and the
-- cap that trims a conversation to its newest offers.
CREATE INDEX IF NOT EXISTS skill_offers_conversation_idx
    ON skill_offers (user_id, conversation_id, offered_at DESC);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so each one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS skill_use_stats_user_isolation ON skill_use_stats;
ALTER TABLE skill_use_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY skill_use_stats_user_isolation ON skill_use_stats
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS skill_offers_user_isolation ON skill_offers;
ALTER TABLE skill_offers ENABLE ROW LEVEL SECURITY;
CREATE POLICY skill_offers_user_isolation ON skill_offers
    USING (user_id = current_setting('app.user_id', true));
