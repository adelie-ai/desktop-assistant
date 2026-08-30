-- #1350: the episode use log - which past turns the `[Recall]` block put in
-- front of the model, and which of those it went on to open.
--
-- The block offers episodes nobody searched for, so the two facts that judge
-- such an offer have to be recorded somewhere: an episode surfaced on twenty
-- prompts and opened on none is an offer the arm would be better without, and
-- one opened again and again has earned its place at the top of the arm.
-- Ranking reads the second (`domain::activation`), and a person reads the
-- first. An arm without this log ranks on semantic distance alone, which is
-- search rather than activation, and nothing would ever record whether an
-- offered episode was useful.
--
-- Why not the tables migration 044 created, and why not migration 048's.
-- Same shape, different key each time. `knowledge_use_stats` and
-- `knowledge_offers` carry a foreign key to `knowledge_base(id)`, and an
-- episode has no row in that table. `skill_use_stats` is keyed on a catalog
-- name and deliberately carries no foreign key, because a skill is never
-- deleted - an episode is, whenever its conversation is, so its use rows need
-- a cascade of their own.
--
-- Keyed on the digest's row id, scoped to the reading user. The id is what
-- `turn_digests` is read by and what the block puts on a line, so it is the
-- only identity the model can act on. The use is per-user because one
-- person's opens say nothing about another's.
--
-- The foreign key is what the deletion story rests on. Deleting a conversation
-- deletes its digests (migration 062), and this cascade then frees their use
-- rows - without it, a person who deleted a conversation would leave counters
-- behind naming turns that no longer exist.
--
-- No marks table. The knowledge log records a third act - "this helped" or
-- "this was wrong" - and no tool sets one on an episode. A table with no
-- writer is a table nobody maintains; the act arrives with the tool that
-- performs it.
--
-- No situation table either, for the same reason: nothing records the
-- situation an episode was opened in, so the recall arm answers that term with
-- its "no signal" constant rather than with a guess.
--
-- Both tables are personal data. Each carries `user_id`, every query scopes by
-- it, both enable their own RLS policy below (migration 029's list is static
-- and does not reach a table created later), and both are registered in
-- `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query tool
-- grafts a `user_id` predicate onto any LLM-supplied SQL that names them.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS episode_use_stats (
    user_id         TEXT NOT NULL,
    -- The digest's row id, which is the handle the episode is fetched by.
    episode_id      TEXT NOT NULL REFERENCES turn_digests (id) ON DELETE CASCADE,
    -- How many times the episode appeared in a [Recall] block. Mainly a
    -- denominator: surfaced is not useful.
    offered_count   BIGINT NOT NULL DEFAULT 0,
    -- How many times the model read the turn after it was offered.
    opened_count    BIGINT NOT NULL DEFAULT 0,
    -- When the episode first entered the log. With the counters, this is what
    -- the tail approximation reads for the uses that fell out of the window.
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_offered_at TIMESTAMPTZ,
    -- The most recent use timestamps, newest first. The writer caps the
    -- length, so this array is bounded however long the episode lives.
    recent_uses     TIMESTAMPTZ[] NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, episode_id)
);

-- The cascade above deletes by `episode_id` alone, which the primary key leads
-- with `user_id` instead - so that delete would scan. One index on the
-- referencing column is what keeps deleting a conversation cheap.
CREATE INDEX IF NOT EXISTS episode_use_stats_episode_idx
    ON episode_use_stats (episode_id);

-- The standing episode offers: which past turns are in front of the model
-- right now, and where.
--
-- An open counts only against a standing offer, and counting it deletes the
-- row - so a second read of the same episode in the same turn is one open, and
-- a read of an episode nothing offered is not an open at all.
--
-- Bounded by how it is written. A [Recall] block renders once per turn and
-- deletes this conversation's rows before inserting its own, so a conversation
-- holds one turn's offers. Nothing else offers an episode today; the writer
-- caps the set anyway, on the same rule the knowledge log's writer follows.
CREATE TABLE IF NOT EXISTS episode_offers (
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    episode_id      TEXT NOT NULL REFERENCES turn_digests (id) ON DELETE CASCADE,
    offered_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, conversation_id, episode_id)
);

-- The two writes that are not keyed on the episode: the per-turn clear, and
-- the cap that trims a conversation to its newest offers.
CREATE INDEX IF NOT EXISTS episode_offers_conversation_idx
    ON episode_offers (user_id, conversation_id, offered_at DESC);

-- And the cascade's own, for the reason the stats table's index gives.
CREATE INDEX IF NOT EXISTS episode_offers_episode_idx
    ON episode_offers (episode_id);

-- Row-level security. Migration 029's list is static and does not reach a
-- table created later, so each one enables its own. `current_setting(
-- 'app.user_id', true)` is NULL when the GUC is unset, and `user_id = NULL` is
-- NULL, so a read path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS episode_use_stats_user_isolation ON episode_use_stats;
ALTER TABLE episode_use_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY episode_use_stats_user_isolation ON episode_use_stats
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS episode_offers_user_isolation ON episode_offers;
ALTER TABLE episode_offers ENABLE ROW LEVEL SECURITY;
CREATE POLICY episode_offers_user_isolation ON episode_offers
    USING (user_id = current_setting('app.user_id', true));
