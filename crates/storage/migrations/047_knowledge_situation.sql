-- #1125 (absorbing #238): the situations a knowledge entry has been seen in.
--
-- Retrieval is keyed on prompt text, which for a life assistant is the weakest
-- cue available. Encoding specificity says recall depends on the overlap
-- between the cue present when a memory was written and the cue present when it
-- is sought, and the situation - where the person is, what part of the day it
-- is, which day of the week - is a cue the system already knows without asking.
-- This table is what an entry remembers of that, so a recurring situation can
-- reach it again.
--
-- One row per (entry, field, value), which is what makes the whole design work:
--
-- * The match is presence, not frequency. `times` and `last_seen_at` are
--   recorded for eviction and for a later reader, and no ranking rule reads
--   them. Weighting the match by the count would put the use log's own signal
--   into the score twice, and it would leave the retrieve-record-retrieve loop
--   open. As it stands the loop closes after one step: recording a value the
--   row already holds changes nothing any ranking reads.
-- * Accumulation never touches `knowledge_base`. An entry that had to be
--   rewritten to learn where it is useful would restate its own content, move
--   `updated_at`, and put itself back in the embedding backfill queue. #238's
--   accumulation rule is an upsert on this table and nothing else.
-- * The fan of a value is a `count(*)` under one index, which is what lets the
--   read weight each cue value by how much it separates one entry from another.
--   A value the whole store carries separates nobody and is weighted at zero.
--
-- Bounded per entry by the writer, not by a reaper: two of the three fields are
-- closed sets already (four parts of a day, seven days of a week), and the open
-- one is capped at `MAX_SITUATION_VALUES_PER_FIELD` values per field with the
-- least recently seen evicted first.
--
-- Personal data. The table carries `user_id`, every query scopes by it, and it
-- enables its own row-level security below - migration 029's policy list is
-- static and does not reach a table created later. The name is also registered
-- in `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query
-- tool grafts a `user_id` predicate onto any LLM-supplied SQL that names it.
--
-- The foreign key gives the record the entry's own lifetime: a hard reap frees
-- these rows with it, and no row can name an entry that does not exist. Soft
-- deletion is not covered by the key, so a retired entry keeps its record.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS knowledge_situation (
    user_id       TEXT NOT NULL,
    entry_id      TEXT NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    -- The dimension: 'host', 'time_of_day' or 'weekday' today. Stored as text
    -- rather than an enum type so a later dimension is a new value and not a
    -- schema change; the reader skips a name it does not know, because an
    -- unknown dimension is one it cannot score rather than a corrupt row.
    field         TEXT NOT NULL,
    -- The value seen for that dimension. Free-form for a host; one of a closed
    -- set for the two clock fields.
    value         TEXT NOT NULL,
    -- How many observations have landed on this pair. Recorded, and read by
    -- nothing that ranks - see the header.
    times         BIGINT NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, entry_id, field, value)
);

-- The fan read: how many of this user's entries carry one (field, value). The
-- cue asks this once per field per turn, so it must not be a sequential scan.
CREATE INDEX IF NOT EXISTS knowledge_situation_fan_idx
    ON knowledge_situation (user_id, field, value);

-- The eviction read: the least recently seen value of one field on one entry.
CREATE INDEX IF NOT EXISTS knowledge_situation_evict_idx
    ON knowledge_situation (user_id, entry_id, field, last_seen_at DESC);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so this one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS knowledge_situation_user_isolation ON knowledge_situation;
ALTER TABLE knowledge_situation ENABLE ROW LEVEL SECURITY;
CREATE POLICY knowledge_situation_user_isolation ON knowledge_situation
    USING (user_id = current_setting('app.user_id', true));
