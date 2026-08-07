-- #698: the use log - what a knowledge entry was offered for, what was opened,
-- and what was marked.
--
-- Retrieval ranks entries with weights, and nothing recorded that an entry was
-- ever put in front of the model or ever taken up. Without those two facts
-- every coefficient in the retrieval path is chosen by hand and stays chosen by
-- hand, fitted to one store and carried to every other. These two tables are
-- what lets a deployment measure its own.
--
-- Bounded per entry, deliberately. `knowledge_use_stats` holds ONE row per
-- entry: aggregate counters, a first-seen stamp, and the most recent use
-- timestamps in an array capped by the writer. That is the standard hybrid for
-- ACT-R base-level activation - exact over the recent window, approximated over
-- the tail - and it is why there is no per-event table here. An unbounded event
-- log would grow with every prompt and hold nothing the score reads.
--
-- `knowledge_use_marks` is bounded the same way: the primary key includes the
-- source, so each entry holds at most one standing mark per source. A mark is a
-- current opinion, not an event, so a second mark from the same source replaces
-- the first.
--
-- Both tables are personal data. Each carries `user_id`, every query scopes by
-- it, and both are added to the RLS backstop below - migration 029's policy
-- list is static, so a user-scoped table added later must enable its own.
-- Both names are also registered in `PERSONAL_DATA_TABLES`
-- (crates/storage/src/database.rs) so the db_query tool grafts a `user_id`
-- predicate onto any LLM-supplied SQL that names them.
--
-- The foreign key to `knowledge_base` gives the log the entry's own lifetime:
-- a hard reap of a retired entry frees its use rows with it, and no row can
-- name an entry that does not exist. Soft deletion is NOT covered by the key -
-- a retired entry keeps its record, which is what lets a later reader see that
-- it was offered often and never opened.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS knowledge_use_stats (
    user_id         TEXT NOT NULL,
    entry_id        TEXT NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    -- How many times the entry appeared in a [Recall] block or a search
    -- result. Mainly a denominator: surfaced is not useful.
    offered_count   BIGINT NOT NULL DEFAULT 0,
    -- How many times something fetched the entry by id after it was offered.
    opened_count    BIGINT NOT NULL DEFAULT 0,
    -- How many times a mark was set on the entry, of either polarity.
    marked_count    BIGINT NOT NULL DEFAULT 0,
    -- When the entry first entered the log. With the counters, this is what
    -- the tail approximation reads for the uses that fell out of the window.
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_offered_at TIMESTAMPTZ,
    -- The most recent use timestamps, newest first. The writer caps the length,
    -- so this array is bounded however long the entry lives.
    recent_uses     TIMESTAMPTZ[] NOT NULL DEFAULT '{}',
    -- The conversation this entry is currently standing offered in, and when
    -- the offer was made. An open counts only against a standing offer, and
    -- counting it takes the offer down - so a second fetch in the same turn is
    -- one open, and a read of an entry nothing offered is not an open at all.
    offer_conversation_id TEXT,
    offered_at      TIMESTAMPTZ,
    PRIMARY KEY (user_id, entry_id)
);

-- The [Recall] block clears a conversation's standing offers before it makes
-- its own, which is a write keyed on the conversation rather than the entry.
-- Partial, because a row with no standing offer is never the target.
CREATE INDEX IF NOT EXISTS knowledge_use_stats_offer_idx
    ON knowledge_use_stats (user_id, offer_conversation_id)
    WHERE offer_conversation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS knowledge_use_marks (
    user_id     TEXT NOT NULL,
    entry_id    TEXT NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    -- 'model' or 'person'. A person's mark outranks the model's, and the two
    -- are held apart rather than averaged. No client offers a person's mark
    -- yet; the value exists so a human judgement has somewhere to go the day
    -- one does, rather than a schema change standing in the way.
    marked_by   TEXT NOT NULL,
    -- 'positive' or 'negative'. A negative mark is not the absence of a
    -- positive one: "offered, opened, and it was wrong" is the strongest
    -- evidence for retiring an entry that this log can hold.
    polarity    TEXT NOT NULL,
    reason      TEXT,
    marked_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, entry_id, marked_by),
    CONSTRAINT knowledge_use_marks_marked_by_known
        CHECK (marked_by IN ('model', 'person')),
    CONSTRAINT knowledge_use_marks_polarity_known
        CHECK (polarity IN ('positive', 'negative'))
);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so each one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS knowledge_use_stats_user_isolation ON knowledge_use_stats;
ALTER TABLE knowledge_use_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY knowledge_use_stats_user_isolation ON knowledge_use_stats
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS knowledge_use_marks_user_isolation ON knowledge_use_marks;
ALTER TABLE knowledge_use_marks ENABLE ROW LEVEL SECURITY;
CREATE POLICY knowledge_use_marks_user_isolation ON knowledge_use_marks
    USING (user_id = current_setting('app.user_id', true));
