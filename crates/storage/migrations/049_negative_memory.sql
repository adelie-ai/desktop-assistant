-- #1126: negative memory - the actions that went badly, and what they went
-- badly with.
--
-- An action that produced a bad outcome in a particular context has to come
-- back BEFORE the same action is taken again. A burn recalled afterwards taught
-- nothing, so this is not read at prompt time with everything else: it is read
-- at the decision point, keyed on the act about to be taken.
--
-- Two tables, and the split is the design. `negative_memory` is the lesson;
-- `negative_memory_facet` is what the lesson is scoped to. Scope is a set of
-- required facets rather than a column per dimension, because the set shrinks:
-- a second occurrence somewhere else drops the facets it disagrees with, which
-- is the only thing in the whole feature that makes a burn wider.
--
-- Four properties are carried by the schema rather than by the code above it.
--
-- * **One live lesson per identity.** The partial unique index below allows one
--   un-extinguished burn per (user, action, fingerprint), and any number of
--   extinguished ones beside it. The fingerprint is taken over the ACTION's own
--   arguments and never over the situation - the situation is what widens, so
--   it cannot also be what identifies. Without the partial predicate, writing a
--   correction over a burn would collide with the burn it corrects.
-- * **Extinction is an overlay, not a delete.** A burn that stopped applying
--   keeps every column it had; a `correction` row is written beside it and
--   `superseded_by` names that row. "This went badly, and later it stopped
--   going badly" is knowledge, and deleting it lets the same lesson be learned
--   again from nothing.
-- * **Strength is not stored.** It is a function of `last_confirmed_at` and the
--   clock, computed by `domain::negative_memory`. A stored number would be
--   wrong the moment nothing wrote it, and two readers would then disagree
--   about whether a burn is still loud enough to interrupt anything.
-- * **The writer bounds the table.** There is no sweep. On the next write path
--   a row the reader could never act on is deleted, the same way migration
--   047's situation record is bounded by its own writer, and the foreign key
--   then takes its facets with it. Two ways a row qualifies, and both are
--   stated in `domain::negative_memory` as well, because what a reader believes
--   and what the store does have to be the same thing: nothing has confirmed it
--   for long enough, or its stamp sits so far AHEAD of the reader's clock that
--   the strength arithmetic can never raise it.
--
-- Both tables are personal data. Each carries `user_id`, every query scopes by
-- it, both enable their own RLS policy below (migration 029's policy list is
-- static and does not reach a table created later), and both are registered in
-- `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query tool
-- grafts a `user_id` predicate onto any LLM-supplied SQL that names them. What
-- a person's assistant tried and how it failed is as personal as the work it
-- was doing.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS negative_memory (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    -- The tool the memory is about. A tool call is the one act in a turn with
    -- an identity that can be matched exactly, and exact matching is what holds
    -- the scope narrow.
    action            TEXT NOT NULL,
    -- Digest of the argument facets alone: the burn's handle. Two failures
    -- share a lesson when their action and their fingerprint both match.
    fingerprint       TEXT NOT NULL,
    -- 'burn' (the lesson) or 'correction' (the overlay written over one that
    -- stopped applying). Stored as text rather than an enum type so a later
    -- kind is a new value and not a schema change; the reader skips a name it
    -- does not know, because an unknown kind is one it cannot act on.
    kind              TEXT NOT NULL,
    -- What went wrong, in the words of whatever recorded it. On a correction,
    -- what changed.
    outcome           TEXT NOT NULL,
    -- How many times this lesson has been recorded. Read by the order the
    -- warning is written in, and by nothing that decides whether a burn fires
    -- or how wide it is: widening is decided by which facets a later
    -- occurrence disagreed with, never by a count.
    occurrences       BIGINT NOT NULL DEFAULT 1,
    written_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- The last time the same lesson was recorded. Decay runs from here, so a
    -- confirmation restores full strength by moving this and nothing else.
    last_confirmed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- The correction that extinguished this, when one has.
    superseded_by     TEXT,
    superseded_at     TIMESTAMPTZ
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'negative_memory'::regclass
          AND conname = 'negative_memory_kind_chk'
    ) THEN
        ALTER TABLE negative_memory
            ADD CONSTRAINT negative_memory_kind_chk
            CHECK (kind IN ('burn', 'correction'));
    END IF;
END $$;

-- One live lesson per identity, and any number of extinguished ones beside it.
-- The predicate is what lets an overlay share its original's identity.
CREATE UNIQUE INDEX IF NOT EXISTS negative_memory_live_identity_idx
    ON negative_memory (user_id, action, fingerprint)
    WHERE kind = 'burn' AND superseded_by IS NULL;

-- The per-turn read: this user's live burns, loudest first.
CREATE INDEX IF NOT EXISTS negative_memory_live_idx
    ON negative_memory (user_id, last_confirmed_at DESC)
    WHERE kind = 'burn' AND superseded_by IS NULL;

-- The history read: everything ever recorded against one action, overlays
-- included.
CREATE INDEX IF NOT EXISTS negative_memory_action_idx
    ON negative_memory (user_id, action, last_confirmed_at DESC);

-- What a memory is scoped to: one row per required facet.
--
-- One value per facet, unlike migration 047's situation record, which holds
-- every value an entry has been seen in. The difference is the point: a record
-- is a history, and a scope is a condition.
CREATE TABLE IF NOT EXISTS negative_memory_facet (
    user_id   TEXT NOT NULL,
    memory_id TEXT NOT NULL REFERENCES negative_memory(id) ON DELETE CASCADE,
    -- 'argument' (part of the action's identity, never dropped) or 'situation'
    -- (circumstance, dropped by an occurrence that disagrees with it).
    kind      TEXT NOT NULL,
    -- The argument's name, or the situation field's stored name ('host',
    -- 'time_of_day', 'weekday').
    name      TEXT NOT NULL,
    -- The value that must match. Compared by equality, so it is stored whole
    -- and a value too long to store is not a facet at all.
    value     TEXT NOT NULL,
    PRIMARY KEY (user_id, memory_id, kind, name)
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'negative_memory_facet'::regclass
          AND conname = 'negative_memory_facet_kind_chk'
    ) THEN
        ALTER TABLE negative_memory_facet
            ADD CONSTRAINT negative_memory_facet_kind_chk
            CHECK (kind IN ('argument', 'situation'));
    END IF;
END $$;

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so each one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS negative_memory_user_isolation ON negative_memory;
ALTER TABLE negative_memory ENABLE ROW LEVEL SECURITY;
CREATE POLICY negative_memory_user_isolation ON negative_memory
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS negative_memory_facet_user_isolation ON negative_memory_facet;
ALTER TABLE negative_memory_facet ENABLE ROW LEVEL SECURITY;
CREATE POLICY negative_memory_facet_user_isolation ON negative_memory_facet
    USING (user_id = current_setting('app.user_id', true));
