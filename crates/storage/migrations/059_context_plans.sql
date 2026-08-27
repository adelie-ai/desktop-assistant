-- #1327: one row per turn recording the retrieval plan - every candidate
-- the [Recall] lookup considered, each one's activation score broken down
-- by term, which candidates it offered, and which the model later opened.
--
-- `context_breakdowns` (054) records what filled the prompt. This table
-- records what retrieval considered before any of it filled anything. The
-- two rows differ in size (a breakdown row is small; a plan can run tens
-- of KB) and in when they are written (the plan on the first round, the
-- breakdown at the end of the turn), so they stay two tables rather than
-- one column added to 054.
--
-- Keyed on (user_id, request_id): the turn's own correlation id, the same
-- shape 054 uses and for the same reason. The key makes a repeat of one
-- turn's write replace its row rather than add a second. It does NOT make
-- the row count a property of the schema: the value is the client's own
-- turn id, so a client that reuses one id for two turns is writing the
-- same key twice on purpose or by mistake. The upsert therefore refuses to
-- move a row between conversations - see the WHERE on the DO UPDATE in
-- `crates/storage/src/context_plans/mod.rs` - so a reused id leaves the
-- first conversation's record intact rather than silently relocating it,
-- and the daemon warns.
--
-- `weights` and `arms` are JSONB objects, and `candidates` and `opened` are
-- JSONB arrays, for the drift argument 054 makes for its own JSONB columns:
-- the candidate shape has one definition, in Rust (`PlannedCandidate` in
-- `crates/core/src/ports/context_plan.rs`), and a column list in SQL would
-- be a second definition able to drift from it.
--
-- `candidates` is cut to 512 entries (`MAX_PLANNED_CANDIDATES`) however
-- many the lookup considered; `considered_count` keeps the true count and
-- `truncated` says whether the array was cut, so a cut plan never reads as
-- a smaller turn than it was. The cut happens before this table is ever
-- reached - see `build_context_plan` in `crates/core/src/recall.rs`.
--
-- `opened` starts empty and gains entries after the first write, as the
-- model fetches candidates during the turn.
--
-- `recall_ran = false` is a recorded turn, not a missing one: a turn whose
-- first round rendered no recall lookup still gets a row, with every array
-- empty. "No retrieval" is a fact about the turn, and this table states it
-- rather than leaving the reader to guess from an absent row.
--
-- Personal data. The row carries `user_id`, every query scopes by it, the
-- table is registered in `PERSONAL_DATA_TABLES`
-- (`crates/storage/src/database.rs`) so the db_query tool grafts a
-- `user_id` predicate onto LLM-supplied SQL, and RLS is enabled below -
-- migration 029's policy list is static and does not reach a table
-- created later.
--
-- No foreign key to `conversations`, for the same reason 054 has none: a
-- conversation deleted while a turn is in flight must not hold this row
-- open, and the read is scoped by conversation id anyway.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS context_plans (
    user_id               TEXT NOT NULL,
    -- The turn's correlation id, the same value 054 keys on.
    request_id            TEXT NOT NULL,
    conversation_id       TEXT NOT NULL,
    -- Whether a recall lookup ran at all this turn.
    recall_ran            BOOLEAN NOT NULL,
    -- The prompt text the lookup embedded, bounded to 8 KiB by the writer.
    -- NULL when `recall_ran` is false.
    query_text            TEXT,
    query_text_truncated  BOOLEAN NOT NULL DEFAULT FALSE,
    -- RECALL_BAR as applied this turn.
    bar                    DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- The ActivationWeights every candidate below was scored under.
    weights                JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Which shape of ActivationTerms `weights` produced. A reader comparing
    -- an old turn's terms against today's ranking reads this to know
    -- whether the two are the same computation.
    scorer_version         TEXT NOT NULL DEFAULT '',
    -- The three arms' scan summaries: entry, note, skill.
    arms                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Every candidate the lookup considered, ranked order within each arm,
    -- cut to MAX_PLANNED_CANDIDATES.
    candidates              JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- The true candidate count before the array above was cut.
    considered_count        INTEGER NOT NULL DEFAULT 0,
    truncated                BOOLEAN NOT NULL DEFAULT FALSE,
    -- Ids fetched by the model during this turn, in the order opened.
    opened                   JSONB NOT NULL DEFAULT '[]'::jsonb,
    recorded_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, request_id)
);

-- The conversation read, newest first by write time.
CREATE INDEX IF NOT EXISTS context_plans_conversation_idx
    ON context_plans (user_id, conversation_id, recorded_at);

-- The retention sweep's own access path: delete by age, across every user.
CREATE INDEX IF NOT EXISTS context_plans_recorded_at_idx
    ON context_plans (recorded_at);

-- Row-level security. Migration 029's list is static and does not reach a
-- table created later, so this table enables its own, on the same terms
-- 054 states: `current_setting('app.user_id', true)` is NULL when the GUC
-- is unset, and `user_id = NULL` is NULL, so an unpinned read sees zero
-- rows. This backstop binds the un-privileged `adele_query` role the
-- db_query tool runs as, not the daemon's own connection, which owns the
-- table and is exempt from a non-FORCE policy; what holds the daemon's
-- reads to one tenant is the `user_id = $1` predicate each of them binds.
DROP POLICY IF EXISTS context_plans_user_isolation ON context_plans;
ALTER TABLE context_plans ENABLE ROW LEVEL SECURITY;
CREATE POLICY context_plans_user_isolation ON context_plans
    USING (user_id = current_setting('app.user_id', true));
