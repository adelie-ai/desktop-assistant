-- #588: one row per turn saying what filled its prompt.
--
-- The per-part measurement already happens on every turn - the assembler fills
-- it as it lays each block out, the turn span carries it, the metrics facade
-- accumulates it - and is then dropped. So "what ate my context?" is
-- answerable for a turn somebody is watching and for no turn that has already
-- ended. This table is where that measurement stops being discarded.
--
-- The budget tier is kept for the same reason. The daemon resolves every
-- budget as a purpose override, a connector's curated table, the universal
-- fallback or a learned cap, and nothing leaves the daemon: a curated 200k and
-- a silent fallback 200k are the same number and a different situation, and
-- nothing downstream could tell them apart.
--
-- Keyed on (user_id, request_id), which is the turn's own correlation id and
-- also its trace id. That makes the write idempotent by construction - a
-- retried or re-driven turn replaces its row rather than adding a second one -
-- and gives the single-turn read a key a client already holds, because the
-- same id is on every event the turn streamed.
--
-- ## Two measurements, and they must not be read as one
--
-- `estimated_parts` is what the assembler measured, with the estimator the
-- context budget itself uses. `provider_used_tokens` is what the provider
-- reported for that same prompt. They are two counters over one thing and they
-- do not agree; the difference is itself the signal. Nothing in this schema
-- sums them, derives one from the other, or defaults one to the other, and the
-- column names say which is which rather than leaving it to be inferred.
--
-- The absence rules differ, and both are deliberate. A part that did not render
-- stores a zero, because the assembler always knows whether it emitted a block.
-- A provider that declined to report stores NULL, because a zero would invent a
-- measurement.
--
-- ## Why the parts are one JSONB column and not ten
--
-- The ten parts have exactly one definition, `PromptPart` in
-- `crates/core/src/telemetry/prompt.rs`, and every writer and reader derives
-- its names from that enum. Ten columns would be a second list of parts, in
-- SQL, able to drift from the first - and a drifted part list reports one
-- part's cost under another part's name while every total still adds up, which
-- is the failure this whole feature exists to prevent. The object is keyed by
-- the enum's own stable labels, so `estimated_parts->>'transcript'` works for
-- ad-hoc queries and a part added later needs no migration.
--
-- `budget_source` is stored as its stable label for the same reason, with no
-- CHECK constraint: a list of tiers in SQL is a second definition of the tier
-- set, and the writer is the only thing that puts values here.
--
-- Personal data. The row carries `user_id`, every query scopes by it, the
-- table is registered in `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs)
-- so the db_query tool grafts a `user_id` predicate onto LLM-supplied SQL, and
-- RLS is enabled below - migration 029's policy list is static and does not
-- reach a table created later.
--
-- No foreign key to `conversations`. A conversation deleted while a turn is in
-- flight must not be held open by this row, and the read is scoped by
-- conversation id anyway, so a row naming a conversation that has gone is
-- simply unreachable rather than wrong.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS context_breakdowns (
    user_id               TEXT NOT NULL,
    -- The turn's correlation id, which is also its trace id.
    request_id            TEXT NOT NULL,
    conversation_id       TEXT NOT NULL,
    -- Where the turn begins in the conversation: the message ordinal its user
    -- prompt took. Lets a reader jump from a record to the messages it
    -- describes, the same handle the tool-usage view uses.
    turn_ordinal          INTEGER NOT NULL,
    -- The model the turn actually ran on, as the route resolved it.
    model                 TEXT NOT NULL,
    -- Prompt tokens the PROVIDER reported. NULL means the provider said
    -- nothing, which is not zero.
    provider_used_tokens  BIGINT,
    -- The input-token budget the turn resolved, and which tier resolved it.
    -- Both NULL for a turn that ran with no budget installed.
    budget_tokens         BIGINT,
    budget_source         TEXT,
    -- Whether proactive compaction ran on this turn.
    compaction_active     BOOLEAN NOT NULL DEFAULT FALSE,
    -- ESTIMATED tokens per prompt part, keyed by `PromptPart::as_label`.
    estimated_parts       JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- How many tool schemas the prompt advertised. A count, not a token figure.
    advertised_tool_count INTEGER NOT NULL DEFAULT 0,
    -- How many messages the turn read as a pointer, a head or a notice instead
    -- of as their stored content. What the transcript figure is not charging
    -- for; the transcript itself still holds every byte.
    projected_messages    INTEGER NOT NULL DEFAULT 0,
    recorded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, request_id)
);

-- The conversation read, in conversation order. Pages stay stable while the
-- conversation grows, because a new turn appends past the end of every page a
-- caller has already read.
CREATE INDEX IF NOT EXISTS context_breakdowns_conversation_idx
    ON context_breakdowns (user_id, conversation_id, turn_ordinal);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so each one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS context_breakdowns_user_isolation ON context_breakdowns;
ALTER TABLE context_breakdowns ENABLE ROW LEVEL SECURITY;
CREATE POLICY context_breakdowns_user_isolation ON context_breakdowns
    USING (user_id = current_setting('app.user_id', true));
