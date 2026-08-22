-- #1252: the full text of every turn - the request as sent, the reply, the
-- tool calls and their results.
--
-- The conversation is already stored. The PROMPT is not: the system prompt,
-- the `[Recall]` block, the scratchpad injection and the post-eviction window
-- exist in memory for one provider call and are then gone, so the exact bytes
-- the model was shown are otherwise unrecoverable. That is the difference
-- between "what was said" and "what the assistant was reading when it said
-- it", and only the second answers why it acted as it did.
--
-- Two tables, and the split follows what is known when. `turn_records` is one
-- row per turn, written before the first round, so a turn that then fails
-- still has a record saying it happened and where it dispatched.
-- `turn_round_records` is one row per round inside it, written as soon as the
-- provider answers and completed when that round's tool calls resolve.
--
-- Three properties are carried by the schema rather than by the code above it.
--
-- * **A round belongs to exactly one turn, and goes with it.** The foreign key
--   cascades, so the retention sweep deletes turns and the rounds follow. A
--   sweep that dropped the turn row and left the rounds would report a window
--   it does not keep, and the rounds are where the content is.
-- * **A turn and a round are each written at most once.** The primary keys are
--   the identities - (user_id, correlation_id) and (user_id, correlation_id,
--   round) - so a retry, a redelivery or a second daemon replaying the same
--   turn leaves one record rather than a second copy of somebody's
--   conversation.
-- * **Every record names its user and its conversation, and the two agree.**
--   Both columns are on both tables, because the profiler this replaces wrote
--   entries carrying neither and a file full of unattributable prompts answers
--   nothing. The correlation id is the CLIENT's to choose, so a client that
--   reuses one across two conversations would otherwise file rounds of the
--   second under a turn row naming the first. Two mechanisms refuse that, and
--   both are needed: the foreign key below carries `conversation_id`, which
--   stops a NEW round attaching to a turn of another conversation; and the
--   round upsert in `crates/storage/src/turn_records/mod.rs` guards its update
--   on the same column, because an update that leaves the referencing columns
--   alone re-checks no foreign key at all. Either way the write is declined
--   and reported, never stored as a record that contradicts itself.
--
-- Both tables are personal data of the widest kind in this schema: a round row
-- holds the assembled system prompt, every injected block, the person's own
-- words and whatever a tool read on their behalf. Each carries `user_id`,
-- every query scopes by it, both enable their own RLS policy below (migration
-- 029's policy list is static and does not reach a table created later), and
-- both are registered in `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs)
-- so the db_query tool grafts a `user_id` predicate onto any LLM-supplied SQL
-- that names them.
--
-- Retention is not optional and not unbounded: `sweep_expired_turn_records`
-- deletes by `started_at`, and the daemon's `[inspector] retention_days` holds
-- a floor of one day. The index below is what makes that sweep an index scan
-- rather than a full table read.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS turn_records (
    -- The turn's own correlation id: the value the client stamps on its own
    -- event stream, so a person quoting a reply and this store use one
    -- identifier. It is the trace id too on a turn nobody handed a trace,
    -- which is the ordinary case; a caller that forwarded a `traceparent` to
    -- be continued makes the two differ, and this follows the client's.
    correlation_id  TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    -- Where the turn dispatched. All three are nullable because the daemon's
    -- routing has a documented fall-through: when no concrete live connection
    -- resolves, the turn goes to the statically-configured primary client and
    -- the daemon does not know which connection or model that is. NULL is that
    -- state, recorded rather than guessed at.
    connection_id   TEXT,
    provider        TEXT,
    model           TEXT,
    -- The tool policy this turn resolved to, as its stable spelling. Stored as
    -- text rather than an enum type so a later level is a new value and not a
    -- schema change.
    tool_policy     TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, correlation_id)
);

-- What the round table's foreign key references. The primary key alone would
-- let a round name a conversation its turn does not; this is what makes the
-- pair referenceable, so a disagreement cannot be written at all.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'turn_records'::regclass
          AND conname = 'turn_records_identity_key'
    ) THEN
        ALTER TABLE turn_records
            ADD CONSTRAINT turn_records_identity_key
            UNIQUE (user_id, correlation_id, conversation_id);
    END IF;
END $$;

-- The retention sweep: every turn older than the window, across all users.
CREATE INDEX IF NOT EXISTS turn_records_started_at_idx
    ON turn_records (started_at);

-- A person's own turns, most recent first - the read a conversation-scoped
-- inspector makes.
CREATE INDEX IF NOT EXISTS turn_records_conversation_idx
    ON turn_records (user_id, conversation_id, started_at DESC);

CREATE TABLE IF NOT EXISTS turn_round_records (
    correlation_id  TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    -- One-based, matching the round span and the round's log line, so a record
    -- and a trace read the same way. One value sits past the tool loop's last
    -- round: the wind-down call that closes out a turn whose budget is spent.
    -- See `WIND_DOWN_ROUND` in crates/core/src/service.rs.
    round           INTEGER NOT NULL,
    -- The request exactly as handed to the connector: a JSON array of
    -- messages, each with its role, in the order it was sent, including the
    -- system prompt and every injected block. This is the assembled prompt and
    -- not the conversation - the two differ on every turn.
    request         JSONB NOT NULL,
    -- The reply text the provider streamed, whole. Not a preview: a preview is
    -- what the profiler this replaces stored, and 200 characters of a prompt
    -- answers no question anybody asks of it.
    response_text   TEXT NOT NULL,
    -- The tool calls the model asked for, with their arguments as the model
    -- wrote them.
    response_tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- What those calls returned, as the rows the turn stored. Empty for a
    -- round that answered without calling anything, and for one that was
    -- stopped before its first call resolved.
    tool_results    JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- What the provider reported this round cost, where it reported anything.
    -- Named `token_usage` rather than `usage`, which reads as a SQL keyword at
    -- every call site even though Postgres allows it.
    token_usage     JSONB,
    -- Why the round failed, where it did. NULL is a round the provider
    -- answered.
    error           TEXT,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, correlation_id, round),
    FOREIGN KEY (user_id, correlation_id, conversation_id)
        REFERENCES turn_records (user_id, correlation_id, conversation_id)
        ON DELETE CASCADE
);

-- Row-level security. Migration 029's list is static and does not reach a
-- table created later, so each one enables its own. `current_setting(
-- 'app.user_id', true)` is NULL when the GUC is unset, and `user_id = NULL` is
-- NULL, so a read path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS turn_records_user_isolation ON turn_records;
ALTER TABLE turn_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY turn_records_user_isolation ON turn_records
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS turn_round_records_user_isolation ON turn_round_records;
ALTER TABLE turn_round_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY turn_round_records_user_isolation ON turn_round_records
    USING (user_id = current_setting('app.user_id', true));
