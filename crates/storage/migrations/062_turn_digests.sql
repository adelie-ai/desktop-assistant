-- #1349: the episodic turn index - one digest per turn, scoped to the person.
--
-- The conversation is the one memory surface that cannot be recognized.
-- `messages` carries no embedding, only a lexical `tsvector` with tool rows
-- excluded, so a past turn is reachable by position, by an always-injected
-- rolling summary, or by a lexical search the model must think to call. All
-- three need the model to already suspect that a past turn matters.
--
-- The harness already builds the artifact that fixes this: a bounded,
-- provenance-stamped digest of each turn, written with no model call (#1207).
-- It used to land on the conversation's own scratchpad, where it duplicated
-- the transcript beside it and was invisible from every other conversation.
-- This table is its home instead.
--
-- Four properties are carried by the schema rather than by the code above it.
--
-- * **A digest belongs to one person and one conversation, and goes with the
--   conversation.** Once the store is user-scoped a digest's lifecycle is no
--   longer the conversation's, so the cascade has to be stated: without it,
--   deleting a conversation would be a promise the product does not keep.
--   `user_id` is carried on the row as well, the same way `messages` carries
--   it, so every read scopes without a join.
-- * **One digest per turn, per person.** The identity is (user_id,
--   conversation_id, opening_message_id) - the message that opened the turn -
--   so a re-run of the capture, a redelivery, or a backfill pass over a turn
--   the harness already captured leaves one row rather than a second copy of
--   somebody's conversation (AGENTS.md 8.4).
--
--   `user_id` leads the key so the conflict target cannot cross tenants by
--   construction. Without it the key is (conversation_id, opening_message_id)
--   and the foreign key only requires the conversation to EXIST, not to
--   belong to the writer - so a second tenant could insert against another
--   person's conversation and squat the row, after which the rightful
--   owner's upsert conflicts, is refused by the `EXCLUDED.user_id` guard on
--   the DO UPDATE, and returns nothing. The write would be silently dropped.
--   With `user_id` in the key the guard becomes defence in depth rather than
--   the only thing standing there.
-- * **A digest can be dispositioned.** A knowledge entry carries a
--   disposition and a soft-deletion story so the machinery that withholds and
--   retires a claim can reach it; an episode that outlives its conversation
--   needs the same hooks, with the same vocabulary and the same rule that
--   'superseded' and 'redundant' must name their successor. Migration 056
--   holds the reasoning for both, and this table follows it rather than
--   inventing a second vocabulary. Disposition is what this change delivers;
--   `deleted_at` is reserved for a trash path a later ticket adds, and is
--   written by nothing today - see the column's own comment below.
-- * **A digest is embedded.** `embedding` / `embedding_model` take the same
--   shape as every other embedded table (migration 040), so the stale sweep
--   and the backfill in `crates/storage/src/embedding_backfill.rs` reach it
--   through `EMBEDDED_TABLES` with no special case.
--
-- `superseded_by` is deliberately not a foreign key, for the reason migration
-- 038 gives for `knowledge_base`: a successor can be hard-deleted, so a
-- dangling id is expected and is read as such.
--
-- This table is personal data of the widest kind in this schema: a digest
-- holds the person's own words and what the assistant answered. It carries
-- `user_id`, every query scopes by it, it enables its own RLS policy below
-- (migration 029's policy list is static and does not reach a table created
-- later), and it is registered in `PERSONAL_DATA_TABLES`
-- (crates/storage/src/database.rs) so the db_query tool grafts a `user_id`
-- predicate onto any LLM-supplied SQL that names it.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

-- The key `turn_digests`'s foreign key points at. `conversations` has only
-- `id` as its primary key, and a foreign key can reference a unique
-- constraint only - so pointing at (user_id, id) needs that pair declared
-- unique first.
--
-- It cannot fail on existing data. `id` is already the primary key, so it is
-- unique on its own, and a pair whose second column is unique is unique
-- whatever the first column holds. No duplicate is constructible, and the
-- statement adds an index rather than rejecting a row.
--
-- ADD CONSTRAINT has no IF NOT EXISTS form, hence the catalog guard. The
-- owner of `conversations` can add this, so it stays inside the owner-only
-- migration set that migration 029 established (see
-- `run_migrations_succeeds_as_unprivileged_table_owner`).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'conversations'::regclass
          AND conname = 'conversations_user_id_id_key'
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_user_id_id_key UNIQUE (user_id, id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS turn_digests (
    id                 TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    conversation_id    TEXT NOT NULL,
    -- The message that opened the turn: the digest's identity, and the handle
    -- a reader follows back into the transcript.
    opening_message_id TEXT NOT NULL,
    content            TEXT NOT NULL,
    -- Whether the turn that produced this text had already read content from
    -- outside the trust boundary (#1247). The assistant's closing text is in
    -- the digest, and a turn that read a page routinely quotes it.
    after_outside_read BOOLEAN NOT NULL DEFAULT FALSE,
    disposition        TEXT NOT NULL DEFAULT 'active',
    disposition_reason TEXT,
    superseded_by      TEXT,
    embedding          vector[],
    embedding_model    TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- RESERVED, and written by nothing today. There is no trash or restore
    -- path for a digest yet; the three reads in
    -- `crates/storage/src/turn_digest.rs` already filter on this column so
    -- that the ticket which adds one changes a writer and not every reader.
    -- Until then every row has it NULL, and `deleted_at` keeps exactly one
    -- meaning when it arrives: the row is in the trash. Disposition is
    -- orthogonal to it, so a dispositioned row is normally live.
    deleted_at         TIMESTAMPTZ,
    UNIQUE (user_id, conversation_id, opening_message_id),
    -- Composite on purpose. A single-column reference to `conversations(id)`
    -- asks only whether the conversation EXISTS, not who owns it, so a write
    -- naming another person's conversation is accepted by the database and
    -- refused only by whatever the application remembers to check. Referencing
    -- (user_id, id) makes the pair itself the thing that must exist, so a
    -- cross-tenant digest cannot be stored at all - by the schema, not by a
    -- caller's care.
    --
    -- No ON UPDATE clause, so it defaults to NO ACTION: moving a conversation
    -- to another owner is refused while it holds digests, rather than silently
    -- carrying somebody's episodes across to a different person.
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations (user_id, id) ON DELETE CASCADE
);

-- The person's own episodes, newest first, across every conversation they
-- own - the read that makes the record reachable at all.
CREATE INDEX IF NOT EXISTS turn_digests_user_created_idx
    ON turn_digests (user_id, created_at DESC)
    WHERE deleted_at IS NULL;

-- The cascade needs no index of its own. Because the foreign key is composite,
-- deleting a conversation runs
-- `DELETE FROM turn_digests WHERE user_id = ... AND conversation_id = ...`,
-- and the unique key above leads with exactly those two columns - measured,
-- not assumed: the planner uses
-- `turn_digests_user_id_conversation_id_opening_message_id_key` with both
-- columns as index conditions. A single-column index on `conversation_id`
-- would be used with `user_id` demoted to a filter, which is a worse plan for
-- the same work, so there is deliberately no second index here.

-- The same vocabulary migration 056 pins for knowledge_base. ADD CONSTRAINT
-- has no IF NOT EXISTS form, hence the catalog guard.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'turn_digests'::regclass
          AND conname = 'turn_digests_disposition_chk'
    ) THEN
        ALTER TABLE turn_digests
            ADD CONSTRAINT turn_digests_disposition_chk
            CHECK (disposition IN
                ('active', 'refuted', 'superseded', 'redundant', 'obsolete', 'trivial'));
    END IF;
END $$;

-- 'superseded' and 'redundant' resolve through the link, so they must name
-- one. Nothing requires that they are the only dispositions that may - see
-- migration 056 for why the reverse rule refused rows carrying real
-- information.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'turn_digests'::regclass
          AND conname = 'turn_digests_superseded_by_chk'
    ) THEN
        ALTER TABLE turn_digests
            ADD CONSTRAINT turn_digests_superseded_by_chk
            CHECK (disposition NOT IN ('superseded', 'redundant') OR superseded_by IS NOT NULL);
    END IF;
END $$;

-- Row-level security. Migration 029's list is static and does not reach a
-- table created later, so this one enables its own. `current_setting(
-- 'app.user_id', true)` is NULL when the GUC is unset, and `user_id = NULL` is
-- NULL, so a read path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS turn_digests_user_isolation ON turn_digests;
ALTER TABLE turn_digests ENABLE ROW LEVEL SECURITY;
CREATE POLICY turn_digests_user_isolation ON turn_digests
    USING (user_id = current_setting('app.user_id', true));
