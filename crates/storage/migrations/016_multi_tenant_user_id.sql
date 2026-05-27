-- Issue #102: multi-tenant schema — add `user_id` to personal-data tables.
--
-- Every personal-data table gains a `user_id TEXT NOT NULL` column so the
-- daemon can scope queries by user (issue #105 wires the extraction from
-- JWT `sub`). Single-tenant desktop installs collapse to the sentinel
-- `'default'` user_id, which is also the backfill value for pre-existing
-- rows. The auth-jwt-shared-crate branch carries `sub: String` in its
-- `Claims` struct, so `TEXT` matches the eventual extraction shape.
--
-- Backfill strategy
--   We use the standard `ADD COLUMN … DEFAULT 'default' NOT NULL` form,
--   which Postgres materializes for existing rows. The DEFAULT is left
--   in place so single-tenant deploys continue to work without #105 —
--   inserts that omit `user_id` resolve to `'default'`. #105 will drop
--   the defaults once every call site supplies a real `user_id`. The
--   regression test `inserting_conversation_without_user_id_fails`
--   explicitly drops the default before its INSERT to prove the NOT
--   NULL constraint is in place.
--
-- Reversibility
--   The existing migration tooling (`pool::run_migrations`) is forward-
--   only — it has no `down` concept. Reversing this change requires a
--   manual `ALTER TABLE … DROP COLUMN user_id` plus dropping the new
--   indexes and restoring `tag_registry`'s old single-column PRIMARY KEY.
--   The PR description records this as a known limitation; future
--   migration tooling (sqlx-cli or refinery) would gain a paired
--   down-migration without changing the forward shape.
--
-- Idempotency
--   Every ALTER uses `IF NOT EXISTS` / DO-block guards so a partial
--   apply (or a fresh-install re-apply) is safe. The tag_registry PK
--   swap checks the current PK shape before mutating.

-- ---------------------------------------------------------------------------
-- conversations
-- ---------------------------------------------------------------------------
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Per-user listing ordered by recency — the conversation list hot path.
CREATE INDEX IF NOT EXISTS conversations_user_id_updated_at_idx
    ON conversations (user_id, updated_at DESC);

-- Per-user active (non-archived) filter — the dreaming archival scan and
-- the UI's "active conversations" view both need this.
CREATE INDEX IF NOT EXISTS conversations_user_id_archived_at_idx
    ON conversations (user_id, archived_at);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Per-user, per-conversation, ordered scan — every message-load path.
-- This duplicates information already in `conversations.user_id` (each
-- message inherits its conversation's user) but lets us scope queries
-- without a JOIN, and the index keeps the user check cheap.
CREATE INDEX IF NOT EXISTS messages_user_id_conv_ordinal_idx
    ON messages (user_id, conversation_id, ordinal);

-- ---------------------------------------------------------------------------
-- knowledge_base — covers both KB entries and what was historically
-- `factual_memory` (consolidated into the same table; see
-- `migrate_json::LegacyMemory`).
-- ---------------------------------------------------------------------------
ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Per-user listing / pagination by recency.
CREATE INDEX IF NOT EXISTS knowledge_base_user_id_created_at_idx
    ON knowledge_base (user_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- message_summaries — collapsible summary anchors, child of conversations.
-- ---------------------------------------------------------------------------
ALTER TABLE message_summaries
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS message_summaries_user_id_conv_idx
    ON message_summaries (user_id, conversation_id);

-- ---------------------------------------------------------------------------
-- dreaming_watermarks — per-conversation extraction watermarks.
-- ---------------------------------------------------------------------------
ALTER TABLE dreaming_watermarks
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS dreaming_watermarks_user_id_idx
    ON dreaming_watermarks (user_id);

-- ---------------------------------------------------------------------------
-- tag_registry — formal tag vocabulary learned during dreaming
-- consolidation (#108). Tag names must now be unique per user, not
-- globally, so the PRIMARY KEY moves from `(name)` to `(user_id, name)`.
-- ---------------------------------------------------------------------------
ALTER TABLE tag_registry
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Swap the PK from `(name)` to `(user_id, name)`. Guarded by inspecting
-- the current PK so a re-run is a no-op. The `tag_registry` table's
-- self-referential `deprecated_for_tag TEXT REFERENCES tag_registry(name)`
-- FK had to be dropped before we could mutate the PK; we recreate a
-- broader (user_id, name) reference afterward so cross-user deprecation
-- links can't form. Existing rows backfilled to 'default' keep their
-- self-references intact because the rewritten FK lands inside the
-- 'default' user_id partition.
DO $$
DECLARE
    current_pk_cols TEXT;
    dep_fk_name     TEXT;
BEGIN
    SELECT string_agg(a.attname, ',' ORDER BY array_position(i.indkey, a.attnum))
      INTO current_pk_cols
      FROM pg_index i
      JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
     WHERE i.indrelid = 'tag_registry'::regclass
       AND i.indisprimary;

    IF current_pk_cols = 'name' THEN
        -- Drop the old self-referential FK before reshaping the PK.
        SELECT conname
          INTO dep_fk_name
          FROM pg_constraint
         WHERE conrelid = 'tag_registry'::regclass
           AND contype  = 'f';
        IF dep_fk_name IS NOT NULL THEN
            EXECUTE format('ALTER TABLE tag_registry DROP CONSTRAINT %I', dep_fk_name);
        END IF;

        ALTER TABLE tag_registry DROP CONSTRAINT tag_registry_pkey;
        ALTER TABLE tag_registry ADD PRIMARY KEY (user_id, name);

        -- Recreate the deprecation link inside the user's own namespace.
        -- A tag can only deprecate another tag belonging to the same
        -- user — multi-tenant isolation forbids cross-user pointers.
        ALTER TABLE tag_registry
            ADD CONSTRAINT tag_registry_deprecated_for_tag_fkey
            FOREIGN KEY (user_id, deprecated_for_tag)
            REFERENCES tag_registry (user_id, name)
            ON DELETE SET NULL;
    END IF;
END $$;

-- The pre-existing partial index `tag_registry_active_idx` on `(name)`
-- still works for lookups within a single user when combined with a
-- `user_id =` filter, but a leading-user_id variant is cheaper for the
-- common "list this user's active tags" scan.
CREATE INDEX IF NOT EXISTS tag_registry_user_id_active_idx
    ON tag_registry (user_id, name)
    WHERE deprecated_for_tag IS NULL;
