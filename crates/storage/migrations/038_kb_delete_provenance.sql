-- Delete provenance for knowledge-base tombstones (#695).
--
-- Consolidation retires an entry two very different ways and wrote the same row
-- change for both: a MERGE carries the content forward into a canonical row, a
-- PRUNE judges the entry not worth keeping at all. With only `deleted_at` on
-- disk the two are indistinguishable, so "was this fact relocated or
-- destroyed?" could not be answered by any query -- which is what made the
-- 606-of-608 extraction loss on the reference instance impossible to audit or
-- monitor (#694).
--
--   deleted_kind   'merge' | 'prune'. NULL on tombstones written before this
--                  migration, and on deletes that did not come from
--                  consolidation (a person removing an entry, for example).
--   deleted_reason The model's stated reason for a prune, length-bounded by the
--                  writer. NULL for merge members: the model states no
--                  per-member reason there, and `superseded_by` already says
--                  what happened.
--   superseded_by  For a merge member, the id of the canonical row that
--                  absorbed its content.
--
-- `superseded_by` is deliberately NOT a foreign key. Tombstones are hard-reaped
-- once past their retention window, so the target can legitimately disappear;
-- an FK would either block the reap or, with ON DELETE SET NULL, erase the
-- audit link at exactly the moment it is wanted. Dangling ids are expected and
-- are read as such -- the same contract `metadata.source_conversation_id`
-- already carries against archival hard-deletes.
--
-- Every statement here is idempotent: the runner applies each migration at most
-- once, tracked in the `schema_migrations` ledger, but a database migrated
-- before that ledger existed replays the whole set once on its first boot
-- under it.

ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS deleted_kind   TEXT,
    ADD COLUMN IF NOT EXISTS deleted_reason TEXT,
    ADD COLUMN IF NOT EXISTS superseded_by  TEXT;

-- Pin the vocabulary at the database so a later writer cannot invent a third
-- spelling that silently breaks the merge-vs-prune split every audit query
-- depends on. ADD CONSTRAINT has no IF NOT EXISTS, hence the catalog guard.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'knowledge_base'::regclass
          AND conname = 'knowledge_base_deleted_kind_chk'
    ) THEN
        ALTER TABLE knowledge_base
            ADD CONSTRAINT knowledge_base_deleted_kind_chk
            CHECK (deleted_kind IS NULL OR deleted_kind IN ('merge', 'prune'));
    END IF;
END $$;
