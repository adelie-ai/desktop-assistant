-- Widen delete provenance into a disposition vocabulary, decoupled from
-- deletion (#893).
--
-- Migration 038 gave a tombstone a `deleted_kind` and a `deleted_reason`, but
-- both columns meant something only once `deleted_at` was already set:
-- consolidation's one destructive verb was delete, so the only entries ever
-- worth explaining were the ones already gone. On the reference instance that
-- verb collapsed 608 extraction rows to 26 live entries with no restore path.
--
-- The fix widens the same columns rather than adding parallel ones. An entry
-- consolidation judges wrong, stale or redundant is now DISPOSITIONED --
-- marked with what it is and why -- and stays live. `deleted_at` keeps
-- exactly one meaning after this migration: the row is in the trash.
-- Disposition is orthogonal to it, so a dispositioned row is normally live.
--
--   disposition          'active' (default; a live claim) | 'refuted'
--                        (established untrue) | 'superseded' (replaced by
--                        `superseded_by`) | 'redundant' (duplicate of
--                        `superseded_by`) | 'obsolete' (was true, no longer
--                        applies) | 'trivial' (harmless, not worth
--                        surfacing). Renamed from `deleted_kind`.
--   disposition_reason   The model's or the person's stated reason. Renamed
--                        from `deleted_reason`.
--   superseded_by        Unchanged: the id of the row that replaced this one.
--                        Required exactly when disposition is 'superseded' or
--                        'redundant'.
--
-- Old data is backfilled rather than left to default, because a live row and
-- a tombstone written before this migration both read `disposition` as NULL
-- once the rename lands, and NULL cannot satisfy the NOT NULL this migration
-- adds:
--
--   * A merge tombstone (`deleted_kind = 'merge'`) already names its
--     successor in `superseded_by`, so it maps to 'superseded'.
--   * A prune tombstone (`deleted_kind = 'prune'`) was judged not worth
--     keeping at all, which is what 'trivial' means; the model's stated
--     reason survives unchanged in `disposition_reason`.
--   * Everything else reading NULL -- every live row, and a tombstone written
--     before migration 038 ever recorded a kind -- maps to 'active', the
--     column's own default. For a pre-038 tombstone this records that no
--     judgement was ever captured, not that the row is current: `deleted_at`
--     still hides it from every read path.
--
-- The old `knowledge_base_deleted_kind_chk` constraint is dropped before the
-- backfill runs, because its vocabulary ('merge', 'prune') is exactly what
-- the backfill replaces -- leaving it in place would reject the very values
-- the migration writes.
--
-- `superseded_by` is deliberately not touched by the rename or a foreign key;
-- see migration 038's own comment for why (tombstones are hard-reaped, so a
-- dangling id is expected and read as such). The `negative_memory` table
-- carries its own unrelated `superseded_by` column and this migration does
-- not touch it.
--
-- Every statement here is idempotent: the runner applies each migration at
-- most once, tracked in the `schema_migrations` ledger, but a database
-- migrated before that ledger existed replays the whole set once on its first
-- boot under it.

-- Rename, guarded two ways. ALTER TABLE ... RENAME COLUMN has no IF EXISTS
-- form, so a plain guard on the old name is not enough on its own: a replay
-- of the WHOLE migration history (an emptied ledger on an already-migrated
-- database, not the ordinary ledgered path) reruns migration 038 first, and
-- its `ADD COLUMN IF NOT EXISTS deleted_kind` sees no column under that name
-- -- it was renamed away here -- and adds it back, empty. Without the first
-- branch below, this migration's own replay would then try to rename that
-- empty column onto a `disposition` that already holds the real data, and
-- collide with it. The first branch drops that replay artifact instead;
-- the second is the ordinary first-run rename.
--
-- The existence checks read `pg_attribute` through a `regclass` cast rather
-- than `information_schema.columns` by name: a name-only lookup is not
-- schema-qualified and would match a same-named table in a different schema
-- on the same database (harmless in a single-tenant `public` deployment, but
-- wrong the moment more than one schema on the connection can hold a table
-- called `knowledge_base` — the test suite's per-suite private schemas are
-- exactly that). Casting `'knowledge_base'::regclass` resolves through the
-- connection's own `search_path`, the same table every other statement here
-- reaches unqualified.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'knowledge_base'::regclass
          AND attname = 'disposition' AND NOT attisdropped
    ) THEN
        ALTER TABLE knowledge_base DROP COLUMN IF EXISTS deleted_kind;
    ELSIF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'knowledge_base'::regclass
          AND attname = 'deleted_kind' AND NOT attisdropped
    ) THEN
        ALTER TABLE knowledge_base RENAME COLUMN deleted_kind TO disposition;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'knowledge_base'::regclass
          AND attname = 'disposition_reason' AND NOT attisdropped
    ) THEN
        ALTER TABLE knowledge_base DROP COLUMN IF EXISTS deleted_reason;
    ELSIF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'knowledge_base'::regclass
          AND attname = 'deleted_reason' AND NOT attisdropped
    ) THEN
        ALTER TABLE knowledge_base RENAME COLUMN deleted_reason TO disposition_reason;
    END IF;
END $$;

-- Drop the old vocabulary's constraint before the backfill writes values it
-- does not recognize.
ALTER TABLE knowledge_base DROP CONSTRAINT IF EXISTS knowledge_base_deleted_kind_chk;

-- Backfill. Each statement matches nothing on a replay, once the first run
-- has already rewritten every value it targets.
UPDATE knowledge_base SET disposition = 'superseded' WHERE disposition = 'merge';
UPDATE knowledge_base SET disposition = 'trivial'    WHERE disposition = 'prune';
UPDATE knowledge_base SET disposition = 'active'      WHERE disposition IS NULL;

ALTER TABLE knowledge_base
    ALTER COLUMN disposition SET DEFAULT 'active',
    ALTER COLUMN disposition SET NOT NULL;

-- Pin the new vocabulary at the database, the same way migration 038 pinned
-- the old one -- ADD CONSTRAINT has no IF NOT EXISTS, hence the catalog
-- guard.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'knowledge_base'::regclass
          AND conname = 'knowledge_base_disposition_chk'
    ) THEN
        ALTER TABLE knowledge_base
            ADD CONSTRAINT knowledge_base_disposition_chk
            CHECK (disposition IN
                ('active', 'refuted', 'superseded', 'redundant', 'obsolete', 'trivial'));
    END IF;
END $$;

-- A successor id is meaningful for exactly the two dispositions that resolve
-- through it, and only those two: an entry cannot be 'superseded' or
-- 'redundant' without naming what replaced it, and no other disposition names
-- anything.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'knowledge_base'::regclass
          AND conname = 'knowledge_base_superseded_by_chk'
    ) THEN
        ALTER TABLE knowledge_base
            ADD CONSTRAINT knowledge_base_superseded_by_chk
            CHECK ((disposition IN ('superseded', 'redundant')) = (superseded_by IS NOT NULL));
    END IF;
END $$;

-- The browse and report paths read "everything not in its default state",
-- which for a store where most rows are 'active' is a small slice of the
-- table.
CREATE INDEX IF NOT EXISTS knowledge_base_dispositioned_idx
    ON knowledge_base (user_id)
    WHERE disposition <> 'active' AND deleted_at IS NULL;
