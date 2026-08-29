-- Widen knowledge_base_superseded_by_chk on a store that already applied
-- migration 056's old, tighter constraint (#1345).
--
-- 056 originally pinned the successor link with a biconditional: a
-- disposition of 'superseded' or 'redundant' required a `superseded_by`,
-- AND no other disposition was allowed to name one. The second half
-- refused rows that carry real information: a prune tombstone that also
-- names a successor is one such row, it occurs in practice, and
-- ADD CONSTRAINT on it aborts the migration and blocks daemon boot. No
-- read path depends on forbidding a successor id on any other
-- disposition. 056 has since been corrected in place to add only the
-- forward implication:
--
--   CHECK (disposition NOT IN ('superseded', 'redundant') OR superseded_by IS NOT NULL)
--
-- A store provisioned fresh gets the corrected constraint straight from
-- 056. This migration exists only for a store that already ran the old
-- 056 and so already has the tight constraint on disk -- 056's own
-- catalog guard (`IF NOT EXISTS ... conname = 'knowledge_base_superseded_by_chk'`)
-- will not re-add it once it is present, so nothing short of a separate
-- migration converges such a store onto the corrected definition.
--
-- No data repair runs here, and none is needed: every row on a store that
-- applied the old, tighter constraint already satisfies it, and the
-- tighter constraint implies the looser one (dropping the "no other
-- disposition may name a successor" half of an AND cannot turn a row that
-- passed into one that fails). The migration is schema-only.
--
-- Idempotent on replay: the runner applies each file at most once, but a
-- database migrated before the ledger existed replays the whole set once
-- on its first boot under it. DROP CONSTRAINT IF EXISTS always leaves the
-- constraint absent regardless of which definition (old, new, or none) was
-- there beforehand, so the unconditional ADD CONSTRAINT that follows the
-- diagnostic always installs exactly one, current, definition.

ALTER TABLE knowledge_base DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk;

-- Diagnostic, mirroring 056's own: this drop-then-add always finds the
-- constraint absent immediately below, so this count exists for the same
-- future-mismatch case 056 guards against, not because the drop above can
-- leave a violating row behind.
DO $$
DECLARE
    offending_count bigint;
BEGIN
    SELECT count(*) INTO offending_count
    FROM knowledge_base
    WHERE disposition IN ('superseded', 'redundant') AND superseded_by IS NULL;

    IF offending_count > 0 THEN
        RAISE EXCEPTION
            'migration 061: % row(s) would violate knowledge_base_superseded_by_chk '
            '(disposition superseded or redundant with no superseded_by). '
            'Give each such row a successor, or change its disposition to one '
            'that does not require one.', offending_count;
    END IF;
END $$;

ALTER TABLE knowledge_base
    ADD CONSTRAINT knowledge_base_superseded_by_chk
    CHECK (disposition NOT IN ('superseded', 'redundant') OR superseded_by IS NOT NULL);
