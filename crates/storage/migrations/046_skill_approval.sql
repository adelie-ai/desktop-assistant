-- Approval axis for the skill catalog (#1155): CONSENT, kept separate from
-- provenance. `trust_tier` records where a skill came from. These two columns
-- record whether a person has said it may be followed.
--
-- `approved_at` is the instant a person approved the skill. NULL means nobody
-- has approved it yet, so the skill must not be followed. `approved_by` names
-- the approver; NULL is normal on a single-person deployment, where the user
-- is the only possible approver, and the column is meaningless while
-- `approved_at` is NULL.
--
-- The backfill below must run only on the branch that adds the column, never
-- on a later boot. A database migrated before the `schema_migrations` ledger
-- existed replays every migration once on its first boot under this runner
-- (see AGENTS.md, "Storage & migrations"). A plain
-- `UPDATE skill_index SET approved_at = indexed_at WHERE approved_at IS NULL`
-- would be wrong on that replay: by then the column already exists and its
-- NULLs are real, meaning "a person left this skill unapproved" -- most of
-- all for a self-authored skill, which is unapproved by design. Re-running
-- the backfill on that replay would silently approve every one of them. The
-- `IF NOT EXISTS` guard makes the backfill run exactly once, on the pass that
-- actually adds the column.
--
-- Existing rows are backfilled to `indexed_at`, because every skill in the
-- catalog before this migration arrived by a person putting a file in a
-- skill root -- the same act migration 035's `present_on_disk` default
-- reasons from, and the same act `reconcile_scan` now stamps approval from
-- going forward.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'skill_index'
           AND column_name = 'approved_at'
    ) THEN
        ALTER TABLE skill_index ADD COLUMN approved_at TIMESTAMPTZ;
        ALTER TABLE skill_index ADD COLUMN approved_by TEXT;
        UPDATE skill_index SET approved_at = indexed_at WHERE approved_at IS NULL;
    END IF;
END $$;

-- Browse/audit surfaces filter on approval ("what's indexed but not yet
-- approved?"), mirroring `idx_skill_index_present` from migration 035.
CREATE INDEX IF NOT EXISTS idx_skill_index_approved_at ON skill_index (approved_at);
