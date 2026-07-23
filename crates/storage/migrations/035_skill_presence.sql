-- Cumulative skill catalog (#639): record whether a skill's files were on disk
-- at the last scan of its scope, instead of deleting rows a scan didn't see.
--
-- The catalog is the authoritative copy of a skill, not a shadow of the last
-- scan: skills accrete, and one disappearing from disk is never a reason to
-- forget the procedure. What absence does change is what still works -- the
-- body reads fine, but `disk_path` and `attachments` no longer resolve, so
-- bundled scripts cannot be run. `present_on_disk` carries exactly that, and
-- `last_seen_at` records when the scope was last scanned with the skill in it.
--
-- Defaults are deliberately "present": a row that predates presence tracking
-- was, by construction, on disk when it was indexed, so absent evidence must
-- not read as missing.
ALTER TABLE skill_index
    ADD COLUMN IF NOT EXISTS present_on_disk BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE skill_index
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;

-- Backfill: pre-existing rows were last seen when they were indexed.
UPDATE skill_index SET last_seen_at = indexed_at WHERE last_seen_at IS NULL;

-- Browse/audit surfaces filter on presence ("what's indexed but gone?").
CREATE INDEX IF NOT EXISTS idx_skill_index_present ON skill_index (present_on_disk);
