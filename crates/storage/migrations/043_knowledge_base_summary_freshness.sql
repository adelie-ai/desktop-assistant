-- When a knowledge entry's one-line summary was last written.
--
-- The summary is a condensation of `content`, so an edit to the body makes the
-- stored line describe something the entry no longer says. The write path
-- preserves a stored summary when an update names none, deliberately, so that
-- drift is a normal outcome rather than an unusual one. A summary that is
-- confidently wrong is worse than no summary at all, because a reader believes
-- it and never opens the entry.
--
-- This column is the freshness stamp that lets the dream cycle find those rows,
-- mirroring `embeddings_updated_at`: work is due when the stamp is absent or
-- older than `updated_at`.
--
-- Nullable, because a row that has no summary has no time at which one was
-- written. Rows that already carry a summary are stamped with their own
-- `updated_at`, so an entry summarised before this column existed reads as
-- current and is not rewritten on the first pass.

ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS summary_updated_at TIMESTAMPTZ;

UPDATE knowledge_base
    SET summary_updated_at = updated_at
    WHERE summary IS NOT NULL
      AND summary_updated_at IS NULL;
