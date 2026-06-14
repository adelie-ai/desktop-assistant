-- Dream-cycle overhaul, foundation migration.
--
-- Two new columns on knowledge_base:
--
-- * `embeddings_updated_at` — when the row's embedding vectors were last
--   (re)generated. Embedding generation is now fully decoupled from content
--   writes: writes never embed inline; a background task regenerates vectors
--   for rows where `embedding IS NULL` or `embeddings_updated_at < updated_at`
--   (content changed since the last embed). Existing embedded rows are
--   backfilled to `updated_at` so they are treated as fresh and do not get
--   spuriously re-embedded.
--
-- * `source` — first-class provenance, replacing the `source:dreaming` tag
--   convention. One of:
--     'extraction'    — pulled from a conversation by the dreaming extraction phase
--     'consolidation' — synthesized/edited by the holistic consolidation phase
--     'explicit'      — written during a live conversation turn (the user asked,
--                       or Adele consciously decided it was worth storing)
--   Left NULL on existing rows by design; they keep their current tags until
--   Adele or a consolidation pass rewrites them.

ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS embeddings_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS source                TEXT;

-- Assume current vectors match current content for already-embedded rows so the
-- widened backfill predicate does not regenerate everything on first run.
UPDATE knowledge_base
    SET embeddings_updated_at = updated_at
    WHERE embedding IS NOT NULL
      AND embeddings_updated_at IS NULL;
