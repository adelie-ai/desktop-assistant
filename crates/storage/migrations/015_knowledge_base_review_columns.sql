ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS reviewed_at        TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS review_generation  SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS deleted_at         TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS knowledge_base_needs_review_idx
    ON knowledge_base (created_at)
    WHERE reviewed_at IS NULL AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS knowledge_base_soft_deleted_idx
    ON knowledge_base (deleted_at)
    WHERE deleted_at IS NOT NULL;
