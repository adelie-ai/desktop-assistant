-- Learned error-classification cache (epic #178, tier 2).
--
-- When the deterministic tier-1 matchers don't recognize an opaque backend
-- error, the cheap task LLM (tier 3) classifies it once into a normalized
-- cause and a distinctive signature substring, which is persisted here so the
-- next occurrence is recognized locally — no LLM call.
--
-- GLOBAL, not per-user: this is connector knowledge (how to read a provider's
-- error), not personal data, so there is deliberately no `user_id` column and
-- no per-user scoping. The table is small (one row per learned error shape)
-- and human-auditable/editable.
--
-- `signature` is a case-insensitive substring of the error message; lookup is
-- "does the incoming message contain this signature for this connector". The
-- UNIQUE (connector, signature) constraint makes `record` an idempotent upsert.

CREATE TABLE IF NOT EXISTS error_classifications (
    id              BIGSERIAL PRIMARY KEY,
    connector       TEXT NOT NULL,
    signature       TEXT NOT NULL,
    cause           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'learned',
    hit_count       BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_matched_at TIMESTAMPTZ,
    UNIQUE (connector, signature)
);

CREATE INDEX IF NOT EXISTS idx_error_classifications_connector
    ON error_classifications (connector);
