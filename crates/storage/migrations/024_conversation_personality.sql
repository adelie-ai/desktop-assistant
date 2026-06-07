-- Issue #227: per-conversation personality override (Phase 2).
--
-- Stores an optional partial `PersonalityOverride` (per-trait
-- `Option<PersonalityLevel>`) for a conversation so a client can pin a
-- disposition (e.g. a "no-nonsense" client forcing humor=never) that survives
-- daemon restart and conversation switching. NULL = no override (fall back to
-- the global personality on every send). JSONB keeps the column
-- forward-compatible, mirroring `last_model_selection` (migration 011).
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS personality_override JSONB;
