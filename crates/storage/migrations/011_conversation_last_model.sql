-- Issue #11: per-conversation model selection.
--
-- Stores the last `{connection_id, model_id, effort?}` selection used on a
-- conversation so the user's choice survives daemon restart and conversation
-- switching. JSONB keeps the column forward-compatible (extra fields like
-- per-connector params can be added without another migration).
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS last_model_selection JSONB;
