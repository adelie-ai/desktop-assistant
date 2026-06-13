-- Learned effective context-window observations (issue #343): an adaptive
-- safety net that complements #342.
--
-- When a turn overflows and the provider's error string yields a concrete
-- `max_tokens` (parsed in `error_classify::first_two_numbers`), we persist that
-- observed ceiling per `(connector, model)`. Budget resolution then `min()`s it
-- into the resolved budget so the NEXT turn starts under the real observed
-- limit instead of re-overflowing.
--
-- Down-only ratchet: this table only ever records a window we *saw* reject a
-- prompt. Raising a window is a deliberate config action (#342), never inferred
-- from success, so `record` keeps the SMALLEST observed limit seen for a given
-- configured window.
--
-- Invalidation by configured window: `configured_window` stores the effective
-- configured budget that was in force when the overflow happened. Resolution
-- ignores a learned row whose `configured_window` differs from the current
-- effective budget — so when the user raises (or lowers) the configured window
-- the stale learned cap is naturally discarded and the next turn starts fresh
-- from the new ceiling rather than being pinned to the old observation.
--
-- GLOBAL, not per-user: like `error_classifications` this is connector/model
-- knowledge (how big a window a hosted model actually accepts), not personal
-- data, so there is deliberately no `user_id` column and no per-user scoping.

CREATE TABLE IF NOT EXISTS context_window_observations (
    connector         TEXT   NOT NULL,
    model             TEXT   NOT NULL,
    -- Smallest provider-rejected window observed for this configured_window.
    observed_limit    BIGINT NOT NULL,
    -- The effective configured budget in force when the overflow was observed;
    -- a learned row is treated as stale once this no longer matches.
    configured_window BIGINT NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (connector, model)
);
