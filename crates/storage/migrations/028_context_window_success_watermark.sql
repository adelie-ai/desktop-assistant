-- Success high-water mark for learned context windows (issue #425).
--
-- #343 (migration 025) only ever recorded a window we *saw reject* a prompt and
-- capped DOWN to it forever. A single mis-parsed overflow error then pinned the
-- budget at 534 tokens with no way to recover. This migration adds the other
-- half of the bracket: the largest input-token count we've seen a model
-- ACCEPT.
--
--   * `max_success_input` — the high-water mark. Budget resolution floors the
--     learned cap by this, so a garbage-low `observed_limit` can never pin the
--     budget below a size the model has demonstrably handled, and the budget can
--     climb back as larger prompts succeed. It is provider-MEASURED
--     (`usage.input_tokens`), not scraped from an error string, so it is the
--     value we trust most. Independent of `configured_window`: proven-good stays
--     proven-good across config changes.
--
-- `observed_limit` and `configured_window` become NULLABLE: a row can now exist
-- with only success data (a model that has never overflowed) or only overflow
-- data. NULL `observed_limit` means "no overflow observed yet".

ALTER TABLE context_window_observations
    ALTER COLUMN observed_limit    DROP NOT NULL,
    ALTER COLUMN configured_window DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS max_success_input BIGINT;
