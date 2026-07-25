-- Issue #599: recover a message's creation time from its id.
--
-- `messages.id` has been a UUIDv7 since migration 005, and a UUIDv7's first 48
-- bits ARE the creation time in milliseconds. So every message already carries
-- its own timestamp -- there is no need for a timestamp column, and this works
-- retroactively on every row already stored.
--
-- Exposed as a reusable function rather than inlined at one call site because
-- the same "when did this message happen" question is worth answering anywhere
-- messages are listed, not just in the tool-usage aggregate.
--
-- Returns NULL for anything that is not a UUIDv7 (a pre-id daemon's row, or a
-- v4 id), so callers degrade to "unknown time" instead of erroring.
--
-- The migration runner (pool.rs) re-executes every migration on every startup
-- with no version table, so every statement here MUST be idempotent.

CREATE OR REPLACE FUNCTION uuidv7_ts(id TEXT)
RETURNS TIMESTAMPTZ
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
RETURNS NULL ON NULL INPUT
AS $$
    SELECT CASE
        WHEN id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-7[0-9a-fA-F]{3}-'
        THEN to_timestamp(
                 (('x' || translate(substring(id, 1, 13), '-', ''))::bit(48)::bigint)
                 / 1000.0
             )
        ELSE NULL
    END
$$;
