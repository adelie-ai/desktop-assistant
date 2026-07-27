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
-- The migration runner (pool.rs) applies each migration at most once, tracked
-- in the `schema_migrations` ledger, but every statement here MUST still be
-- idempotent: a database migrated before that ledger existed replays the whole
-- set once on its first boot under it.

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
