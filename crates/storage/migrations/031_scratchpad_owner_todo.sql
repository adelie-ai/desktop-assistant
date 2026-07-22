-- Issue #287: namespace the per-conversation scratchpad by owner_todo, a
-- materialized subagent-tree path (e.g. '1.1'; root sentinel '' for the
-- top-level session's own notes). Subagent writes are stamped + confined to
-- their owner_todo; reads snapshot by the spawn marker. Top-level notes stay
-- owner_todo='' so behavior is unchanged for the non-subagent case.
--
-- The migration runner (pool.rs) re-executes every migration on every startup
-- with no version table, so every statement here MUST be idempotent.

ALTER TABLE scratchpads ADD COLUMN IF NOT EXISTS owner_todo TEXT NOT NULL DEFAULT '';

-- Drop the old 2-column unique (conversation_id, note_key) by COLUMN SET, not a
-- hard-coded name: a restored/renamed DB could carry a non-default constraint
-- name, and if the old 2-col unique survived, the first subagent write reusing
-- a root note_key under a different owner_todo would raise a unique violation
-- (500 on every such write) — exactly the case this epic depends on.
DO $$
DECLARE conname text;
BEGIN
    SELECT con.conname INTO conname
    FROM pg_constraint con
    WHERE con.conrelid = 'scratchpads'::regclass
      AND con.contype = 'u'
      AND (SELECT array_agg(att.attname::text ORDER BY att.attname::text)
           FROM unnest(con.conkey) AS k
           JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = k)
          = ARRAY['conversation_id', 'note_key'];
    IF conname IS NOT NULL THEN
        EXECUTE format('ALTER TABLE scratchpads DROP CONSTRAINT %I', conname);
    END IF;
END $$;

-- New 3-column uniqueness. A UNIQUE INDEX (not ADD CONSTRAINT) is used so
-- IF NOT EXISTS makes it idempotent AND ON CONFLICT (conversation_id,
-- owner_todo, note_key) can still infer from it. Backfilled rows are
-- owner_todo='', so for root rows this is byte-identical in effect to the old
-- 2-col unique and cannot fail on existing data.
CREATE UNIQUE INDEX IF NOT EXISTS scratchpads_conv_owner_key_uidx
    ON scratchpads (conversation_id, owner_todo, note_key);

-- Backs the owner-subtree prefix reads/deletes (owner_todo = $me OR
-- owner_todo LIKE $me || '.%') under any collation. Leads with user_id so it
-- also serves the user_id-first scoping every query applies.
CREATE INDEX IF NOT EXISTS scratchpads_user_conv_owner_idx
    ON scratchpads (user_id, conversation_id, owner_todo text_pattern_ops);

-- owner_todo is a system-generated dot-delimited numeric path; enforce it at
-- the DB so the `owner_todo LIKE $me || '.%'` subtree match never needs
-- LIKE-escaping (defense in depth; $me is always a bound parameter too).
-- Idempotent add (ADD CONSTRAINT has no IF NOT EXISTS, and the runner re-runs).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'scratchpads'::regclass
          AND conname = 'scratchpads_owner_todo_numeric_chk'
    ) THEN
        ALTER TABLE scratchpads
            ADD CONSTRAINT scratchpads_owner_todo_numeric_chk CHECK (owner_todo ~ '^[0-9.]*$');
    END IF;
END $$;
