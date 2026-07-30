-- Per-conversation override for the tool-provenance gate (#1007).
--
-- The gate (#741, `crates/core/src/tool_provenance.rs`) refuses acting tools
-- for the rest of a turn once that turn has read externally-controlled
-- content. This column is the stored escape hatch: when `true`, the daemon
-- constructs the turn's gate disabled, so a conversation that legitimately
-- wants to read a page and then act on it in the same turn can say so.
--
-- Fail-closed by construction: `NOT NULL DEFAULT FALSE` means an absent value
-- and an unset value are the same reading, and every layer above this column
-- (the storage accessor, the daemon resolver, the task-local) also maps a
-- missing/cross-user row or a store error onto `false`. Only an explicit
-- stored `true` disables the gate.
--
-- Every statement here is idempotent: the runner applies each migration at
-- most once, tracked in the `schema_migrations` ledger, but a database
-- migrated before that ledger existed replays the whole set once on its
-- first boot under it.

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS tool_gate_disabled BOOLEAN NOT NULL DEFAULT FALSE;
