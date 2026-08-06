-- Issue #1104: a scratchpad note can attach a knowledge entry, so pinning that
-- note keeps the entry's live content in view.
--
-- The attachment is additive, beside `content` and not in place of it: the note
-- says in the model's words why the entry matters right now, and the entry
-- carries the durable fact. A typed column (rather than an id encoded into
-- `content` behind a magic `note_type`) is checkable by the database and
-- greppable by a reader.
--
-- The foreign key is the structural half of "a reference never outlives its
-- entry": ON DELETE SET NULL clears the attachment the moment the entry row
-- goes, so a dangling id cannot exist. It does not cover a soft delete - a
-- trashed entry keeps its row - so the render path drops an attachment it
-- cannot resolve as well.
--
-- The migration runner (pool.rs) applies each migration at most once, tracked
-- in the `schema_migrations` ledger, but every statement here MUST still be
-- idempotent: a database migrated before that ledger existed replays the whole
-- set once on its first boot under it.

ALTER TABLE scratchpads
    ADD COLUMN IF NOT EXISTS knowledge_entry_id TEXT
        REFERENCES knowledge_base(id) ON DELETE SET NULL;

-- Deleting a knowledge entry has to find the rows that reference it. Without an
-- index that is a sequential scan of the whole pad per deleted entry, and
-- emptying the trash deletes many at once. Partial, because an attachment is
-- rare: the pin cap keeps the referencing set tiny while plain notes are the
-- overwhelming majority.
CREATE INDEX IF NOT EXISTS scratchpads_knowledge_entry_idx
    ON scratchpads (knowledge_entry_id)
    WHERE knowledge_entry_id IS NOT NULL;
