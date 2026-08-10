-- #1186: keep what widened a burn, so a person can see it widen.
--
-- Migration 049 dropped a facet by deleting its row. That is correct for the
-- match - the burn stops requiring it, which is exactly what a second
-- occurrence in another circumstance should do - and it is wrong for a reader.
-- Over-generalization is negative memory's dangerous failure, and it presents
-- as reticence rather than as an error: nobody sees a mistake, they see an
-- assistant that quietly will not do something. Widening is the only mechanism
-- that produces it, and the dropped facet was the only trace it left.
--
-- So a widened facet is now marked instead of deleted. `dropped_at IS NULL` is
-- the facet a burn still requires; a stamped one is what it stopped requiring
-- and when. This is 049's own rule about corrections - keep every column, so
-- the original stays readable - applied to the row below it.
--
-- Nothing about the match changes. Every scope read filters on
-- `dropped_at IS NULL`, so a burn requires exactly what it required before,
-- and a facet is never added after the write that created it, so a stamped row
-- can never collide with a later insert on the primary key.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

ALTER TABLE negative_memory_facet
    ADD COLUMN IF NOT EXISTS dropped_at TIMESTAMPTZ;
