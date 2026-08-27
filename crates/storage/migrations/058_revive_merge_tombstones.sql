-- Revive merge-member tombstones, the recoverable half of the migration
-- question (#694).
--
-- Consolidation's one destructive verb, before migration 056 replaced it,
-- collapsed a group of near-duplicate entries into one canonical row and
-- soft-deleted the rest, recording each deleted member's successor in
-- `superseded_by`. Migration 038 gave that link a name; migration 056
-- renamed the column that carries it to `disposition` and mapped every
-- merge tombstone's kind to the `superseded` value. None of that revived a
-- single row -- it only made the shape of the loss queryable.
--
-- What is still on disk for a merge-member tombstone: its full content, and
-- a `superseded_by` id naming the row that absorbed it. Two things can have
-- happened to that id since:
--
--   * The successor row still exists. The member comes back live, still
--     `superseded`, because the successor's disposition is the more useful
--     fact about it now -- search resolves a match through the link, and
--     the provenance (which entries a merge actually drew on) is preserved.
--   * The successor row was hard-reaped. The member is the only surviving
--     copy of that content, so it comes back `active` with no successor to
--     point at; a dangling `superseded_by` would only assert a link this
--     database cannot follow.
--
-- Nothing else is touched. A prune tombstone (`disposition = 'trivial'`
-- after migration 056's backfill) was a judgement that the content was not
-- worth keeping at all, not a relocation -- reviving those wholesale would
-- overturn that judgement by script instead of by review, so they stay in
-- the trash, restorable individually through the existing restore path. A
-- tombstone written before migration 038 ever recorded a kind backfilled to
-- `disposition = 'active'` (migration 056's default for a NULL kind, since
-- no judgement was ever captured for it) and so does not match this
-- migration's `disposition = 'superseded'` filter either: its merge linkage,
-- if it ever had any, is gone, and this migration does not guess at it.
--
-- Both updates are idempotent on their own: each is scoped to
-- `disposition = 'superseded' AND deleted_at IS NOT NULL`, and both clear
-- `deleted_at`, so a row they touch no longer matches on a second pass.
-- Nothing here changes the table shape, so there is nothing to guard
-- against on replay beyond that.

-- Arm 1: the successor still exists. Bring the member back as a live
-- `superseded` row; leave `superseded_by` exactly as it was.
UPDATE knowledge_base AS member
SET deleted_at = NULL
WHERE member.disposition = 'superseded'
  AND member.deleted_at IS NOT NULL
  AND EXISTS (
      SELECT 1 FROM knowledge_base AS successor
      WHERE successor.id = member.superseded_by
  );

-- Arm 2: the successor is gone. The member is the only surviving copy of
-- its content, so it stands on its own: live, `active`, no successor link.
UPDATE knowledge_base AS member
SET deleted_at = NULL,
    disposition = 'active',
    superseded_by = NULL
WHERE member.disposition = 'superseded'
  AND member.deleted_at IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM knowledge_base AS successor
      WHERE successor.id = member.superseded_by
  );
