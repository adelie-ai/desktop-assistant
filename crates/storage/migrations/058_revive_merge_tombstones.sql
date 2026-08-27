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
-- if it ever had any, is gone, and this migration does not guess at it. A
-- tombstone whose `superseded_by` names its own id is left alone the same
-- way, on the same principle: nothing on disk says what such a row means,
-- so this migration does not guess at that either, and reviving it would
-- produce a `superseded` row pointing at itself -- a link with nothing on
-- the other end to resolve to.
--
-- A chain (C superseded by B superseded by A, A the only live row) resolves
-- fully in this one pass, not just one hop. Arm 1's `EXISTS` only checks
-- that a row with the named id is present in the table -- it does not
-- require that row to already be live. B's own row was on disk before this
-- migration ran regardless of its disposition, so `EXISTS` for C's link to
-- B is true independent of whether B has been revived yet in this same
-- statement; the same holds for B's link to A. Every member of a chain of
-- any depth is therefore revived together, and the order the two `UPDATE`
-- statements run in does not matter.
--
-- Both updates are idempotent on their own: each is scoped to
-- `disposition = 'superseded' AND deleted_at IS NOT NULL`, and both clear
-- `deleted_at`, so a row they touch no longer matches on a second pass.
-- Nothing here changes the table shape, so there is nothing to guard
-- against on replay beyond that.

-- Arm 1: the successor still exists and is not the member itself. Bring the
-- member back as a live `superseded` row; leave `superseded_by` exactly as
-- it was. The self-reference exclusion is deliberate: without it, a row
-- naming its own id would satisfy `EXISTS` against its own row and revive
-- as `superseded` pointing at itself, an unresolvable loop. No known write
-- path produces such a row, but this migration touches rows a user already
-- lost once, so an unidentifiable shape is left untouched rather than
-- guessed at.
UPDATE knowledge_base AS member
SET deleted_at = NULL
WHERE member.disposition = 'superseded'
  AND member.deleted_at IS NOT NULL
  AND member.superseded_by <> member.id
  AND EXISTS (
      SELECT 1 FROM knowledge_base AS successor
      WHERE successor.id = member.superseded_by
  );

-- Arm 2: the successor is gone. The member is the only surviving copy of
-- its content, so it stands on its own: live, `active`, no successor link.
-- A self-referencing row never reaches this arm either: `EXISTS` against
-- its own id is always true, so `NOT EXISTS` is always false for it.
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
