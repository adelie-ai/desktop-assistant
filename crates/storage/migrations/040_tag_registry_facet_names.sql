-- Restore the facet colon in tag-registry names written before #1069.
--
-- The registry used to normalize a name by removing every non-alphanumeric
-- character, so the facet tag `project:adelie-ai` was stored as
-- `projectadelie-ai`. The knowledge base kept the colon, so a registry key
-- could never match the tag on the row it described. The registry now uses the
-- knowledge base's own normalizer; this repairs the rows written before that.
--
-- The colon position cannot be read back out of a stored name, so the repair
-- reads the closed facet vocabulary the knowledge-base prompt defines --
-- `project`, `tool`, `topic`, `person`
-- (`crates/core/src/prompts/sections/knowledge_base.txt`). None of the four is
-- a prefix of another, so a name matches at most one. What follows the facet
-- word decides the reading, because the old normalizer mapped a space onto a
-- dash before it removed the colon:
--
--   `project:adelie-ai`   -> `projectadelie-ai`    glued, so a facet tag
--   `Project : Adelie AI` -> `project--adelie-ai`  double dash, so a facet tag
--   `Project: Adelie AI`  -> `project-adelie-ai`   single dash -- see below
--
-- A single dash is left alone. It is the same shape a plain multi-word tag
-- takes (`project context` and `project-context` both normalized to
-- `project-context`), and a real registry holds families of those. Renaming
-- one would put a key in the registry that no row carries, which is the defect
-- this migration exists to remove, so an unrepaired row is the cheaper error.
--
-- The glued reading has its own limit: a one-word tag that begins with a facet
-- word is split as well (`toolchain` becomes `tool:chain`). Nothing in the
-- stored name separates the two cases. Every rename is therefore named in a
-- RAISE NOTICE, which sqlx logs at INFO on the `sqlx::postgres::notice` target,
-- so a daemon started at the shipped `RUST_LOG=info` records what moved and an
-- operator can undo a wrong split.
--
-- A renamed row loses its embedding. The dedup vector is built from
-- `"<name>: <description>"`, so a rename makes the vector stale, and a stale
-- vector matches the wrong concepts. A row with a NULL embedding still matches
-- by exact name, which is the common path; what it loses is the near-duplicate
-- vector search. Nothing re-embeds a tag row today, so a renamed row stays out
-- of that search until #516 lands.
--
-- Two rows can want the same repaired name, in two ways: a row may already
-- hold the correct name, and two mangled spellings of one tag (`topicdeploy`
-- and `topic--deploy`) repair to the same thing. Both are collisions, and both
-- resolve the same way -- one row survives and the rest are removed. A row
-- already holding the correct name always wins; between mangled spellings the
-- first by `C` collation wins, an arbitrary but stable choice that does not
-- move with the database's own collation. Deprecation pointers and
-- `distinguish_from` entries follow in every case, so no reference is left on
-- a name no row holds, and no row is left naming itself.
--
-- The self-referential foreign key is dropped for the duration and rebuilt
-- afterwards, so the rows may move in any order inside the one transaction the
-- migration runner wraps this in.
--
-- Idempotent: a repaired name contains a colon, and a name containing a colon
-- is never selected, so a replay finds nothing to do.

DO $$
DECLARE
    dropped_count      INTEGER := 0;
    renamed_count      INTEGER := 0;
    repointed_count    INTEGER := 0;
    distinguish_count  INTEGER := 0;
    rename_row         RECORD;
BEGIN
    DROP TABLE IF EXISTS pg_temp.tag_registry_facet_rename;

    CREATE TEMP TABLE tag_registry_facet_rename ON COMMIT DROP AS
    WITH split AS (
        SELECT r.user_id,
               r.name AS old_name,
               f.facet,
               CASE
                 WHEN substring(r.name FROM length(f.facet) + 1) LIKE '--%'
                     THEN substring(r.name FROM length(f.facet) + 3)
                 WHEN substring(r.name FROM length(f.facet) + 1) LIKE '-%'
                     THEN NULL
                 ELSE substring(r.name FROM length(f.facet) + 1)
               END AS facet_value
          FROM tag_registry r
          JOIN (VALUES ('project'), ('person'), ('topic'), ('tool')) AS f(facet)
            ON r.name LIKE f.facet || '%'
           AND length(r.name) > length(f.facet)
         WHERE strpos(r.name, ':') = 0
    ),
    candidate AS (
        SELECT user_id,
               old_name,
               facet || ':' || facet_value AS new_name
          FROM split
         WHERE facet_value IS NOT NULL
           AND facet_value <> ''
    )
    SELECT user_id,
           old_name,
           new_name,
           row_number() OVER (PARTITION BY user_id, new_name
                              ORDER BY old_name COLLATE "C") = 1 AS is_winner
      FROM candidate;

    ALTER TABLE tag_registry
        DROP CONSTRAINT IF EXISTS tag_registry_deprecated_for_tag_fkey;

    -- Remove every mangled row that cannot keep its repaired name: one whose
    -- facet-correct row already exists, and one that lost to another mangled
    -- spelling of the same tag. Renaming a loser would give two rows the same
    -- primary key and abort the migration on every boot.
    DELETE FROM tag_registry t
     USING tag_registry_facet_rename m
     WHERE t.user_id = m.user_id
       AND t.name = m.old_name
       AND (NOT m.is_winner
            OR EXISTS (SELECT 1
                         FROM tag_registry k
                        WHERE k.user_id = m.user_id
                          AND k.name = m.new_name));
    GET DIAGNOSTICS dropped_count = ROW_COUNT;

    FOR rename_row IN
        SELECT m.user_id, m.old_name, m.new_name
          FROM tag_registry_facet_rename m
          JOIN tag_registry t
            ON t.user_id = m.user_id AND t.name = m.old_name
         ORDER BY m.user_id, m.old_name
    LOOP
        RAISE NOTICE 'tag_registry: renaming % to % (user %)',
            rename_row.old_name, rename_row.new_name, rename_row.user_id;
    END LOOP;

    UPDATE tag_registry t
       SET name = m.new_name,
           embedding = NULL,
           embedding_model = NULL
      FROM tag_registry_facet_rename m
     WHERE t.user_id = m.user_id
       AND t.name = m.old_name;
    GET DIAGNOSTICS renamed_count = ROW_COUNT;

    -- Carry every deprecation pointer onto the new name. This covers both the
    -- renamed rows and the dropped duplicates, whose survivor holds that same
    -- new name.
    UPDATE tag_registry t
       SET deprecated_for_tag = m.new_name
      FROM tag_registry_facet_rename m
     WHERE t.user_id = m.user_id
       AND t.deprecated_for_tag = m.old_name;
    GET DIAGNOSTICS repointed_count = ROW_COUNT;

    -- `distinguish_from` names siblings a tag must stay apart from. It is a
    -- plain array with no foreign key, so it needs the same rewrite; a stale
    -- entry there points the extractor at a tag that no longer exists.
    --
    -- A row that named its own mangled duplicate would otherwise end up naming
    -- itself, so a resolved sibling equal to the row's own name is dropped, and
    -- two siblings that resolve to one name collapse to a single entry at the
    -- position of the first. This runs after the rename, so `t.name` is already
    -- the repaired name.
    UPDATE tag_registry t
       SET distinguish_from = ARRAY(
               SELECT resolved.name
                 FROM (SELECT COALESCE(m.new_name, sibling.name) AS name,
                              min(sibling.ord) AS ord
                         FROM unnest(t.distinguish_from)
                                  WITH ORDINALITY AS sibling(name, ord)
                         LEFT JOIN tag_registry_facet_rename m
                           ON m.user_id = t.user_id AND m.old_name = sibling.name
                        GROUP BY 1) resolved
                WHERE resolved.name <> t.name
                ORDER BY resolved.ord)
     WHERE EXISTS (SELECT 1
                     FROM unnest(t.distinguish_from) AS sibling(name)
                     JOIN tag_registry_facet_rename m
                       ON m.user_id = t.user_id AND m.old_name = sibling.name);
    GET DIAGNOSTICS distinguish_count = ROW_COUNT;

    ALTER TABLE tag_registry
        ADD CONSTRAINT tag_registry_deprecated_for_tag_fkey
        FOREIGN KEY (user_id, deprecated_for_tag)
        REFERENCES tag_registry (user_id, name)
        ON DELETE SET NULL;

    RAISE NOTICE
        'tag_registry facet repair: % renamed (embedding cleared, refilled by #516), % mangled duplicates dropped, % deprecation pointers moved, % distinguish_from lists rewritten',
        renamed_count, dropped_count, repointed_count, distinguish_count;
END $$;
