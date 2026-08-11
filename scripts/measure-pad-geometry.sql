-- Measure a real scratchpad's geometry against the recall bar (#1243).
--
-- The pad arm reads its own measured median and median absolute deviation
-- rather than a stated estimate, and whether that is the right call depends on
-- what real pads actually measure. This is the query that answers it. Run it
-- again once several pads sit above RECALL_DISPERSION_MIN_ROWS, because the
-- first run found only one that did, and one pad cannot settle the question.
--
--   psql "$DATABASE_URL" -f scripts/measure-pad-geometry.sql
--
-- Reads only. Touches no row and creates nothing.
--
-- BOTH GUARDS ARE APPLIED HERE, and that is the point of the file. A first
-- measurement that applied neither produced a conclusion that was wrong in its
-- direction, not merely in its magnitude: it reported that the measured read
-- was the more generous of the two on an unrelated prompt, when in fact every
-- pad showing that effect was below the row floor and never measures at all.
-- The guards are:
--
--   rows      >= RECALL_DISPERSION_MIN_ROWS            (20)
--   deviation >= RECALL_DISPERSION_MIN_RELATIVE_SPREAD (0.02) * median
--
-- A pad failing either one falls back to the stated estimate, and its measured
-- geometry describes a code path it does not take. Keep these in step with
-- crates/core/src/ports/recall.rs if the constants there move.
--
-- Two probe families bracket a real prompt, which is not stored and so cannot
-- be replayed directly:
--
--   on  - each note's own vector, standing for a prompt squarely on the pad's
--         subject. The closest a probe can be to the pad.
--   off - unrelated real text from the knowledge store, embedded by the same
--         model, standing for a prompt about something else.
--
-- A real prompt sits between them, so a conclusion that holds at both ends
-- holds wherever the real one lands.

\set bar 6.8
\set est_median 0.65
\set est_deviation 0.05

WITH pad AS (
  SELECT id, conversation_id, embedding, (embedding)[1] AS probe
  FROM scratchpads
  WHERE embedding IS NOT NULL AND embedding_model IS NOT NULL
    AND note_key <> 'goal'          -- excluded from the arm, so excluded here
),
sized AS (
  SELECT conversation_id, count(*) AS pad_n FROM pad GROUP BY 1 HAVING count(*) >= 8
),
onprobe AS (SELECT conversation_id, id AS probe_id, probe FROM pad),
offprobe AS (
  SELECT NULL::text AS conversation_id, k.id AS probe_id, (k.embedding)[1] AS probe
  FROM knowledge_base k
  WHERE k.embedding IS NOT NULL
    AND k.embedding_model = (SELECT embedding_model FROM pad LIMIT 1)
  ORDER BY k.id LIMIT 25
),
dist AS (
  SELECT 'on' AS fam, s.conversation_id, z.pad_n, pr.probe_id, s.id AS note_id,
         (SELECT MIN(chunk <=> pr.probe) FROM unnest(s.embedding) chunk) AS distance
  FROM pad s
  JOIN sized z ON z.conversation_id = s.conversation_id
  JOIN onprobe pr ON pr.conversation_id = s.conversation_id AND pr.probe_id <> s.id
  UNION ALL
  SELECT 'off', s.conversation_id, z.pad_n, pr.probe_id, s.id,
         (SELECT MIN(chunk <=> pr.probe) FROM unnest(s.embedding) chunk)
  FROM pad s JOIN sized z ON z.conversation_id = s.conversation_id CROSS JOIN offprobe pr
),
per AS (
  SELECT fam, conversation_id, pad_n, probe_id, count(*) AS n,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS med
  FROM dist GROUP BY 1,2,3,4
),
mad AS (
  SELECT d.fam, d.conversation_id, d.pad_n, d.probe_id, p.n, p.med,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - p.med)) AS dev
  FROM dist d JOIN per p USING (fam, conversation_id, probe_id)
  GROUP BY 1,2,3,4,5,6
),
eff AS (
  SELECT m.*,
         (m.n >= 20 AND m.dev >= 0.02 * m.med) AS measured,
         CASE WHEN m.n >= 20 AND m.dev >= 0.02 * m.med THEN m.med ELSE :est_median END AS emed,
         CASE WHEN m.n >= 20 AND m.dev >= 0.02 * m.med THEN m.dev ELSE :est_deviation END AS edev
  FROM mad m
)
SELECT e.fam AS probe_family,
       e.pad_n AS pad_notes,
       count(*) AS probes,
       sum(CASE WHEN e.measured THEN 1 ELSE 0 END) AS probes_measuring,
       round(avg(e.dev / e.med)::numeric, 3) AS deviation_over_median,
       round(avg((SELECT count(*) FROM dist d
                   WHERE d.fam = e.fam AND d.conversation_id = e.conversation_id
                     AND d.probe_id = e.probe_id
                     AND (e.emed - d.distance) / e.edev >= :bar))::numeric, 2) AS renders_now,
       round(avg((SELECT count(*) FROM dist d
                   WHERE d.fam = e.fam AND d.conversation_id = e.conversation_id
                     AND d.probe_id = e.probe_id
                     AND (:est_median - d.distance) / :est_deviation >= :bar))::numeric, 2)
         AS renders_under_estimate
FROM eff e
GROUP BY 1, 2
ORDER BY 1 DESC, 2 DESC;
