-- Convert knowledge_base.embedding from a single vector to an array of vectors
-- so that long content can be chunked and each chunk embedded separately.
-- Existing single vectors are wrapped into single-element arrays; NULLs stay NULL.
ALTER TABLE knowledge_base
  ALTER COLUMN embedding TYPE vector[]
  USING CASE WHEN embedding IS NOT NULL THEN ARRAY[embedding] ELSE NULL END;
