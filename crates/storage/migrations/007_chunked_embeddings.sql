-- Convert embedding columns from a single vector to an array of vectors
-- so that long content can be chunked and each chunk embedded separately.
-- Existing single vectors are wrapped into single-element arrays; NULLs stay NULL.
--
-- Idempotent: the ALTER only runs when the column is still plain `vector`
-- (data_type = 'USER-DEFINED'). Once converted to `vector[]` (data_type =
-- 'ARRAY'), subsequent runs are no-ops — without this guard each restart
-- wraps the column in another ARRAY[...] layer and eventually trips
-- Postgres's 6-dimension array limit.
DO $$
BEGIN
    IF (SELECT data_type
          FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'knowledge_base'
           AND column_name = 'embedding') = 'USER-DEFINED' THEN
        ALTER TABLE knowledge_base
          ALTER COLUMN embedding TYPE vector[]
          USING CASE WHEN embedding IS NOT NULL THEN ARRAY[embedding] ELSE NULL END;
    END IF;

    IF (SELECT data_type
          FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'tool_definitions'
           AND column_name = 'embedding') = 'USER-DEFINED' THEN
        ALTER TABLE tool_definitions
          ALTER COLUMN embedding TYPE vector[]
          USING CASE WHEN embedding IS NOT NULL THEN ARRAY[embedding] ELSE NULL END;
    END IF;
END $$;
