-- Repair embedding columns damaged by repeated non-idempotent runs of
-- migration 007 before it was made idempotent. Each re-run wrapped the
-- column in another ARRAY[...] layer; after 6 runs the type exceeds
-- Postgres's array-dimension limit.
--
-- `pg_attribute.attndims` is the declared number of array dimensions for
-- the column: `vector` is 0, `vector[]` is 1, `vector[][]` is 2, etc.
-- Anything above 1 is damaged and gets dropped + re-added. Embeddings are
-- regenerated on demand, so no other data is affected.
DO $$
DECLARE
    kb_ndims INT;
    td_ndims INT;
BEGIN
    SELECT attndims INTO kb_ndims
      FROM pg_attribute
     WHERE attrelid = 'knowledge_base'::regclass
       AND attname = 'embedding';
    IF kb_ndims IS NOT NULL AND kb_ndims > 1 THEN
        ALTER TABLE knowledge_base DROP COLUMN embedding;
        ALTER TABLE knowledge_base ADD COLUMN embedding vector[];
    END IF;

    SELECT attndims INTO td_ndims
      FROM pg_attribute
     WHERE attrelid = 'tool_definitions'::regclass
       AND attname = 'embedding';
    IF td_ndims IS NOT NULL AND td_ndims > 1 THEN
        ALTER TABLE tool_definitions DROP COLUMN embedding;
        ALTER TABLE tool_definitions ADD COLUMN embedding vector[];
    END IF;
END $$;
