-- A one-line summary for each knowledge entry.
--
-- An entry had no short form: `content` is the whole body, and a reader that
-- wants to show many entries at once either prints all of it or cuts it at a
-- byte count. Both are wrong where the list is the point - the knowledge
-- browser in the clients, and the pre-prompt recall block that puts candidate
-- entries in front of the model before it acts.
--
-- Nullable on purpose. Every existing row has no summary, and the migration
-- cannot read the content to write one, so NOT NULL would need a value
-- invented here. Rows keep NULL until a knowledge-maintenance pass fills them.
--
-- The summary is not embedded. Recall matches against the content embedding
-- that already exists; a second vector over text derived from the first would
-- double the embedding work per entry.

ALTER TABLE knowledge_base
    ADD COLUMN IF NOT EXISTS summary TEXT;
