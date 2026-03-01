ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE tool_definitions ADD COLUMN IF NOT EXISTS embedding_model TEXT;
