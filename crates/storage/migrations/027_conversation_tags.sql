ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_conversations_tags ON conversations USING gin(tags);
