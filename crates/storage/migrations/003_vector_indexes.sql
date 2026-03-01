CREATE INDEX IF NOT EXISTS idx_kb_tags ON knowledge_base USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_kb_tsv ON knowledge_base USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_tool_defs_core ON tool_definitions(is_core) WHERE is_core = TRUE;
CREATE INDEX IF NOT EXISTS idx_tool_defs_tsv ON tool_definitions USING GIN(tsv);
