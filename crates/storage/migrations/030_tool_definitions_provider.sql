-- Provider-level tool surfacing (Phase 1).
--
-- Every tool belongs to a `provider`: either an MCP server (its stable
-- namespace/name identity, the same value `tool_namespaces()` groups on) or a
-- builtin group (knowledge, scratchpad, database, recall, system, tool-meta).
-- Alongside the real tools, the daemon registers one synthetic, searchable
-- `provider:<provider>` row per provider. When that synthetic row matches a
-- `builtin_tool_search` query, the member tools sharing its `provider` value get
-- their fused search score boosted, so a whole server's/group's tools surface
-- together (see `PgToolRegistryStore::search_tools`).
--
-- `provider` is NULL only for as-yet-unclassified rows; the index supports the
-- boost's `LEFT JOIN ... ON f.provider = m.provider` and the per-provider sweeps.
ALTER TABLE tool_definitions ADD COLUMN IF NOT EXISTS provider TEXT;
CREATE INDEX IF NOT EXISTS idx_tool_defs_provider ON tool_definitions(provider);
