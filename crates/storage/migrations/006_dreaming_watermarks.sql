CREATE TABLE IF NOT EXISTS dreaming_watermarks (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    last_processed_ordinal INTEGER NOT NULL DEFAULT 0,
    last_scanned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
