CREATE TABLE IF NOT EXISTS tool_definitions (
    name          TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    parameters    JSONB NOT NULL,
    source        TEXT NOT NULL,
    is_core       BOOLEAN NOT NULL DEFAULT FALSE,
    embedding     vector,
    tsv           tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', name || ' ' || description)
                  ) STORED,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
