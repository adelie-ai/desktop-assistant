-- #1328: freeze the corpus so two measurements are comparable, and hold a
-- labelled set that supplies ground truth.
--
-- Background writers (consolidation, extraction, embedding backfill) run
-- continuously, so the knowledge store measured today is not the store
-- measured tomorrow. Without a frozen substrate, a difference between two
-- rankings could be the corpus moving rather than the change under test.
-- Without ground truth, "recall improved" cannot be stated, only felt.
--
-- **This is a regression suite, not a fitting corpus.** The labelled set
-- starts small and self-selected, seeded from real failures. That is enough
-- to prove a change made a known case worse. It is not enough to fit a
-- coefficient: a handful of cases from one failure mode overfits on the
-- first attempt. Any reader adding weight to a small win here should read
-- this paragraph again before they do.
--
-- ## Five tables
--
-- `recall_snapshots` is the manifest: one row per snapshot, naming the
-- embedding model it was taken under. The model is part of the snapshot's
-- identity — embeddings are not portable across models, and a snapshot
-- replayed under a different one is a different experiment, not a repeat of
-- the same one. A row whose embedding model differs from the snapshot's
-- majority is excluded from the snapshot and counted in
-- `excluded_count`, never silently included as if it compared.
--
-- `recall_snapshot_entries` holds the frozen copy of every included
-- knowledge row: the columns retrieval reads, verbatim, so a ranked scan
-- over the snapshot uses the same shape the live scan uses. `embedding` is
-- `vector[]`, the same chunked representation `knowledge_base.embedding`
-- carries (migration 007), so the same distance expression applies
-- unchanged.
--
-- `recall_snapshot_uses` holds the frozen use-log inputs for each included
-- entry — the counters and timestamps `KnowledgeUseRecord` is built from, plus
-- the standing marks as a JSON array (`{source, polarity, reason,
-- marked_at}` per element). A snapshot table, not a live one, so a compact
-- serialized copy is the right shape here even where the live schema (migration
-- 044) keeps marks in their own table: nothing here is queried by mark alone,
-- and a snapshot need not carry a second table to be trusted.
--
-- `recall_cases` is the labelled set: a query, the entry that should win,
-- and where the case came from. `source_request_id` names the turn a real
-- failure was diagnosed from; `note` carries a stated reason where there is
-- no turn to point at. A case must carry one or the other — enforced by the
-- write path, not by this schema, because a CHECK cannot express "at least
-- one of these was chosen for a reason". `baseline_snapshot_id` names the
-- snapshot a case's "currently gets" rank was last measured against, so a
-- snapshot a case still depends on cannot be dropped out from under it — the
-- foreign key below has no CASCADE or SET NULL, so Postgres itself refuses
-- the drop.
--
-- `recall_case_embeddings` caches each case's query embedding per model, so
-- `replaying_a_set_against_a_snapshot_twice_gives_identical_ranks` holds by
-- construction: the second replay reads the same vector the first one wrote,
-- rather than trusting the embedder to answer bit-for-bit the same text twice.
--
-- ## Personal data
--
-- All five tables carry `user_id`, every query scopes by it, and each enables
-- its own RLS policy below (migration 029's policy list is static and does
-- not reach a table created later). All five are registered in
-- `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query
-- tool grafts a `user_id` predicate onto any LLM-supplied SQL that names
-- them. A snapshot freezes one person's knowledge base and use history; a
-- case is a real query someone asked, and a real failure someone hit — as
-- personal as the entries and the turns they came from.
--
-- The migration runner applies each file at most once, but every statement
-- here must still be idempotent: a database migrated before that ledger
-- existed replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS recall_snapshots (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL,
    taken_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- The embedding model every included row was embedded under. Part of the
    -- snapshot's identity, not a descriptive field — replay refuses a
    -- mismatch rather than warning about it.
    embedding_model TEXT NOT NULL,
    entry_count     INTEGER NOT NULL,
    use_count       INTEGER NOT NULL,
    -- Rows read from knowledge_base whose own embedding_model did not match
    -- the majority and were left out of recall_snapshot_entries. A snapshot
    -- never claims a row it cannot compare.
    excluded_count  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS recall_snapshots_user_idx
    ON recall_snapshots (user_id, taken_at DESC);

CREATE TABLE IF NOT EXISTS recall_snapshot_entries (
    snapshot_id     TEXT NOT NULL REFERENCES recall_snapshots(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    -- The knowledge_base id this row was copied from. Kept even after the
    -- live row is edited or deleted — the snapshot is a copy, not a pointer.
    entry_id        TEXT NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    embedding       vector[],
    embedding_model TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    source          TEXT,
    summary         TEXT,
    disposition     TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (snapshot_id, entry_id)
);

CREATE INDEX IF NOT EXISTS recall_snapshot_entries_user_idx
    ON recall_snapshot_entries (user_id, snapshot_id);

CREATE TABLE IF NOT EXISTS recall_snapshot_uses (
    snapshot_id     TEXT NOT NULL REFERENCES recall_snapshots(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    entry_id        TEXT NOT NULL,
    offered_count   BIGINT NOT NULL DEFAULT 0,
    opened_count    BIGINT NOT NULL DEFAULT 0,
    marked_count    BIGINT NOT NULL DEFAULT 0,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_offered_at TIMESTAMPTZ,
    recent_uses     TIMESTAMPTZ[] NOT NULL DEFAULT '{}',
    -- The standing marks, frozen: a JSON array of
    -- {"source": "model"|"person", "polarity": "positive"|"negative",
    --  "reason": string|null, "marked_at": timestamp}.
    marks           JSONB NOT NULL DEFAULT '[]',
    PRIMARY KEY (snapshot_id, entry_id)
);

CREATE INDEX IF NOT EXISTS recall_snapshot_uses_user_idx
    ON recall_snapshot_uses (user_id, snapshot_id);

CREATE TABLE IF NOT EXISTS recall_cases (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL,
    query_text           TEXT NOT NULL,
    expected_entry_id    TEXT NOT NULL,
    -- The turn this failure came from, where one exists. Nullable only when
    -- `note` states why there is none — the write path refuses a case with
    -- neither.
    source_request_id    TEXT,
    note                 TEXT,
    added_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active               BOOLEAN NOT NULL DEFAULT TRUE,
    -- The snapshot this case's "currently gets" rank was last measured
    -- against. No CASCADE and no SET NULL: a snapshot a case still points at
    -- cannot be dropped out from under it. NULL for a case that has never
    -- been anchored to a replay.
    baseline_snapshot_id TEXT REFERENCES recall_snapshots(id),
    CONSTRAINT recall_cases_traceable
        CHECK (source_request_id IS NOT NULL OR note IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS recall_cases_user_idx
    ON recall_cases (user_id, active);

CREATE TABLE IF NOT EXISTS recall_case_embeddings (
    case_id         TEXT NOT NULL REFERENCES recall_cases(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding       vector NOT NULL,
    cached_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (case_id, embedding_model)
);

CREATE INDEX IF NOT EXISTS recall_case_embeddings_user_idx
    ON recall_case_embeddings (user_id, case_id);

-- Row-level security. Migration 029's list is static and does not reach a
-- table created later, so each one enables its own.
-- `current_setting('app.user_id', true)` is NULL when the GUC is unset, and
-- `user_id = NULL` is NULL, so a read path that forgot to pin it sees zero
-- rows.
DROP POLICY IF EXISTS recall_snapshots_user_isolation ON recall_snapshots;
ALTER TABLE recall_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY recall_snapshots_user_isolation ON recall_snapshots
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS recall_snapshot_entries_user_isolation ON recall_snapshot_entries;
ALTER TABLE recall_snapshot_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY recall_snapshot_entries_user_isolation ON recall_snapshot_entries
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS recall_snapshot_uses_user_isolation ON recall_snapshot_uses;
ALTER TABLE recall_snapshot_uses ENABLE ROW LEVEL SECURITY;
CREATE POLICY recall_snapshot_uses_user_isolation ON recall_snapshot_uses
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS recall_cases_user_isolation ON recall_cases;
ALTER TABLE recall_cases ENABLE ROW LEVEL SECURITY;
CREATE POLICY recall_cases_user_isolation ON recall_cases
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS recall_case_embeddings_user_isolation ON recall_case_embeddings;
ALTER TABLE recall_case_embeddings ENABLE ROW LEVEL SECURITY;
CREATE POLICY recall_case_embeddings_user_isolation ON recall_case_embeddings
    USING (user_id = current_setting('app.user_id', true));
