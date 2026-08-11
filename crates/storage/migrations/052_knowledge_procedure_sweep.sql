-- #1175: what the mis-filed-procedure sweep has already judged.
--
-- The knowledge store holds routines written as facts. They read as neither,
-- they compete with real facts for the knowledge arm's attention budget, and
-- nothing can follow them. The sweep reads an entry, asks whether it is really
-- a method, and where it is, writes an UNAPPROVED skill proposing the split -
-- leaving the entry exactly as it stands, because it is the person's own
-- writing and a background pass does not get to rewrite it.
--
-- This table is the sweep's memory of what it has looked at. Without it the
-- pass would re-judge every entry every cycle: a store of a thousand entries
-- would spend a thousand entries' worth of model calls a night to re-derive an
-- answer it already had, forever.
--
-- One row per (user, entry), and the row records the entry's `updated_at` as it
-- was when the judgement was made. An entry whose text changes afterwards is
-- judged again, because the answer was about the old text.
--
-- `proposed_skill` names the skill the sweep proposed, and is NULL where the
-- entry read as an ordinary fact. Both outcomes are recorded, because "we
-- looked and it was a fact" is exactly the answer the ledger exists to avoid
-- paying for twice.
--
-- The foreign key gives the ledger the entry's own lifetime: a hard reap frees
-- these rows with it, and no row can name an entry that does not exist. Soft
-- deletion is not covered by the key, so a retired entry keeps its row - which
-- is right, since restoring it should not re-open a question already answered.
--
-- Personal data. The table carries `user_id`, every query scopes by it, and it
-- enables its own row-level security below - migration 029's policy list is
-- static and does not reach a table created later. The name is also registered
-- in `PERSONAL_DATA_TABLES` (crates/storage/src/database.rs) so the db_query
-- tool grafts a `user_id` predicate onto any LLM-supplied SQL that names it.
--
-- The migration runner applies each file at most once, but every statement here
-- must still be idempotent: a database migrated before that ledger existed
-- replays the whole set once on its first boot under it.

CREATE TABLE IF NOT EXISTS knowledge_procedure_sweep (
    user_id           TEXT NOT NULL,
    entry_id          TEXT NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    -- When the sweep looked at it.
    judged_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- The entry's own `updated_at` at that moment. A later edit moves
    -- `knowledge_base.updated_at` past this and puts the entry back in the
    -- worklist, because the judgement was about the text that has now changed.
    judged_content_at TIMESTAMPTZ NOT NULL,
    -- The skill the sweep proposed, or NULL where the entry read as a fact.
    proposed_skill    TEXT,
    PRIMARY KEY (user_id, entry_id)
);

-- Row-level security. Migration 029's list is static and does not reach a table
-- created later, so this one enables its own. `current_setting('app.user_id',
-- true)` is NULL when the GUC is unset, and `user_id = NULL` is NULL, so a read
-- path that forgot to pin it sees zero rows.
DROP POLICY IF EXISTS knowledge_procedure_sweep_user_isolation ON knowledge_procedure_sweep;
ALTER TABLE knowledge_procedure_sweep ENABLE ROW LEVEL SECURITY;
CREATE POLICY knowledge_procedure_sweep_user_isolation ON knowledge_procedure_sweep
    USING (user_id = current_setting('app.user_id', true));
