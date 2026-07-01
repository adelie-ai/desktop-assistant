-- #434: Postgres Row-Level Security backstop for the LLM-facing
-- `execute_database_query` read path.
--
-- The AST grafter (#141) rewrites every SELECT to append
-- `user_id = $caller` onto personal-data tables, but that rewriter is the
-- *only* thing standing between a hostile/manipulated LLM and another
-- user's rows. This migration makes the database itself enforce tenant
-- scoping regardless of the SQL text: the tool's read path switches into a
-- dedicated, un-privileged role (`adele_query`) that has neither table
-- ownership nor BYPASSRLS, then pins the caller's id in `app.user_id`. The
-- policies below then filter every personal-data table to the caller's own
-- rows — the hard backstop underneath the grafter.
--
-- Trusted daemon code paths are UNAFFECTED. RLS here is non-FORCE, so the
-- table owner is exempt; the daemon connects as the role that ran these
-- migrations (and therefore owns every table), so its own queries — which
-- already scope by user_id in their WHERE clauses — see all rows as before.
-- Only the `SET LOCAL ROLE adele_query` read path is filtered.
--
-- This migration is re-run on every daemon start (like all of them), so it
-- is written to be idempotent: guarded role creation, repeatable grants,
-- and DROP POLICY IF EXISTS before each CREATE POLICY (Postgres has no
-- CREATE POLICY IF NOT EXISTS).

-- 1. The restricted role the tool read path runs under. NOLOGIN — it is
--    only ever entered via `SET LOCAL ROLE` inside the daemon's own
--    session, never connected to directly. NOBYPASSRLS is the default, but
--    is stated explicitly for the reader and asserted by the
--    `rls_role_cannot_bypass` test.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'adele_query') THEN
        CREATE ROLE adele_query NOLOGIN NOBYPASSRLS;
    END IF;
END
$$;

-- Let whichever role runs the migrations / owns the daemon session assume
-- adele_query via SET ROLE. A superuser can SET ROLE regardless; this makes
-- it work for a non-superuser owner too.
GRANT adele_query TO CURRENT_USER;

-- 2. Read privileges. The tool read path only ever SELECTs. Grant SELECT on
--    all current tables, plus default privileges so a table added by a
--    later migration (run by the same owner role) is readable without a
--    follow-up grant. USAGE on the schema is required to resolve objects.
GRANT USAGE ON SCHEMA public TO adele_query;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO adele_query;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO adele_query;

-- 3. Enable RLS + a per-user isolation policy on every user-scoped table.
--    `current_setting('app.user_id', true)` returns NULL when the GUC is
--    unset (the `true` = missing_ok), and `user_id = NULL` is NULL, so a
--    read path that forgot to pin app.user_id sees ZERO rows — fail-closed,
--    never a leak. user_id is `text`, matching current_setting's return
--    type, so no cast is needed.

DROP POLICY IF EXISTS background_tasks_user_isolation ON background_tasks;
ALTER TABLE background_tasks ENABLE ROW LEVEL SECURITY;
CREATE POLICY background_tasks_user_isolation ON background_tasks
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS conversations_user_isolation ON conversations;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY conversations_user_isolation ON conversations
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS dreaming_watermarks_user_isolation ON dreaming_watermarks;
ALTER TABLE dreaming_watermarks ENABLE ROW LEVEL SECURITY;
CREATE POLICY dreaming_watermarks_user_isolation ON dreaming_watermarks
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS idempotency_keys_user_isolation ON idempotency_keys;
ALTER TABLE idempotency_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY idempotency_keys_user_isolation ON idempotency_keys
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS knowledge_base_user_isolation ON knowledge_base;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
CREATE POLICY knowledge_base_user_isolation ON knowledge_base
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS message_summaries_user_isolation ON message_summaries;
ALTER TABLE message_summaries ENABLE ROW LEVEL SECURITY;
CREATE POLICY message_summaries_user_isolation ON message_summaries
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS messages_user_isolation ON messages;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY messages_user_isolation ON messages
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS scratchpads_user_isolation ON scratchpads;
ALTER TABLE scratchpads ENABLE ROW LEVEL SECURITY;
CREATE POLICY scratchpads_user_isolation ON scratchpads
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS tag_registry_user_isolation ON tag_registry;
ALTER TABLE tag_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY tag_registry_user_isolation ON tag_registry
    USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS turns_user_isolation ON turns;
ALTER TABLE turns ENABLE ROW LEVEL SECURITY;
CREATE POLICY turns_user_isolation ON turns
    USING (user_id = current_setting('app.user_id', true));
