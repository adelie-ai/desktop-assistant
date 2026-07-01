-- #434: Postgres Row-Level Security backstop for the LLM-facing
-- `execute_database_query` read path — OWNER-LEVEL half.
--
-- The AST grafter (#141) rewrites every SELECT to append `user_id = $caller`
-- onto personal-data tables, but that rewriter is the *only* thing standing
-- between a hostile/manipulated LLM and another user's rows. This migration
-- makes the database itself enforce tenant scoping regardless of the SQL
-- text: the tool's read path switches into a dedicated, un-privileged role
-- (`adele_query`) that has neither table ownership nor BYPASSRLS, then pins
-- the caller's id in `app.user_id`. The policies below then filter every
-- personal-data table to the caller's own rows — the hard backstop underneath
-- the grafter.
--
-- IMPORTANT — this migration is run automatically at every daemon startup,
-- AS THE DAEMON'S OWN (deliberately un-privileged) DATABASE ROLE. It must
-- therefore only do things a plain table OWNER can do: ENABLE ROW LEVEL
-- SECURITY and CREATE POLICY on its own tables. It must NOT create roles or
-- grant role membership / schema privileges — an un-privileged app role
-- cannot, and attempting it would crash-loop the daemon.
--
-- The privileged, cluster-level half — creating the `adele_query` role,
-- granting the app role membership in it, and granting it SELECT — lives in
-- `crates/storage/bootstrap/rls_role.sql` and is run ONCE by a superuser/DBA.
-- The read path fails closed until that bootstrap has run (a `SET LOCAL ROLE`
-- to a missing role errors, returning no rows rather than un-scoped rows).
-- The two halves are independent and converge in any order.
--
-- Trusted daemon code paths are UNAFFECTED. RLS here is non-FORCE, so the
-- table owner is exempt; the daemon connects as the role that owns every
-- table, so its own queries — which already scope by user_id in their WHERE
-- clauses — see all rows as before. Only the `SET LOCAL ROLE adele_query`
-- read path is filtered.
--
-- Idempotent (re-run every startup): DROP POLICY IF EXISTS before each
-- CREATE POLICY (Postgres has no CREATE POLICY IF NOT EXISTS), and
-- ENABLE ROW LEVEL SECURITY is a no-op when already enabled.
--
-- `current_setting('app.user_id', true)` returns NULL when the GUC is unset
-- (the `true` = missing_ok), and `user_id = NULL` is NULL, so a read path
-- that forgot to pin app.user_id sees ZERO rows — fail-closed, never a leak.
-- user_id is `text`, matching current_setting's return type, so no cast.

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
