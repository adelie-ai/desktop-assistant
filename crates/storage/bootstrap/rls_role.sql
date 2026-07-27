-- #434 RLS backstop — PRIVILEGED bootstrap (run ONCE by a superuser/DBA).
--
-- This is the cluster-level half of the RLS backstop. It is deliberately
-- NOT part of the daemon's auto-run migrations (`run_migrations`): the daemon
-- connects as an un-privileged, least-privilege role that owns its tables but
-- cannot create roles or grant schema/role privileges. Those operations live
-- here and are applied once, out of band, by an administrator.
--
-- What it provisions:
--   * `adele_query` — the role BOTH db_query paths assume via
--     `SET LOCAL ROLE`. NOLOGIN (entered only via SET ROLE, never connected
--     to directly), NOBYPASSRLS + non-superuser + non-owner so the row-level
--     policies from migration 029 actually apply to it. The write path uses
--     the same role so an LLM-supplied statement runs with no ownership of,
--     and no write privilege on, anything the application manages.
--   * membership: the app role is granted `adele_query` so it can SET ROLE
--     into it. WITH ADMIN OPTION so the grant is self-healing if re-run.
--   * SELECT on every current table + USAGE on the schema (only the read path
--     reads application tables), plus DEFAULT PRIVILEGES so tables added by
--     future migrations (created by the app role) are readable without
--     re-running this script.
--
-- Deliberately NOT provisioned here: the `scratch` write sandbox. The daemon
-- creates that schema on demand and grants itself USAGE + CREATE on it, so a
-- database that has run this script needs no further privileged step for
-- writes to work.
--
-- Usage (defaults to the app role `adele_dave`; override with -v app_role=...):
--
--   psql "postgres://<superuser>@host/<db>" \
--     -v app_role=adele_dave \
--     -f crates/storage/bootstrap/rls_role.sql
--
-- Idempotent: safe to re-run. Run it against each database that the daemon
-- uses. Order relative to `run_migrations` does not matter — the two halves
-- converge; the read path fails closed (zero rows) until this has run.

\if :{?app_role}
\else
    \set app_role adele_dave
\endif

\echo Provisioning RLS role adele_query, granting to app role :'app_role'

-- The restricted role. Attempt-and-swallow so a re-run (or a shared cluster
-- where it already exists, or a concurrent creator) is a no-op, not an error.
DO $$
BEGIN
    CREATE ROLE adele_query NOLOGIN NOBYPASSRLS;
EXCEPTION WHEN duplicate_object OR unique_violation THEN
    NULL;
END
$$;

-- Let the app role assume adele_query via SET ROLE. ADMIN OPTION makes the
-- grant idempotent across re-runs.
GRANT adele_query TO :"app_role" WITH ADMIN OPTION;

-- Read access. The read path only ever SELECTs; grant on all current tables
-- plus default privileges so a table added by a later migration (created by
-- the app role) is readable without re-running this script.
GRANT USAGE ON SCHEMA public TO adele_query;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO adele_query;
ALTER DEFAULT PRIVILEGES FOR ROLE :"app_role" IN SCHEMA public
    GRANT SELECT ON TABLES TO adele_query;

\echo Done. Restart the daemon (or it will pick this up on next migration run).
