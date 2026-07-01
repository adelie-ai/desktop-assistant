-- Auto-loaded by the Postgres image on first boot: any *.sql / *.sh placed
-- in /docker-entrypoint-initdb.d/ runs before the server accepts client
-- connections. The `just test-db` recipe mounts this directory there.
--
-- The storage migrations use pgvector's `vector` type but do NOT create the
-- extension themselves (they assume it is pre-installed, as it is in the
-- production database). Without this, `run_migrations` fails with
-- `type "vector" does not exist`. Creating it here — declaratively, as a
-- fixture — is the drift-proof way to satisfy that assumption in tests.
CREATE EXTENSION IF NOT EXISTS vector;
