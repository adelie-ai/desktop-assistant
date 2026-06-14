# Data store

The desktop-assistant daemon keeps conversations, knowledge base entries,
per-conversation state, and related data in one of two backends:

1. **JSON file store** (the fallback) — a single-user, local-disk store.
2. **PostgreSQL** — the supported store for multi-user, persistent, and
   containerized deployments.

This document explains how the daemon picks between them, why containers
should use PostgreSQL, how the schema is created, and how to back up and
restore the data.

## Which store is used, and how selection works

The daemon uses PostgreSQL when a database URL is configured, and otherwise
falls back to the JSON file store. The selection is made once at startup in
`resolve_database_config` (`crates/daemon/src/config/resolution.rs`) and wired
up in `crates/daemon/src/main.rs`:

- If a database URL resolves to a non-empty value, the daemon connects to
  PostgreSQL, runs migrations, and uses `PgConversationStore`. A failure to
  connect or to migrate aborts startup (it does **not** silently fall back to
  JSON).
- If no database URL is configured, the daemon logs
  `no database URL configured; Postgres features disabled` and uses the JSON
  `PersistentConversationStore`.

### Resolution precedence

The URL is resolved with this precedence (see `resolve_database_config`):

1. The `[database].url` value from `daemon.toml`, if present.
2. Otherwise, the `DESKTOP_ASSISTANT_DATABASE_URL` environment variable.

The resolved value is trimmed; an empty or whitespace-only value is treated as
"not set" and the daemon falls back to the JSON store. Note the precedence:
when `[database].url` is set in the config file, it **wins** over the
environment variable. For containers, the common pattern is to leave
`[database].url` unset and supply `DESKTOP_ASSISTANT_DATABASE_URL` via the
environment so it can be injected per-deployment (compose env, k8s Secret).

### Connection string form

```
postgres://<user>:<pass>@<host>:5432/<db>
```

For example: `postgres://adelie:secret@postgres:5432/desktop_assistant`.

## Why containers should use PostgreSQL

In containerized and Kubernetes deployments the pod filesystem is **ephemeral**:
when a container restarts or is rescheduled, anything written to its local
filesystem is lost. The JSON file store writes to
`$XDG_DATA_HOME/desktop-assistant/conversations.json` (see below), which lives
on that ephemeral filesystem. Relying on it in a pod means conversations and
knowledge vanish on the next restart.

PostgreSQL externalizes the data into a durable, separately-managed service
(backed by a persistent volume or a managed database), so daemon pods can be
restarted, rescheduled, or scaled without losing data. **For any
containerized deployment, configure PostgreSQL.**

## The `[database]` configuration

In `daemon.toml`:

```toml
[database]
# PostgreSQL connection URL. If omitted, the daemon falls back to the JSON
# file store (single-user, local disk only).
url = "postgres://adelie:secret@postgres:5432/desktop_assistant"

# Maximum number of connections in the pool. Default: 5.
max_connections = 5
```

Both fields are optional. `max_connections` defaults to `5`
(`default_database_max_connections` in `crates/daemon/src/config/mod.rs`).

### Environment override

```
DESKTOP_ASSISTANT_DATABASE_URL=postgres://adelie:secret@postgres:5432/desktop_assistant
```

This is only consulted when `[database].url` is absent from the config file
(see precedence above).

## Schema is created automatically (no manual DDL)

There is **no manual schema setup step**. On startup with a PostgreSQL URL, the
daemon calls `run_migrations` (`crates/storage/src/pool.rs`), which applies the
embedded migrations in order. The migration SQL lives in
`crates/storage/migrations/` (`001_initial_schema.sql` through
`025_context_window_observations.sql`) and is compiled into the binary via
`include_str!`, so the running container needs no external migration files.

`run_migrations` also runs `CREATE EXTENSION IF NOT EXISTS vector`, so the
target database needs the **pgvector** extension available (the migrations
create `vector`/`tsvector` columns for hybrid search). Use a pgvector-enabled
PostgreSQL image, and ensure the role used at first startup can create the
extension. If migrations fail, the daemon logs
`failed to run database migrations` and exits.

### One-time JSON → PostgreSQL import

If JSON files already exist on disk the first time the daemon starts against an
empty PostgreSQL database, it performs a one-time import (see `main.rs`):

- `conversations.json` → conversations table (only if the table is empty).
- `preferences.json` and `factual_memory.json` → knowledge base (only if that
  table is empty).

This is convenience migration for users moving an existing single-user install
into PostgreSQL; it is not a sync and does not run again once the tables are
populated.

## Where PostgreSQL comes from in the reference deployments

This document does not duplicate the deployment manifests — each reference
deployment provisions PostgreSQL and points the daemon at it:

- **Compose** (issue #380, C-2): `deploy/compose/` defines the full reference
  system, including a PostgreSQL service the daemon connects to. See
  `deploy/compose/README.md` and `deploy/compose/daemon.toml`.
- **Kubernetes** (issue #382, C-4): `deploy/k8s/` provides the cluster
  manifests, including how PostgreSQL is provisioned (or referenced) and how the
  connection string is injected into the daemon (typically via a Secret feeding
  `DESKTOP_ASSISTANT_DATABASE_URL`). See `deploy/k8s/` when it lands.

Refer to those directories for the concrete YAML; the contract from the
daemon's side is only "give me a reachable, pgvector-enabled PostgreSQL and a
connection URL".

## Backup and restore

Because the data lives in PostgreSQL, use the standard PostgreSQL tools. The
exact host/port/credentials come from your connection URL.

### Back up

```sh
# Plain SQL dump
pg_dump "postgres://adelie:secret@postgres:5432/desktop_assistant" \
  > desktop-assistant-backup.sql

# Or the custom compressed format (recommended for pg_restore)
pg_dump -Fc "postgres://adelie:secret@postgres:5432/desktop_assistant" \
  -f desktop-assistant-backup.dump
```

### Restore

```sh
# From a plain SQL dump
psql "postgres://adelie:secret@postgres:5432/desktop_assistant" \
  < desktop-assistant-backup.sql

# From a custom-format dump
pg_restore -d "postgres://adelie:secret@postgres:5432/desktop_assistant" \
  --clean --if-exists desktop-assistant-backup.dump
```

When restoring into a fresh database, ensure the pgvector extension is
available; restoring into an empty database the daemon has not yet migrated is
also fine — the daemon will run any missing migrations on next startup.

## The JSON store (single-user desktop)

For a single-user desktop install (the daemon running directly on your
machine, not in a container), the JSON file store is a reasonable default and
needs no external service. Leave `[database].url` unset and
`DESKTOP_ASSISTANT_DATABASE_URL` unset; the daemon writes to the XDG data
directory:

- `$XDG_DATA_HOME/desktop-assistant/conversations.json`, or
- `$HOME/.local/share/desktop-assistant/conversations.json` when
  `XDG_DATA_HOME` is not set.

(Companion files `preferences.json` and `factual_memory.json` live alongside
it.) Note that some features that depend on the database — including
persisted model/personality selection and the dreaming background task — are
only available with PostgreSQL; the JSON backend keeps a subset of state
in-memory and drops it on shutdown. Back up the JSON store by copying the
`desktop-assistant/` data directory.
