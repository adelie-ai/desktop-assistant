# `deploy/` — containerized deployment

This directory holds everything for building and running desktop-assistant as
containers, from a single image up to a full Kubernetes cluster deployment. It
is part of the containerization epic (#378).

## Directory map

| Directory   | Issue       | What it is |
|-------------|-------------|------------|
| `docker/`   | #379 (C-1)  | Container **image build** for the daemon: `Dockerfile`, entrypoint, and image-level docs. |
| `compose/`  | #380 (C-2)  | A **reference full system** wired with Compose (Podman/Docker): the daemon, PostgreSQL, MCP servers, and supporting services, with example `daemon.toml`, `mcp_servers.toml`, and OIDC config. The easiest way to run the whole stack on one host. |
| `k8s/`      | #382 (C-4)  | **Cluster manifests** for running on Kubernetes (Deployments/Services/Secrets/PVCs, PostgreSQL provisioning, and the daemon connection wiring). |
| `podman/`   | #385 (C-7)  | A desktop **quadlet pod** for running the stack as a systemd-managed Podman pod on a single machine. **Deferred** — not yet present. |

See each subdirectory's own `README.md` for the details specific to that
deployment style.

## PostgreSQL is the data store for containers

Pod and container filesystems are **ephemeral** — data written to local disk is
lost on restart or reschedule. The daemon's JSON file-store fallback writes to
that ephemeral disk, so **containerized deployments must use PostgreSQL**, which
externalizes the data into a durable service.

The daemon selects PostgreSQL automatically when a connection URL is configured
(via `[database].url` in `daemon.toml`, or the `DESKTOP_ASSISTANT_DATABASE_URL`
environment variable), and otherwise falls back to the JSON store. The database
schema is created and migrated automatically at startup — there is no manual
DDL step — but the target PostgreSQL must have the **pgvector** extension
available.

The `compose/` and `k8s/` deployments each provision PostgreSQL and point the
daemon at it; this README does not duplicate their configuration.

For the full reference — store selection and precedence, the `[database]`
config block, auto-migration behavior, connection-string form, and
backup/restore with `pg_dump`/`pg_restore` — see
[`docs/data-store.md`](../docs/data-store.md).
