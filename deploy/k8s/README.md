# Kubernetes manifests for the daemon

A kustomize base plus per-environment overlays, running the `desktop-assistant`
daemon (the "brain") with its own pgvector Postgres and an in-cluster Ollama.

**The deployment guide is [`docs/k8s-deployment.md`](../../docs/k8s-deployment.md)** -
how to build and push the image, deploy an instance step by step, run a second
instance alongside the first, and day-two operations. This file covers what is
specific to the manifests themselves.

> Every hostname, registry, namespace, and model name here is a **placeholder**.
> This repo is public - real values belong in a private overlay, never in a
> commit.

## Layout

```
base/                    namespace-agnostic; no hostnames, no registry, no creds
  kustomization.yaml
  postgres.yaml          pgvector Postgres + PVC + initdb hook (creates `vector`)
  ollama.yaml            in-cluster Ollama + model PVC
  daemon.yaml            the daemon, its /state PVC, Service
  rls-bootstrap.yaml     Job provisioning the adele_query RLS role
  daemon.toml            seed config; an overlay replaces it
overlays/
  example/               the shape of an environment, with placeholder values
secret.example.yaml      documents the Secret shape; created imperatively
check-rls-bootstrap.sh   named shape/anti-drift assertions for the RLS Job
```

The base names no environment. An overlay supplies the namespace, the image, and
the seed `daemon.toml`:

```sh
kubectl kustomize overlays/example | kubectl apply -f -
```

## Private overlays

This repo is public, so it ships only `overlays/example`. Keep real overlays
outside the repo and point them at `base/` by relative path - ideally through a
symlink, so moving the checkout is a one-line fix. The deploy recipes take the
target from the environment:

```sh
ADELE_K8S_NAMESPACE=<namespace> just deploy-rls-bootstrap
```

Full pattern, with a worked overlay:
[`docs/k8s-deployment.md`](../../docs/k8s-deployment.md#private-overlays).

## Validation

```sh
just check-deploy
```

Renders the base and the example overlay, schema-validates the output
client-side, dry-runs the generated `rls-bootstrap-sql` ConfigMap, and runs the
RLS shape assertions below. Entirely offline - never contacts an API server, so
it is safe in CI.

## RLS role bootstrap

The `db_query` read tool runs as a restricted `adele_query` role (`SET LOCAL
ROLE`) so Postgres row-level security applies to it. That role and its grants are
the privileged half of the RLS backstop in
`crates/storage/bootstrap/rls_role.sql`. It is deliberately **not** part of the
daemon's auto-run migrations: the daemon connects as a least-privilege role that
cannot `CREATE ROLE`/`GRANT`, so nothing in the pod provisions it. Without this
step a fresh database ships a **dead `db_query`** that fails closed on every call.

`base/rls-bootstrap.yaml` is a Job that runs that SQL as the app role `adele`,
gated on `pg_isready` via a `wait-for-postgres` initContainer. `just
deploy-rls-bootstrap` drives it:

- **No SQL duplication / drift.** The SQL is never hand-copied into a manifest.
  The recipe generates the `rls-bootstrap-sql` ConfigMap straight from the
  canonical `crates/storage/bootstrap/rls_role.sql`, and the Job mounts it at
  `/bootstrap`. The running SQL is always byte-for-byte the source.
- **Idempotent / re-runnable.** The SQL swallows a duplicate role and its grants
  self-heal (`WITH ADMIN OPTION` + `ALTER DEFAULT PRIVILEGES`); the recipe clears
  any prior Job first (a Job's pod template is immutable, so a bare re-apply
  would error). Re-run it freely.
- **Not folded into `postgres-init`.** That initdb hook (`base/postgres.yaml`)
  runs once on empty `PGDATA` before any app tables exist, so `GRANT SELECT ON
  ALL TABLES` would grant on nothing. A ready-gated, re-runnable Job avoids that.

Run it after the daemon has migrated so the explicit grant lands on real tables;
the `ALTER DEFAULT PRIVILEGES` clause also covers tables added by later
migrations, so ordering is not critical and a re-run is always safe.

`check-rls-bootstrap.sh` asserts the above as named checks -
`rls_bootstrap_manifest_runs_rls_role_sql`, `rls_bootstrap_passes_app_role_adele`,
`rls_bootstrap_gated_on_postgres_ready`, `rls_bootstrap_is_rerunnable`,
`rls_bootstrap_configmap_from_canonical_sql` - so a refactor that breaks one
fails the gate by name.

## Smoke test

```sh
NS=<namespace>

# Reach the WS door from the desktop
kubectl -n "$NS" port-forward svc/adele-daemon 11339:11339 &

# Grab the login password
PW=$(kubectl -n "$NS" get secret adele-secrets \
       -o jsonpath='{.data.WS_LOGIN_PASSWORD}' | base64 -d)

# Connect the desktop TUI to the remote brain and send a prompt
adele-tui --transport ws --service ws://127.0.0.1:11339/ws \
  --ws-login-username adele --ws-login-password "$PW"
```

Expected: a real reply. Then `kubectl -n "$NS" rollout restart deploy/adele-daemon`
and reconnect - conversation history persists (it is in Postgres, not the pod),
and so does anything changed via the settings API (it is on the state PVC).

## Config persistence and the seed

A `local-path` PVC (`adele-daemon-state`) is mounted at `/state`, with
`XDG_CONFIG_HOME`/`XDG_DATA_HOME` pointed under it, so everything the daemon
persists survives restarts and rollouts: `daemon.toml`, `mcp_servers.toml`,
service accounts, the per-connection secret files set from a client
(`SetConnectionSecret`), and the system-id.

`daemon.toml` in an overlay is therefore a **seed**, not live config. An init
container copies it onto a *fresh* volume only (non-clobbering) and chowns the
volume to the daemon uid; after first boot the on-volume config wins, so runtime
edits made through the settings API persist. Editing the overlay and re-applying
does **not** reconfigure a running daemon.

To reset the baseline, either edit the live file in place:

```sh
kubectl -n "$NS" exec deploy/adele-daemon -- \
  sh -c 'cat > /state/config/desktop-assistant/daemon.toml' < my-daemon.toml
kubectl -n "$NS" rollout restart deploy/adele-daemon
```

or wipe the volume to re-seed from the overlay - this also drops any client-set
credentials and MCP enable/disable state on the PVC:

```sh
kubectl -n "$NS" scale deploy/adele-daemon --replicas=0
kubectl -n "$NS" delete pvc adele-daemon-state   # recreated on next apply
kubectl kustomize <your-overlay> | kubectl apply -f -
```

Two storage notes, both learned the hard way:

- **`local-path`, not NFS.** NFS `root_squash` blocks the Postgres entrypoint's
  `chown` of `PGDATA`, and blocks the init container's chown of the state volume.
  Single-replica state belongs on node-local storage anyway.
- **One mounted dir, not deep XDG subpaths.** kubelet creates a mount's parents
  as root, which would stop the daemon (uid 10001) creating sibling directories
  like `~/.local/share/adelie`. Mounting `/state` alone keeps every write a child
  of a directory the daemon owns.

## Credentials

Two different things, stored two different ways:

| | Where | How set |
| --- | --- | --- |
| Postgres + web login passwords | k8s Secret `adele-secrets` | `kubectl create secret` (see `secret.example.yaml`) |
| Per-connection LLM credential | daemon's own store on the `/state` PVC | `SetConnectionSecret` from a client |

A cloud provider key in `adele-secrets` does **not** wire anything up. The daemon
reads it from its own store at
`/state/data/desktop-assistant/secrets/<account>`, where `<account>` comes from
the connection's `[connections.<name>.secret]` block. Each namespace has its own
PVC, so each instance needs its own.

## What this deployment is not

- **Auth:** an interim static password (`/login` -> HS256 token). OIDC is the
  later, multi-tenant path.
- **TLS on the daemon port:** off. The LAN/tailnet provides transport encryption,
  so CA distribution is skipped. Not for the public internet. (The web UI's
  ingress terminates TLS separately.)
- **Inference:** whatever the overlay's `daemon.toml` points at. The example
  keeps embeddings on the in-cluster Ollama so vectors never leave the cluster,
  and sends reasoning purposes to a cloud connection.
