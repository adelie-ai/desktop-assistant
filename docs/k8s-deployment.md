# Deploying the daemon to Kubernetes

How to build the daemon image, push it to a registry your cluster can pull from,
and deploy one or more instances with kustomize.

This covers the **brain**: the daemon, its Postgres, and an in-cluster Ollama.
The web UI is a separate repo with its own deployment
([`adele-web-ui`](https://github.com/adelie-ai/adele-web-ui), `docs/k8s-deployment.md`)
that renders into the same namespace.

> Every hostname, registry, namespace, and model name below is a
> **placeholder**. This repo is public - real values belong in a private
> overlay, never in a commit. See [Private overlays](#private-overlays).

## Contents

- [Which image do I want?](#which-image-do-i-want)
- [Build and push](#build-and-push)
- [How the manifests are laid out](#how-the-manifests-are-laid-out)
- [Deploy an instance](#deploy-an-instance)
- [Private overlays](#private-overlays)
- [Worked example: a second instance](#worked-example-a-second-instance)
- [Day-two operations](#day-two-operations)
- [Troubleshooting](#troubleshooting)

## Which image do I want?

Two Dockerfiles at the repo root build two different things:

| File | Builds | Use when |
| --- | --- | --- |
| `Dockerfile` | daemon only | You want conversation + persistence and no tools. Smallest image, builds from this repo alone. |
| `Dockerfile.fleet` | daemon **+ the MCP server fleet** | You want tools (web, files, terminal, tasks, weather, ...). This is the image to deploy, and the one to derive from. |

`Dockerfile.fleet` is the usual answer. It bundles the fleet binaries at
`/opt/adele/mcp/<name>-mcp` and seeds a curated default config on first boot, so
a fresh instance comes up with tools already wired. Its full contract - the
on-disk layout, which servers ship enabled, and how to derive a downstream image
with one `COPY` - is documented in [`deploy/mcp/README.md`](../deploy/mcp/README.md).

## Build and push

### Daemon-only image

Builds from this repo alone:

```sh
podman build -t localhost/adele-daemon:dev -f Dockerfile .
```

### Fleet image

The daemon and the fleet servers each build from their own source tree, so the
build context is a **staged directory** holding `desktop-assistant/` and the
`*-mcp` repos as siblings. Stage clean copies - no `target/`, no `.git/`:

```sh
# ADELE = your checkout root, holding desktop-assistant + the *-mcp repos.
ADELE=<path-to-your-checkout-root>
CTX=$(mktemp -d)/fleet-ctx
mkdir -p "$CTX"

for repo in desktop-assistant command-mcp cve-mcp fileio-mcp geocode-mcp \
            homeassistant-mcp internet-radio-mcp openstreetmap-mcp skills-mcp \
            tasks-mcp terminal-mcp timeclock-mcp weather-forecast-mcp web-mcp; do
  rsync -a --exclude target --exclude .git --exclude .worktrees \
        "$ADELE/$repo/" "$CTX/$repo/"
done

podman build -t localhost/adele-daemon:fleet -f "$CTX/desktop-assistant/Dockerfile.fleet" "$CTX"
```

See [`deploy/mcp/README.md`](../deploy/mcp/README.md) for the exclude list to use
and for adding a server to the fleet.

### Tag and push

Tag with something **immutable and traceable** - a short commit SHA, optionally
prefixed with what changed. A moving tag like `latest` makes a rollout
unreproducible and a rollback guesswork.

```sh
REGISTRY=registry.example.com:5000
TAG=fleet-$(git -C "$ADELE/desktop-assistant" rev-parse --short HEAD)

podman tag localhost/adele-daemon:fleet "$REGISTRY/adele/adele-daemon:$TAG"
podman push "$REGISTRY/adele/adele-daemon:$TAG"
```

If your registry allows anonymous pull and serves a cert the nodes trust, no
`imagePullSecret` is needed. Otherwise create one and add it to the overlay.

## How the manifests are laid out

`deploy/k8s/` is a kustomize base plus per-environment overlays:

```
deploy/k8s/
  base/                    namespace-agnostic; no hostnames, no registry, no creds
    kustomization.yaml
    postgres.yaml          pgvector Postgres + PVC + initdb hook
    ollama.yaml            in-cluster Ollama + model PVC
    daemon.yaml            the daemon, its /state PVC, Service
    rls-bootstrap.yaml     Job provisioning the adele_query RLS role
    daemon.toml            seed config (an overlay replaces it)
  overlays/
    example/               the shape of an environment, with placeholder values
      kustomization.yaml
      namespace.yaml
      daemon.toml
  secret.example.yaml      documents the Secret shape; created imperatively
```

The base names no environment. An overlay supplies:

- the **namespace** (and the `Namespace` object),
- the **image** registry and tag,
- the **seed `daemon.toml`** (connections, purposes, models).

Render any overlay to see exactly what would be applied:

```sh
kubectl kustomize deploy/k8s/overlays/example
```

Validate offline, without touching a cluster - renders the base and the example,
schema-checks the output, and runs the RLS-bootstrap shape assertions:

```sh
just check-deploy
```

## Deploy an instance

Namespace `adele-example` throughout; substitute your own.

### 1. Credentials

Never committed - created imperatively:

```sh
kubectl create namespace adele-example

kubectl -n adele-example create secret generic adele-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=WS_LOGIN_PASSWORD="$(openssl rand -hex 24)"
```

### 2. Apply

```sh
kubectl kustomize deploy/k8s/overlays/example | kubectl apply -f -

kubectl -n adele-example rollout status deploy/postgres
kubectl -n adele-example rollout status deploy/ollama
```

### 3. Pull the embedding model

The daemon's embedding purpose points at the in-cluster Ollama, so vectors never
leave the cluster. Nothing pulls the model for you:

```sh
kubectl -n adele-example exec deploy/ollama -- ollama pull nomic-embed-text
```

Skipping this is the classic silent failure: embeddings come back empty, vector
search quietly degrades to full-text only, and nothing errors.

### 4. Provision the RLS role

The `db_query` tool runs as a restricted `adele_query` role so Postgres
row-level security applies to it. That role is deliberately **not** created by
the daemon's auto-migrations - the daemon connects as a least-privilege role
that cannot `CREATE ROLE`. Without this step a fresh database ships a **dead
`db_query`** that fails closed on every call:

```sh
ADELE_K8S_NAMESPACE=adele-example just deploy-rls-bootstrap
```

Idempotent and re-runnable. Run it after the daemon has migrated, so the grants
land on real tables.

### 5. Connection secrets

A cloud LLM credential is **not** a Kubernetes Secret. The daemon keeps
per-connection secrets in its own store on the `/state` PVC, at
`/state/data/desktop-assistant/secrets/<account>`, where `<account>` is the
`account` field of the connection's `[connections.<name>.secret]` block.

Set it from any connected client via `SetConnectionSecret` (the settings UI
exposes this). `backend = "auto"` resolves to that file store inside a pod,
where no desktop keyring exists.

Until it is set, the daemon starts and serves, but every turn on that connection
fails to authenticate.

### 6. Verify

```sh
kubectl -n adele-example get pods
kubectl -n adele-example logs deploy/adele-daemon --tail=30

# WS door from your workstation
kubectl -n adele-example port-forward svc/adele-daemon 11339:11339
```

Expect `WebSocket listening on ws://0.0.0.0:11339` and the tool inventory in the
startup log.

## Private overlays

**This repo is public.** Real namespaces, registries, image tags, hostnames, and
model choices must not be committed. Keep a private overlay outside the repo and
point it at the in-repo base by relative path:

```
~/deploy-env/                       (private; not a git repo, or a private one)
  _bases/
    desktop-assistant -> symlink to <checkout>/desktop-assistant/deploy/k8s/base
  prod/
    daemon/
      kustomization.yaml
      namespace.yaml
      daemon.toml
```

```yaml
# ~/deploy-env/prod/daemon/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: adele-production

resources:
  - namespace.yaml
  - ../../_bases/desktop-assistant

images:
  - name: registry.example.com:5000/adele/adele-daemon
    newName: registry.internal.example:5000/adele/adele-daemon
    newTag: fleet-a1b2c3d

configMapGenerator:
  - name: adele-daemon-config
    behavior: replace
    files:
      - daemon.toml

generatorOptions:
  disableNameSuffixHash: true
```

A **symlinked base** keeps the private overlay independent of where the checkout
lives - repointing it after moving or merging is one `ln -sfn`. kustomize
follows it.

The deploy recipes take the environment from the environment, so the same
recipes drive any instance:

```sh
ADELE_K8S_NAMESPACE=adele-production just deploy-rls-bootstrap
```

## Worked example: a second instance

Running a test and a production instance side by side. They share a cluster and
differ only in their overlays.

| | test | prod |
| --- | --- | --- |
| namespace | `adele-staging` | `adele-production` |
| daemon image tag | whatever is being tried | pinned, immutable |
| `dreaming_enabled` | `false` - no surprise background inference mid-smoke | your call; `true` costs money continuously |
| web UI hostname | `adele-staging.example.com` | `adele.example.com` |
| daemon LB nodePort | `31000` | `31001` - must differ |

Everything else - the base, the recipes, the bootstrap - is identical.

Two things are namespace-scoped and easy to forget on the second instance:

1. **Credentials do not carry over.** Each namespace needs its own
   `adele-secrets`, and its own connection secret on its own `/state` PVC. A
   fresh PVC means a fresh secret store.
2. **NodePorts are cluster-global.** Two `LoadBalancer`/`NodePort` Services
   cannot claim the same port. Pin them explicitly per environment rather than
   letting both auto-allocate, so a re-apply is reproducible.

Exposing the daemon directly (for native GTK/TUI/KDE clients, which do not go
through the web UI's ingress) is environment-specific - cluster support for
`LoadBalancer` varies - so it belongs in the overlay, not the base:

```yaml
# prod/daemon/daemon-lb.yaml, added to the overlay's `resources:`
apiVersion: v1
kind: Service
metadata:
  name: adele-daemon-lb
  labels:
    app: adele-daemon
spec:
  type: LoadBalancer
  selector:
    app: adele-daemon
  ports:
    - name: ws
      port: 11339
      targetPort: 11339
      nodePort: 31001
```

## Day-two operations

### Rolling out a new image

Bump the tag in the overlay and re-apply:

```sh
kubectl kustomize ~/deploy-env/prod/daemon | kubectl apply -f -
kubectl -n adele-production rollout status deploy/adele-daemon
```

Prefer this over `kubectl set image`: an out-of-band `set image` leaves the
overlay claiming one tag while the cluster runs another, and the drift is
invisible until the next apply quietly reverts it.

### Changing the baseline config

`daemon.toml` in the overlay is a **seed**, not live config. An init container
copies it onto a *fresh* `/state` volume and never clobbers an existing one, so
runtime edits made through the settings API win and persist.

Editing the overlay therefore does **not** reconfigure a running instance. To
change a live one, either edit it through the settings UI, or re-seed:

```sh
kubectl -n adele-production exec deploy/adele-daemon -- \
  rm /state/config/desktop-assistant/daemon.toml
kubectl -n adele-production rollout restart deploy/adele-daemon
```

Re-seeding discards runtime config changes, including MCP server enable/disable
state held in `mcp_servers.toml`.

### What survives what

| | pod restart | re-apply | PVC delete |
| --- | --- | --- | --- |
| conversations, knowledge base | yes | yes | no |
| `daemon.toml`, `mcp_servers.toml`, connection secrets | yes | yes | no |
| Ollama models | yes | yes | no (re-pull) |
| web-UI browser sessions | no (emptyDir key) | no | n/a |

## Troubleshooting

**`db_query` fails on every call.** The `adele_query` role was never created -
run `just deploy-rls-bootstrap` (step 4). A fresh database does not have it.

**Embeddings are silently empty and search is worse than expected.** Either the
embedding model was never pulled (step 3), or the embedding purpose points at a
*generation* model rather than an embedding model. A generation model returns a
501 from the embeddings endpoint and the daemon degrades to full-text search
rather than failing loudly. Check `[purposes.embedding]` names something like
`nomic-embed-text`.

**Daemon starts but every turn fails to authenticate upstream.** The connection
secret is not set on this instance's `/state` PVC - see step 5. Each namespace
needs its own; it does not come from the image or a Kubernetes Secret.

**`rls-bootstrap` Job pod hangs in `Init`.** It gates on `pg_isready`. If
Postgres is up, check that the `rls-bootstrap-sql` ConfigMap exists - the Job
mounts it, and `just deploy-rls-bootstrap` is what generates it. Applying the
Job manifest alone leaves the pod unable to mount.

**Postgres will not start on NFS-backed storage.** `root_squash` blocks the
entrypoint's `chown` of `PGDATA`. The base pins `local-path` for this reason;
keep single-replica state on node-local storage.
