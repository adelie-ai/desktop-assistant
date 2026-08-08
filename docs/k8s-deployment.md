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
- [TLS on the WebSocket door](#tls-on-the-websocket-door)
- [Who may administer the instance](#who-may-administer-the-instance)
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

### Telemetry export: one build argument, off by default

Both Dockerfiles take `--build-arg OTEL=1`, which compiles the OTLP exporter
into the daemon - and, for the fleet image, into all 13 bundled MCP servers as
well. Without it the image contains no exporter, and the `OTEL_*` variables the
manifests set do nothing at all.

```sh
podman build --build-arg OTEL=1 -t localhost/adele-daemon:otel -f Dockerfile .
```

The argument takes `0` or `1` and nothing else; any other value fails the build
rather than producing an image that looks instrumented and exports nothing. The
finished image carries the answer as a label:

```sh
podman image inspect --format '{{index .Config.Labels "ai.adelie.otel"}}' <image>
```

The `OTEL=1` build takes noticeably longer and produces a larger binary. It
needs no extra packages in the builder.

An image that can export still exports nothing until an overlay opts the
collector in and points it at a backend. Both steps, and how to confirm the
signals arrive, are in
[`deploy/k8s/README.md`](../deploy/k8s/README.md#telemetry).

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
  rsync -aL --exclude target --exclude .git --exclude build \
        --exclude '.flatpak-builder' --exclude .venv --exclude .worktrees \
        --exclude .claude --exclude '.env' --exclude '.envrc' \
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
  components/
    telemetry/             opt-in; an overlay lists it under `components:`
      otel-collector.yaml  OpenTelemetry Collector DaemonSet + Service
      otel-collector-config.yaml  collector pipeline (backend from the environment)
      daemon-telemetry.yaml  patch: the daemon's OTEL_* wiring
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

The `db_query` tool runs as a restricted `adele_query` role - on reads so
Postgres row-level security applies to it, and on writes so an LLM-supplied
statement owns nothing outside its `scratch` sandbox. That role is deliberately
**not** created by the daemon's auto-migrations - the daemon connects as a
least-privilege role that cannot `CREATE ROLE`. Without this step a fresh
database ships a **dead `db_query`** that fails closed on every call:

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
startup log. That is a plaintext `ws://` listener because the shipped
`daemon.toml` sets `[tls] enabled = false` for this smoke-test example - see
the next section before you point a real deployment at it.

## TLS on the WebSocket door

`[tls] enabled` defaults to `true`. Both `deploy/k8s/base/daemon.toml` and
`deploy/k8s/overlays/example/daemon.toml` set `[tls] enabled = false`
deliberately, which is why step 6 above shows plain `ws://` with no
certificate anywhere in the example. Turn TLS on for a real deployment by
removing that line (or setting it back to `true`) and either:

- leaving `cert_file` and `key_file` unset, so the daemon generates and
  reuses a local CA and server certificate under
  `$XDG_DATA_HOME/desktop-assistant/tls/` - `/state/data/desktop-assistant/tls/`
  in this base's Deployment, which is on the `/state` PVC and so survives a
  pod restart; or
- pointing `cert_file` and `key_file` at a certificate you manage yourself,
  for example a `cert-manager` Secret mounted into the pod.

**Behaviour change.** If `[tls] enabled` is `true` and the daemon cannot set
up TLS - a missing or unreadable cert/key file, a certificate that fails to
parse, or a PVC the daemon cannot write to on the auto-generate path - the
daemon now **refuses to start** instead of falling back to plaintext. Before
this change it logged one error line and served the WebSocket door
(including `/login`) in plaintext on the same address instead. That mattered
too little to fail startup over before the authorization tier; it matters
enough to refuse now, because that door carries the bearer token for an
administrator (`[authz] admin_subjects`, see
[Who may administer the instance](#who-may-administer-the-instance)).

If a pod goes into `CrashLoopBackOff` after enabling TLS or rotating a
certificate, check `kubectl logs` for a `TLS setup failed` line naming the
underlying error, then restore a readable, valid cert/key pair and roll the
pod. If you want this instance to serve plaintext deliberately - for example
behind an ingress or a tailnet that already terminates TLS - set
`[tls] enabled = false` (or `DESKTOP_ASSISTANT_WS_TLS=false`) instead of
leaving a broken TLS configuration in place.

## What the assistant says about where its tools run

The daemon's own tools - the MCP servers it spawns, its built-ins - act on the
daemon's filesystem. In a pod that is the container, not the user's computer. So
the assistant is told which machines exist before it is told which tools it has,
in a `Where things run` section of its system prompt.

The daemon works this out for itself. It reports a container when any of these
is true: `KUBERNETES_SERVICE_HOST` is set, `/.dockerenv` exists,
`/run/.containerenv` exists, or the generic `container` variable is set. Every
pod sets the first, so a normal deployment needs no configuration.

State it yourself where detection cannot help - a virtual machine, a bare-metal
server, or a container you genuinely treat as the user's own workstation:

```toml
# daemon.toml, in your overlay
[deployment]
on_workstation = false
```

`DESKTOP_ASSISTANT_ON_WORKSTATION` sets the same value and wins over the file.
Absent both, container detection decides, and a daemon that is not in a
container reports a workstation.

What changes for the person using it: with a split like this, the assistant no
longer offers to read files it cannot reach. It names the two machines, uses a
client-side tool for the user's own files when the client registered one, and
says plainly that it can act only on the pod when the client registered none.
Client-side tools are configured in
[client-mcp-host.md](client-mcp-host.md).

## Who may administer the instance

The daemon separates a **tenant** from an **administrator**. A tenant owns their
own conversations, knowledge and preferences. An administrator additionally owns
the service: provider credentials, connectors and purposes, the database, the
WebSocket auth posture, and the MCP servers the daemon spawns. The full command
split is in [API_TRANSPORT.md](API_TRANSPORT.md).

On a desktop this needs no configuration: the local Unix-socket peer runs the
daemon, so it administers the daemon. **In a pod there is no such peer.** Every
client arrives over the WebSocket door, and a WebSocket client is a tenant
unless you name it:

```toml
# daemon.toml, in your overlay
[authz]
# The JWT `sub` of each administrator. For the built-in /login door this is the
# login username (DESKTOP_ASSISTANT_WS_LOGIN_USERNAME); under OIDC it is the
# subject your provider issues.
admin_subjects = ["adele"]
```

Name the real subject. `"default"` is not one: it is the storage schema's own
sentinel, and the daemon drops it from this list with a warning. Under OIDC,
check that your provider puts the caller's identity in `sub` - a token with no
`sub` claim is now refused at the handshake with `401` rather than connecting as
that sentinel.

Leave it out - the default - and **nobody** can change this instance's
configuration over the network. That is the right setting for an instance whose
config is fully declared in the overlay and re-seeded on change. Set it when you
administer the instance through a client's settings UI.

The section is deliberately file-only. No command writes it, so a leaked WS
login password no longer hands over the whole admin surface. It is read once at
startup: edit it and restart the pod. A config reload reports `authz` in
`restart_required` rather than pretending the edit is live.

One caveat worth knowing before you enable extra MCP servers on a shared
instance: the daemon-side `fileio`, `terminal` and `command` servers run inside
the daemon process as its own uid, so a tool call could write `daemon.toml`
directly. They ship disabled. Constraining daemon-side tool execution is the
other half of `docs/design/multi-tenancy-boundary.md` (decisions 3 to 5).

Remember the seed rule below: editing `daemon.toml` in the overlay does not
reconfigure a running instance. Re-seed, or set the value before first boot.

The shipped `deploy/k8s/base/daemon.toml` and
`deploy/k8s/overlays/example/daemon.toml` both leave `[authz]` out on purpose.
The base is the smoke-test shape and the example overlay is a public template;
neither should ship a subject name that looks like a real account.

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

`[authz] admin_subjects` is the exception that proves the rule: it is file-only,
so it can *only* change this way, and the daemon must restart to pick it up.

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
Reads fail with a permission error and writes report that the role is missing;
the `scratch` sandbox itself needs no privileged step, the daemon provisions
it.

**Embeddings are silently empty and search is worse than expected.** Either the
embedding model was never pulled (step 3), or the embedding purpose points at a
*generation* model rather than an embedding model. A generation model returns a
501 from the embeddings endpoint and the daemon degrades to full-text search
rather than failing loudly. Check `[purposes.embedding]` names something like
`nomic-embed-text`.

**Daemon starts but every turn fails to authenticate upstream.** The connection
secret is not set on this instance's `/state` PVC - see step 5. Each namespace
needs its own; it does not come from the image or a Kubernetes Secret.

**Daemon pod `CrashLoopBackOff` after enabling TLS or rotating a
certificate.** `[tls] enabled = true` and TLS setup failed - see
[TLS on the WebSocket door](#tls-on-the-websocket-door). Check `kubectl logs`
for the `TLS setup failed` line naming the underlying error, then fix the
cert/key and roll the pod. This is deliberate: the daemon refuses to serve
the remote WebSocket door in plaintext when TLS was requested and cannot be
delivered.

**`rls-bootstrap` Job pod hangs in `Init`.** It gates on `pg_isready`. If
Postgres is up, check that the `rls-bootstrap-sql` ConfigMap exists - the Job
mounts it, and `just deploy-rls-bootstrap` is what generates it. Applying the
Job manifest alone leaves the pod unable to mount.

**Postgres will not start on NFS-backed storage.** `root_squash` blocks the
entrypoint's `chown` of `PGDATA`. The base pins `local-path` for this reason;
keep single-replica state on node-local storage.
