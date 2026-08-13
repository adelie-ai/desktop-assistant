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
base/                       namespace-agnostic; no hostnames, no registry, no creds
  kustomization.yaml
  postgres.yaml             pgvector Postgres + PVC + initdb hook (creates `vector`)
  ollama.yaml               in-cluster Ollama + model PVC
  daemon.yaml               the daemon, its /state PVC, Service
  rls-bootstrap.yaml        Job provisioning the adele_query RLS role
  daemon.toml               seed config; an overlay replaces it
components/
  telemetry/                opt-in; an overlay lists it under `components:`
    kustomization.yaml
    otel-collector.yaml     OpenTelemetry Collector DaemonSet + Service
    otel-collector-config.yaml  collector pipeline; backend from the environment
    daemon-telemetry.yaml   patch: the daemon's OTEL_* wiring + grace period
overlays/
  example/                  the shape of an environment, with placeholder values
secret.example.yaml         documents the Secret shape; created imperatively
check-rls-bootstrap.sh      named shape/anti-drift assertions for the RLS Job
check-telemetry.sh          named shape/anti-drift assertions for the telemetry deploy
check-telemetry-render.sh   the same opt-in property, read from what kustomize renders
```

Everything in `base/` is deployed by every overlay. `components/` is opt-in: an
overlay that says nothing about a component renders exactly as it did before that
component existed.

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
RLS and telemetry shape assertions below. Entirely offline - never contacts an
API server, so it is safe in CI.

`just check-deploy` needs `kubectl`, so the main gate does not run it. The
telemetry assertions need only `python3`, so they also run in `just check`
through `scripts/tests/k8s-telemetry.test.sh`, which additionally breaks each
requirement in turn and fails if the matching check does not notice.

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

## Telemetry

The daemon produces traces, metrics and log records. Console output on stderr is
the default and needs nothing here: `kubectl logs` works as it always has, and
the periodic metrics summary appears in it. Export to a collector is
**additional**, and it takes three deliberate steps - an image built with the
exporter compiled in, an overlay that opts the collector in, and a backend for it
to forward to. Until all three are done, an existing install is unchanged.

Nothing is on by default, and nothing about telemetry is in the base. All of it -
the collector, and the daemon's own `OTEL_*` wiring - lives in
`components/telemetry`, so an overlay that does not list it renders **byte for
byte** what it rendered before this existed. Applying this change to such an
overlay restarts nothing.

The daemon's variables belong with the component rather than the base for a
concrete reason: `OTEL_EXPORTER_OTLP_ENDPOINT` names the `otel-collector`
Service, which only the component creates. In the base it would be a manifest
describing something that is not there. The grace period goes with them, because
it exists to protect a flush that only happens when there is something to export.

What the component deploys: an OpenTelemetry Collector as a **DaemonSet**, one pod
per node, receiving OTLP on 4317 (gRPC) and 4318 (HTTP), batching, and forwarding
all three signals to one backend; plus a patch adding the daemon's `OTEL_*`
variables and `terminationGracePeriodSeconds`. The variables point at the
`otel-collector` Service, which sets `internalTrafficPolicy: Local` so the name
resolves to the collector pod on the daemon's own node.

`adelie-telemetry`'s side of this - the `RUST_LOG` contract, what may appear at
each level, the full `OTEL_*` table, and how a failed export looks in the log -
is in [`docs/logging.md`](../../docs/logging.md). This section covers only what
Kubernetes adds.

### 1. Build an image that can export

Export lives behind an off-by-default Cargo feature, so the default image
contains no exporter at all and setting the variables in a pod does nothing.
Pass the build argument to compile it in:

```sh
podman build --build-arg OTEL=1 -t localhost/adele-daemon:otel -f Dockerfile .
```

`Dockerfile.fleet` takes the same argument, and applies it to the daemon **and**
to all 12 bundled MCP servers, so a tool call is instrumented too:

```sh
podman build --build-arg OTEL=1 -t localhost/adele-daemon:fleet-otel \
  -f "$CTX/desktop-assistant/Dockerfile.fleet" "$CTX"
```

`OTEL` takes `0` or `1` and nothing else. Any other value fails the build rather
than quietly producing an image that looks instrumented and exports nothing.

**To tell whether a given image has it**, read the label:

```sh
podman image inspect --format '{{index .Config.Labels "ai.adelie.otel"}}' <image>
```

The label carries the build argument. The binary is the ground truth, and
answers even for an image built elsewhere:

```sh
podman run --rm --entrypoint sh <image> \
  -c 'grep -c opentelemetry /usr/local/bin/desktop-assistant-daemon'
```

A non-zero count means the exporter is compiled in. Zero means it is not,
whatever the label says.

Expect the `OTEL=1` build to cost real time and size. Most of that is the OTLP
client - tonic, hyper, prost, reqwest, tower - rather than TLS. The builder stage
needs no extra packages: the TLS backend compiles native code through
`aws-lc-rs`, which wants a C compiler and an assembler, and the Rust base image
already carries both. It does **not** need `cmake`.

The runtime stage's `ca-certificates` is load-bearing once a backend is reached
over `https`, because both OTLP transports read the operating system trust store.
Do not move either image to a distroless or scratch base without adding a CA
bundle first.

### 2. Opt the overlay in

One block, and it is the only thing that turns the collector on:

```yaml
# <your-overlay>/kustomization.yaml
components:
  - <relative path>/components/telemetry
```

A private overlay outside this repo reaches the base through a symlink. Give the
components directory one too, so the path stays short and moving the checkout
stays a one-line fix:

```sh
ln -sfn <checkout>/desktop-assistant/deploy/k8s/components _bases/desktop-assistant-components
```

Remove the block and the overlay goes back to what it rendered before telemetry
existed - no collector, and a daemon pod with no telemetry on it.

### 3. Point the collector at a backend

Two values, patched onto the collector. The pipeline reads both through
`${env:...}`, so an overlay never copies it:

```yaml
# <your-overlay>/otel-backend.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector
spec:
  template:
    spec:
      containers:
        - name: collector
          env:
            - name: OTLP_BACKEND_ENDPOINT
              value: <backend-service>.<backend-namespace>.svc.cluster.local:4317
            - name: OTLP_BACKEND_INSECURE
              value: "true"      # plaintext inside the cluster; keep TLS off-cluster
```

```yaml
# <your-overlay>/kustomization.yaml
patches:
  - path: otel-backend.yaml
```

`overlays/example` does exactly this, with a placeholder backend. Which backend,
and its real endpoint, belong to the private overlay - they are not in this repo.

**Do not copy the pipeline file to change the endpoint.** Two reasons, and the
second one is why the endpoint is an environment variable at all:

- It is sixty lines that would drift from the component's copy the first time a
  processor changed there.
- The obvious way to do it does not work. `configMapGenerator` with
  `behavior: replace` cannot target a ConfigMap generated inside a component -
  kustomize has no generator to merge against at that point and the render fails
  with `does not exist; cannot merge or replace`.

An overlay that genuinely needs a different pipeline - a vendor exporter, auth
headers, an extra processor - patches the generated ConfigMap instead, with a
`patches:` entry targeting `ConfigMap/otel-collector-config` and supplying the
whole of `data.config.yaml`. That works, and it keeps the content-hash roll.

**Do this before you move the overlay's image tag to an `otel` build, not after.**
A collector whose backend is still the placeholder does not complain. Its exporter
is lazy - it dials the backend only when something gives it data to forward - so a
collector with nothing to do starts, reports `Everything is ready. Begin running
and processing data.`, and sits at about 2m CPU and 22Mi with no errors and no
restarts, however wrong its endpoint is.

That silence is the trap. The moment an `otel` image arrives in an overlay whose
backend was never set, every signal is received, batched, and dropped at the last
hop, while the collector still looks healthy. The symptom is "telemetry is not
working" and nothing in the cluster points at the cause. Set the backend first,
and the two failures stay distinguishable.

One thing about the collector's ConfigMap is worth knowing: it keeps its
content-hash name suffix, unlike the seed `daemon.toml`, so applying a change to
the pipeline rolls the DaemonSet automatically. An overlay's own
`generatorOptions: {disableNameSuffixHash: true}` does not reach into a
component, so an overlay that sets one globally still gets the roll - verified by
rendering, not assumed. Adding one to the component itself would take the suffix
off, though: kustomize folds a global `true` into every generator and a local
`false` cannot override it, so the roll would stop happening and nothing would
say so.

What each part of the pipeline does is commented in the component's own file. In
short: the `otlp` receiver answers on both transports because a workload chooses
between them at run time; `memory_limiter` sheds load before the container is
killed; `batch` turns many small exports into few; the exporter is the single hop
off the node.

### 4. Deploy and confirm it works

```sh
NS=<namespace>
kubectl kustomize <your-overlay> | kubectl apply -f -
kubectl -n "$NS" rollout status daemonset/otel-collector
```

Then read the system, in this order - each step tells you which half is wrong.

**The collector is listening.** Its own log names both receivers at start:

```sh
kubectl -n "$NS" logs daemonset/otel-collector | grep -i "starting.*server"
```

**The daemon is exporting.** With an `otel` image and the collector up, nothing
appears in the daemon's log at all - a working export is silent. What appears
when it is *not* working is an SDK line naming the transport:

```sh
kubectl -n "$NS" logs deploy/adele-daemon | grep -i "ExportError\|BatchSpanProcessor\|telemetry could not"
```

**The signals arrive.** Ask the backend, not the collector's log. Send a prompt
through the daemon (see the smoke test below), then query the backend for
`service.name = "adele-daemon"`. All three signals should be there: spans for the
turn, the metrics the daemon records, and its log records at whatever `RUST_LOG`
allows.

A quiet collector log proves nothing at this step, because a collector forwarding
into a wrong endpoint is quiet until it has data and then reports its own failure.
When the backend has nothing and the daemon shows no `ExportError`, read the
collector's own counters - they separate "received nothing" from "received and
could not forward":

```sh
kubectl -n "$NS" port-forward daemonset/otel-collector 8888:8888 &
curl -s http://127.0.0.1:8888/metrics \
  | grep -E "otelcol_(receiver_accepted|exporter_sent|exporter_send_failed)"
```

**A rollout does not lose the last window.** `terminationGracePeriodSeconds` is
set explicitly on the daemon pod. Kubernetes stops a pod with `SIGTERM`, the
daemon handles it, and the telemetry guard flushes as it drops - so the final
metrics summary reaches the collector rather than dying with the pod. Delete a
pod and watch that summary arrive:

```sh
kubectl -n "$NS" delete pod -l app=adele-daemon
```

### Two things to know before turning this on

**`RUST_LOG=debug` ships conversation content to the collector.** One filter
governs the console layer and the OTLP log exporter together, deliberately, so
raising the verbosity of a pod also raises what leaves it. `info` never carries
content. Raise a single target rather than the whole process.

**An `otel` image always tries to export, and no environment variable stops it.**
`OTEL_SDK_DISABLED` and the per-signal `OTEL_TRACES_EXPORTER=none` are defined by
the OpenTelemetry specification and are not implemented in the Rust SDK, so
setting them does nothing at all. Deleting the endpoint variable does not stop it
either: the specification's default endpoint is `localhost`, so the exporter then
tries the pod's own loopback, finds nothing, and writes one line per failed batch:

```text
ERROR opentelemetry_sdk: name="BatchLogProcessor.ExportError" error="Operation
  failed: TonicLogsClient export failed with gRPC code: Unavailable: transport
  error: tcp connect error: Connection refused (os error 111)"
```

The daemon is unaffected - it starts, serves turns and writes its normal console
output throughout - but the log gains that line every few seconds. So an `otel`
image wants a reachable collector, and "turn it off" means the image, not the
environment.

### Turning it off again

**Stop exporting**: deploy an image built without `OTEL=1`. It contains no
exporter at all, so the `OTEL_*` variables in the manifest become inert and no
export is attempted. This is the only way to stop it.

**Remove the collector**: delete it and nothing breaks. Turns are served as
before and the console output and metrics summary continue; the daemon's exports
fail and are dropped, with the error line above in the log.

```sh
kubectl -n "$NS" delete daemonset otel-collector
kubectl -n "$NS" delete service otel-collector
```

**Silence the failures without changing the image**: point the endpoint at a
collector that exists. There is no quieter option, which is the reason the base
ships the collector beside the daemon rather than leaving it to the overlay.

### The shape assertions

`check-telemetry.sh` holds the above as named checks, so a refactor that breaks
one fails by name rather than silently: `otel_collector_receives_otlp_over_grpc_and_http`,
`otel_collector_pipes_all_three_signals`, `otel_collector_batches_before_it_exports`,
`otel_collector_backend_is_an_overlay_placeholder`,
`the_base_alone_carries_no_telemetry`,
`otel_collector_is_a_component_an_overlay_opts_into`,
`otel_collector_config_change_rolls_the_pods`,
`daemon_exports_to_the_node_local_collector`,
`daemon_labels_its_telemetry_with_namespace_pod_and_node`,
`daemon_has_time_to_flush_on_termination` and
`telemetry_manifests_name_no_real_environment`.

Those read the manifests. `check-telemetry-render.sh` reads what kustomize
actually renders, because a manifest can be right and a render wrong - this
change produced one of those, where a global generator option silently overrode a
local one. Its four checks - `rendered_base_carries_no_telemetry`,
`rendered_opted_in_overlay_deploys_one_collector`,
`rendered_opted_in_daemon_gets_its_telemetry_wiring` and
`rendered_overlay_backend_patch_reaches_the_collector` - need `kubectl`, so they
run in `just check-deploy` rather than in `just check`. The last two exist because
a kustomize patch that stops applying is silent: the render simply comes out
without it, and the first mechanism this README documented for naming a backend
turned out not to work at all.

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
