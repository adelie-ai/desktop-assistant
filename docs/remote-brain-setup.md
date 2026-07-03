# Running Adele's brain remotely (k8s), with a thin desktop client

This guide walks through the "split" setup end to end: **Adele's brain (the
daemon) runs on a Kubernetes cluster**, and your desktop is a **thin IO client**
that talks to it over WebSocket. Tools that must act on *your* machine still run
locally, via the client-side MCP host — so a brain in the cluster can read and
write files on your desktop.

It's the same daemon binary and the same clients you already run; only the
*placement* changes. The pieces:

```
  desktop (thin IO)                      kubernetes
  ┌─────────────────────┐                ┌──────────────────────────────┐
  │ adele (tui / --prompt)│  ws:// + JWT  │  adele-daemon  ── http ──▶ ollama│
  │  ├ client-mcp host   │◀─────────────▶│      │                        │
  │  │  (fileio, git, …) │  ClientToolCall│      └── sql ──▶ postgres(pgvector)│
  │  └ built-in tools    │                │                              │
  └─────────────────────┘                └──────────────────────────────┘
   tools here touch YOUR box              tools here touch the POD
```

Two localities, one model:
- **Server-side tools** (builtins + the MCP fleet) run *with the daemon* — in the
  pod.
- **Client-side tools** (`client-mcp.toml`) run *on the edge* — your desktop.

See also: [`deploy/k8s/README.md`](../deploy/k8s/README.md) (the raw manifest
reproduce recipe) and [`docs/client-mcp-host.md`](./client-mcp-host.md) (the
`client-mcp.toml` reference).

---

## 0. Prerequisites

- A Kubernetes cluster with a default `StorageClass` and outbound network.
- `kubectl` pointed at it; a namespace to work in (`adele-test` below).
- A container registry the cluster can pull from. Anything works; on the lab
  cluster we push to the TrueNAS registry (`truenas.lab.spadea.tech:30095`),
  which serves a Let's Encrypt cert and allows anonymous pull — so no
  `imagePullSecret` is needed. Adjust the image refs below for your registry.
- `podman` (or `docker`) to build the image.

Everything below assumes namespace `adele-test`; change to taste.

---

## 1. Build and push the daemon image

The `Dockerfile` builds a daemon-only image (non-root, `EXPOSE 11339`).

```sh
podman build -t truenas.lab.spadea.tech:30095/adele/adele-daemon:latest -f Dockerfile .
podman push  truenas.lab.spadea.tech:30095/adele/adele-daemon:latest
```

> The daemon links `libpam`, so the builder installs `libpam0g-dev` and the
> runtime installs `libpam0g`. If you fork the Dockerfile, keep those.

Point `deploy/k8s/30-daemon.yaml`'s `image:` at whatever you pushed.

---

## 2. Postgres (pgvector)

The daemon auto-runs its migrations on boot, but the migrations do **not**
`CREATE EXTENSION vector` — so use the `pgvector/pgvector` image and create the
extension via an initdb hook (`deploy/k8s/10-postgres.yaml` does this).

Two gotchas baked into that manifest:
- Use a **node-local** storageclass (`local-path`), not NFS — NFS `root_squash`
  blocks the postgres entrypoint's `chown` of `PGDATA`.
- If your cluster has more than one default storageclass, name it explicitly.

---

## 3. In-cluster inference (Ollama)

So the daemon has a reachable backend with no external dependency, run Ollama in
the cluster (`deploy/k8s/40-ollama.yaml`) and pull a small model:

```sh
kubectl apply -f deploy/k8s/40-ollama.yaml
kubectl -n adele-test rollout status deploy/ollama
kubectl -n adele-test exec deploy/ollama -- ollama pull llama3.2:1b
```

CPU-only nodes are slow, so the daemon config (§4) tunes the Ollama connection:

```toml
[connections.default]
type = "ollama"
base_url = "http://ollama:11434"   # the in-cluster Service
connect_timeout_secs = 600         # cold CPU model-load can exceed the 30s default
stream_timeout_secs  = 300         # slow inter-token gaps
keep_warm            = true        # keep the interactive model resident (no cold load per turn)
max_context_tokens   = 4096        # clamp context so prompt-eval stays cheap on CPU
```

> The default first-response stall-timeout is 30s. On slow nodes a cold
> model-load blows past it and you get `LLM error: Ollama stream stalled` — the
> four knobs above fix it. For production, swap this block for a Bedrock/OpenAI
> connection + a secret instead.

---

## 4. Config + secrets

Structural blocks (`[connections]`, `[purposes]`, `[personality]`, `[ws_auth]`)
have no env override, so `daemon.toml` ships via a **ConfigMap**
(`deploy/k8s/20-daemon-config.yaml`), mounted read-only into the pod. Transport,
DB, and login knobs are env on the Deployment.

The **secret** carries the DB password and the WS login password. Create it
imperatively so nothing lands in git:

```sh
kubectl -n adele-test create secret generic adele-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=WS_LOGIN_PASSWORD="$(openssl rand -hex 16)"
```

> The daemon's secret backend already falls through file → systemd-credentials →
> Secret Service → keyring, so a mounted k8s Secret works with **no code change**.

### Auth: how a remote client gets in

There is **no token-less WS mode** — the door always needs a bearer token. Two
ways to get one:
- **Static password (used here):** set `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME` +
  `DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD` on the daemon, and `[ws_auth] methods =
  ["password"]` in `daemon.toml`. The client does `POST /login` (HTTP Basic) and
  gets an HS256 JWT.
- **OIDC (production / multi-tenant):** point `[ws_auth.oidc]` at your issuer.

TLS is **off** here (`DESKTOP_ASSISTANT_WS_TLS=false`) because the LAN/tailnet
already provides transport encryption and it avoids shipping a self-signed CA.
**Do not** expose this to the public internet as-is — put it behind Tailscale or
an ingress that terminates TLS.

---

## 5. Deploy

```sh
kubectl apply -f deploy/k8s/00-namespace.yaml
kubectl apply -f deploy/k8s/10-postgres.yaml
kubectl apply -f deploy/k8s/40-ollama.yaml
kubectl apply -f deploy/k8s/20-daemon-config.yaml
kubectl apply -f deploy/k8s/30-daemon.yaml
kubectl -n adele-test rollout status deploy/adele-daemon
```

---

## 6. Connect a client

The daemon's WS door is a `ClusterIP` Service on `:11339`. Reach it from your
desktop with a port-forward (simplest) — or expose it on your tailnet for a
persistent address.

```sh
kubectl -n adele-test port-forward svc/adele-daemon 11339:11339 &
PW=$(kubectl -n adele-test get secret adele-secrets -o jsonpath='{.data.WS_LOGIN_PASSWORD}' | base64 -d)
```

### Interactive (the TUI)

```sh
adele --transport ws --service ws://127.0.0.1:11339/ws \
      --ws-login-username adele --ws-login-password "$PW"
```

### Headless — one-shot `--prompt`

For scripting (and CI), send a single prompt and print the reply to stdout, no
TUI:

```sh
adele --prompt "In one sentence, what is the capital of France?" \
      --ws ws://127.0.0.1:11339/ws \
      --ws-login-username adele --ws-login-password "$PW"
```

The password can also come from `DESKTOP_ASSISTANT_TUI_WS_PASSWORD` (and
username / jwt from their env vars) so it doesn't land in your shell history.

That's the split proven: your desktop is pure IO; the brain runs in k8s.

---

## 7. Wire in local tools (the client-side MCP host)

The point of a remote brain is that it can still act on *your* machine. Drop a
`~/.config/adele/client-mcp.toml` describing local MCP servers, and the client
hosts them and advertises their tools to the brain:

```toml
[[servers]]
name = "filesystem"
command = "fileio-mcp"
args = ["serve"]
namespace = "fs"          # tools exposed as fs__<tool> (avoids name collisions)

[surfaces.tui]
enabled = ["filesystem"]  # which surface hosts which servers; `default` applies to the rest
```

Now the same commands as above can *do things locally*:

```sh
adele --prompt "Write a markdown file at /tmp/states.md listing the US states and their capitals." \
      --ws ws://127.0.0.1:11339/ws --ws-login-username adele --ws-login-password "$PW"
# → the cluster brain calls the client-hosted fileio tool, which writes the file on THIS machine.
```

Which surface exposes which servers is per-client (`[surfaces.<name>]`), and the
file is **per machine** — so a laptop, a desktop, and a Raspberry Pi each give
the same remote brain a different set of local capabilities. Full schema:
[`docs/client-mcp-host.md`](./client-mcp-host.md).

---

## Troubleshooting

- **`LLM error: Ollama stream stalled`** — slow nodes vs. the 30s default. Bump
  `connect_timeout_secs` / `stream_timeout_secs`, set `keep_warm = true`, and
  clamp `max_context_tokens` (§3). Confirm the model is resident:
  `kubectl -n adele-test exec deploy/ollama -- ollama ps`.
- **`LLM error: failed to check Ollama models …`** — the daemon can't reach the
  backend. If you point at an Ollama outside the cluster, make sure it binds
  `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`), not just localhost.
- **Client build fails on `reqwest` feature `webpki-roots`** — reqwest 0.13.2+
  removed it while `mcp-client` still requests it. Pin the daemon's version:
  `cargo update -p reqwest@<0.13.x> --precise 0.13.1`.
- **Postgres `CrashLoopBackOff` with `chown … Operation not permitted`** — the
  PVC is on NFS. Use `local-path` (§2).
- **401 on connect** — no/blank bearer. Check the username/password match the
  daemon's env and `[ws_auth] methods = ["password"]` is set.
