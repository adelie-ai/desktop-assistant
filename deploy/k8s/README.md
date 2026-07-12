# Prove-the-split: Adele daemon on k8s, desktop as WS IO client

A minimal reference deployment that runs the `desktop-assistant` daemon (the
"brain") in a pod with its own pgvector Postgres, and drives it from a desktop
client over the WebSocket transport. It proves the **remote-brain split** —
transport, `/login` auth, and DB persistence — without OIDC, TLS, or the MCP
tool fleet (all deliberately out of scope; separate projects).

## What this is / isn't

- **Auth:** interim static password (`/login` → HS256 token). OIDC is the
  later, multi-tenant path (epic C-4/C-5).
- **TLS:** off. The LAN/tailnet provides transport encryption, so we skip
  self-signed-cert CA distribution. Not for the public internet.
- **Tools:** none. The image is daemon-only; the MCP fleet is a separate image
  (epic C-1). This proves conversation + persistence, not tool use.
- **Inference:** an in-cluster Ollama pod (`40-ollama.yaml`) with a small CPU
  model (`llama3.2:1b`), so the smoke test gets a real reply with no external
  host or cloud creds. Swap to Bedrock/OpenAI by editing `20-daemon-config.yaml`
  (`[connections]`/`[purposes]`) and adding the credential to the
  `adele-secrets` Secret.
- **Config persistence:** a `local-path` PVC (`adele-daemon-state` in
  `30-daemon.yaml`) is mounted at `/state`, and `XDG_CONFIG_HOME`/`XDG_DATA_HOME`
  point under it, so everything the daemon persists — `daemon.toml`,
  `mcp_servers.toml`, service accounts, the per-connection secret files set from
  a client (`SetConnectionSecret`, #484), and the system-id — survives restarts.
  `20-daemon-config.yaml` is now a **seed**: an init container copies its
  `daemon.toml` onto a *fresh* volume (non-clobbering) and chowns the volume to
  the daemon uid. After the first boot the on-volume config wins — see "Changing
  the baseline config" below.

## Deploy

```sh
# 1. Build the daemon-only image (from the repo root)
podman build -t localhost/adele-daemon:prove-split -f Dockerfile .

# 2. Push to a registry your cluster can pull from. This example uses one that
#    serves a trusted cert and allows anonymous pull, so no imagePullSecret is
#    needed; replace registry.example.com:5000 with your own.
podman tag localhost/adele-daemon:prove-split \
  registry.example.com:5000/adele/adele-daemon:prove-split
podman push registry.example.com:5000/adele/adele-daemon:prove-split

# 3. Create the Secret (random passwords; creds never committed)
kubectl -n adele-test create secret generic adele-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=WS_LOGIN_PASSWORD="$(openssl rand -hex 16)"

# 4. Apply (Ollama first so the daemon has a backend; pull the model once it's up)
kubectl apply -f deploy/k8s/00-namespace.yaml
kubectl apply -f deploy/k8s/40-ollama.yaml
kubectl -n adele-test rollout status deploy/ollama
kubectl -n adele-test exec deploy/ollama -- ollama pull llama3.2:1b
kubectl apply -f deploy/k8s/10-postgres.yaml
kubectl apply -f deploy/k8s/20-daemon-config.yaml
kubectl apply -f deploy/k8s/30-daemon.yaml
```

## Smoke test

```sh
# Reach the WS door from the desktop
kubectl -n adele-test port-forward svc/adele-daemon 11339:11339 &

# Grab the login password
PW=$(kubectl -n adele-test get secret adele-secrets \
       -o jsonpath='{.data.WS_LOGIN_PASSWORD}' | base64 -d)

# Connect the desktop TUI to the remote brain and send a prompt
adele-tui --transport ws --service ws://127.0.0.1:11339/ws \
  --ws-login-username adele --ws-login-password "$PW"
```

Expected: a real reply. Then `kubectl -n adele-test rollout restart deploy/adele-daemon`
and reconnect — conversation history persists (it's in Postgres, not the pod),
and so does anything you changed via the settings API (it's on the state PVC).

## Changing the baseline config

Because `20-daemon-config.yaml` only *seeds* a fresh volume, editing the
ConfigMap and re-applying does **not** change a daemon that already has a
`daemon.toml` on its PVC. To reset the baseline, either edit the live file in
place:

```sh
kubectl -n adele-test exec deploy/adele-daemon -- \
  sh -c 'cat > /state/config/desktop-assistant/daemon.toml' < my-daemon.toml
kubectl -n adele-test rollout restart deploy/adele-daemon
```

or wipe the volume to re-seed from the (edited) ConfigMap — this also drops any
client-set credentials on the PVC:

```sh
kubectl -n adele-test scale deploy/adele-daemon --replicas=0
kubectl -n adele-test delete pvc adele-daemon-state   # recreated on next apply
kubectl apply -f deploy/k8s/30-daemon.yaml
```
