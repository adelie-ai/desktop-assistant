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
and reconnect — conversation history persists (it's in Postgres, not the pod).
