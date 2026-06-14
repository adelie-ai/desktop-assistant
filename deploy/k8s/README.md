# Kubernetes deployment

Run the desktop-assistant **daemon** (the C-1 image) in Kubernetes: WebSocket
front door gated by **OIDC RS256** bearer tokens, **Postgres**-backed, **no
D-Bus / no UDS**. TLS is terminated at the Ingress.

| File | Purpose |
|------|---------|
| `deployment.yaml` | The daemon (`replicas: 1`), initContainer config copy, env, TCP probes. |
| `service.yaml` | ClusterIP on `11339`. |
| `ingress.yaml` | nginx Ingress: WS upgrade + long timeouts, terminates TLS. |
| `configmap.yaml` | `daemon.toml` (OIDC + LLM + purposes) and `mcp_servers.toml`. |
| `secret.example.yaml` | Placeholder DB URL + LLM creds — copy to `secret.yaml`. |
| `kustomization.yaml` | Ties the resources together. |

## Prerequisites

- A Kubernetes cluster with the **nginx ingress controller** installed.
- **Postgres** the daemon can reach (in-cluster Service or a managed instance —
  see "Postgres options" below). The daemon creates its own schema on startup.
- An **OIDC IdP** that issues **RS256** JWTs (Cognito / Okta / Auth0 / Dex /
  Keycloak). HMAC-signed tokens are rejected.
- An LLM backend: either an in-cluster/remote **Ollama** (default, no creds) or
  a cloud provider whose credentials you put in the Secret.
- A **TLS cert** Secret named `assistant-tls` for the ingress host (provision it
  yourself or with cert-manager).
- The **C-1 image** built and pushed to a registry your cluster can pull.

## Build & push the image

The image is built from the repo-root `Containerfile` (issue C-1):

```sh
# from the repo root
podman build -t registry.example.com/adelie/desktop-assistant:<tag> -f Containerfile .
podman push   registry.example.com/adelie/desktop-assistant:<tag>
```

Then set that reference as the `image:` in `deployment.yaml` (both the
initContainer and the daemon container use it). Pin a tag or digest in
production rather than `:latest`.

## Configure

1. **OIDC** — edit `[ws_auth.oidc]` in `configmap.yaml`. The placeholder is
   `https://idp.example.com`. Swap in your IdP (see "Swapping the OIDC issuer").
2. **LLM** — `configmap.yaml` ships a credential-free Ollama connection at
   `http://ollama.default.svc.cluster.local:11434` using model `llama3.2`. Point
   `base_url` at your Ollama Service and make sure the model is pulled, OR
   replace `[connections.default]` with an `anthropic` / `openai` / `bedrock`
   block (commented examples are in the ConfigMap) and update
   `[purposes.interactive].model`.
3. **Secret** — `cp secret.example.yaml secret.yaml`, fill in the real
   `DESKTOP_ASSISTANT_DATABASE_URL` and the credential for your chosen LLM, then
   keep `secret.yaml` out of git (`.gitignore` already excludes it).
4. **Ingress host / TLS** — set the real host in `ingress.yaml` and provision the
   `assistant-tls` Secret.

## Apply

```sh
kubectl apply -k deploy/k8s/
```

`kustomization.yaml` applies, in order: ConfigMap, Secret (`secret.yaml`),
Deployment, Service, Ingress. Watch it come up:

```sh
kubectl rollout status deploy/desktop-assistant
kubectl logs deploy/desktop-assistant -c daemon -f   # expect "listening on 0.0.0.0:11339"
```

## Swapping the OIDC issuer to a real IdP

The placeholder `[ws_auth.oidc]` points at `https://idp.example.com`. Replace
all four URLs with your provider's values (most expose them at
`<issuer>/.well-known/openid-configuration`):

| Provider | `issuer_url` |
|----------|--------------|
| Cognito | `https://cognito-idp.<region>.amazonaws.com/<user-pool-id>` |
| Okta | `https://<org>.okta.com` (or `.../oauth2/<server-id>`) |
| Auth0 | `https://<tenant>.us.auth0.com/` |
| Keycloak | `https://<host>/realms/<realm>` |

Then set `client_id` and — critically — `audience` to the value your IdP stamps
into the token's `aud` claim (Cognito/Dex: the client id; Auth0: the API
identifier). **`audience` must be non-empty** or the daemon refuses to start. A
real `https://` issuer has no loopback restriction.

## Obtaining a Bearer token

Depends on your IdP; two common shapes:

```sh
# Auth0 / generic OIDC client-credentials (machine-to-machine):
TOKEN=$(curl -s https://<tenant>.us.auth0.com/oauth/token \
  -H 'content-type: application/json' \
  -d '{"client_id":"<id>","client_secret":"<secret>","audience":"desktop-assistant","grant_type":"client_credentials"}' \
  | jq -r .access_token)

# Cognito (user pool, password grant via the token endpoint or AWS CLI):
TOKEN=$(aws cognito-idp initiate-auth \
  --auth-flow USER_PASSWORD_AUTH \
  --client-id <client-id> \
  --auth-parameters USERNAME=<user>,PASSWORD=<pass> \
  --query 'AuthenticationResult.IdToken' --output text)
```

Decode the JWT (e.g. jwt.io) and confirm `iss` and `aud` match your config.

## Test with port-forward + wscat

```sh
kubectl port-forward svc/desktop-assistant 11339:11339

# in another shell:
wscat -c ws://localhost:11339/ws -H "Authorization: Bearer $TOKEN"
```

Frames are JSON envelopes `{"id":"<cid>","command":{...}}` (see
`docs/WEBSOCKET_API.md`):

```json
{"id":"1","command":{"ping":{}}}
{"id":"2","command":{"create_conversation":{"title":"k8s smoke test"}}}
{"id":"3","command":{"send_message":{"conversation_id":"<cid>","content":"Hello from k8s"}}}
```

No/invalid token is rejected with `401` during the WebSocket upgrade. (To test
through the real ingress with TLS, use `wss://assistant.example.com/ws`.)

## Postgres options

- **In-cluster Postgres** — quickest: deploy a Postgres StatefulSet + Service and
  point `DESKTOP_ASSISTANT_DATABASE_URL` at it. You own backups, upgrades, and
  storage. Fine for dev/single-tenant; data lives on a PVC.
- **Managed Postgres** (RDS / Cloud SQL / Neon / etc.) — recommended for
  anything real: durability, backups, and HA are handled for you, and the daemon
  is then truly stateless apart from in-memory live conversation state. Just put
  the managed connection string in the Secret. **No manifest here provisions
  Postgres** — that is intentionally out of scope for this issue.

## Security note: `terminal-mcp`

`terminal-mcp` gives the model a **shell inside the daemon pod**. It is left OUT
of the default `mcp_servers.toml` here. In a single-user desktop deployment that
shell is the user's own machine; in a **server-side / shared / multi-tenant**
cluster it is a meaningful escalation surface (a prompt can run arbitrary
commands in the pod, read mounted secrets, and reach anything the pod's network
policy and ServiceAccount allow). Only add `terminal-mcp` if you accept that, and
if so harden the pod first: drop the ServiceAccount token, a restrictive
NetworkPolicy, read-only root filesystem, non-root user (the image already runs
as `assistant`), and seccomp. The same caution applies to `fileio-mcp` for paths
outside the pod's ephemeral storage.

## Why `replicas: 1`

Live conversation/turn state is held per-daemon in memory and WebSocket sessions
are sticky to the pod that accepted them. More than one replica would split
clients across daemons that don't share that state. Scale by running independent
daemons, not by raising the replica count; the Deployment uses the `Recreate`
strategy so a rollout never briefly runs two.
