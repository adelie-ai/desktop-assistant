# Compose reference stack

A complete, runnable reference system: the desktop-assistant **daemon** wired to
**PostgreSQL** (persistence) and **Dex** (OIDC), with the daemon's WebSocket
endpoint gated by OIDC bearer tokens. This is the "run it and see it work" stack
a newcomer starts with.

It uses `podman-compose`; the file is plain compose v3 syntax, so
`docker compose` works too (substitute the command name).

## What's here

| File | Purpose |
|------|---------|
| `podman-compose.yml` | The three services (`postgres`, `dex`, `assistant`) + optional `bridge`. |
| `dex-config.yaml` | Dev OIDC provider: one static client + one static user, password grant enabled. |
| `daemon.toml` | Daemon config: OIDC auth, an Ollama LLM connection, purposes. |
| `mcp_servers.toml` | A representative subset of the MCP fleet baked into the image. |
| `.env.example` | Template for `.env` (Postgres creds, Ollama URL, Dex client secret). |

## Prerequisites

- `podman` + `podman-compose` (or Docker + compose).
- A reachable **Ollama** with the chat model pulled:
  ```sh
  ollama pull llama3.2
  ```
  By default the assistant reaches Ollama at `http://host.containers.internal:11434`
  (the podman host alias). Adjust `OLLAMA_BASE_URL` in `.env` AND `base_url` in
  `daemon.toml` if your Ollama lives elsewhere. (Ollama is the default because it
  needs no cloud credentials. Swap in an `anthropic` / `openai` / `bedrock`
  connection later if you prefer.)
- The C-1 daemon image. The compose file builds it from the repo-root
  `Containerfile` on first `up --build`. (First build is slow — it compiles the
  daemon and installs the full MCP fleet.)
- `wscat` for the WebSocket test: `npm i -g wscat`. `curl` and `jq` for the token.

## Issuer / hostname (the one subtle bit)

OIDC tokens carry an `iss` claim equal to Dex's `issuer`. The daemon's validator
pins acceptance to its configured `issuer_url` (exact match) and fetches
discovery + JWKS from it. Crucially, the validator **only accepts an `https://`
issuer or an explicit loopback `http://`** (localhost / 127.0.0.1 / ::1). A
plaintext `http://dex:5556/...` issuer is **rejected** because `dex` is not
loopback.

So the issuer must be `http://localhost:5556/dex`. But inside the daemon
container, `localhost` would normally be the daemon itself, not Dex. This stack
resolves that by running the `assistant` service in **Dex's network namespace**
(`network_mode: "service:dex"`). Then:

- inside the daemon container, `localhost:5556` → Dex (token validation works);
- on your host, `localhost:5556` → Dex (published by the `dex` service);
- the daemon's published WebSocket port `11339` rides the same namespace.

One issuer URL, agreed on by daemon, host, and wscat.

**Tradeoff vs. the alias approach.** You could instead keep the assistant on its
own network and add a host alias so `localhost` (or a chosen name) maps to Dex —
e.g. `extra_hosts`/an alias plus matching issuer. That keeps services isolated
but is fiddlier to get byte-identical across daemon/host/client, and `localhost`
specifically can't be aliased to another host cleanly. The shared-namespace
approach is the simplest thing that makes a single issuer URL valid everywhere,
so that's what's wired here. The cost: the assistant has no `ports:` of its own
(11339 is published on `dex`) and reaches Postgres at `localhost:5432` rather
than `postgres:5432` — both already set up in the compose file.

## 1. Bring it up

```sh
cp .env.example .env
podman-compose up --build
```

Wait until `postgres` and `dex` report healthy and the assistant logs that it is
listening on `0.0.0.0:11339`.

## 2. Verify Dex discovery

```sh
curl -s http://localhost:5556/dex/.well-known/openid-configuration | jq .
```

You should see `issuer: "http://localhost:5556/dex"` and the `authorization_endpoint`,
`token_endpoint`, and `jwks_uri` under `…/dex/…`.

## 3. Obtain a token (password grant, non-interactive)

The dev user is `dev@example.com` / `password` (from `dex-config.yaml`), the
client is `desktop-assistant` with the secret from `.env`.

```sh
TOKEN=$(curl -s http://localhost:5556/dex/token \
  -d grant_type=password \
  -d client_id=desktop-assistant \
  -d client_secret=desktop-assistant-dev-secret \
  -d username=dev@example.com \
  -d password=password \
  -d scope=openid \
  | jq -r .id_token)
echo "$TOKEN"
```

Use the **`id_token`** (a signed JWT with `aud=desktop-assistant`), not the
opaque `access_token`. Decoding it (e.g. at jwt.io) should show
`iss=http://localhost:5556/dex` and `aud=desktop-assistant` — exactly what the
daemon validates.

## 4. Connect over WebSocket

```sh
wscat -c ws://localhost:11339/ws -H "Authorization: Bearer $TOKEN"
```

Every frame is a JSON envelope `{"id": "<correlation-id>", "command": {...}}`
(see `docs/WEBSOCKET_API.md`). Paste these one line at a time into the wscat
prompt.

A quick liveness check:

```json
{"id":"1","command":{"ping":{}}}
```

→ a `result` frame with a `pong`. Now create a conversation and send a message:

```json
{"id":"2","command":{"create_conversation":{"title":"compose smoke test"}}}
```

→ a `result` frame whose payload is `{"conversation_id":"<cid>"}`. Use that id:

```json
{"id":"3","command":{"send_message":{"conversation_id":"<cid>","content":"Hello from the compose stack"}}}
```

You should get streamed `assistant_delta` event frames followed by
`assistant_completed` (the reply served by your Ollama `llama3.2`).

### Bad / absent token → 401

```sh
# No header:
wscat -c ws://localhost:11339/ws
# Garbage token:
wscat -c ws://localhost:11339/ws -H "Authorization: Bearer not-a-real-token"
```

Both are rejected with HTTP `401 Unauthorized` during the WebSocket upgrade —
the daemon validates the bearer token (signature via Dex's JWKS, `iss`, `aud`,
`exp`) before accepting the connection.

## 5. Persistence check

Conversations live in Postgres, not in the assistant container.

1. Send at least one message (step 4) so a conversation exists.
2. Restart just the assistant:
   ```sh
   podman-compose restart assistant
   ```
3. Reconnect (steps 3–4) and list/open conversations — your history is still
   there, because it was read back from the `postgres` service (whose data sits
   on the `pgdata` named volume).

To prove it's really in Postgres:

```sh
podman-compose exec postgres \
  psql -U assistant -d desktop_assistant -c '\dt'
```

You'll see the daemon's tables (conversations, messages, etc.).

## Optional: the D-Bus bridge

```sh
podman-compose --profile dbus up
```

This starts the `bridge` service (same image, `bridge` role) sharing the
daemon's local UDS via a named volume. It is here to **document** how the bridge
attaches to the daemon — a full session-bus deployment where the bridge owns
`org.desktopAssistant` on a real D-Bus is the **quadlet pod (issue C-7)**, not
this compose stack.

## Tearing down

```sh
podman-compose down          # keep data
podman-compose down -v       # also drop the pgdata volume (wipes history)
```

## Going beyond dev

- **TLS:** this stack sets `DESKTOP_ASSISTANT_WS_TLS=false` for plain `ws://`.
  For real use, terminate TLS (real cert) and drop that override; the issuer
  should then be `https://…`.
- **LLM:** replace the `[connections.default]` Ollama block in `daemon.toml`
  with a cloud connector (`anthropic` / `openai` / `bedrock`) and supply its
  credentials (see `docs/connectors/` and `docs/cloud-providers.md`).
- **Secrets:** `.env` here holds dev-grade secrets. Use real secret management
  for anything else, and rotate the Dex client secret + user password.
