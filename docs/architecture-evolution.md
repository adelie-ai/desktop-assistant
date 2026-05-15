# Architecture Evolution

`architecture.md` describes the current crate layout. This document describes
where the system is heading and the design rules that get us there. Decided
in conversation; not yet implemented (see the task list for near-term work).

## Target shape

A **server-first, multi-tenant assistant** deployable across:

- Bare process / systemd user service — single-tenant desktop dev. Current
  shape; kept indefinitely as the development driver.
- Kubernetes pod — multi-tenant production with persistent connections.
- Knative service / serverless container — multi-tenant production, scales
  to zero, persistent-ish connections.
- AWS Lambda (request/response) — multi-tenant production.
- AWS Lambda + API Gateway WebSocket — streaming production.

The "desktop assistant" framing remains the day-to-day development driver and
shapes the UX. The deployment story is server-first: every architectural
decision has to survive Lambda, and the desktop case falls out as the trivial
single-tenant deploy of the same server.

## Design rules

1. **Stateless request handling.** Every request loads its dependent state
   from DB or object store and writes back. No daemon-resident state that's
   load-bearing for correctness. Warm-process caches are valid as opportunistic
   optimization only. Forced by Lambda; healthy for k8s scaling.

2. **JWT-only auth.** The daemon validates JWTs and extracts `sub` for
   `user_id`. It trusts a configured set of issuers' JWKS. Production uses an
   external IdP (Cognito, Authentik, Keycloak, etc.); desktop installs use a
   local JWT minter that authenticates the OS user via `SO_PEERCRED`. No
   peer-cred or session-based auth in the daemon's request path.

3. **Shared Postgres, user_id-scoped.** All personal-data tables carry
   `user_id`; queries are always scoped. Single-tenant desktop installs
   collapse to a fixed default user_id.

4. **Single UID daemon.** Daemon runs as one process user (container UID, or
   the desktop user). No `setresuid`, no MCP-as-target-user, no `CAP_SETUID`.
   Multi-tenancy is enforced in code, not by the kernel.

5. **Pluggable transport.** The core handler logic (`AssistantApiHandler`,
   `EventSink`) is transport-agnostic. Implementations: raw WebSocket
   (k8s/desktop), API Gateway WebSocket via `postToConnection` (Lambda), UDS
   (local clients without JWTs).

6. **Background tasks externalized.** Embedding backfill, dreaming,
   summarization, etc., run as separate workers — EventBridge cron (Lambda),
   CronJob (k8s), or in-process tokio task (desktop bare-process). Same
   handler functions, different invocation models.

7. **MCPs are HTTP/SSE primary.** Stdio MCPs survive only as a single-tenant
   desktop convenience; multi-tenant servers and Lambda can't safely run them
   (shared UID = no isolation; no long-lived child processes on Lambda).

8. **Client-side execution for client-local MCPs.** Client-local MCPs (file
   access, personal info, terminal, the user's own laptop tooling) execute on
   the user's machine, not on the server. The daemon emits a
   `client_tool_call` event; the chat client executes locally and posts the
   result back. Conversation turn state is persisted to DB at each step so it
   can suspend on a client tool call and resume on response — this is also
   the persistence shape Lambda needs.

## Phased evolution

### Phase 0 — Foundations (in progress)

- Shared `auth-jwt` crate (claim shape, encode/decode, key file IO).
- Local JWT minter on UDS (desktop dev convenience; group-gated for
  multi-user hosts).
- Multi-tenant DB schema (add `user_id` to personal-data tables).

### Phase 1 — Multi-tenant single-deploy

- Daemon validates JWT on every request, extracts `user_id`.
- Personal-data queries scoped to `user_id`.
- D-Bus interface becomes a separate per-user binary that talks HTTP/WS +
  JWT to the daemon. Daemon drops the `dbus-interface` dep.
- HTTP/SSE MCP support in `mcp-client`.
- Stdio MCPs marked single-tenant-only.

### Phase 2 — Stateless turn execution

- Conversation turn becomes a DB-persisted state machine.
- Turn states: `pending_llm`, `pending_tool_dispatch`,
  `pending_client_tool`, `complete` (sketch — actual schema TBD).
- Background workers (Lambda or in-process) drive transitions.
- `client_tool_call` event + client-side execution protocol on the WS API.

### Phase 3 — Lambda deployment

- API Gateway WebSocket integration (`postToConnection` event sink).
- EventBridge-driven background workers.
- Cold-start optimization (lazy DB pool, connection reuse).
- Cognito (or equivalent) as JWT issuer; local minter not used in this
  deployment.

### Phase 4 — Knative / serverless container

- Same code as Phase 3, but persistent connections; in-process tokio task
  path is the runtime, not API Gateway.
- Auto-scale-to-zero forces the same statelessness as Phase 3.

## Open questions (deferred)

1. **Conversation turn state schema.** What columns capture the LLM-loop
   state cleanly? How are tool results threaded back to a suspended turn?
   How do we bound DB churn for chatty turns?
2. **Client capability advertisement.** How does the daemon learn which
   client-local MCPs the connected client has? WS handshake? Registration
   RPC? A claim in the JWT? Affects tool-list returned to the LLM.
3. **MCP registration / discovery.** For HTTP/SSE MCPs in multi-tenant
   servers — do users self-register endpoints, or does the admin? How are
   credentials passed?
4. **Streaming on Lambda.** API Gateway WebSocket has per-frame invocations
   and `postToConnection` for outbound. The `EventSink` impl needs to
   accommodate both that and direct WS without leaking the model into core.
5. **Background task triggers.** When does an embedding backfill run on
   Lambda? Scheduled cron, post-conversation event, some other trigger?
6. **Credential storage in production.** Per-user LLM API keys: stored in
   DB encrypted with a system key, or stored unencrypted in a backend that's
   already encrypted at rest (Vault, AWS Secrets Manager, k8s Secret)? Pick
   when we have a real deployment target.

## Decisions explicitly *not* taken

- **systemd-creds for at-rest encryption.** Too systemd-specific; fails
  Lambda and Knative. Bare-metal installs that want at-rest encryption can
  opt into it as a deployment-time wrapping concern, not a code dependency.
- **Per-user filesystem isolation enforced by kernel UIDs.** Same reason —
  doesn't survive container deployments.
- **Per-user database connections.** Unnecessary at the scales we care
  about; user_id scoping in code is the standard SaaS pattern and works at
  both desktop and multi-tenant scales.
- **Custom OIDC IdP.** Too much engineering. Use Cognito / Authentik /
  Keycloak in production. The local JWT minter is a desktop dev convenience,
  not a real IdP.
- **Peer-cred auth in the daemon's request path.** UDS endpoints stay
  available for local clients but auth is still JWT (issued by the local
  minter). Keeps the auth path uniform across deployments.
