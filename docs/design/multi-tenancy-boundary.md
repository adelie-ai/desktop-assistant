# Multi-tenancy boundary: config, credentials, and tool execution

Status: proposed. Epic: #680. Related: #531 (runner axis), #538 (built-in MCP in
clients), #490 (composable image), #260 (client-tool session scoping), #583
(handler lifecycle), #413 (scheduled routines)

## Problem

The daemon began as a single-user desktop application, and tenancy was never
designed into it because the operating system supplied it. One home directory
held the config. One keyring held the credentials. File and shell tools ran as
the user's uid, against the user's filesystem. The kernel enforced every one of
those boundaries, for free, and correctly.

Running the same daemon as a shared service removes the enforcer but not the
assumptions that depended on it. Nothing in the code was wrong when it was
written; it is wrong now only because the deployment changed underneath it.

The data layer already made this transition. Conversations, knowledge, scratchpads,
turn state, and background tasks are `user_id`-scoped, with a Postgres RLS backstop
(migration 029) as defense in depth, and per-turn identity is available as a
`current_user_id()` task-local. The pattern is proven and in production. It simply
was never extended past storage.

Three areas still assume a single human:

**Configuration.** No method on `SettingsService` (`crates/core/src/ports/inbound.rs`)
accepts a caller identity. `set_llm_settings`, `set_embeddings_settings`,
`set_api_key`, and `set_persistence_settings` all mutate one global `daemon.toml`.
Any client that can reach the daemon reconfigures it for every tenant. This is an
authorization gap, not only a modeling one.

**Credentials.** Secrets resolve as `(service, account)` where the account derives
from the connector name, so one provider key serves the entire instance. Every
tenant's spend lands on it, and any client can rotate it away from everyone else.

**Tool execution.** `terminal`, `command`, `fileio`, and `skills` run inside the
daemon process, as one uid, against one filesystem. On the desktop that is exactly
right - the sandbox is the user's own machine. In a shared pod nothing scopes it.
`web-mcp` is a subtler case of the same thing: it holds no state, so it passes a
naive check, but it browses from the pod's network position, which in-cluster
reaches the database, the model server, the k8s API, and cloud metadata endpoints.

Centrally-hosted stateful servers sit alongside these. `timeclock-mcp` writes JSONL
under a single `data_dir()` with no user scoping, and `homeassistant-mcp` carries
per-user credentials for one specific home.

## Constraint

Single-user desktop usage stays first-class. It is not a compatibility mode or a
degraded path, and it must not acquire new concepts to learn, new configuration to
set, or new UI to navigate as a result of anything here.

This is enforced mechanically rather than by intention: every change lands with a
named test asserting that when no per-user row exists, resolution returns exactly
the `daemon.toml` value. A change that cannot satisfy that test is the wrong change.

## Decisions

### 1. Per-user settings are an override layer, not a replacement

`daemon.toml` remains the source of truth and becomes the *default* layer.
Resolution runs conversation override -> user setting -> file default.

A single-user daemon never writes a user-level row, so every lookup falls through
to the file and behaves as it does today. No migration, no new rows, nothing to
explain.

This is not a new pattern. `skill_index.owner_user_id` is NULL for global with a
generated `owner_key` mirror so one unique index covers both cases;
`conversations.personality_override` is a partial per-trait override falling back to
the global on every send; `last_model_selection` is per-conversation. Three call
sites already prove the shape.

### 2. Identity comes from the transport, not from a new concept

Peer-cred supplies the uid on UDS and D-Bus; JWT supplies the subject remotely; the
daemon already exposes `current_user_id()` per turn. `SettingsService` is the only
layer that discards it. Threading it through is plumbing an existing value one layer
further, which costs the desktop case nothing.

### 3. Ship every MCP server; let the operator hard-disallow

Rather than deciding at packaging time which servers are safe, ship them all and
give the operator an admin-level denylist.

A self-hoster running the container for themselves keeps full capability, including
on clients that cannot host tools in-process. A hosted operator locks the instance
down to what they are willing to stand behind. The project does not make that call
on either party's behalf, and there is no image variant matrix to deploy wrong.

An empty denylist is the default, so the desktop case never learns the feature
exists.

### 4. The denylist unit is (server, runner), not server

Blocking a server by name alone is cosmetic: an operator denies daemon-side `fileio`
and a client registers its own `fileio` as a client tool, restoring the capability
through the other door.

Both answers are also legitimately wanted at once. On a hosted instance, client-side
`fileio` is fine - it runs on the user's own machine, where the kernel still
enforces isolation - while daemon-side `fileio` is not. Policy must be able to say
so. This is the runner axis from #531 carrying real weight rather than labeling UI
chips.

### 5. Enforce at spawn and advertisement, never at call time

A denied server does not start, and its tools never reach the tool index or the
model's tool list. Blocking at invocation is too late: the model has already planned
around a capability it cannot use, and the observed failure mode there is
confabulation rather than a clean refusal.

Enforcement therefore has to reach `tool_definitions`, or tool search will surface
rows for tools that cannot be called.

A denied server renders as denied-by-policy in the MCP panel rather than silently
vanishing, consistent with the honest-state design already used there and with the
project's capability-degradation rule: surface *why* something is off.

### 6. Operator config and tenant config are different things

`[database]`, `[tls]`, `[ws_auth]`, `[connections]` and their credentials,
`[backend_tasks]`, and `[profiling]` describe how the service runs. A tenant should
neither set nor see them. These do not become per-user; they become admin-gated.

Preferences - model choice per purpose, personality, speech mode - become per-user
through decision 1.

The denylist depends on this split. Without an admin/tenant distinction, an "admin
denylist" is a global setting any tenant can switch off.

## Consequences

Most of the hot-apply problem dissolves. Settings resolved per turn from the
database need no file watcher and no process-wide client swap, the way
`resolve_turn()` already works for the LLM. What still needs a genuine hot-swap is
only what stays global: TLS certificates, `ws_auth`, and the host-global tool and
skill index embedder.

That matters because restart-as-reconfiguration is not viable for a shared instance.
Prod runs one replica with `strategy=Recreate`, shutdown does not drain in-flight
turns, and an abandoned turn is a known data-loss path (#583). Reconfiguring should
never cost every tenant their connection.

The epic shrinks. Sandboxing server-side file and shell access disappears as a work
item, replaced by a policy list. The MCP OAuth ownership problem mostly follows the
per-user servers that carry the tokens. Skill scanning's user-roots path simply does
not run where it does not apply.

`[purposes.embedding]` turns out to be two settings under one name. A per-user
knowledge-base embedder is viable, because `knowledge_base` is user-scoped and its
rows already carry an `embedding_model` stamp. The tool and skill index embedder
cannot be, because `tool_definitions` and `skill_index` are deliberately host-global.
Splitting them is a prerequisite for making either one configurable per user.

## Open questions

**Credential model.** Per-tenant BYO keys, or operator-pays with quota? Both are
defensible and the choice determines the schema.

**Runner for clients that cannot host tools.** The wasm web client and the voice
client cannot host in-process `fileio` or `terminal`. Where an operator denies
daemon-side file access, those surfaces have no runner at all. Candidates: borrow a
runner from the same user's other registered client, add a per-user server-side
root, or accept the gap. Borrowing needs tool scoping keyed to (user, host) rather
than (user, session), and #260 deliberately made registration session-scoped to stop
the voice `say_this` leak - so the distinction to draw is between session-affine
capabilities, which stay scoped, and host-affine ones, which belong to a machine.

**Centrally-hosted stateful servers.** `timeclock`, `tasks`, and `homeassistant`
genuinely want to stay central: a single-user self-hoster reaches the same data from
the TUI, the GTK client, or a phone, and a phone cannot run them at all. Scope
inside each server, or give each a per-user data root?

**Do embeddings go per-user at all,** or does the operator choose the embedder for
everyone?

**Headless work has no client runner.** Dreaming and consolidation do not need one.
Scheduled routines (#413) plausibly will.

## Rejected alternatives

**Build-time image variants** (a desktop fleet and a stripped shared fleet). Makes
the policy call at packaging time on behalf of an operator who might reasonably
decide otherwise, and removes capability from single-user self-hosters who had no
reason to lose it. The denylist achieves the same safety without either cost.

**Per-tenant sandboxing for server-side file and shell tools.** Substantial
machinery to make something safe that the client already does correctly, using a
kernel that is already enforcing it on the user's own machine.

**Leaving reconfiguration to a restart.** Acceptable when the only person affected
is the one who made the change. Not acceptable when it disconnects every tenant and
can lose whoever was mid-turn.

**A general multi-user mode flag.** Conditioning behavior throughout the code on a
deployment mode reintroduces exactly the complexity this design is trying to keep out
of the single-user path. The resolution chain needs no flag, and policy differences
are expressed as data in the denylist rather than as branches in the code.
