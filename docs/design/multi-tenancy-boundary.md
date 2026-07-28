# Multi-tenancy boundary: config, credentials, and tool execution

Status: accepted. Epic: #680. Work: #689 (operator/tenant split), #690 (MCP denylist), #692 (per-user data root), #693 (daemon linking, deferred), #686 (restart-required reporting). Related: #531 (runner axis), #538 (built-in MCP in
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
from the connector name, so one provider key serves the entire instance. Decision 8
accepts the sharing: users in one organization share the organization's account. The
defect is narrower. Any client could rotate that key away from everyone else, because
`set_api_key` took no caller identity. That is the operator/tenant gap above, not a
missing per-tenant credential store. Closed by #728: the write now needs the
administrator capability, granted by the local peer uid or by
`[authz] admin_subjects`.

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

**As built (#728).** The capability is resolved at the transport and enforced by one
gate in `dispatch_loop`, driven by an exhaustive `required_capability(&Command)` match
with no wildcard arm. Two grants: a Unix-socket peer whose kernel-attested uid equals
the daemon's own, and a subject named in the file-only `[authz] admin_subjects`. The
desktop needs no configuration, which is this document's Constraint.

The split landed as *writes* admin, *reads* tenant. Reading the operator sections is
still open, for two reasons: the credentials they used to carry are redacted on the way
out (#727), and the same values reach every client through `GetConfig`, which every
settings panel and the personality surface already read. Gating the reads coherently
means partitioning `Config` itself, which is decision 1's per-user override layer.
Tracked separately rather than half-done here.

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

`[purposes.embedding]` describes two things under one name. A per-user knowledge-base
embedder is technically viable, because `knowledge_base` is user-scoped and its rows
already carry an `embedding_model` stamp. The tool and skill index embedder is not,
because `tool_definitions` and `skill_index` are deliberately host-global.

Decision 9 makes the split unnecessary for now: one embedder serves the instance. The
observation still matters in one place. A force rebuild must re-embed the rows a user
owns without rebuilding the host-global indexes, because those affect every user and
cost the operator money.

### 7. The tenant boundary is intra-organization, not hostile isolation

Multiple users share one instance. They belong to one organization. They are not
adversaries.

This sets the bar for every other decision here. The goal is that one user does not
see or disturb another user's data and settings. The goal is not to defend against a
tenant who attacks the instance. When two groups need real separation, run a second
instance in Kubernetes. That is cheaper and stronger than building hostile isolation
into one process.

### 8. Connectors and credentials stay global

One provider credential serves the instance. The operator owns it and pays for it.

Per-tenant credentials would change the connection registry, the secret store, the
purpose resolver, and every call site that resolves a client. That is a large
architectural change. Decision 7 makes it unnecessary: users in one organization can
share the organization's provider account. Revisit this if a real per-user billing or
per-user provider-account requirement appears.

### 9. Purposes stay global

The model for each purpose - interactive, dreaming, embedding, and the rest - is one
setting for the instance. The operator sets it.

Per-user purposes would need a resolution chain on the hot path of every turn, and
per-user embedders would split the vector space. The cost is high and the benefit is
small inside one organization. Personality and speech mode are still good candidates
for the per-user override layer in decision 1, because they are cheap and personal.

### 10. Daemon-hosted stateful servers get a per-user data root

`timeclock`, `tasks`, and `homeassistant` stay in the daemon. Each user gets a
private data directory. The server is told which root to use for the current user.

This keeps the property that makes central hosting worth it: one dataset, reachable
from the TUI, the GTK client, or a phone. It also gives `web-mcp` somewhere to put
per-user Chromium profile data, which is per-user state that is shared today.

## Open questions

**Headless work has no client runner.** Dreaming and consolidation do not need one -
they only read and write the database. Scheduled routines (#413) are different. A
routine runs a full agent turn on a timer with no client attached. If file and shell
tools only run client-side, a routine cannot use them. Decision 10 suggests an
answer: give routines the same per-user data root. That needs a decision before
routines ship.

**Runner push-down to a desktop client** is deferred, not rejected. See "Deferred"
below.

## Deferred

**Linking Adelie daemons and pushing work down to a desktop client.** A voice or web
session cannot host `fileio` or `terminal` in-process. Borrowing a runner from the
same user's desktop client would fix that, and it is a useful feature in its own
right - ask by voice, act on your desktop.

It needs three things this epic does not cover: a permission model for one surface
acting on another, tool scoping keyed to (user, host) rather than (user, session)
- #260 deliberately made registration session-scoped to stop the voice `say_this`
leak, so the distinction to draw is between session-affine capabilities and
host-affine ones - and the remote-daemon link itself, which has an unused KCM today.

That is a separate effort. Until it lands, a surface that cannot host a tool reports
the capability as unavailable and names the runner that would provide it.

## The pattern recurs

The framing above - that the operating system *was* the multi-tenancy, and that lifting the
daemon into a shared service removed the enforcer while leaving every assumption that
depended on it - turned out not to be specific to tenancy.

The same shape has since appeared twice more:

**Machine identity.** When the daemon ran on your machine, "the home directory" was
unambiguous because there was only one. In a cluster serving several clients on several
machines, the daemon's own hostname identifies a pod and means nothing to a user. Co-location
*was* the machine scope, and it dissolved the same way.

**Working environment.** One person may run several OS accounts on one machine, each a
separate environment with its own paths and checkouts, all connecting as the same assistant
user. Being the only account on the machine *was* the environment identity.

In each case the assumption is still in the data, unstated, because at the time it was
written there was only one possible value. The general lesson is worth carrying: when a
deployment change removes something that was previously implicit, the code does not break -
it keeps working against an assumption that has quietly stopped being true, and the failure
shows up much later as data that cannot be interpreted.

The memory-architecture consequences are tracked in #694, with the scoping work in #894.
They are not part of this epic, but they share its root cause.

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
