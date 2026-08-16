# WebSocket API

This document describes the desktop-assistant WebSocket API exposed by the daemon.

## Endpoint

- Path: `/ws`
- Default bind: `127.0.0.1:11339` (set with `DESKTOP_ASSISTANT_WS_BIND`)
- URL example: `ws://127.0.0.1:11339/ws`
- Login path: `/login` (HTTP `POST`, Basic auth)

## Authentication

The WebSocket handshake requires a bearer token:

- Header: `Authorization: Bearer <jwt>`
- Missing or invalid token: HTTP `401 Unauthorized` during handshake
- A valid token that names no subject: HTTP `401 Unauthorized` during handshake

The last case is worth stating on its own, because a token can be signed
correctly and still fail here. The daemon resolves the caller's identity from
the token as part of accepting the connection, and the identity is the `sub`
claim. A token with no `sub` (or a blank one) names nobody, so the upgrade is
refused and the daemon logs why.

This matters most with an external identity provider. An RS256 access token is
validated against the issuer's keys, and that validation only requires `exp`, so
a machine or client-credentials token with no `sub` claim passes it. The daemon
used to file such a connection under the shared `"default"` identity - the same
partition a single-tenant install keeps all its data in. Configure the issuer to
put the caller's identity in `sub`.

If an instance was running that way, its conversations and knowledge entries are
already filed under `"default"`. Those rows stay where they are: once the issuer
names a subject, the connection is scoped to that subject and no longer sees
them. Move the data with a `user_id` update if it should follow the new
identity, and do it before the new subject starts writing.

Local clients need no token: the UDS door (which the D-Bus bridge also goes
through) authenticates by kernel peer credentials, and the identity is the
connecting peer's OS user. A JWT is a network-door concern, so `/login` is the
way to get one.

Use `/login` with HTTP Basic auth to mint a bearer JWT:

```http
POST /login HTTP/1.1
Host: daemon.example.com
Authorization: Basic <base64(username:password)>
```

Successful response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "subject": "alice"
}
```

`/login` credential validation modes:
- **Static password.** Validates against the daemon's own env credentials
  (`DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`, `DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD`).
  This is the mode for any deployment that is reachable over a network. Setting
  the password selects it, whatever else is configured.
- **OS password (PAM).** Validates against the host's own account password and
  uses the current OS username (ignores `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`).
  It is a local convenience - the point is that a person on their own machine
  logs in with the password they already have - so the daemon enables it only
  when it is listening on loopback, and never in a container. A daemon bound
  past loopback (`DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339`) leaves it off unless
  `DESKTOP_ASSISTANT_WS_LOGIN_LOCAL_SYSTEM_AUTH=true` says otherwise, because
  the same mode on a reachable port is a password oracle for a real system
  account. Setting that variable to `false` turns it off everywhere.
- **Off.** With no static password and no local door, `/login` answers `404`.
  The startup log says which of the three conditions applies.

### Failed-attempt throttling

`/login` counts attempts, per source address and per username. The first few are
answered normally, so a mistyped password costs nothing. After that the endpoint
answers `429 Too Many Requests` with a `Retry-After` header in whole seconds,
and the wait doubles with each further attempt up to a ceiling of one minute.
One successful login clears both counters.

The two counters do not refuse the same callers. The per-source counter refuses
the address that spent it. The per-username counter refuses only a caller that
has itself failed recently, for the reason below; where the server reports no
source address it applies to every caller, because nothing else can tell them
apart.

A `429` is not a credential verdict - the daemon did not read the password. A
client must wait for `Retry-After` and try again rather than re-prompting or
retrying on its own schedule: an early retry spends from the same budget and
pushes the wait out again, so a client left running with a stale password would
hold the door shut for whoever has the right one. In `client-common` the
auto-reconnect loop already waits; a first connect, and any caller of
`request_ws_login_token`, gets the error back and must wait itself.
`auth::login_retry_after` reads the wait off the error without matching its
text.

Two rules keep the throttle from becoming a way to lock the account's owner out.
The wait is capped at one minute, so a client with a stale password recovers
quickly. And the per-username counter refuses only a caller that has itself
failed recently: the account name is not a secret, so without that rule anyone
who could reach the port could spend the budget and then send one request per
lockout, and the owner's correct password would be refused before it was read.

The cost of the second rule is that a caller from an address with no failure
record of its own gets one attempt before its own counter exists, so an attacker
with an endless supply of source addresses buys one guess per address. That is
the deliberate trade: the person who owns the account cannot be shut out by
somebody else's guessing.

Failed attempts are logged at warn level with the source address and the
username tried. Refusals are logged at debug, because they cost the caller
nothing and one warn line each would let an attacker flood the log.

Two limits worth knowing. The counters live in the daemon process, so a restart
clears them. And the daemon does not read `X-Forwarded-For` or any other
forwarding header, because a caller writes those: behind a reverse proxy every
request carries the proxy's address and the per-source counter becomes one
shared bucket, leaving the per-username counter to do the work.

`/login` authenticates exactly one account, and the `subject` in the response is
the account it authenticated — the same value carried in the token's `sub` claim.
That subject is the user identity the daemon files the connection's data under:
conversations, knowledge entries and scratchpad notes are all scoped by it. So in
container mode the tenant is whatever `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME` names,
and changing that value moves subsequent writes to a different partition. A
request for any other username is rejected (`401`), never issued a token under a
different name.

## Message Model

All payloads are JSON text frames.

### Client -> Server

Envelope:

```json
{
  "id": "req-123",
  "command": { "ping": {} }
}
```

- `id`: client-generated request correlation ID
- `command`: command variant payload

### Server -> Client

Server frames are one of:

1. Result frame

```json
{
  "result": {
    "id": "req-123",
    "result": { "pong": { "value": "pong" } }
  }
}
```

2. Error frame

```json
{
  "error": {
    "id": "req-123",
    "error": "conversation not found"
  }
}
```

An error frame may also carry an optional `detail` object classifying the
outcome, so a caller can act on it programmatically instead of reading prose:

```json
{
  "error": {
    "id": "req-123",
    "error": "not authorized: 'set_api_key' requires the administrator capability; this connection holds tenant",
    "detail": {
      "code": "not_authorized",
      "description": "not authorized: 'set_api_key' requires the administrator capability; this connection holds tenant",
      "message": "Only a daemon administrator can do that. This connection is a tenant.",
      "retryable": false
    }
  }
}
```

- `code` is the stable identifier to branch on: `not_authorized`,
  `unsupported`, `not_found`, `already_terminal`. Never match the message text.
  A handful of rules-based refusals carry a feature-specific code instead, via
  the same field: `CreateConnection`, `UpdateConnection`, and
  `UpsertMcpServer` refuse a `base_url` or MCP server `url` that fails the
  shared remote-URL policy (#804, #895) with one of `url_malformed`,
  `url_scheme_not_allowed`, `url_insecure_scheme`, or `url_target_blocked` —
  see `docs/mcp-services.md`. An older client that does not recognize a code
  still round-trips it unchanged (below), so this never breaks parsing.
- `description` is for logs; `message` is fit to show a person; `retryable`
  says whether repeating the identical request could plausibly succeed.
- `detail` is **absent** when the daemon cannot classify the failure honestly.
  Read that as "unclassified", not as "not an authorization problem".

Backward compatible by construction: `detail` is an optional *field* on the
existing `error` frame, not a new frame variant, so a client that knows only
`{id, error}` parses it unchanged, and `error` repeats `detail.description`. A
`code` a client does not recognize round-trips instead of failing the parse.

3. Event frame (unsolicited/streaming)

```json
{
  "event": {
    "event": {
      "assistant_delta": {
        "conversation_id": "c1",
        "request_id": "srv-req-abc",
        "chunk": "Hello"
      }
    }
  }
}
```

## Commands

Current command variants:

- `ping`
- `get_status`
- `get_config`
  - The config view carries `restart_required`: what is in the daemon's config
    file that the running process is not acting on, as stable area keys
    (`"database"`, `"embeddings"`, `"persistence"`, `"ws_auth"`, `"tls"`,
    `"authz"`, `"recall"`). Absent or empty means every
    configured value is live. It
    reports *area names only* and never the configured value, so a pending
    `[ws_auth]` or `[tls]` change is visible without disclosing what it was
    changed to. Treat the set as open and render an unrecognized key verbatim
    rather than dropping it. Additive and backward-compatible: an older daemon
    that omits the field deserializes as empty, and an empty report is omitted
    on the wire.
  - The config view also carries `caller_capability`: `"admin"` or `"tenant"`
    for the connection that asked. Read it at connect time and render the
    sections this connection may not change as unavailable, with the reason,
    rather than letting a write fail on submit. Absent from a daemon that
    predates the authorization tier, which means "not reported" - keep the
    prior behaviour then, not "tenant". See "Authorization" below.
  - One key is not an area: `"config_load_failed"` means `daemon.toml` itself
    would not load, so nothing in it is in force and the daemon is running
    built-in defaults. A settings UI should show that state instead of
    presenting the defaults as the user's configuration - an empty connections
    list here means "could not read your config", not "you have none". While it
    is reported, the daemon refuses config-changing commands (`set_config`,
    connection and purpose writes) with an error naming the file, so it cannot
    overwrite a file it could not read. It clears once the file loads again.
- `set_config { changes }`
- `create_conversation { title }`
- `list_conversations { max_age_days }`
- `get_conversation { id }`
- `delete_conversation { id }`
- `clear_all_history`
- `send_message { conversation_id, content, turn_id?, traceparent? }`
  - `turn_id`: the client's own correlation id for this turn, as a uuid. A
    turn starts when a person presses send, so the client mints this and the
    daemon adopts it; the ack returns the value the daemon actually uses. An
    absent, malformed or nil value means the daemon mints its own, which is
    what keeps an older client working unchanged. It reaches no authorization
    decision and it is not `idempotency_key`.
  - `traceparent`: a W3C trace context, for a caller that is already inside a
    trace and wants the daemon to continue it rather than start one. The web
    BFF is the case this exists for. An unusable value is discarded and never
    fails the turn. `tracestate` is not carried.
- `get_llm_settings`
- `set_llm_settings { connector, model?, base_url? }`
- `set_api_key { api_key }`
- `get_embeddings_settings`
  - The embeddings view (also carried in `get_config`) includes a `health`
    field with the backend's capability-detected state *as of the moment you
    ask*: `{ "status": "disabled" }` (no embedding backend configured, vector
    search off by design), `{ "status": "ok" }` (the backend is producing real
    embeddings), `{ "status": "unavailable", "reason": "..." }` (a backend is
    configured but cannot embed right now, so vector search has degraded to
    full-text search), or `{ "status": "unknown" }` (health was not determined —
    configured but not probed). A startup probe seeds the value; from then on it
    tracks live embed outcomes and a background re-probe that runs while the
    backend is `unavailable`, so a backend that recovers flips back to `ok`
    without a daemon restart and a backend that breaks flips to `unavailable`
    without one either. Changing the `[embeddings]` configuration itself still
    requires a restart. The legacy `available` boolean remains a shallow
    connector check; `health` is the honest signal. Additive and
    backward-compatible:
    `health` defaults to `unknown`, so an older daemon that omits the field
    deserializes as `unknown` (not `disabled` — a working-but-unreported backend
    must not read as off), and a future `status` an older client does not
    recognize also deserializes as `unknown` rather than failing the payload.
- `set_embeddings_settings { connector?, model?, base_url? }`
- `get_connector_defaults { connector }`
- `get_database_settings`
- `set_database_settings { url, max_connections }`
- `list_negative_memories`
- `get_negative_memory { id }`
- `clear_negative_memory { id, note? }`
- `list_skills { limit? }`
- `set_skill_approval { name, approved }`
- `list_context_breakdowns { conversation_id, limit?, offset? }`
- `get_context_breakdown { request_id }`

Result payloads are typed variants (`pong`, `status`, `conversation_id`, `conversations`, `conversation`, `config`, `ack`, etc.).

### Context breakdown: what filled a turn's prompt

`list_context_breakdowns` returns one entry per turn of a conversation, oldest
turn first; `get_context_breakdown` returns the one turn whose `request_id` you
name, or `null`. `limit` defaults to 50 when the field is absent and is capped
at 500, so page by how many entries you received rather than by how many you
asked for. An explicit `limit` of 0 returns an empty page - it is a value, not
an absent field, so the default does not apply to it. Both are tenant reads scoped to the calling user, and a
`request_id` is not a capability: another user's turn reads as absent rather
than as a refusal.

`request_id` is the same correlation id the turn's own events carried
(`assistant_delta`, `assistant_completed`, `assistant_status`), so a client can
open the breakdown for a turn straight from its own event log with nothing to
look up.

One case breaks that, and it is the one worth knowing. When a client re-attaches
to a turn already running, or replays a completed reply through an
`idempotency_key`, the daemon re-stamps the events it forwards with the
*asking* client's `turn_id` - so the id on those events is not the id the turn
was recorded under, and `get_context_breakdown` answers `null` for it. That is
an absent record, not a lost one: the entry is filed under the id of the turn
that actually ran, and `list_context_breakdowns` on the conversation still
returns it.

Each entry carries:

- `turn_ordinal` - the message ordinal the turn's user prompt took, so the
  entry can be lined up against the transcript.
- `model` - what the turn actually ran on.
- `estimated_parts` - a list of `{ part, estimated_tokens }`, in the order a
  prompt renders them. The names come from a closed set the daemon owns, and
  this document does not repeat that set: a second copy of it would drift, and
  would then name a part nothing sends. Read the names off a live reply, and
  render one you do not recognize as it is - a dropped part is prompt cost that
  appears to have come from nowhere.
- `estimated_total_tokens` - those parts summed.
- `advertised_tool_count` - how many tool schemas the prompt carried. A count,
  not a token figure; what the schemas cost is the `tool_schemas` part.
- `provider_used_tokens` - what the provider itself reported for the prompt.
- `budget_tokens` and `budget_source` - the input-token budget the turn ran
  under, and which tier resolved it.
- `compaction_active` - whether the turn shrank its own window under token
  pressure and summarised what it dropped.
- `projected_messages` - how many messages this prompt read as a compaction
  pointer, the head of an oversized result, or a truncation notice, rather than
  as their stored content. The transcript itself still holds every byte.
- `recorded_at` - when the daemon wrote the entry, RFC3339.

**The estimate and the provider's count are two measurements, and a client must
not merge them.** `estimated_parts` is measured by the daemon's own context
assembler, with the estimator the context budget uses. `provider_used_tokens`
is what the provider's tokenizer reported for that same prompt. They do not
agree, and the difference between them is itself worth showing. Do not add one
to the other, do not present either as a component of the other, and do not
compute a percentage of one against the parts of the other. Show both.

Their absence rules differ, deliberately. A part that rendered nothing reports
`0`, because the daemon always knows whether it emitted the block. A provider
that reported no count leaves `provider_used_tokens` off the payload entirely,
because a `0` there would invent a measurement - so treat an absent field as
"not reported", never as zero. `budget_tokens` and `budget_source` are absent
together for a turn that ran with no budget installed.

Both figures describe the turn's opening prompt, and only that one. A turn runs
one prompt per round of its tool loop, and each later prompt carries the tool
traffic the rounds before it produced; the entry reports the standing bill the
turn opened with, which is what an operator acts on, not the tail of the loop.
So `provider_used_tokens` is absent - never borrowed from a later round - when
the opening round reported no usage, and when the provider refused the opening
prompt for overflow and the daemon re-assembled a smaller one.

`budget_source` is one of `purpose_override`, `connector_table`,
`universal_fallback` or `learned_cap`, as a plain string rather than an enum so
a tier added later does not break a client that has not been rebuilt. It is the
field that separates a curated limit for this model from the conservative
fallback the daemon uses when nothing supplied one - the same number, and a
different situation. Treat the set as open and render an unrecognized value
verbatim.

Not every turn has an entry. A turn that ran without a correlation id (an agent
run, a scheduled job) has no key to be recorded under, and a turn cancelled
before it assembled a prompt measured nothing. A deployment with no database
keeps no entries at all and answers both commands with an error rather than an
empty list, so "this conversation has no entries" and "this daemon keeps none"
stay different answers.

### Skills: what the library holds, and what may be followed

A skill is a named procedure. The assistant can write one for itself - from a
plan it finished, or from a method it found while consolidating - and every
such skill is recorded **unapproved**. An unapproved skill is offered by
nothing and its body is refused by the skill-read tool, so until a person
approves it, it exists and does nothing. These two commands are the whole of
what a person does about that.

**`list_skills { limit? }`** returns `skills`, one row per skill this person
can see: the host-global ones plus their own.

```json
{"skills": [
  {"name": "deploy-the-lab",
   "description": "Roll a new image out to the cluster and watch it settle.",
   "kind": "workflow",
   "trust_tier": "local",
   "source": "self-authored",
   "own": true,
   "present_on_disk": false,
   "approved": false,
   "tags": ["ops"]}
]}
```

A row carrying `proposed_from_entry_id` was not written by the assistant from
its own work: it was **proposed** from a knowledge entry that turned out to be a
procedure. The entry is untouched, so that field is the other half of the split
- it names the entry a person may want to retire once they approve the skill.

An unapproved skill is **listed**, with `approved: false`. That is the point of
the command: silence would be indistinguishable from a library with nothing in
it, and a skill nobody can see is a skill nobody will ever approve. `approved_at`
and `approved_by` are present only once consent has been recorded.

`trust_tier` is `local`, `github`, `well_known` or `unknown` - a string rather
than an enum, so a tier added later reaches an older client as an unfamiliar
word instead of a parse failure. On anything but `local`, `description` is text
somebody outside this machine wrote: render it as a quotation, not as the
assistant's own words. The body is not carried; read it with the skill tools.

**`set_skill_approval { name, approved }`** returns
`{"skill_approval_set": {"approved": true, "changed": true}}`. `approved` is
the state after the call, so a client can render the result without a second
read; `changed` says whether this call is the one that moved it. Asking for the
state a skill is already in answers `changed: false` and writes nothing - not
an error, and the skill is in the state you asked for either way.

Two things it deliberately does not do.

**It carries no approver.** The approver is the connection's authenticated
subject. A payload field naming somebody else would let a record of consent be
written in another person's name.

**It reaches only your own skills.** A host-global skill was approved by
somebody putting a file in a skill root, and withdrawing that from one person's
session would decide it for every other tenant on the host. A name you own no
row for is refused with `not_found` rather than silently ignored, because you
asked for a state change that did not happen.

A deployment with no skill catalog - no database, or no skill roots - answers
both commands with `unsupported` rather than with an empty list.

### Negative memory: what is holding a call back

The daemon holds a tool call back when the same act went badly before, and the
model reads the stored lesson in place of the tool result. A person sees only
that the call did not run, so these three commands are what make the reason
visible - and what lets a person overrule it.

Read them together with the activity feed. A held call reports as a finished
tool call with `ok: false` and an output of the form
`held by a stored lesson (<id>): <what went wrong>`. That id is what
`get_negative_memory` takes.

**`list_negative_memories`** returns `negative_memories`, one row per live
memory, strongest first:

```json
{"result": {"negative_memories": [{
  "id": "0199...", "action": "terminal_run",
  "arguments": [{"name": "command", "value": "rm -rf build"}],
  "circumstances": [{"name": "host", "value": "workshop"}],
  "outcome": "build is a mount point", "occurrences": 2,
  "strength": 0.71, "firing": true,
  "written_at": "2026-08-01T09:00:00Z",
  "last_confirmed_at": "2026-08-05T09:00:00Z",
  "goes_quiet_at": "2026-09-02T09:00:00Z",
  "cleared": false
}]}}
```

`strength` is a fraction of full at the moment the daemon answered; it halves
every two weeks without a repeat. **`firing`, not `strength`, is what says
whether a person's work is held**: the two come apart, because clearing a
memory leaves its confirmation stamp alone, so a memory cleared a second ago
still reads at full strength and holds nothing. `goes_quiet_at` is in the
future only when `firing` is true - a memory silenced by decay reports the day
it fell silent, and one silenced any other way reports the moment it was read -
so a date still to come always means work still held.

`arguments` and `circumstances` are separate lists on purpose: an argument is
the act's identity and never widens, a circumstance is provisional, and an
argument may itself be *named* `host`.

**The list is capped at 200 rows**, ordered by last confirmation, newest first.
It is deliberately the same read a turn makes to decide what may hold a call,
with the same bound and the same order, so a memory this list omits is one the
daemon would not have fired either. A person cannot therefore be shown a list
that is missing the memory holding their work: reaching the cap needs 200
memories confirmed more recently than the one dropped, and a memory that holds
anything was confirmed inside the last four weeks.

**`get_negative_memory { id }`** returns `negative_memory`, either `null` or one
memory in full. It answers for a cleared memory as well as a live one, and it
answers `null` for a correction's own id - a correction is the record of a
lesson that stopped applying rather than a lesson, and it is readable on the
memory it corrects. Two fields a list row does not carry:

- `dropped` - the circumstances the memory once required and no longer does,
  each with the value it was born requiring and the date it was dropped. A
  memory widens in exactly one way: a later occurrence of the same act,
  somewhere else, drops the circumstances it disagreed with. **This list is the
  whole history of a memory getting wider, and an empty `circumstances` beside
  a long `dropped` is what a memory that now fires everywhere looks like.**
- `correction` - `{ id, outcome, written_at }` when the memory has been
  cleared. `outcome` says why it stopped applying and `written_at` is when.

**`clear_negative_memory { id, note? }`** returns
`{"negative_memory_cleared": {"cleared": true}}`. Nothing is deleted: this
writes the same `correction` overlay that the act succeeding writes, so the
original stays readable and `get_negative_memory` keeps answering for it.
`note` is what the correction says; omit it and the daemon writes its own line
recording that a person cleared it, which is what tells a later reader a
person's judgement from an observed success. `cleared: false` means the memory
was already cleared or this user does not hold it - neither is an error, and
the memory is in the state you asked for either way.

All three need only the tenant capability, and all three report
`{"error": {"code": "unsupported"}}` on a deployment with no database, where
negative memory does not run at all. That is deliberately not an empty list: a
person asking why the assistant will not do something would read an empty list
as "nothing is holding it".

### Credentials in connection URLs

Every connection URL a reply carries — the database `url` in
`get_database_settings` and `get_config` — has its password replaced by `***`:

```json
{"result": {"database_settings": {"url": "postgres://adele:***@postgres:5432/adele", "max_connections": 5}}}
```

The rest of the URL (scheme, user, host, port, database, options) is intact, so
a settings UI can still show what is configured. Both the inline
`user:password@` form and the libpq `?password=` parameter are redacted.

Writes round-trip. Posting back a URL that still carries the `***` placeholder
keeps the stored password, provided nothing else in the URL changed — so a form
that only ever saw the redacted value can still save an unrelated field.
Editing any other part of the URL means re-entering the password: the daemon
refuses such a write rather than splicing its stored credential into a URL the
client has edited, and it never stores the placeholder as if it were a
password.

## Events

Current event variants:

- `config_changed { config }`
- `assistant_delta { conversation_id, request_id, chunk }`
- `assistant_completed { conversation_id, request_id, full_response }`
- `assistant_error { conversation_id, request_id, error }`
- `assistant_status { conversation_id, request_id, message, capability_change? }`

### Tool-provenance gating (behaviour change)

A turn that has taken in content from outside the trust boundary - a fetched
web page, a third-party API result, a file read at a path the model chose -
refuses the tools that could act on it for the remainder of that turn. The
closed tiers are `mutate`, `network_egress`, `code_execution`, and anything the
daemon cannot classify. Reading, and output to the user's own session, stay
open. The next turn starts clean.

**What an integrator must do.** A model-chosen tool call that would previously
have run may now come back refused. The refusal arrives as an ordinary tool
result, not an error, and the turn keeps going - so a client that does nothing
still works and simply sees the assistant take another path. To handle it
deliberately:

1. Watch `assistant_status` for `capability_change`. It is present only on the
   one status per turn that reports the narrowing, and absent on ordinary
   progress:

   ```json
   {"assistant_status": {
     "conversation_id": "c1",
     "request_id": "r1",
     "message": "Read outside content - sending, changing and running are off for the rest of this turn",
     "capability_change": {
       "reason": "external_content_ingested",
       "closed_tool_tiers": ["mutate", "network_egress", "code_execution", "unclassified"]
     }
   }}
   ```

   The change holds for the turn named by that `request_id`.

2. Read `tool_tier` on `ToolUsageView` to know in advance which of the tools a
   conversation uses can be refused. Tier values: `read`, `present`, `mutate`,
   `network_egress`, `code_execution`, `unclassified`. The last four are the
   ones that close.

3. To get a refused action done, start a new turn that does not read outside
   content first.

Both fields are optional additions. A client built before them keeps parsing
every frame unchanged, and a tier or reason string it does not recognise must
be treated as `unclassified` / `unknown` - which is to say, assume it can be
refused.

#### The tool policy, and what closes by default

**Behaviour change.** The daemon now runs every turn at a *tool policy*, and
the shipped default is `standard`, which refuses nothing. A turn that reads
outside content keeps every tier and reports the fact once on the status
channel. An integrator that relied on tiers closing after ingest must set
`tool_policy = "aggressive"` under `[security]` in `daemon.toml` to get the
previous behaviour back. The three levels:

| Level | A turn that has read outside content |
|---|---|
| `aggressive` | The four tiers above close for the rest of the turn |
| `standard` (default) | Nothing closes |
| `lax` | Nothing closes, and nothing is reported |

The daemon says which level it resolved in its startup log. A value it does
not recognise is named in an error there, and the daemon runs at `standard`.

#### Text a turn wrote after reading outside content

`[security] hard_withhold` decides what happens to the words a turn writes to
durable storage after it has read content from outside the trust boundary: a
plan step's goal and outcome, and a negative memory's account of what went
wrong.

```toml
[security]
tool_policy = "standard"
hard_withhold = false
```

| Value | What is stored | What the model reads |
|---|---|---|
| `false` (default) | the words, with the fact recorded | the placeholder at `aggressive`, the words at `standard` and `lax` |
| `true` | a placeholder | the placeholder, at every level |

The level that decides is the one in force when the block is **rendered**, not
the one in force when the note was written. So moving `tool_policy` changes what
the model sees of everything already stored.

A person reads the words at every level and under either setting, except where
`hard_withhold` destroyed them - then there is nothing to read.

**Behaviour change.** Before this, the words were destroyed at write time at
every level, which is `hard_withhold = true`. A daemon that upgrades without
setting the key gets `false` and stops discarding text it wrote. The daemon
states the resolved mode in its startup log on every boot, set or not, so an
operator can read the current state rather than infer it from an absent key.

There is deliberately no per-conversation or per-user override. The per-turn
control is `tool_policy`.

#### Per-conversation override

A conversation can ask for the level that refuses nothing, whatever the
daemon default is. Send
`set_conversation_tool_gate { conversation_id, disabled: true }` and expect
`result -> conversation_tool_gate { disabled: true }` echoing the stored
value. `disabled: true` selects `lax`; `disabled: false` stores no level, so
the conversation follows the daemon default. The value is resolved fresh on
every send, so flipping it takes effect starting with the conversation's next
turn; `GetConversation` also reports the live value as `tool_gate_disabled`
on the conversation view (omitted from the wire when `false`).

A conversation cannot yet select `aggressive` for itself - the stored value is
still the boolean this command has always carried. Naming all three levels per
conversation is issue #1199.

With the override on, a gated tool call runs even after the turn has read
outside content - `check()` never refuses. The turn still tracks whether it
ingested outside content, so the person watching is told once per turn, on
the same `assistant_status` channel, with no `capability_change` payload
(nothing actually closed, so `closed_tool_tiers` would be empty and
misleading):

```json
{"assistant_status": {
  "conversation_id": "c1",
  "request_id": "r1",
  "message": "Live dangerously is on for this conversation: a tool that would normally be refused after reading outside content ran anyway."
}}
```

This is a deliberate, per-conversation hole in the gate, not a bug: with it
on, a turn that reads attacker-controlled content and then acts on it is
indistinguishable from a turn the user actually asked to act. Fails closed at
every layer: an unset value, a missing conversation row, a cross-user row, or
a store error all resolve to the gate staying enforced. Only an explicit
stored `true` disables it.

### `evicted_results` on `ToolUsageView` (behaviour change)

**This changes what an existing reader of the tool-usage aggregate sees.**

A completed agentic step drops its large tool results from the model's view
and reads them as a short pointer to the scratchpad note it distilled them
into. That decision is now recorded on the message row, so every later turn
reads the pointer while the stored transcript keeps every byte the tool
returned. Those rows are counted in `evicted_results`.

Before, `evicted_results` could only be non-zero for a conversation compacted
by an old build that overwrote the row. A non-zero count therefore meant "the
bytes are gone and `result_bytes` under-reports what this tool cost". That is
no longer what it means on its own.

What an integrator must do:

- Read `evicted_results` as "results the model reads as a pointer", not as
  "results whose bytes are lost".
- Do not add `evicted_results` to `result_bytes` to reconstruct an original
  size. For a row evicted by a completed step the bytes are already counted in
  full, because the conversation still holds them; for a row from an old build
  they are unrecoverable and count zero. The two cannot be told apart in this
  view.
- A rising `evicted_results` with steady `result_bytes` is the normal, healthy
  shape for a conversation doing long agentic work. It is not data loss.

No field was added or removed, and the wire bytes are unchanged.

## Typical Session Flow

1. Acquire JWT (local clients)
- Call D-Bus `GenerateWsJwt("my-client")` (token subject is current OS username).

2. Acquire JWT (remote clients, no D-Bus)
- `POST /login` with Basic auth.
- Receive `token`.
- On `429`, wait for the seconds named in `Retry-After` and repeat. Do not
  re-prompt for the password: the door is shut, and the credential may be right.

3. Open WebSocket
- Connect to `ws://127.0.0.1:11339/ws`.
- Include `Authorization: Bearer <token>`.

4. Health check
- Send `ping`.
- Expect `result -> pong`.

5. Discover or create a conversation
- Send `list_conversations`.
- If needed, send `create_conversation`.

6. Send a user message
- Send `send_message`.
- First response is `result -> ack`.
- Then receive streamed events:
  - one or more `assistant_delta`
  - terminal `assistant_completed` (or `assistant_error`)

7. Refresh conversation state
- Send `get_conversation` if you need the full canonical message list.

8. Optional live configuration
- Send `set_config`.
- Expect:
  - `result -> config`
  - followed by `event -> config_changed` with the same config snapshot.
- `set_ws_auth_settings` still answers with a bare `ack`, and is followed by an
  `event -> config_changed`. Its `restart_required` is how the client that made
  the change learns that the new authentication methods, OIDC config, or allowed
  origins are written but not yet in force.

## Authorization

Authenticating a token says who you are. It does not say that you own the
service. A WebSocket connection is a **tenant** unless `[authz] admin_subjects`
in `daemon.toml` names its subject (the JWT `sub`), in which case it is an
**administrator**. The allowlist is empty by default, and **no command writes
it**: it is set by editing `daemon.toml`, so the API surface offers a tenant no
way to grant themselves the capability. (Daemon-side file and shell MCP servers
run as the daemon's own uid and could reach the file; they ship disabled, and
constraining them belongs to the tool-execution design, not this tier.)

A tenant may do everything with its own data: conversations, messages,
knowledge, scratchpads, background tasks, personality, and every read on this
API, including `list_connections`, `list_available_models` and `get_purposes`,
which the model picker needs.

An administrator is additionally required for the service-configuration writes:

`set_api_key`, `set_embeddings_settings`, `set_database_settings`,
`set_backend_tasks_settings`, `set_ws_auth_settings`,
`create_connection`, `update_connection`, `delete_connection`,
`set_connection_secret`, `set_purpose`, `add_mcp_server`, `remove_mcp_server`,
`set_mcp_server_enabled`, `upsert_mcp_server`, `set_mcp_secret`,
`upsert_service_account`, `remove_service_account`,
`start_knowledge_maintenance`, and `mcp_server_action` for every verb except
`"status"`.

`set_config` is decided by its payload, and every field it carries writes
daemon-global state, so any non-empty change set needs the administrator
capability - the personality traits included, because they are one global block
that every conversation without an override resolves against. A tenant sets
their own disposition with `set_conversation_personality`.

`set_conversation_tool_gate` is a per-conversation, tenant-level lever from
the start - there is no global counterpart to weigh against here, unlike the
personality traits' staged path through `set_config` above.

`mcp_server_action` is split by its verb: `"status"` is a read and stays tenant;
`"start"`, `"stop"`, `"restart"` and anything added later need the
administrator capability.

`start_knowledge_maintenance` needs the administrator capability: every
operation reaches past the caller's own rows, up to nulling and re-deriving
every tenant's embeddings through the operator's provider.

Discover your own level from `caller_capability` on the `get_config` reply (and
on every `config_changed` event) rather than probing with a write.

A refused command returns the error frame shown above with
`detail.code = "not_authorized"` and `retryable: false`. The connection stays
open, and the command has no effect - the check runs before the handler.

**This is a behaviour change for existing API consumers.** A remote caller that
could previously issue any of the commands above will now be refused unless it
is listed. If your integration configures the daemon, add its subject to
`[authz] admin_subjects` and restart the daemon. Integrations that only send
messages and read their own data are unaffected. So are local clients on the
daemon's own account, which are administrators by construction over the Unix
socket and D-Bus.

## Notes

- The command `id` correlates only `result`/`error` frames.
- Streaming assistant events are correlated by server-generated `request_id` inside event payloads.
- Multiple requests can be in flight concurrently; clients should match by `id` and event metadata.
