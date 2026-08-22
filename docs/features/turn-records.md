# Turn records

What the assistant was reading when it answered.

The conversation is already stored. The prompt is not. A turn assembles its
system prompt, its `[Recall]` block, its scratchpad injection and its
post-eviction window for one provider call, hands the result to a connector,
and drops it. So "what was said" is recoverable and "what the model was shown"
is not, and only the second answers why the assistant acted as it did.

Turn records close that gap. One record per turn, one per round inside it,
carrying the request exactly as sent, the reply, the tool calls with their
arguments, and each tool result as the turn stored it.

## Two tables, and what is in each

`turn_records` is one row per turn: its correlation id, the user and the
conversation, the connection, connector and model it dispatched on, the tool
policy it resolved to, and when it started.

`turn_round_records` is one row per round within it:

| Column | What it holds |
|---|---|
| `request` | Every message handed to the connector, with its role, in order - the system prompt and every injected block included. |
| `response_text` | The reply the provider streamed, whole. Not a preview. |
| `response_tool_calls` | What the model asked to run, with the arguments it wrote. |
| `tool_results` | What those calls returned, as the `Role::Tool` rows the turn appended. |
| `token_usage` | What the provider reported the round cost. |
| `error` | Why the round failed, where it did. |

The request is the assembled prompt, not the conversation. The two differ on
every turn: the window is trimmed, an oversized tool result is projected down
to its head, and the injected blocks exist nowhere else.

Rounds are numbered one-based, matching the round span and the round's log
line. A turn that spends its whole tool budget makes one further provider call
- the wind-down that turns an exhausted turn into a closing the person can read
- and that call is recorded too, one past the loop's last round. Its request
exists nowhere else at all: the wrap-up instruction it carries is dropped
before the reply is persisted.

A round the turn abandoned part-way through its calls records the calls that
already ran. They committed their side effects, so a record saying no tool ran
would be a wrong answer rather than a gap.

## One identifier

A record is keyed by the turn's correlation id: the value the client stamps on
its own event stream, so a person quoting a reply and the store reading it use
one identifier.

That value is usually the trace id too, because a turn nobody handed a trace
derives one from it. It is not the trace id when a caller forwarded a
`traceparent` to be continued - the trace is then the caller's, and the two
differ. The record follows the client's id in both cases, because that is the
one a person can actually quote.

A turn that reached the loop by another door - an agent run, a scheduled job, a
test - carries no client-minted id, so its records are keyed by its own trace
id spelled as a uuid.

## This is a store, not a log

Nothing here writes to the console. A record is queryable, scoped to its owner
and removed on a schedule; a log line is none of those. The daemon's spans keep
their own job - latency and correlation - and carry no content.

That distinction is what this feature replaces. `ProfilingLlmClient` was once
the only way to see inside a turn, and it paid for that with an unrotated JSONL
file on the pod's ephemeral disk, a 200-character preview instead of the text,
and no user or conversation id on any entry. Each of those is a constraint
here: the record is in the database, it holds the whole text, and every row
names its user and its conversation.

## Configuration

```toml
[inspector]
# Whether the full text of every turn is kept. Absent means the deployment
# decides - see below.
enabled = true
# How long a record is kept before the sweep removes it. Held to at least one
# day; there is no keep-forever value.
retention_days = 7
```

Read once at startup, so an edit needs a restart.

### The default follows the deployment

The record is what makes the assistant debuggable at all, and it is also a
second copy of somebody's whole conversation. On a single-person desktop those
two facts do not conflict; on a shared daemon they do. So an absent `enabled`
resolves differently in the two places:

- **Reachable only over the Unix socket** - one local person, who already owns
  every byte of it. Capture is **on**, with a seven-day window.
- **Remote WebSocket door open** (`[transports] ws_enabled`) - more than one
  principal can arrive. Capture is **off** until an operator turns it on.

A stated `enabled` wins in both directions. The daemon says which it resolved
to on every boot, and why:

```
turn capture: on (the deployment) - the full text of every turn is kept for 7 days, then deleted
turn capture: off ([inspector] enabled) - no turn text is kept
```

Turn capture also needs a database. An operator who turns it on without one
gets a warning naming both halves rather than an empty table.

### Retention is not optional

The sweep runs hourly wherever there is a database, and deletes every turn
whose `started_at` is past the window. Rounds go with their turn by the foreign
key's cascade, because the rounds are where the content is - a sweep that
dropped the turn row and left them would report a window it does not keep.

It runs whether or not capture is on now, which is the case that matters most:
an operator who ran with capture on and has just turned it off is the one
person whose records need removing, and a sweep gated on capture would strand
every one of them while the boot line said no turn text is kept.

There is no configuration that keeps records forever. `retention_days = 0` is
read as the one-day floor rather than obeyed: zero reads as "keep nothing" to
one operator and "keep everything" to the next, and only one of those is safe
to guess at.

## What the schema refuses

The correlation id is the client's to choose. Two things follow, and the schema
carries both rather than the code above it.

A turn and a round are each written at most once: the primary keys are the
identities, so a retry or a redelivery leaves one record rather than a second
copy of somebody's conversation.

A round's conversation must be its turn's. The foreign key carries
`conversation_id`, so a client that reused one correlation id across two
conversations gets its second round refused by the database and a warning in
the log, rather than a stored record that names one conversation on the turn
and another on the round with nothing to say which half is right.

## What it costs on disk

Each round records the whole prompt, and round N's prompt contains rounds 1 to
N-1, so one turn's records grow with the square of its round count. A long
tool-using turn over a large context can therefore write several megabytes.
Nothing bounds this but the retention window: there is no per-record or
per-turn size cap. Size the window against the disk, and read `[inspector]
enabled = false` as the answer where the record is not worth that.

## Multi-tenancy

Both tables carry `user_id`, every statement binds it, both enable their own
row-level security policy in migration 055, and both are registered in
`PERSONAL_DATA_TABLES` so the `execute_database_query` tool grafts a `user_id`
predicate onto any model-supplied SQL that names them. They are the widest
personal-data surface in the schema, so the scoping audit names them by hand as
well (`turn_records_are_user_scoped`).

Reading a turn record that is not yours is a capability of its own,
`transport_dispatch::authz::READ_ANY_USER_TURN_RECORDS` - the read-everything
switch, on the same axis as every other privileged read rather than in a
parallel model. The daemon ships no command that reads across users, and the
store's own statements bind the caller's id, so nothing today can widen a read.

## What a person sees

Nothing. There is no new setting on a desktop install, no new concept and no
client change: the default follows the deployment and the record is written
behind the turn. It becomes visible when a reader ships, under the Context
Inspector epic.

## The cost of capture

A wired recorder clones the assembled request once per round, and writes one
statement per turn plus two per round - the request and reply as soon as the
provider answers, then the tool results once its calls resolve. An unwired one
does none of that: the turn loop clones nothing and issues no write, so a
daemon with capture off runs exactly the turn it ran before this feature
existed. `capture_changes_nothing_the_model_sees` holds the other half - a
captured turn sends the provider the same bytes an uncaptured one does.

## Where the code is

| Piece | Path |
|---|---|
| Port types and the recorder trait | `crates/core/src/ports/turn_record.rs` |
| The three write sites in the turn loop | `crates/core/src/service.rs` |
| Postgres adapter | `crates/storage/src/turn_records/mod.rs` |
| Retention sweep | `crates/storage/src/turn_records/retention.rs` |
| Schema | `crates/storage/migrations/055_turn_records.sql` |
| `[inspector]` and the startup line | `crates/daemon/src/config/mod.rs` |
| Store, handler and sweeper wiring | `crates/daemon/src/main.rs` |
| The read-everything capability | `crates/transport-dispatch/src/authz.rs` |
| Acceptance tests | `crates/core/tests/turn_records.rs`, `crates/storage/tests/turn_records.rs` |
