# Logging and telemetry

The daemon and the D-Bus bridge produce traces, metrics and logs through
[`adelie-telemetry`](https://github.com/adelie-ai/adelie-telemetry), the crate
every Adelie Rust binary shares. One setup, one set of knobs, so an operator
meets the same behaviour in the daemon, the bridge, the voice service and every
MCP server.

Console output is the default and needs no collector. Export to an
OpenTelemetry collector is *additional*, not a replacement, and is behind an
off-by-default Cargo feature.

## Where it goes

**stderr, always. Never stdout.** stdout carries a process's own output: the
daemon prints an operator-facing result there for its two command-line escape
hatches, an MCP server frames JSON-RPC there, and a client writes the model's
reply there. One log line mixed into that stream corrupts what the reader is
parsing.

Console output is plain text, not JSON, and colour is off. With a collector
doing the real collection, stderr is for a person reading `journalctl` or
`kubectl logs`, and plain text is easier to read there than JSON.

## How much of it

`RUST_LOG` sets the filter, through `EnvFilter`.

```sh
RUST_LOG=info desktop-assistant-daemon
RUST_LOG=info,desktop_assistant_mcp_client=debug desktop-assistant-daemon
```

When `RUST_LOG` is unset or unparseable, each binary falls back to its own
default:

| binary | default filter |
|---|---|
| `desktop-assistant-daemon` | `error` |
| `adelie-dbus-bridge` | `info` |

Every shipped deployment sets `RUST_LOG=info` explicitly - both systemd units,
both container images and the Kubernetes manifests - so the daemon's `error`
default applies only to a bare `cargo run`. One consequence is worth knowing:
the periodic metrics summary is written at INFO, so a run with no `RUST_LOG` at
all does not print it. Ask for `RUST_LOG=info` to see it.

One filter governs the console and the OTLP log exporter together. An operator
who turns the verbosity up expects the same lines wherever they read them.

## What may appear at each level

This contract is a rule, not a preference.

| Level | Carries |
|---|---|
| INFO | ids, counts, byte sizes, durations, model names, tool names, error kinds, token counts. **Never content.** |
| DEBUG | prompts, the assembled context, tool arguments, tool results, tool failure messages, search queries, extracted facts. |

"Content" is anything derived from what a person or a model wrote: a prompt, a
message body, a tool-call argument, a tool result, a search query, an extracted
personal fact. Ids, counts, sizes, durations, and the *names* of tools, models
and providers are not content.

**A failing tool's message is content**, and it is the case that catches people
out. A tool says what it could not do, and that sentence quotes what it was
given - "failed to read `<path>`: permission denied". An MCP server's message
arrives here verbatim. So a log line carries `CoreError::kind()`, which names
the variant and nothing else, and the message goes to DEBUG beside the tool
result.

**A spawned MCP server's stderr is buffered, never streamed to the log.** The
stdio transport keeps the last few lines of each server's stderr in memory so
that a server which dies during the handshake can say why (see
[MCP services](mcp-services.md#startup-behaviour)). No line is logged as it
arrives, at any level: stderr is a server's whole unfiltered output rather than
a chosen field, and a server is free to print a credential or a piece of the
user's own content into it.

The tail reaches one place, and only when a connection actually fails: the text
of the error, which the executor then logs at ERROR beside the server's name.
That is a deliberate, bounded exposure, taken because the alternative is an
operator who can see only an exit code and has to attach to the process to
learn anything more. It is bounded twice over - a failed connect, and 10 lines
plus an unterminated final fragment, of at most 512 bytes each - and scrubbed
of the characters that would let a server rewrite the log line it lands in.
`server_stderr_is_buffered_and_never_logged` in
`crates/mcp-client/tests/stderr_diagnostics.rs` holds the streaming half of
that line.

Two things follow.

**`RUST_LOG=debug` means conversation content reaches the journal, and reaches
the collector when one is configured.** That is deliberate, and it is why the
default is `info`. Raise a single target rather than the whole process when you
are debugging one thing:

```sh
RUST_LOG=info,desktop_assistant_mcp_client=debug
```

**A new log line is checked against the contract before it lands.** Named tests
hold the line where content used to leak: `no_content_at_info` and
`content_appears_at_debug` in `crates/core/tests/log_content_contract.rs`,
`no_search_query_at_info` in `crates/mcp-client/src/builtin.rs` - which drives
every search builtin, not a chosen few -
`no_extracted_fact_content_at_info` in
`crates/storage/tests/dreaming_db_paths.rs`, and `turn_span_records_no_content`
in `crates/core/tests/turn_telemetry.rs`, which reads span fields as well as
console text.

`scripts/tests/systemd-logging.test.sh` holds the shipped systemd units to a
global `RUST_LOG` of `info` or quieter, and forbids `debug` on the targets that
carry content. A target that starts logging content belongs on that list the
same day.

## Following one turn

A user reports "it took four minutes to answer at 14:20". Everything below
exists so that report can be followed to the call that was slow, without
turning anything on and without reproducing the turn.

### The identifier to start from

One `request_id` per turn, stamped on every event that turn streams, so a
client already shows it. That value is a field on the turn span, and every line
the turn writes carries it through span scope:

```text
INFO turn{request_id="4bf92f35-..." trace_id="4bf92f3577b34da6a3ce929d0e0e4736" conversation_id="c-91" user_id="alice"}: executing tool tool=web_fetch arg_bytes=214
```

The `trace_id` beside it is normally the same value. A uuid is 16 bytes and a
W3C trace id is 16 bytes, so the request id becomes the trace id directly, with
no mapping table and no second identifier. One string therefore greps the pod
log, appears in the client's own event stream, and finds the trace in a
backend, and it can be pasted from any one of the three into any other.

That holds with the `otel` feature off. What the feature adds is export, not
correlation.

**One case makes them differ, and it is worth knowing before you go looking.**
When a caller supplies a `traceparent` - the web BFF forwarding a browser's
turn - the daemon continues *that* trace, so `trace_id` is the caller's and
`request_id` stays the correlation id the client reads its own stream by.
Pasting the request id into a backend then finds nothing. The turn's own log
line carries both, so the trace id is one grep away.

### Who mints it

A turn starts when a person presses send or stops speaking, and that happens in
the client. So the client mints the id and the daemon adopts it. The daemon
mints its own only when none arrives, which is what keeps an older client
working unchanged - a client that sends nothing is a supported configuration,
not a degraded one.

The daemon accepts a well-formed, non-nil uuid and falls back to minting for
anything else. A malformed value never fails the turn.

**`request_id` is no longer unique by construction.** The daemon used to mint it
per send, so nothing could collide. A client now chooses it, and the daemon
checks its shape and not its novelty. Nothing inside the daemon keys on it - the
in-flight index and the idempotency store key on the user, the conversation and
the idempotency key, and event delivery is scoped by user id - so a repeated
value costs a confusing trace and nothing else there.

Two consequences reach a client, and both are worth stating for an integrator
writing against this API:

- **Mint one id per turn, not one per session.** A client dedupes the echoed
  `UserMessageAdded` on `request_id`, because it already rendered that bubble
  optimistically. Reusing an id across two turns in one conversation therefore
  suppresses the second bubble in the client that sent it. The shipped clients
  mint per send and never hit this.
- **A process that multiplexes several callers over one daemon connection must
  mint its own value for the daemon hop** rather than forward what it was
  given, because it demultiplexes the reply stream on what comes back. The web
  BFF is the one that does this, and it carries the caller's trace in
  `traceparent` instead, which loses nothing: the daemon joins that trace
  either way.

The id is a correlation id and nothing else. It grants no capability and names
no user, so it reaches no authorization or tenancy decision. It is also not the
idempotency key, which is a separate field on the same command and is what the
exactly-once retry path reads.

### Which boundaries carry the trace, and which do not

A trace is only useful if it survives a process boundary, and none of these
boundaries is plain HTTP, so none of them gets it for free.

| Boundary | How the context travels |
|---|---|
| Client to daemon, over UDS, WebSocket or D-Bus | `turn_id` and `traceparent`, two optional fields on the `SendMessage` command. The frames are our own and have no header concept. |
| Daemon to a stdio MCP server | `params._meta.traceparent` on the JSON-RPC request. A pipe has no headers, and `_meta` is the MCP spec's own place for protocol metadata. |
| Daemon to a remote MCP server, over Streamable HTTP | A real `traceparent` request header, which is what a server nobody here owns understands. |
| Browser to the web BFF to the daemon | The browser mints an id and sends it. The BFF mints its own `turn_id` for the daemon hop and passes the browser's trace on in `traceparent`, so the daemon joins the browser's trace. Three processes, one trace. |

Two boundaries deliberately carry nothing, and both are worth knowing about
before an operator goes looking for the missing half:

- **An LLM provider.** No provider continues our trace and none ever will. The
  useful move there is capture rather than propagation: the provider's own
  request identifier is recorded on the `llm.call` span as
  `provider_request_id`. That is the value to quote when opening a support
  ticket with the provider, and it is the closest thing to end-to-end that this
  boundary allows. Ollama reports no such identifier, so its calls carry none.
- **A D-Bus caller.** `SendPrompt(conversation_id, prompt)` has no room for a
  caller-supplied id and no options dictionary to add one to, and widening that
  signature would break every existing caller. So the bridge mints, and the
  bridge is the top of the trace for a D-Bus caller such as the KDE plasmoid.
  The caller still gets a correlatable turn: the id the method returns is the
  one the daemon stamps on every streamed event.

`tracestate` is not sent on any boundary. Nothing in this fleet sets one, and
an empty one carries no information a receiver can use.

### A conversation is not a trace

A conversation lives for days and holds an unbounded number of turns, which is
not a shape any backend renders usefully. So the conversation id is an
attribute on every span in the turn rather than a trace of its own, and one
backend query still returns every turn in a conversation.

### The shape of a turn

```text
turn                       one turn, the root
  recall.lookup            the turn's one embedding round-trip
  turn.round               one iteration of the tool loop
    llm.call               the provider call for that round
    tool.call              one tool dispatch, one span each
  turn.round
    llm.call
```

An MCP server's own spans join this tree. A stdio server hangs its work under
the `tool.call` that dispatched it; with the `otel` feature off the daemon has
no span id to name, so the server's spans hang under the turn instead. The
trace is the same either way, and only the shape differs.

Field summary. Every span carries `conversation_id`.

| span | fields |
|---|---|
| `turn` | `request_id`, `trace_id`, `conversation_id`, `user_id`, `connection_id`, `provider`, `model`, then `rounds`, `outcome`, `duration_ms`, the `prompt.*` breakdown and the `context.*` census below when it ends |
| `turn.round` | `round` (one-based), `conversation_id`, `tools`, `outcome`, and the four token counts the provider reported |
| `llm.call` | `purpose`, `provider`, `model`, `conversation_id`, `provider_request_id`, the four `gen_ai.usage.*` token counts below, plus `round` and `outcome` for a round's own call |
| `tool.call` | `tool`, `runner` (`client` or `server`), `conversation_id`, `outcome` |
| `recall.lookup` | `conversation_id` |

`connection_id` reads `unset` when routing fell through to the statically
configured primary client, because there is no configured connection to name.
`provider` and `model` still say what ran: the primary is built from `[llm]`
through the same resolver, so a `[llm]`-only install - the ordinary desktop
shape - is attributed like any other. A sentinel rather than an empty field: an
empty field renders as nothing and reads as absent.

A turn also spends provider time outside its rounds - naming a new
conversation, summarising to fit the window, sorting a large tool fleet into
namespaces, the wind-down when the round budget runs out. Each of those is an
`llm.call` span too, hung from the turn rather than from a round, and the
`purpose` field says which it is. Without them a turn whose four minutes went
into compaction would decompose into a gap.

Each is measured at the provider call itself rather than around the helper that
makes it, because several of those helpers can return without calling at all -
an empty compaction range, a recovery ladder that freed enough at its first
step. A measurement taken at the helper's boundary would record a call that
never happened, and the histogram's count is read as how many calls there
were.

### What a call cost

Every `llm.call` span carries the token counts the provider reported for that
call, under the OpenTelemetry GenAI semantic-convention names:

| attribute | what it counts |
|---|---|
| `gen_ai.usage.input_tokens` | prompt tokens |
| `gen_ai.usage.output_tokens` | completion tokens |
| `gen_ai.usage.cache_creation.input_tokens` | input tokens written into the provider's prompt cache |
| `gen_ai.usage.cache_read.input_tokens` | input tokens served from that cache |

The convention's names rather than this project's, so a backend that
special-cases GenAI attributes renders a provider call natively. The
`llm.tokens.*` metrics keep their own names: renaming a metric breaks the
queries already reading it, so the convention is adopted where it is new.

The counts are on the span and not on the metrics for a reason. A metric
labelled by conversation would answer "what did this turn cost", and a
conversation id is unbounded - the registry caps a metric at 64 label sets with
no eviction, so one such label burns that dimension until the process restarts.
A span attribute has no cardinality budget, and the span already carries the
conversation id and the round.

**An absent count is absent, never zero.** A provider that reports no usage
leaves all four attributes off the span, and a provider that does no prompt
caching leaves the two cache attributes off. A recorded zero would sum into a
total that reads as a real measurement, which is the distinction
`llm.tokens.unreported` exists to keep. A failed call reports nothing and so
carries none of the four.

### What filled the input

`gen_ai.usage.input_tokens` says a round cost 40k. It cannot say whether that
was the transcript, the pinned notes or eighty tool schemas, and each of those
has a different fix: compact the transcript, prune the notes, drop a server's
tools, narrow the recall. So the turn span also carries a breakdown of the
prompt the turn opened with, part by part.

| field | what it counts |
|---|---|
| `prompt.system_tokens` | the cached system instruction - standing guidance, personality, client context, machine topology, the tool-listing note, any one-turn refinement - and the ambient `[Now]` line |
| `prompt.summary_tokens` | the `[Summary of earlier conversation]` block |
| `prompt.turn_index_tokens` | the `[Earlier turns]` index: one line per turn before this one |
| `prompt.current_task_tokens` | the `[Current task]` anchor |
| `prompt.working_state_tokens` | the `[Working state]` line |
| `prompt.plan_tokens` | the `[Plan]` block |
| `prompt.pinned_tokens` | the `[Pinned]` block, notes and attached knowledge together |
| `prompt.scratchpad_tokens` | the `[Scratchpad]` key index |
| `prompt.recall_tokens` | the `[Recall]` block |
| `prompt.transcript_tokens` | the conversation history this prompt carries |
| `prompt.tool_schema_tokens` | the tool schemas, sent out of band in the request's `tools` array |
| `prompt.total_tokens` | every part above, summed |
| `prompt.tool_count` | how many tools those schemas describe |
| `prompt.tool_schema_tokens_max` | the tool-schema cost of the largest block any of the turn's rounds sent |
| `prompt.tool_count_max` | how many tools that largest block carried |

Every part sums to `prompt.total_tokens` and nothing is left over, so a figure
that looks wrong can be checked rather than guessed at. `prompt.tool_count` is
the one figure that is not a token count, and it is named as the count it is:
the tool bill is a pair, because a schema cost without a tool count says
nothing about whether to drop a server.

**The tool figures come in three places, because one of them cannot answer the
question.** The parts above are the prompt the turn *opened* with. Within a turn
the advertised tool set only grows - a tool search activates what it found, and
nothing retires until the bound forces it - so the opening figure is the floor
of that growth. `prompt.tool_schema_tokens_max` and `prompt.tool_count_max` are
its ceiling, taken from the round whose schemas cost the most, and the pair
travels from that one round rather than being two independent maxima. Every
round's own pair is on its own `turn.round` span:

| field | span | what it counts |
|---|---|---|
| `prompt.tool_count` | `turn.round` | how many tools this round advertised |
| `prompt.tool_schema_tokens` | `turn.round` | what this round's schemas cost |

A round that stopped before it assembled a prompt carries neither, which is an
unmeasured round rather than one that advertised nothing.

**These are estimates, and they will not sum exactly to the provider's own
input count.** They come from this daemon's estimator - the same one the
context budget's pre-flight check reads, so the number an operator sees is the
number the shrink decision was taken on - while the provider tokenises its own
way. The provider's reported count stays the authority; this says where that
number went.

**Zero is a measurement here, unlike a provider's count.** A turn with nothing
pinned reports `prompt.pinned_tokens=0` rather than leaving the field off. That
is the opposite of the rule above for `gen_ai.usage.*`, and deliberately: a
provider can decline to say, so an absent count there is unknowable, while the
assembler always knows whether it emitted a block. An absent field here could
only mean the part went unmeasured, which is the one thing a reader must be
able to tell from an empty block.

**The figures are the prompt the turn opened with, not the one it ended with.**
A turn assembles a prompt per round and each round adds its own tool traffic,
so a last-round figure would report the tail of a tool loop under a name that
reads as the turn's own. What this answers is the standing bill - what the turn
cost before it did anything. A turn cancelled before its first round assembles
no prompt and carries none of these fields.

### What eviction did not reach

The breakdown above says what the turn opened with. This says what it was still
carrying when it ended, and it is the figure the working-memory work is measured
against: the turn span carries a census of its own tool traffic, taken from the
stored bytes at the end of the last completed round.

| field | what it counts |
|---|---|
| `context.tool_bytes` | stored bytes of every tool result **in the window**, as of the end of the last completed round |
| `context.tool_bytes_carried` | what the round actually reads for those same results |
| `context.tool_bytes_evicted` | of the difference, the bytes a compaction pointer saved |
| `context.tool_bytes_reduced` | of the difference, the bytes a recall reduction saved |
| `context.tool_bytes_shrunk_elsewhere` | of the difference, what some other mechanism saved |
| `context.tool_carried_pct` | `carried` as whole percent of `tool_bytes` |

**The window, not the conversation.** A conversation holds every message it ever
had, and the store loads all of them. Counting those would make this figure
track how old a conversation is rather than what a turn is carrying.

The census runs at the end of each tool round, so the round's own results are
already in the window and no prompt has carried them yet. They have no
projection entry, so they count as fully carried and push
`context.tool_carried_pct` up - the figure is what the turn is holding at that
moment, not what the last prompt sent.

**Every figure is stored-minus-read, so a mechanism gets credit for what it
actually saved.** A reduction saves the ENVELOPE: a 40 KB entry whose
scaffolding was 1 KB counts 1 KB as reduced and leaves 39 KB in `carried`.
Counting the stored size instead would report a result the prompt still carries
whole as fully reached.

**Evicted, reduced and shrunk-elsewhere are three different things.** An evicted
result left the turn's view and a pointer stands in its place. A reduced one is
a recall - a knowledge-base or scratchpad read - that kept the entry text and
lost only its envelope, because nothing else holds what it carried and a pointer
would send the model back to free recall. The third is the oversized-head notice
and overflow recovery, which shrink a result without leaving a prefix this
census owns; their work is reported rather than folded into a bucket that did
not do it.

`context.tool_carried_pct` carries no `_bytes` in its name on purpose: it is the
one figure here that is not a byte count, and a unit check by substring would
pass it regardless.

Per model needs no label: the turn span already carries `model`, so the census
groups by it like every other field on that span. The four figures below the
total reach the metrics facade as `llm.context.tool.bytes`, labelled `state`
with `carried`, `evicted`, `reduced` and `shrunk_elsewhere`, against the
`llm.context.measured` denominator. The four sum to the total.

A turn that answered without calling a tool carries none of these fields, which
is a turn that held no tool bytes rather than one that measured none.

### The two lines an operator greps

One per round, and one per turn. Both carry fields, never an interpolated
sentence, so a backend can group and sort by any of them.

```text
INFO turn{...}:turn.round{round=2 input_tokens=8120 output_tokens=96 tools="web_fetch" outcome="tools_called"}: round finished round=2 duration_ms=1840 outcome="tools_called" input_tokens=8120 output_tokens=96 cache_write_tokens=- cache_read_tokens=-
INFO turn{... rounds=17 outcome="answered" duration_ms=241033}: turn finished duration_ms=241033 model="claude-example" rounds=17 input_tokens=214800 output_tokens=3311 cache_write_tokens=- cache_read_tokens=- outcome="answered"
```

Each line carries its own fields *and* the fields of every span above it, so a
round line names its round and its turn without either being threaded by hand.

A `-` means the provider did not report that count. It is never `0`, because a
zero and an absence are different facts and nothing downstream could tell them
apart afterwards.

### Span timing, and how much of it

Both binaries turn span-close events on, so every closing span writes a line
carrying how long it was open:

```text
INFO turn{request_id="4bf92f35-..."}:turn.round{round=2}:llm.call{round=2 provider="anthropic" model="claude-example" outcome="ok"}: close time.busy=1.81s time.idle=74.2µs
```

That is what makes turn timing readable in `journalctl` or `kubectl logs`,
where there is no trace backend to open.

A turn may run up to 200 rounds, and each round closes its own span plus one
for the provider call and one per tool, so a pathological turn writes several
hundred close lines. **That is a deliberate trade.** The close line *is* the
per-round duration, which is the one thing the report above needs; a round
already writes several lines of its own, so the close line is a small addition;
and putting these spans below INFO would remove the round from the trace in
every shipped deployment, which is the opposite of what they are for.

### Nothing here is content

Ids, counts, durations, outcomes, and the names of tools, models and providers.
No prompt, no assembled context, no tool argument, no search query, no model
reply - on a log line or on a span field. The span half matters more than it
looks: nothing prints a span field unless an event fires inside the span, so a
captured argument is invisible locally and still exports over OTLP. Every span
in the turn path is therefore built by hand. There is no `#[instrument]`, which
would capture each argument by default.

`crates/core/tests/turn_telemetry.rs` holds that line from both directions: it
reads span fields back in process, and it reads console text, because neither
can see what the other does.

A third thing that only an in-process reader can see is **where a span ends**.
A span's extent is decided by where its last handle drops, and the code that
measures the same work sits somewhere else; when the two part company the
histogram reports one number and the exported trace draws another, and only the
trace is wrong - in the direction that blames whatever the span is named after.
Nothing on the console shows it. `each_instrumented_call_closes_its_span_with_its_own_measurement`
asserts each span closes before the work that follows its measurement, as an
order rather than as a duration, because the two elapsed times overlap and a
timing assertion would pass whichever way the code was written.

One field in the turn path is written by the model: the tool name. It goes
through `adelie_telemetry::Safe`, which caps it and replaces the characters
that change what a reader sees. Without that, a newline in a name produces what
reads as a second genuine log line - its own timestamp column, its own level -
and an ANSI escape survives even with colour off.
`a_model_chosen_tool_name_cannot_forge_a_log_line` holds it.

## Metrics

Call sites record through `adelie_telemetry::metrics` and never through an
opentelemetry meter directly. A direct call would make the crate that records
depend on opentelemetry whether or not export is on.

```rust
use adelie_telemetry::metrics::{self, Label};

metrics::increment("llm.requests", &[Label::new("provider", "example")]);
metrics::record_duration("dreaming.scan.duration", elapsed, &[]);
```

**Label values are names, not content.** A prompt or a tool argument used as a
label would be both a disclosure and an unbounded memory leak in a process that
runs for weeks. One metric may carry 64 distinct label sets; past that, further
label sets fold into one series labelled `cardinality=other`.

The registry runs whether or not export is on. It keeps counters and
fixed-bucket histograms in process and writes a summary every 10 minutes, so a
desktop install running a default-feature build gets real numbers in its
journal. Each summary reports the window that just closed beside a running
total, because on a pod that has run for a month a cumulative number is
dominated by history and stops moving.

```text
INFO metrics summary window_seconds=600 uptime_seconds=8400 counters=1 histograms=2
INFO duration metric="dreaming.scan.duration" labels=outcome=ok window_count=1 window_p95_ms=2500 total_count=14 total_p95_ms=5000
```

Durations go into fixed-bucket histograms rather than a count and a sum,
because a mean hides the tail: "the average turn took 3 seconds" and "one turn
in twenty took four minutes" are the same mean, and only the second is the
report a user files.

### What the turn path records

| metric | kind | labels |
|---|---|---|
| `turn.duration` | histogram | `outcome` |
| `turn.rounds` | counter | `outcome` |
| `turn.round.duration` | histogram | `outcome` |
| `llm.call.duration` | histogram | `provider`, `model`, `purpose`, and `outcome` for a round's own call |
| `tool.call.duration` | histogram | `tool`, `outcome` |
| `llm.tokens.input` | counter | `provider`, `model` |
| `llm.tokens.output` | counter | `provider`, `model` |
| `llm.tokens.cache_write` | counter | `provider`, `model` |
| `llm.tokens.cache_read` | counter | `provider`, `model` |
| `llm.tokens.unreported` | counter | `provider`, `count` |
| `llm.prompt.part.tokens` | counter | `part` |
| `llm.prompt.tools` | counter | none |
| `llm.prompt.measured` | counter | none |
| `llm.prompt.tool.tokens` | counter | `server` |
| `llm.prompt.round.tools` | counter | none |
| `llm.prompt.round.measured` | counter | none |
| `dreaming.scan.duration` | histogram | `outcome` |
| `dreaming.facts.written` | counter | none |
| `consolidation.scan.duration` | histogram | `outcome` |

Tokens are recorded **per round**, not per turn, and the turn's total is the
sum of its rounds. The useful question is not what a turn cost but which round
blew up: a turn that re-sends a growing transcript ten times has a very
different shape from one that answers immediately, and only per-round numbers
show it.

All four counts are recorded separately. The two cache counts are the whole
cost story on a caching provider, where a cache read costs a fraction of a
fresh input token, so reporting input alone makes a well-cached turn look
identical to a cold one.

**A count the provider did not report is not zero.** It is skipped, and
`llm.tokens.unreported` is incremented instead with a `count` label naming
which one. So a total that looks low can be checked against how many calls said
nothing, which a silent `0` would make impossible.

The three `llm.prompt.*` counters are the breakdown above, over time. `part` is
one of ten names from a closed set, and no conversation, user, model or
provider axis is added to any of them: the operator question they answer is
what fraction of the input one part is spending, which needs no such axis, and
one of them would burn the label budget. `llm.prompt.measured` counts the
prompts measured and is the denominator that turns the other two back into a
per-turn mean.

`llm.prompt.tool.tokens` is the tool schema bill on a per-connection axis, and
it exists because the aggregate cannot be acted on: an operator reading 23.7k
spent on tools has no way to tell which server to drop. `server` is the
connection's own label - `daemon:built-ins`, `client:built-ins`, and
`daemon:<name>` for each configured MCP server - so it is bounded by the
operator's configuration the way `model` and `provider` are, and unlike a
conversation id. It is recorded **per round**, so a set that grows within a turn
shows the growth; `llm.prompt.round.tools` and `llm.prompt.round.measured` are
its count and its denominator on the same per-round basis.

They are counters rather than histograms because the facade's only histogram is
a *duration* histogram - fixed millisecond buckets, a millisecond sum, and an
export that names its values `ms`. Token counts put through it would be
labelled as milliseconds everywhere they surfaced, so they accumulate the way
`llm.tokens.input` already does instead.

Every `outcome` and `purpose` label is an enum rendering to a `&'static str`,
so an unbounded value cannot be passed: it has the wrong lifetime. `provider`
and `model` come from operator configuration.

`tool` is the one label whose value the **model** writes, and it is bounded at
the call site rather than by its type: a name the daemon's own tool list does
not contain is recorded as `unknown`. Without that, sixty-four invented names - about
sixty-four rounds of one conversation, and reachable by prompt injection - fill
this metric's label budget, which has no eviction, and every real tool
afterwards folds into `cardinality=other` until the process restarts. `runner`
is on the `tool.call` span rather than on the metric for the same budget
reason: a span field has no series key to spend.

The daemon raises that budget from the facade's default of 64 to 512, because
it fronts a fleet of MCP servers and `tool.call.duration` spends one label set
per (tool, outcome) pair. A conversation id, a user id or a request id is never
a label.

## Export to a collector

Off by default. With the feature off, no opentelemetry crate is resolved at
all, so `cargo install` costs a desktop user nothing extra - no crates, no
native code, no C toolchain. `scripts/no-opentelemetry.sh` holds that line in
the gate.

```sh
cargo build -p desktop-assistant-daemon --features otel
cargo build -p desktop-assistant-dbus-bridge --features otel
```

The daemon's feature turns on `desktop-assistant-core`'s own `otel` feature,
which is what gives a turn span the trace id its request id spells. A build
without it still mints and carries the id everywhere described above; it simply
exports no span.

With the feature on, the OTLP layers are added *beside* the console layer,
never in place of it, so an exporting build still prints locally. The feature
compiles the TLS backend, which needs a C compiler and an assembler
(`build-essential` on Debian, `gcc` plus `binutils` elsewhere). It does not
need `cmake`.

### Configuration

Everything comes from the standard `OTEL_*` environment variables. There are no
CLI flags and no Adelie-specific variables. Nothing is passed to the exporter
builders in code, so every variable below reaches them.

| Variable | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Endpoint for all three signals. The signal's path (`/v1/traces` and so on) is appended. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Endpoint for traces, used exactly as written, so it must include the path. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | Endpoint for metrics. Same rule. |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Endpoint for log records. Same rule. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf`, for all three. |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces. Per-signal forms exist for metrics and logs too. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers for all three, as `key=value,key=value`. Per-signal forms exist. |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | Export timeout in milliseconds. Per-signal forms exist. |
| `OTEL_EXPORTER_OTLP_COMPRESSION` | `gzip` or `zstd`. Per-signal forms exist. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes, as `key=value,key=value`. |

A per-signal variable beats the generic one.

**There is no run-time switch that turns export off.** The specification
defines `OTEL_SDK_DISABLED` and the per-signal `OTEL_TRACES_EXPORTER=none`
form, and neither `adelie-telemetry` at the pinned revision nor
`opentelemetry_sdk` 0.32 reads them (`adelie-ai/adelie-telemetry#4`). A build
with the `otel` feature always tries to export; a build without it never does.
Ship the default-feature binary where export is not wanted.

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.example.com:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  desktop-assistant-daemon
```

Each binary reports its own `service.name`: `adele-daemon` and
`adele-dbus-bridge`.

**`OTEL_SERVICE_NAME` does not change it.** The name is passed to the SDK's
resource builder in code, and a value supplied there wins over the variable, so
the variable is read and then overwritten. Verified in a running pod: with
`OTEL_SERVICE_NAME` set to another value, every record still arrived as
`service.name=adele-daemon`. Do not set it, and do not use it to tell two
deployments apart - carry `OTEL_RESOURCE_ATTRIBUTES` for that, which the same
builder merges rather than replaces. Tracked in `adelie-ai/adelie-telemetry#11`.

### Choosing a transport

Two transports are compiled in and `OTEL_EXPORTER_OTLP_PROTOCOL` selects one at
run time. A third value the specification defines, `http/json`, is **not**
compiled in. Asking for it does not fail at startup; it fails every export at
run time instead, so check this value first when nothing arrives.

`http/protobuf` (port 4318) is the safer default. It uses a blocking HTTP
client on the exporter's own thread.

`grpc` (port 4317) needs a running Tokio runtime. Both binaries here are
`#[tokio::main]` and install telemetry inside the runtime, so both transports
work in both.

**Both transports read the operating system trust store**, and neither bundles
a root set of its own. Two things follow:

- A container image that exports over HTTPS needs a CA bundle
  (`ca-certificates`) whichever transport it uses, or every export fails on an
  unknown issuer. Both images this repo ships already install it.
- A private certificate authority installed on the host works with no code
  change, again on either transport.

A bundled root set was tried and removed: it is always one CA rotation away
from rejecting a valid certificate, and fixing that needs an upstream release
and a rebuild rather than an image update.

### When export does not work

A wrong value in the environment costs the process its export and nothing else.
The console layer is installed either way and the metrics summary keeps
running. A typo in one variable must not be able to silence a process.

There are two distinct failures, and they look different in the log.

**The pipeline could not be built** - for example an `https` endpoint in a build
whose TLS backend was compiled out. `init` writes one ERROR line naming the
cause together with the `OTEL_*` variables that were set. Header values are
never printed, because they routinely carry an API key.

**The pipeline was built and each export fails** - an unreachable collector, a
malformed endpoint, or `http/json`. This is *not* caught at startup. Each batch
fails and the SDK writes its own line, which names the transport and little
else:

```text
ERROR opentelemetry_sdk: name="BatchLogProcessor.ExportError" error="Operation failed: HTTP export failed: network error"
```

So a malformed `OTEL_EXPORTER_OTLP_ENDPOINT` and a collector that is simply
down are indistinguishable from the log alone. Re-read the variables before
looking at the network.

### Shutdown

Telemetry is flushed and shut down when the guard drops at the end of `main`,
in the order traces, metrics, logs, with one final metrics summary. The batch
exporters buffer, and a process that exits without that flush loses whatever
was still buffered - usually the part worth having, because a crash is what was
being investigated. Shutdown is capped at 5 seconds.

**Kubernetes:** set `terminationGracePeriodSeconds` to at least 30 in any
deployment that runs with `otel` on, so the pod has room to flush telemetry and
finish whatever else it was doing before SIGKILL arrives.

## The gate

`just check` covers both configurations:

| step | what it holds |
|---|---|
| `just no-otel` | a default build resolves no opentelemetry crate |
| `just lint-otel` | the exporting path compiles, warnings as errors |
| `just doc-otel` | the exporting path's documentation resolves |
| `just test-otel` | the exporting path's tests run |

The three `-otel` steps exist for the same reason `lint-sqlite` and
`lint-mcp-host` do: the feature is off by default and nothing in a workspace
build turns it on, so the workspace steps compile none of that code. A change
that builds with default features can still fail with `otel` on.
