# Bedrock Connector Design

Crate: `desktop-assistant-llm-bedrock`

## Purpose and scope

One connector reaches every model AWS Bedrock offers. Bedrock serves those
models through several different APIs, and no single API reaches all of them.
The connector speaks each API it needs and hides that choice from the user. A
person configures one Bedrock connection and picks a model. They never pick an
API.

The connector owns model discovery, backend selection, request translation, and
the cross-cutting concerns of timeout and cache policy. It does not own
provider-independent behaviour that belongs to the daemon: purpose binding,
context budgeting, retry, and tool dispatch.

## Why several backends

Each Bedrock API surface reaches models the others cannot. That is the reason
for the split. Capability differences between the APIs are real, but they are
secondary: they decide which backend serves a request, not whether the backend
exists at all.

| Backend | SDK operation | Endpoint | Reaches |
|---|---|---|---|
| Converse | `Converse`, `ConverseStream` | `bedrock-runtime` | Text chat, broadly: Anthropic, Amazon Nova, Meta, Mistral, Cohere, DeepSeek, GLM. Mostly through cross-region inference profiles. |
| Responses | Responses API | `bedrock-mantle` | The `openai.gpt-5.6` family. No other API reaches these models. |
| Invoke | `InvokeModel`, `InvokeModelWithResponseStream` | `bedrock-runtime` | Everything Converse refuses: embeddings, image and video generation, reranking, and any model that rejects a Converse request. |

Converse is a text-and-chat API only. Embedding models, image generation models
and rerankers are not addressable through it. The connector already calls
`InvokeModel` for embeddings, so the Invoke path exists today. The backend work
generalises that path; it does not introduce it.

Responses runs on a different endpoint from the other two. It needs its own SDK
client, not a second serialization over the shared one.

## Architecture

```
                    BedrockConnector  (implements LlmClient)
                    - model aggregation and de-duplication
                    - backend selection
                    - capability composition
                    - cache policy
                             |
        +--------------------+--------------------+
        |                    |                    |
   ConverseBackend      InvokeBackend      ResponsesBackend
        |                    |                    |
   bedrock-runtime      bedrock-runtime      bedrock-mantle
```

## The backend trait

```rust
/// One Bedrock API surface.
///
/// An implementor translates the connector's request into its own API shape
/// and translates the response back. It does not retry, and it does not
/// decide which models the user sees.
#[async_trait]
pub trait BedrockBackend: Send + Sync {
    /// Short API name, for logs, notices and model annotation.
    fn api_name(&self) -> &'static str;

    /// Whether this backend can serve the model at all.
    ///
    /// This is the routing primitive. A backend that cannot serve a model
    /// never receives a request for it, and never contributes it to the
    /// catalogue.
    fn can_serve(&self, model_id: &str) -> bool;

    /// The models this backend reaches, with any listing notices.
    async fn list_models(&self) -> Result<ModelListingReport, CoreError>;

    /// What this API surface supports for one model.
    ///
    /// Why the model id: support varies per model inside a single API.
    /// Converse accepts `cachePoint` for Anthropic and Nova models, and
    /// rejects it for Meta, Mistral and Cohere models.
    fn capabilities(&self, model_id: &str) -> BackendApiCapabilities;

    /// Stream a completion.
    async fn stream_completion(
        &self,
        request: BedrockRequest,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError>;
}
```

The trait uses `#[async_trait]`. It does not use `-> impl Future` in return
position, because `BedrockConnector` holds `Vec<Arc<dyn BedrockBackend>>` and
return-position `impl Trait` is not dyn-compatible.

`BedrockBackend` does not extend `LlmClient`. `BedrockConnector` implements
`LlmClient`, and the backends sit behind it. This keeps `core` unaware of
Bedrock internals, and keeps each backend responsible for one API.

`BedrockRequest` carries domain messages and tool definitions, not one API's
wire types, because the Responses and Invoke surfaces do not accept Converse
messages. Each backend translates. Two things stay above the boundary and are
written once: the connector builds the tool-name bijection, so every backend
spells a tool the same way in the request and in the message history, and the
connector races nothing against the cancellation token itself - the token
travels on the request and each backend races its own network waits.

The cache checkpoint is answered in three parts, by whoever owns each part.
The connector answers the operator's policy. The backend's `capabilities`
answers whether the surface accepts a checkpoint for that model. The backend
also holds what it has learned at run time, because a refusal belongs to one
(surface, model) pair: a model can accept Anthropic `cache_control` through
Invoke and reject `cachePoint` through Converse.

## Model aggregation

The connector queries every backend and merges the results into one catalogue.

**A model appears once.** The user picks a model, not a model-and-API pair. When
several backends serve the same model id, the entry is de-duplicated, and it
records the full set of backends that serve it. It does not keep only the
richest one. The discarded backends carry capabilities that the kept one does
not, and dropping them would hide capabilities the product has.

**A failing backend degrades the catalogue. It does not empty it.** Each backend
returns a `ModelListingReport`, which carries `notices` beside `models`. The
connector concatenates both. A backend that fails contributes a notice that
names what is missing, and the models from the other backends still reach the
picker. The daemon puts those notices on every `ListAvailableModels` row for the
connection, so a client can explain a partial catalogue instead of showing an
empty one.

Inside the Converse backend, discovery makes two control-plane calls in
parallel:

| Call | IAM action | Contributes |
|---|---|---|
| `ListFoundationModels` | `bedrock:ListFoundationModels` | Foundation models with on-demand throughput |
| `ListInferenceProfiles`, `type=SYSTEM_DEFINED` | `bedrock:ListInferenceProfiles` | Cross-region inference profiles (`us.anthropic.claude-...`) |
| `ListInferenceProfiles`, `type=APPLICATION` | `bedrock:ListInferenceProfiles` | Application inference profiles the account created for its own cost attribution |

**The type filter is mandatory, not a refinement.** An unfiltered
`ListInferenceProfiles` returns `SYSTEM_DEFINED` profiles and nothing else. The
API reference does not state that default, which is exactly why it reads as "no
filter means everything". Verified against a live account on 2026-08-03: the
unfiltered call and `type=SYSTEM_DEFINED` returned the same 63 profiles, and
`type=APPLICATION` is the only way to ask for the rest. So the connector makes
both calls by name. A connector that makes one unfiltered call sees no
application profile at all, and every model behind one silently loses its
reasoning, its prompt cache, its context window and its streaming path - which
is the defect this resolution exists to remove.

**Both calls follow `nextToken`.** A page missed is a profile missed, and a
profile missed is a model whose capabilities are lost, so the listing is not one
page deep. A live account already returns 63 system-defined profiles per region
before it creates any of its own. The connector stops after 50 pages per type,
which is far more than an account can hold and exists so a repeating token
cannot spin the listing forever; hitting it adds a notice rather than passing a
truncated catalogue off as complete.

A failure of `ListFoundationModels` fails that backend's listing. A failure of
`ListInferenceProfiles` does not: the listing degrades to on-demand foundation
models and adds a partial-catalogue notice that names
`bedrock:ListInferenceProfiles`. Many IAM policies grant only the first call.
The two profile calls are reported apart, because what each one hides is
different - the system-defined call hides models no other route reaches, and the
application call hides the profiles an account made for itself. When both fail,
one notice is emitted, because the second would repeat the first without adding
a fact.

**Foundation models without on-demand support are filtered out.** Their bare ids
are not callable. Selecting one fails at invocation time with a
`ValidationException`. In a current AWS account that filter removes nearly every
modern chat model, because those models are reachable only through an inference
profile. Every backend that lists foundation models applies the same filter. A
backend that skips it advertises models that cannot be called.

**A profile carries its base model's capabilities.** `ListInferenceProfiles`
reports no modalities, and a profile is only a route to a foundation model, so
the profile entry reuses the modality metadata `ListFoundationModels` returned
in the same call, keyed by model id.

**One id, read one way, by both sides.** Extended thinking, prompt caching, the
streaming-with-tools deny list and the context window all read the base model
id, and so does the request builder at dispatch time, because a turn arrives
carrying a model id and nothing else. One function answers for every one of
them, on both paths. A capability recovered from an input the dispatch path
does not have is a capability the picker offers and the request builder
discards - the defect the reasoning work exists to remove.

That function answers in two steps.

1. **The register.** A successful `ListInferenceProfiles` records what each
   profile routes to, taken from the foundation-model ARNs in the profile's
   `models`. A cross-region profile lists one ARN per region; they name one
   model, and that model is the answer. No ARNs, an ARN shape the connector
   cannot read, or two ARNs naming different models all record nothing, because
   a wrong base model attaches another model's capabilities to the profile.
   Each profile is recorded under **both** forms AWS accepts as a `modelId` -
   the bare profile id and the profile ARN - because a turn carries one of them
   and the gates see only what it carried. Which form a turn carries for an
   application profile is not settled here: AWS documents that a cross-region
   profile takes either, and is less explicit for an application profile.
   Registering both makes the question moot rather than leaving it to be
   discovered on a turn.
2. **The prefix rule.** An id the register does not hold is the profile id
   minus its geography prefix. The recognised prefixes are `global.`, `us.`,
   `eu.`, `apac.`, `ap.`, `au.`, `jp.` and `us-gov.`, held in one list and
   pinned by a test. Every entry ends at the separator, so no entry can swallow
   another and the order does not matter. It is an allowlist rather than "drop
   the first dotted segment", because model ids carry dots of their own - that
   rule would strip `openai.gpt-5.6-sol` to `gpt-5.6-sol`.

A bare id is recorded only where the ARN adds something the prefix rule does not
already get right; an ARN is always recorded, because no rule reduces one.
Entries are never removed, so what the register holds is every profile any
connection in this process has listed - a deleted profile is undispatchable at
AWS, so a stale entry cannot grant a capability to a live turn, and a profile
whose route changes is corrected by the next listing.

**The register is process-wide, and that is the load-bearing part.** The two
sides that must agree do not share a client object: the daemon lists models
through the registry's per-connection client and dispatches turns through a
second client built for the interactive purpose. A register held per client
would let the picker see a capability the request builder cannot.

It is keyed by profile id, and narrowing that key was a choice. A daemon can hold
Bedrock connections to several accounts, and a connection-identity key - base
url, aws profile, credential, all shared by both clients because both are built
from one resolved connection - needs no network call. It is not used, because it
would make the resolver a method and change a public signature to close a
collision nobody can demonstrate: an application profile id is a
twelve-character service-generated identifier, and a system-defined id names the
same foundation model in every account. An account-scoped key is the one that is
genuinely unavailable, since a turn carries a model id and a credential and
never an account id.

**Two ids this fixes.** An `APPLICATION` profile has a generated id that no rule
reduces; accounts create them to attribute cost per team or per feature, and
before the register such an account silently got a lesser product on every turn.
A geography newer than the prefix list has the same shape. Both are ordinary
listed profiles, so both are registered.

**One id it does not fix, on purpose.** A turn can dispatch a model the listing
never returned - a per-turn model override for a profile this connection does
not list, or any id the account does not return. The register has no entry, so
the prefix rule alone decides, which is the conservative answer: no thinking
budget (reported at `warn!`, never silently dropped), no cache checkpoint, no
context window, and the streaming path with its runtime fallback. The connector
does **not** refresh the listing to answer, because that would put a
control-plane call, its IAM failure modes and its latency on the turn path. The
boundary is the miss, not the model: the same id resolves fully as soon as any
listing in this process has returned it.

**The register is warmed at startup.** The connection's `warmup` lists the
models once, so a configured model that is an application profile is usually
registered before the first turn arrives - rather than only after whichever
client first opens the model picker. It warms the listing cache at the same
time.

Usually, not always: the registry spawns the warm detached and does not wait for
it, so a turn dispatched during startup can beat it, and a listing that fails
registers nothing. Both land on the conservative answer above, which is why
neither is retried. A warm that fails, or that comes back degraded, is a `warn!`
line naming what the connection lost, so a connection nobody uses cannot fail
startup but a broken one is not silent either.

**A degraded warm is not cached.** `ListInferenceProfiles` can fail while
`ListFoundationModels` succeeds, and the report is then a real listing with a
notice on it. Caching that from the boot warm would hold a partial catalogue for
the whole hour, decided at the moment transient failures are likeliest - several
connections firing control-plane calls at once, or a session credential not yet
refreshed - and with nobody watching. Whatever the call did register is kept; the
cache stays empty, so the next caller makes a real call, sees the notice, and can
press refresh.

Only a profile whose base model this account did not list falls back to a
family guess from the id, which reports vision from the family and treats the
model as generative - an embedding model is reachable by its bare on-demand id,
so a profile for one resolves through the listing or does not exist. Deriving
vision from a curated id list on this path is what made a new vision-capable
model report `vision: false` until somebody edited the list, while the same
model's foundation entry reported it correctly.

Notices are cached with the models, so a cache hit inside the one-hour TTL still
reports the degradation. An explicit refresh re-issues the calls and always
returns a report, so a client can tell a reload that found nothing new from a
reload that failed.

## Backend selection

Selection runs in two steps, in this order.

1. **Reach.** Keep the backends whose `can_serve` accepts the model. If none
   does, the model is not in the catalogue, and no request for it can arrive.
2. **Requested features.** Among those, choose a backend whose
   `BackendApiCapabilities` satisfy what the request asks for: cache control
   when the cache policy is on, vision when the messages carry images, and so
   on.

When more than one backend qualifies, prefer the backend that already served
this conversation, then the first one listed. Stability is worth more than a
small capability gain, because a change of backend inside a conversation
invalidates the prompt cache.

When no single backend satisfies every requested feature, the connector fails
and names the conflict: the model, the features asked for, and the backend that
provides each one. It does not drop a feature and continue. A partial capability
is the user's decision, not the connector's.

## Capability composition

A capability answer belongs to a **(connection, model)** pair. The same model
behaves differently on different connections, and one connection serves models
with different support. The connector composes that answer from three inputs:

- **Model capabilities.** What the model was trained for: vision, reasoning,
  tools, and its kind. This is the existing `ModelCapabilities`.

  `reasoning` answers "can this connector configure reasoning for this model",
  not "does this model reason". The two differ, and the difference is
  load-bearing: DeepSeek R1 reasons on every request and returns the trace, and
  Bedrock's Converse contract for it carries no reasoning field, so an effort
  set against it changes nothing. Only Anthropic Claude 3.7 and the 4.x line
  and later take a thinking budget, through
  `additionalModelRequestFields.thinking`. The capability record and the
  request builder read one function, so a client cannot be shown a control the
  request path will discard. A budget that arrives for a model that takes none
  is reported at `warn!` with the model and the budget, and the request goes
  out without it.
- **Backend API capabilities.** What one API surface supports for that model:
  streaming, cache control, tool search, reasoning configuration. This type is
  Bedrock-local. `docs/design/connector-capabilities.md` records why it stays
  out of `core`.
- **Connector capabilities.** What this connector supports at all, whatever the
  model or the API.

For a model that one backend serves, the composed answer is the intersection.
Any input can block a capability. No input can grant a capability that another
denies.

For a model that several backends serve, the composed answer is the **union
across the qualifying backends**, and each capability records which backends
provide it. A capability that only Invoke delivers is genuinely available on
that model. Reporting it as unavailable because Converse lacks it would
under-report what the product can do.

The union carries one obligation. Two capabilities can each be available while
no single backend provides both. The composed answer therefore keeps the
per-capability backend set, so selection can detect an unsatisfiable combination
by name before the request goes out, instead of meeting it as a
`ValidationException` in the middle of a turn.

Each unavailable capability carries a reason, so a client can show a control as
disabled with an explanation instead of letting a person try and fail.
`docs/design/connector-capabilities.md` defines the states and the reasons.

## Prompt caching

Bedrock supports prompt caching. The connector reaches it through whichever
backend serves the request.

**Converse** accepts `cachePoint` blocks in the `system`, `messages` and `tools`
fields, up to four checkpoints per request for Anthropic models, with a
five-minute or one-hour TTL. The response carries `cacheReadInputTokens` and
`cacheWriteInputTokens`. These map onto `TokenUsage`'s
`cache_read_input_tokens` and `cache_creation_input_tokens`.

**Invoke** accepts Anthropic's native `cache_control` markers for Anthropic
models, and `cachePoint` for Nova models.

**Responses** accepts `prompt_cache_breakpoint` on content blocks for the
GPT-5.6 family, and caches automatically for earlier OpenAI models.

Support is per model, not per API. Anthropic Claude 3.5 and later, and Amazon
Nova, accept cache checkpoints. Claude 3, Meta, Mistral, Cohere and DeepSeek do
not, and reject a request that carries one. The connector detects support from
the model id, with the region prefix stripped so an inference profile resolves
to its base model.

An unrecognised model gets no checkpoint. The two errors are not equal: a
checkpoint the model refuses fails the whole turn, while a checkpoint withheld
only costs input tokens. So the connector withholds when it is unsure.

**Where the checkpoint goes.** One checkpoint, after the stable system prefix.
Not on the tool list. Bedrock evaluates checkpoints in the order `tools`, then
`system`, then `messages`, and a change in an earlier section invalidates the
cache for every later section. Our tool list changes inside a conversation as
tool search activates and deactivates namespaces. A checkpoint on `tools` would
therefore invalidate the system cache on every turn the tool list moves, and
cost more than it saves. The Anthropic connector marks the system prefix only,
for this same reason.

**The policy.** `cache_policy` on a Bedrock connection selects how much of the
request is marked:

```rust
pub enum CachePolicy {
    /// Emit no cache checkpoints, whatever the model supports.
    None,
    /// One checkpoint after the stable system prefix. The default.
    SystemPromptOnly,
}
```

The default keeps the behaviour every connection had before the setting
existed. `none` exists because caching is not free: Bedrock bills a cache
**write** above the uncached input rate, and the write pays back only when a
later turn reads the same prefix. A conversation of several turns reads it
every turn and comes out ahead. A workload of many short one-turn
conversations pays the premium every turn and reads it rarely, and `none` is
how that workload stops paying. `none` is also the way to rule caching out of
a misbehaving turn without a code change.

There is no "system prefix and tool list" value. It would be sound only where
the tool list is fixed for a whole conversation, and this daemon's is not: tool
search activates and deactivates namespaces inside a conversation. Bedrock
evaluates checkpoints in the order `tools` -> `system` -> `messages`, so a
checkpoint on `tools` would invalidate the system cache on every turn the list
moves, and would cost more than it saves. Adding the value would ship a setting
that quietly bills more.

The connector decides where checkpoints go. The backend writes them in its own
API's spelling. A backend serving a model without caching support ignores the
policy.

**Recovery when a model refuses a checkpoint.** The support list above is read
from AWS documentation, and that documentation lists only the models absent
from "Models at a glance", so it is a best reading rather than an enumeration.
A model on the list that rejects a checkpoint would otherwise fail every turn.
So the Converse path catches the `ValidationException` that **names the cache
field**, retries the same turn once without the checkpoint, and records the
model so later turns omit it. The recovery is logged at `warn!` with the model
and the provider's message.

Three limits, each deliberate:

- The error must name the cache feature: `cachePoint`, `cache_control`, or the
  prose forms "cache point", "cache checkpoint" and "prompt caching". A
  validation failure that names anything else is returned to the caller
  unchanged. A status code is not evidence. The bare word "caching" is
  deliberately not a marker: matching is a substring test over the whole
  message, and Bedrock quotes the offending schema path, so a tool whose input
  schema has a property named `caching` would be read as a cache refusal.
- The retry is classified only when the request that failed actually carried a
  checkpoint. The retry does not, so the retry can never be read as a second
  refusal, and the memo can never be written on evidence the fallback itself
  produced.
- The memo is written from the refusal, not from the retry's success. A
  fallback that succeeds proves only that the request without the field works,
  which is true whatever the real cause was.

A model whose refusal names none of those markers is not recovered. That turn
fails, and `cache_policy = "none"` is the remedy. The fail-safe direction is
deliberate: a wrongly-disabled cache costs input tokens, and a wrongly-kept
checkpoint costs the whole turn.

A prefix below the model's caching minimum is not this case. AWS states the
inference still succeeds and simply does not cache.

## Cross-cutting concerns

**Retry** is not implemented here. `RetryingLlmClient` in `core` wraps any
`LlmClient` from outside and already applies exponential backoff to retryable
errors. A second retry loop inside the connector would nest the backoffs and
multiply the attempts.

**Timeouts** are connector-level, because they are network concerns rather than
API concerns:

- `connect_timeout` (30s) - the time to establish the connection.
- `event_timeout` (60s) - the time between streaming events before the stream
  counts as stalled.
- `non_streaming_timeout` (600s) - the whole-request budget for the
  non-streaming path. **Not settable per connection yet**, see below.

Three settings, because there are three questions. The first two bound the
streaming path's two phases: the connect race, then each gap between events.
Neither bounds a whole turn - a stream that keeps producing runs as long as it
likes.

The non-streaming path has no intermediate events to time. `Converse` answers
once, after generation is complete, so its bound is necessarily a bound on the
generation, and it gets its own setting rather than borrowing `event_timeout`.
Reusing that one would give a single name two meanings, and a change to stall
detection would move a generation deadline with it. The default is deliberately
generous: this path is mandatory for Llama 3 and 4 with tools, whose one-shot
answers can run for minutes, so the bound is there to catch a hung request and
nothing else.

Each path also races its request against the cancellation token, so a stop ends
the turn rather than waiting the budget out.

**`non_streaming_timeout` cannot be configured yet.** The client accepts the
value, nothing in the daemon sets it, and there is no connection key for it, so
every Bedrock connection runs the 600-second default and no setting changes
that. Issue #1042 carries it through the wire shape and the resolver to join
the other two; until it lands, a model that needs longer than ten minutes for a
one-shot answer has no remedy short of a code change.

**Tool-schema sanitisation** runs above the backend boundary, in the shared
request conversion. Top-level `oneOf`, `anyOf` and `allOf` are removed and a
`type` is added; empty-string keys are dropped from recorded tool inputs; tool
names pass through a per-request bijection and back. Every backend consumes the
sanitised result, so the rules are written once.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AWS_BEDROCK_API_KEY` | Static credentials, `ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]` | Falls back to the AWS credential chain |
| `AWS_PROFILE` | Named AWS profile | - |
| `AWS_REGION` | AWS region | `us-east-1` |

Credentials come from the static key above, or from the standard AWS provider
chain: environment, profile, SSO, instance role.

```toml
[connections.my-bedrock]
type = "bedrock"
region = "us-east-1"
aws_profile = "production"
connect_timeout_secs = 30
stream_timeout_secs = 120
# "system_prompt_only" (the default) or "none". See "Prompt caching".
cache_policy = "system_prompt_only"

# A connection is an endpoint and a credential. The model is chosen per
# purpose, so the same connection can serve a large interactive model and a
# small one for background work.
[purposes.interactive]
connection = "my-bedrock"
model = "us.anthropic.claude-opus-4-1"
```

The table is `[connections.<name>]`, plural. A misspelled table name configures
nothing: `DaemonConfig` accepts unknown keys, so the whole block is discarded.
The daemon names every discarded key at `warn!` on load, which is the only
signal that a block did nothing.

`cache_policy` is a file setting. The connection commands on the API carry no
field for it, so a client cannot read or write it, and an edit made through a
client leaves the configured value in place rather than clearing it.

## IAM

| Permission | Purpose |
|---|---|
| `bedrock:InvokeModel` | Completions, embeddings, non-chat modalities |
| `bedrock:InvokeModelWithResponseStream` | Streaming completions |
| `bedrock:ListFoundationModels` | Model listing |
| `bedrock:ListInferenceProfiles` | Cross-region model listing |

Chat only, minimum viable:

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "arn:aws:bedrock:*:::foundation-model/*"
    }
  ]
}
```

Add, for model listing:

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    }
  ]
}
```

A policy that grants only `ListFoundationModels` is supported. The catalogue
degrades as "Model aggregation" describes.

## Alternatives considered

**One connector per Bedrock API.** Rejected. The API that serves a model is an
implementation detail of Bedrock, not a choice a user should make. Separate
connectors would make a person learn which API serves which model, configure the
same credentials several times, and choose a new model whenever a model moved
between APIs.

**Invoke to obtain Anthropic prompt caching.** Rejected. Converse supports
prompt caching directly through `cachePoint`. Routing Anthropic models to Invoke
for caching would gain nothing and would add a second serialization path to
maintain. Invoke earns its place through the modalities Converse cannot address.

**A new Anthropic serialization inside this crate.** Rejected. Where a backend
needs the native Anthropic request shape, reuse `llm-anthropic`'s serialization
behind a thin transport adapter that sends the serialized body through
`InvokeModelWithResponseStream`. A second copy of the request builder and the
event parser would leave two implementations to keep in step with one vendor's
API.

### Where the profile-to-base-model mapping lives

Four candidates, scored against two requirements: the capability record and the
dispatch gates must read one answer, and no turn may make a control-plane call.

**A register populated by the listing, read by every gate. Chosen.** Meets both.
The listing already makes the call, so the mapping is free, and a miss has a
defined answer that costs nothing - the prefix rule, which is what every gate
did before. Its cost is a process-wide register, which is what makes the record
and the dispatch path share one answer across two client objects, and it leaves
one case unsolved: an id no listing returned stays conservative.

**Resolution when the connection is built.** Rejected. It resolves the
connection's own model and nothing else, so a per-turn model override or a
cross-connection send is unresolved, and it puts a control-plane call in the
constructor - which then decides whether the daemon starts cleanly.

**Carry the resolved base model on the request.** Rejected on scope and on
reach. `stream_completion` takes a model id because that is the port's shape,
so this means a change to `core::ports::llm` and to every connector, for one
provider's problem. It also only moves the question: whatever sets the field
still has to resolve the id, from the same listing.

**Decline, and document the gap.** Rejected as the whole answer, kept as the
answer for one case. It is what shipped before, and it costs an account that
routes through an application profile its reasoning control, its prompt cache,
its context window and the deny list, on every turn. Where it is still right is
the never-listed id, where the alternative is a control-plane call on the turn
path.

## Migration path

1. **Done.** The Converse code sits behind the trait as `ConverseBackend`, with
   the listing contract, the on-demand filter and the notices unchanged. The
   connector holds one backend and sends every request and the whole catalogue
   to it.
2. Add backend selection and the de-duplicating aggregation, with one backend
   registered. Still no behaviour change.
3. Add `ResponsesBackend` for the GPT-5.6 family on `bedrock-mantle`.
4. Generalise the existing embeddings `InvokeModel` call into `InvokeBackend`,
   then extend it to the other modalities Converse refuses.

Each step lands on its own and leaves the connector working.

One interlock spans steps 1 and 4, and it is invisible in a diff. The Converse
listing returns embedding models, because those models are what the daemon's
embedding path reaches and the catalogue is where a person picks one.
`ConverseBackend::can_serve` reports - correctly - that Converse cannot serve
them. So **the aggregation in step 2 must not filter the catalogue by
`can_serve`** until step 4 lists those models through `InvokeBackend`. Filtering
early takes every embedding model out of the picker.

## References

- `docs/design/connector-capabilities.md` - the capability model and its states
- `docs/connectors/cloud-connector-abstraction.md` - connector uniformity
- `crates/core/src/ports/llm.rs` - `ModelCapabilities`, `ModelListingReport`,
  `LlmClient`
- [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
