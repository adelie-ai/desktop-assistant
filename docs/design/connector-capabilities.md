# Connector Capabilities

## What this answers

For a given **(connection, model)** pair, what can a client enable, and when it
cannot enable something, why not.

That is the whole subject. A person chooses a connection and a model. Every
control the interface offers after that - vision, reasoning effort, hosted tool
search, prompt caching - is either available for that pair or it is not. Today
each such control is gated by its own ad-hoc check, several of those checks
disagree with each other, and none of them carries a reason. The result is the
failure shape the repository rules call out by name: a person sets a value,
presses save, and only then learns the combination does not support it.

Two requirements follow, and they are the design:

1. **The answer is keyed on the pair.** Not on the connector alone, and not on
   the model alone. A connector-level answer cannot describe a model, and a
   model-level answer cannot describe the API that serves it.
2. **The answer reaches the client before the person acts.** A capability that
   is only discovered when a request fails is not a capability system.

## The answer

```rust
/// Whether one capability is available for one (connection, model) pair,
/// and why not when it is unavailable.
pub enum CapabilityState {
    /// Available. Enable the control.
    Supported,
    /// The model was not trained for it. Choosing another model can fix it.
    UnsupportedByModel,
    /// The API surface serving this model does not carry it.
    UnsupportedByApi,
    /// This connector implementation does not provide it at all.
    UnsupportedByConnector,
    /// Not determined. Neither enable nor claim it is unavailable.
    Unknown,
}
```

`Unknown` is a distinct state, not a synonym for unavailable. A model we have
not classified, a listing that degraded, a connector that has not adopted the
mechanism yet - all of these are `Unknown`. A client renders `Unknown` as
available-but-unverified rather than hiding the control, because hiding a
control that in fact works is the worse error and is silent. This is the same
three-state rule the repository applies to optional operating-system services:
"is the capability present?" is a different question from "did my call succeed?",
and a fail-safe that suits one of them is pathological for the other.

The reason is what makes the state useful. `UnsupportedByModel` tells a person
to pick another model. `UnsupportedByApi` tells them nothing they can act on and
should read as a plain statement. `UnsupportedByConnector` tells them to use a
different connection. A single `false` supports none of that.

## Where the inputs come from

Three inputs feed the answer. They are inputs, not a mandated type hierarchy.

**The model.** What the model was trained for: vision, reasoning, tools, and its
kind. This is the existing `ModelCapabilities` on `ModelInfo`, and it is the
input most in need of better data. Several connectors infer it from substrings
of the model id today. Where a provider publishes real capability metadata, read
that instead.

One of those flags does not mean what its name suggests, and the exception is
load-bearing. `reasoning` answers "can this connector configure reasoning for
this model", not "does this model reason". Every consumer already treats it that
way - a client offers a reasoning control, a connector decides whether to send
the field - and a model that reasons but accepts no configuration has to report
`false`, or the control is shown and the budget is dropped. DeepSeek R1 on
Bedrock is exactly that model. So the flag is populated from the same function
the request path reads, never from a second list that can disagree with it.

**The API surface.** What the API serving this model supports. Only a connector
that speaks more than one API needs this input, and only Bedrock does. Azure's
two surfaces differ in URL shape and carry identical capabilities; Google's two
auth modes differ in host and credential. Neither is a capability difference.

**The connector.** What the implementation supports at all. Two questions hide
here and must not share a name:

- *Can a connector of this type ever do X?* A property of the connector type,
  answerable before any connection exists. This is what a settings panel needs
  while a person is still choosing a connector.
- *Does this configured connection do X?* A property of the live instance,
  after configuration. This is what a turn actually obeys, and what the model
  listing must carry.

Hosted tool search is the worked example. `Connector::type_offers_hosted_tool_search`
answers the type question and feeds the connector-defaults view; the `LlmClient`
trait's `hosted_tool_search` answers the instance question and is what a turn
obeys. The names say which is which, because they once did not: both were
called `supports_hosted_tool_search`, and a reader had to open the doc comments
to learn that clients were told one and turns obeyed the other.

Keep both questions. Name each for the axis it answers, and make the pair-keyed
answer carry the instance one.

## Composition

Where one path serves the pair, the answer is the intersection. Any input can
block a capability. No input grants one that another denies. An unclassified
input contributes `Unknown` rather than `false`, and `Unknown` beside
`Supported` yields `Unknown`, never a confident `false`.

Where a connector reaches one model through several API surfaces, the answer is
the **union** across the surfaces that can serve it, and each capability records
which surfaces provide it. A capability that only one surface delivers is
genuinely available on that model; reporting it unavailable because another
surface lacks it under-reports the product.

The union carries one obligation. Two capabilities can each be available while
no single surface provides both. Keeping the per-capability surface set lets the
connector detect an unsatisfiable request before it goes out and name the
conflict, rather than meeting it as a provider error mid-turn.

## Defaults

A connector that has not adopted the mechanism reports `Unknown` for everything.
It does not report "nothing supported".

This matters more than it looks. Under an intersection rule, a default of
"nothing supported" would turn every capability off for every un-adopted
connector the moment the method exists - a silent, fleet-wide regression
delivered by a defaulted trait method that nobody had to call. `Unknown`
preserves current behaviour while adoption proceeds, and it is honest: we have
not determined the answer.

The same caution applies to the decorators. Every wrapper around `LlmClient` -
routing, classifying, profiling, retrying - forwards trait methods by hand. A
decorator that forgets to forward a capability method answers for the wrong
client. The routing decorator did exactly that for hosted tool search: it
reported the static fallback's support rather than the selected connection's,
so a per-turn model override assembled the tool list for one client and sent
it to another. In its static-fallback mode it now resolves the capability
through the same per-turn lookup its dispatch path uses. Its dynamic-purpose
mode still answers a fixed `false` while its dispatch path resolves a real
client, which is safe only because backend tasks do not reach the
hosted-search path at all - a fixed answer, not a resolved one. Every
capability method added later must resolve against the client that will serve
the request, and that second mode is where the resolution is still missing.

## Reaching the client

The pair-keyed answer travels on the model listing, beside the model it
describes. That is the response a client already fetches to populate a picker,
and it is the only response keyed on the pair.

It does not travel on the connector-defaults view. That view answers the
connector-type question, not the pair question, and none of its fields carry a
serde default - adding one there breaks older clients on deserialize.

Wire rules that apply:

- Extend with optional fields. Never add a variant to an existing wire enum.
  Prove an old payload still parses.
- Serde compatibility is not source compatibility. Client code that constructs
  these structs with an explicit field list fails to compile as soon as a field
  exists, and most of those sites are inside test modules - the sweep needs
  `--all-targets` across every client repository.
- One wire capability type is `Copy` today and is copied by value in a client. A
  nested state-and-reason struct is not `Copy`. Expect that site to need a
  change, and find the others the same way.

Finish the rollout that is already in flight before adding to it. The model-kind
axis reached the wire, but every client still reads the older derived boolean
beside it, and the end-to-end fixtures omit the new field entirely. A second
unadopted axis is worse than one.

## Making it load-bearing

A capability value that nothing reads is documentation, not a gate. Today the
model capability record has one consequential consumer in the daemon - the check
that binds a model to a purpose - while the decisions that matter read a
connector-type string match or a connector-private heuristic.

So each capability added to the pair-keyed answer replaces a decision site
rather than sitting beside one. The reasoning-effort mapping is the clearest
first candidate: it branches on the connector-type string today, which is why a
model that advertises reasoning support can have its reasoning budget silently
dropped by the connector that serves it.

## An instance answer that cannot lie

The instance question above has a second requirement the type question does not:
the answer and the behaviour it promises must be the same fact. A capability
answered by a separate boolean is a claim about the code beside it, and a claim
can be wrong.

Hosted tool search showed the cost. The `LlmClient` trait carried a
`supports_hosted_tool_search` boolean and a `stream_completion_with_namespaces`
method whose default body flattened every namespace into one ordinary tool list.
A connector could report the capability and inherit that body. The turn was then
worse than one that claimed nothing, because the service layer strips
`builtin_tool_search` as soon as hosted search is active: the model received the
whole tool fleet inline *and* no way to discover tools. Azure was one edit from
this. A runtime sweep in the daemon narrowed the gap and could not close it,
because it can only probe a connector whose transport it knows how to drive.

### The candidates

Three shapes were compared. The deciding hazard is stated after them, because it
is what separates the two that otherwise look close.

**A. Remove the default body.** Every implementor writes the method; the
flattening becomes a named helper a connector calls on purpose. Cost is about
forty-five one-line delegations, most of them in test doubles.

**B. An extension trait.** `LlmClient` gains
`hosted_tool_search() -> Option<&dyn HostedToolSearch>`, which returns `Some`
only where the extension is implemented. The capability answer and the
implementation become the same object, and the boolean disappears rather than
being policed. A client with no hosted search leaves the method at its `None`
default and costs nothing.

**C. A wrapper applied at construction.** The registry wraps a hosted-search
connector in a `HostedSearchClient<C>` when it wires one, and only that wrapper
carries the namespaced path.

### The hazard

Six decorators wrap an LLM client: retry, profiling, the profiling-or-not
wrapper, error classification, reasoning substitution, and per-turn routing.
Each must stay in the call path for a namespaced turn, exactly as it does for an
ordinary one. A decorator that answers the capability by handing back its
**inner** client's dispatch object is bypassed for that turn - so the turns
carrying the most tools are the ones that lose retry and per-turn routing, and
nothing else in the workspace notices.

### The choice

**B, with each decorator implementing the extension for itself.** A decorator
answers `Some(self)` when, and only when, its inner client has hosted search,
and its implementation decorates and then hands down to its inner client. The
chain stays whole. One named test per decorator observes that decorator's own
effect on a namespaced turn, so a bypass fails a test that names the decorator.

Scored against the requirement - a claim that cannot be wrong:

- **A fails it.** Writing the method is not the same as pairing it with the
  claim. A connector can still report the capability and write a flattening
  body, which is the exact defect, only louder. A cost of forty-five edits buys
  a nudge, not an invariant.
- **C fails it for the same reason the runtime sweep does.** Enforcement sits at
  the wiring site, so a connector built by any other path escapes it, and the
  wrapper adds a layer to a decorator chain that is already six deep.
- **B holds it in the type system.** Returning `Some(self)` requires
  `Self: HostedToolSearch`, and implementing that trait means writing the
  request. Reporting the capability without the implementation does not compile.

B's naive form fails the hazard, and the work to fix it - one small
implementation per decorator - is what its file-count advantage pays for. It is
still far below A's cost, because a test double that does not care implements
nothing at all.

### What the type system still does not cover

Flattening stays representable, deliberately: a connector may implement
`HostedToolSearch` with a body that flattens, and that is a reviewable choice
rather than an inherited default. The compiler requires a body, not a correct
request. So the daemon's cross-connector sweep is kept, with a narrower job:
what a hosted implementation puts on the wire, whether the registry arm still
wires the capability on, and whether a claim quietly disappeared. What it
stopped proving is that a claim has an implementation at all, and that a
decorator forwarding a claim also forwards the dispatch. Neither is
representable now.

## Non-goals

- **A universal API-surface type in `core`.** Only Bedrock has several surfaces
  with differing capabilities. That type stays inside the Bedrock connector.
  Eight other connectors should not carry a layer that describes one.
- **Provider-specific fields on a shared type.** Whether a connector accepts
  named AWS profiles is a Bedrock configuration concern. It is not a capability
  of every connector, and it does not belong on a type they all implement.
- **A health and diagnostics page.** The states and reasons here are the data
  such a page would render, and it should reuse them rather than invent a second
  vocabulary. Building the page is separate work.

## References

- `docs/connectors/bedrock.md` - the multi-surface connector that needs the
  union rule
- `docs/connectors/cloud-connector-abstraction.md` - the uniform connector
  contract
- `crates/core/src/ports/llm.rs` - `ModelCapabilities`, `ModelKind`,
  `LlmClient`, `HostedToolSearch`, `dispatch_namespaced`
