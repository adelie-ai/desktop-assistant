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

Today both are called `supports_hosted_tool_search` - one on the `Connector`
enum, one on the `LlmClient` trait - and clients are told the first while turns
obey the second. They can disagree. Keep both questions, give them names that
say which they answer, and make the pair-keyed answer carry the instance one.

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
  `LlmClient`
