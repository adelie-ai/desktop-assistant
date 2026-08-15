# The verbatim window

How much of a conversation a turn carries word for word.

## Why a count of messages bounds nothing

The window has always been "the most recent 40 messages". A turn is not a unit
of size: one is "thanks" and the next carries 40 KB of tool output, so forty
messages is anywhere between a paragraph and most of a context window. A bound
that cannot say how big it is is not a bound.

The `[context]` section replaces that count with a token target.

## The target

The **lower** of a fraction of the effective per-turn input budget and an
absolute ceiling.

Capacity is not a budget. A third of a million-token window is 330,000 tokens
per turn because the room existed. The fraction protects a small window and the
ceiling protects a large one, and neither does the other's job.

**"Effective" is a claim about which number.** Three figures all read as "the
context window" and they differ:

| figure | what it is |
|---|---|
| the model's nominal window | what the provider advertises |
| the configured ceiling | what an operator wrote in `purposes.<kind>.max_context_tokens` |
| the **effective** budget | what the assembler plans against after the learned-overflow cap |

The fraction is taken from the third. A target measured against either of the
others is a claim in the wrong unit.

## Pressure, not a limit

Below the target nothing happens. Above it the window carries fewer turns, which
leans harder on the mechanisms that already exist: superseded tool results are
evicted, `[Earlier turns]` keeps a dropped turn distinguishable from one that
never happened, and the rolling summary keeps its gist.

**Nothing is refused and nothing is truncated.** The floor is one complete turn:
a turn that costs more than the whole target on its own is carried whole. In
practice the floor is the larger of one complete turn and the window's own
`MIN_CONTEXT_MESSAGES`, because that floor applies whatever this asks for.

The bound only ever narrows. Overflow recovery shrinks the window when the
provider says the prompt was too big, and this never widens it back - that
mechanism is the one that has seen the provider's own count.

`COMPACTION_TOKEN_RATIO` stays where it is as the emergency. Success is that it
stops being reached.

## Turning it on

Off by default, because its failure presents as "she forgot". Off leaves the
window exactly as it was; it does not mean upgrading changes nothing, because
the `[Earlier turns]` index is gated on windowing rather than on this switch.

```toml
[context]
verbatim_window_tokens = true
# Optional; these are the built-in defaults.
verbatim_window_ratio = 0.33
verbatim_window_ceiling_tokens = 60000

# Per model, because both numbers behind the target are per model: what the
# window costs, and how far the model carries a conversation without the
# transcript. An override states only what it changes.
[context.models."a-small-local-model"]
verbatim_window_ceiling_tokens = 4000
```

Read once, when the conversation handler is built, so an edit needs a restart. A
reload reports the section as the `context` restart area.

## The ceiling is chosen, not measured

60,000 tokens is a working number: comfortably above what an ordinary turn
carries, far below what a large window would otherwise permit. The measurement
that replaces it - the point at which adding more transcript stops changing the
answer, per model - is #1209.
