# The replay eval

Measure, per model, how much transcript a turn actually needs.

## Why it exists

[The verbatim window](verbatim-window.md) rests on two figures nobody measured:
how far a model carries a conversation forward without the transcript, and the
point at which adding more transcript stops changing the answer. Picking either
by judgment is hand-fitting. Both sit at conservative defaults until an
experiment sets them.

One experiment answers both. Take real conversations from the store. Replay the
next turn against a ladder of window sizes. Compare the answers.

## Running it

```
desktop-assistant --replay-eval <user-id>
```

It reads that user's conversations, replays each across the ladder, prints the
report, and exits without starting the daemon. The user id is an argument rather
than a guess: every store read is scoped to a user, and a daemon serving several
has no default one.

The ladder is 1k, 4k, 16k, 64k and 256k estimated tokens. It costs
`conversations x rungs` model calls, so a run is bounded at 40 conversations.

## What it reports

```
model: a-local-model
sampled: 12 conversation(s)
skipped: 3
trust setting: 4000
sufficiency ceiling: 16000
sampled ids: …
skipped c7: no_history
conversations in the store: 61
conversations read: 40

Write this into daemon.toml:

# measured over 12 conversation(s); trust setting: 4000
[context.models."a-local-model"]
verbatim_window_ceiling_tokens = 16000
```

**It states what it did not reach.** Conversations in the store, conversations
read, conversations skipped and why, one line each.

**And how the sample was chosen.** The 40 are the most recently updated
conversations, because that is the order the store lists in - a recency bias
rather than a random draw. A daemon whose recent work is all short questions
measures a model on short questions, and the report says so rather than letting
the sample pass as coverage.

## How the two numbers are read off

The largest rung is the reference answer. A rung counts when enough of the
sampled conversations agree with their own reference at that rung.

| number | threshold | means |
|---|---|---|
| sufficiency ceiling | 0.9 similarity | the smallest rung beyond which the answer stops changing |
| trust setting | 0.7 similarity | the smallest rung at which it is still the same answer |

Similarity is Jaccard overlap of lower-cased word sets: deterministic, needs no
model and no embedding backend, moves when an answer loses a fact, and stays
still for wording a model varies between identical runs.

**The top of the ladder cannot be its own evidence.** Every answer agrees with
itself, so a run that only agrees there is reported as unmeasured rather than as
needing exactly the ladder somebody chose. A model with no measurement takes the
conservative default rather than another model's number.

## What it measures, and what it does not

It measures the **model**, not the assembler. The prompt it builds is the
conversation's own messages cut to a window and the next thing the user said -
no tool schemas, no `[Recall]`, no plan. Adding those would measure how much a
particular assembly helps, which changes with every block this project adds and
is not what either number is for.

The cut calls the same function the live window uses, but it is not handed the
same projection: a live turn seeds one from the eviction decisions earlier turns
recorded, and this does not. On a conversation with prior evictions the eval
therefore cuts earlier than live would for the same target, so the ceiling it
reports is conservative - it will hold at least as much as it measured.

The rung cuts the history and never the prompt being replayed: the question is
what the model answers with less context, not what it answers to less of the
question.

## Offline

It reads conversations from the store rather than from any service, and it runs
as a batch job outside the request path. It does call a model - it has to,
because the question is what the model answers with less context - and the
connector it calls is the operator's own. Its tests call nothing: the replay
arrives as a closure, so a scripted answer is the whole fixture.
