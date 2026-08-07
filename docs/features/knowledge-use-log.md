# The knowledge use log

Retrieval ranks knowledge entries with weights, and every weight has to come
from somewhere. A value fitted against one store, one embedding model and one
subject domain does not carry to a second deployment. It does not stay still
inside one deployment either, because a growing store pulls every nearest
neighbour closer.

Nothing recorded that an entry was ever put in front of the model, and nothing
recorded that the model took it up. Without those two facts every coefficient in
the retrieval path is chosen by hand and stays chosen by hand. The use log is
the record that lets a deployment measure its own.

Recording it changes no ranking. Nothing reads the score yet.

## Three records, none of them inferred

| record | what happened | quality |
| --- | --- | --- |
| offered | the entry appeared in a `[Recall]` block or a search result | low |
| opened | something fetched the entry by id after it was offered | high |
| marked | somebody said the entry was useful, or was wrong | highest |

Usefulness is never read out of whether a retrieved fact appears to have shaped
an answer. The `[Recall]` block offers ids and no bodies precisely so that a
fetch is a deliberate act: the model reads a one-line stand-in and chooses to
open the entry.

**The ratio is the interesting number.** An offer is mainly a denominator. An
entry offered fifty times and never opened is ranking too high and earning
nothing, which is the cleanest evidence for retiring one that this log can hold.
An entry offered twice and opened twice is carrying its weight.

**A negative mark is not the absence of a positive one.** "Offered, opened, and
it was wrong" is stronger evidence than silence, and the reason a marker gives
with it is what a later reader needs.

## Where each record is made

| act | where | what it records |
| --- | --- | --- |
| offered by `[Recall]` | `crates/daemon/src/knowledge_use.rs` | the entries the block will show |
| offered by a search | `kb_search`, `crates/mcp-client/src/builtin.rs` | every entry on the page |
| opened | `kb_get`, same file | the entries the response delivered |
| marked | `builtin_knowledge_base_mark`, same file | one standing judgement per entry |

The `[Recall]` offer is recorded by a decorator around the recall lookup, rather
than inside the renderer. The block is assembled during prompt building, where
no tool runs, so the decorator applies the renderer's own rules -
`RecallRelevance::clears_floor`, `KnowledgeEntry::display_line` and
`MAX_RECALL_ENTRIES` - to the candidates the lookup returned.

One rule it cannot apply. The renderer also drops an entry `[Pinned]` is already
carrying in full, and whether a pin resolved is decided later in assembly. On a
turn where a pinned note attaches an entry that also ranks near the prompt, the
decorator records that entry - which was in front of the model, under another
block - and misses the entry that took its place in the line budget. The case
needs a pinned attachment and a near-prompt rank together, and it moves the
count by one.

## An open is a taken-up offer

A fetch by id is only recorded as an open when the entry is standing offered in
the same conversation, and counting it takes the offer down. Two things follow:

- A read that nothing offered records nothing. An id from a pinned note, or one
  the model held from an earlier task, is ordinary bookkeeping rather than
  evidence.
- A second fetch of the same entry in the same turn records one open, so a
  retried tool call adds nothing.

**An offer stands for one turn.** A `[Recall]` block is rendered once per turn,
from the user's prompt, so an offer it makes replaces whatever that conversation
had standing. A search runs inside a turn that is already going, so an offer it
makes is added. "Offered in the same turn" therefore needs no turn identifier:
the standing set is this turn's set, because the turn's first block replaced it.

The one degraded case is a deployment with pre-prompt recall switched off.
Nothing then replaces the set at a turn boundary, so an offer made by a search
stands until it is taken up. That is a wider window than a turn, never a
narrower one, and it still refuses the read that nothing offered.

## Marking

`builtin_knowledge_base_mark` takes a batch of ids, a `useful` flag, and a short
reason. A mark is a standing opinion, one per source per entry, so a second mark
from the same source replaces the first - which makes a retried call safe and a
change of mind ordinary.

The schema carries a second source, `person`, for a mark a human makes. No
client offers one yet. The value exists so a human judgement has somewhere to go
the day one does, and a person's mark outranks the model's when the two disagree.

Marking is the one write on the log that a caller asks for rather than one that
measures a read, so it is awaited and its failure reaches the caller. A caller
that asked to record a judgement has to learn whether the judgement landed.

## Bounded per entry

A spacing term needs per-use timestamps, because when the uses fell is the half
of the signal a lifetime counter cannot express. Keeping every event for ever is
unbounded, so a record keeps two things:

- the most recent `RECENT_USE_WINDOW` use timestamps, exactly
- aggregate counters and a first-seen stamp, for everything older

That is the standard hybrid for ACT-R base-level activation: exact over the
recent window, and the streaming approximation over the tail. There is no
per-event table, and there is no bare lifetime counter.

A "use" is an open or a mark. An offer is not a use: being shown is not being
taken up, and counting it as one would let ranking feed itself.

## The score

`KnowledgeUseRecord::usefulness` states what the log knows, on one scale:

```text
S = sum over the recent window of  age^-d
  + the tail approximation over every older use
  + sum over the marks of  sign * weight * age^-d

score = ln(max(S, MIN_ACTIVATION_SUM))
```

Three properties follow from that shape:

- **Recency weighted.** Every term is an age raised to a negative power, so an
  old use and an old mark both fade.
- **A negative mark lowers the score.** It subtracts rather than failing to add,
  so an entry that was opened and then found wrong ends below one never opened.
- **No rich-get-richer.** The logarithm means doubling the uses adds a constant.
  Marks raise ranking, ranking decides retrieval, and retrieval is a
  precondition for being marked - so the growth has to be sub-linear or that
  loop compounds.

The coefficients live in `UseScoreWeights`, and their defaults are declared
starting points rather than measured values. They are a struct rather than
constants exactly so that a deployment which has kept a use log can fit its own
and pass them in.

## Multi-tenancy

Both tables carry `user_id`, both are on the RLS policy list, and both are
registered in `PERSONAL_DATA_TABLES` so the `db_query` tool grafts a `user_id`
predicate onto any model-supplied SQL that names them.

`knowledge_base.id` is a global primary key, so an id another user owns is an id
this user can name. Every write therefore selects the entry out of
`knowledge_base` under the caller's `user_id` and inserts from that select,
rather than inserting the id it was handed. An id the caller does not own, and
one that names a retired entry, are both silently not recorded - the same answer
`builtin_knowledge_base_get` gives for the same id.

The foreign key to `knowledge_base` gives the log the entry's own lifetime: a
hard reap frees the use rows with the entry. Soft deletion does not, so a retired
entry keeps its record - which is what lets a later reader see that it was
offered often and never opened.

## Failure

Recording never fails a read. An offer and an open are measurements of a read,
and a measurement must not be able to break what it measures, so both run off
the caller's path in their own task and their errors become log lines. The one
exception is the mark, which the caller asked for.

Running off the path costs one guarantee worth stating: a spawned write may
still be in flight when the tool returns. In practice an offer is recorded when
the lookup answers and taken up a model round trip later, which is seconds.

The log is capability-gated like every other knowledge closure. Without a
database the tools behave exactly as they did before it existed, and
`builtin_knowledge_base_mark` reports that the knowledge base is not configured.

## Where the code is

| Piece | Path |
| --- | --- |
| The records, the window, and the score | `crates/core/src/domain/knowledge_use.rs` |
| The port, and the off-the-path write | `crates/core/src/ports/knowledge_use.rs` |
| The two tables | `crates/storage/migrations/044_knowledge_use_log.sql` |
| The adapter | `crates/storage/src/knowledge_use.rs` |
| The `[Recall]` offer decorator | `crates/daemon/src/knowledge_use.rs` |
| The search offer, the open, and the mark | `crates/mcp-client/src/builtin.rs` |
| Multi-tenant and correlation tests | `crates/storage/tests/knowledge_use_log.rs` |
