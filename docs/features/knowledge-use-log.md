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
| offered by `[Recall]` | `ConversationHandler::send_prompt` | the entries the block rendered |
| offered by a search | `kb_search`, `crates/mcp-client/src/builtin.rs` | every entry on the page |
| opened | `kb_get`, same file | the entries the response delivered |
| marked | `builtin_knowledge_base_mark`, same file | one standing judgement per entry |

**The renderer reports what it rendered.** `render_recall` returns the ids of
the lines it emitted alongside the block text, and the turn records those. There
is no second implementation of the selection: the floor, the width, and every
"already in view" drop are applied in one place, and the ids come out of it.

That matters most for a pinned entry. `[Pinned]` carries such an entry in full,
so the block drops it and shows one further entry instead - and a pin is made
precisely for an entry that keeps ranking near the prompt, so the drop repeats
turn after turn. Recording the pinned id would accrue offers against
structurally zero opens, because the model has no reason to fetch an entry it
can already read whole. That is the exact profile of the cleanest prune
candidate, so the strongest endorsement the system holds would read as evidence
to delete.

The same argument settles an entry whose id the line cannot carry whole. An id
is stored as the write tool's caller wrote it, and a line can spend only
`RECALL_ID_MAX_CHARS` characters of one; a read matches an id exactly, so a cut
id resolves to nothing. Before the use log that was a failed fetch the model
could recover from by searching. With the log it is worse, because the offer is
recorded whether or not the fetch can succeed - so the entry would take an offer
every turn it ranked near the prompt and could never take an open. The block
therefore drops such an entry rather than showing an id no read resolves, and a
further entry takes the slot. Bounding the id where it is written is #1136.

The block renders on a turn's first round only, so the offer is recorded there
and nowhere else. An empty list recorded on a later round would take down the
offers the turn had just made.

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

The turn records its offer whether or not the block showed anything, and whether
or not the lookup succeeded. A lookup that timed out, or whose knowledge arm
failed, would otherwise leave the previous turn's offers standing - and the
model still has that block in its transcript, so a fetch on a later turn would
read as taking one up. The window would be "since the last successful lookup"
rather than one turn.

**An offer is keyed on the conversation.** One entry can stand offered in two
conversations at once, and a single column would let the second offer overwrite
the first and drop the open made against it - undercounting exactly the entries
broad enough to surface in two places at once.

The one degraded case is a deployment with pre-prompt recall switched off, where
the block never renders and nothing replaces the set at a turn boundary. An
offer made by a search then stands until it is taken up or until
`MAX_STANDING_OFFERS` pushes it out. That is a wider window than a turn, never a
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

The reason is cut to `MARK_REASON_MAX_CHARS` rather than refused. It comes from
a language model and nothing before storage bounds it, and an over-long reason
should cost its tail, not the mark.

The write retries once, and only when an entry went missing under it. Deleting
an entry removes it outright, and the delete runs in whatever conversation the
user asked from, so an id named in a mark can be gone between the statement's
own read of `knowledge_base` and the foreign key check that follows. The key
check then raises and the whole batch rolls back, which would contradict what
the tool promises: an id that did not land is named, and the rest of the batch
still lands. On the retry the row is definitively gone and the remaining ids are
marked. The check reads the SQLSTATE, not the message.

## Bounded per entry

`044_knowledge_use_log.sql` creates three tables, each bounded by how it is
written. `knowledge_use_stats` holds one row per entry, `knowledge_offers` one
row per `(conversation, entry)` currently in front of the model, and
`knowledge_use_marks` one standing judgement per source per entry.

A `[Recall]` block deletes its conversation's offer rows before inserting its
own, so a conversation normally holds one turn's worth. Where no block runs, the
writer trims to the newest `MAX_STANDING_OFFERS`. The foreign key to
`knowledge_base` frees a reaped entry's rows with it.

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

`KnowledgeUseRecord::use_sum` states what the log knows, on one scale:

```text
S = sum over the recent window of  age^-d
  + the tail approximation over every older use
  + sum over the marks of  sign * weight * age^-d
```

Two readings of `S` exist, and they are for different callers.
`KnowledgeUseRecord::usefulness` answers `ln(max(S, MIN_ACTIVATION_SUM))`, which
is the figure to report or to compare between entries. Retrieval reads `S`
itself, because it has to join it with a term of its own before either is
compressed - see the activation score in
`docs/features/pre-prompt-recall.md`.

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

**Retrieval reads this.** The `[Recall]` block ranks the entries the bar admitted
by an activation score whose reinforcement half is `S` - so an entry the model
keeps opening rises above a slightly nearer one nothing has ever taken up. What
that score is, how the two halves are joined, and what happens when this log
cannot be read are in `docs/features/pre-prompt-recall.md`.

## Multi-tenancy

All three tables carry `user_id`, each enables its own RLS policy in the
migration that creates it, and all three are registered in
`PERSONAL_DATA_TABLES` so the `db_query` tool grafts a `user_id` predicate onto
any model-supplied SQL that names them.

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
the caller's path in their own task and their errors become log lines. Those go
out at `warn`, because a write that fails is a fault rather than an expected
decline: an unmigrated database or an exhausted pool makes every write fail, and
a log the daemon never mentions leaves the tables empty while ranking scores
every entry alike on the use terms - a ranking that looks like a working
ranking, which is the failure this substrate exists to prevent. The one
exception to running off the path is the mark, which the caller asked for.

Running off the path costs one guarantee worth stating: a spawned write may
still be in flight when the tool returns. In practice an offer is recorded
before the turn's LLM call and taken up after the model has answered, so a full
round trip separates them. Under a saturated pool an open could in principle
reach the database before the offer it belongs to, and would then find nothing
standing and be dropped. It errs toward undercounting, which is the safe
direction for a signal that decides what gets retired.

A turn whose block showed nothing records an offer of no entries, because a
recall offer replaces the conversation's standing offers and an empty one is
what ends the previous turn's. A turn whose lookup failed records the same, for
the same reason.

The log is capability-gated like every other knowledge closure. Without a
database the tools behave exactly as they did before it existed, and
`builtin_knowledge_base_mark` reports that the knowledge base is not configured.

## Where the code is

| Piece | Path |
| --- | --- |
| The records, the window, and the score | `crates/core/src/domain/knowledge_use.rs` |
| The port, and the off-the-path write | `crates/core/src/ports/knowledge_use.rs` |
| The three tables | `crates/storage/migrations/044_knowledge_use_log.sql` |
| The adapter | `crates/storage/src/knowledge_use.rs` |
| What the block reported it offered | `render_recall`, `crates/core/src/recall.rs` |
| Recording it, once per turn | `ConversationHandler::send_prompt`, `crates/core/src/service.rs` |
| The search offer, the open, and the mark | `crates/mcp-client/src/builtin.rs` |
| Multi-tenant and correlation tests | `crates/storage/tests/knowledge_use_log.rs` |
