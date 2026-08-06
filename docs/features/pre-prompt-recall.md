# Pre-prompt recall: the `[Recall]` block

The assistant reaches its memory only when it decides to. It has to notice that
a search might help, choose a query, and spend a tool round. When it does not
notice, the knowledge base is a store nobody reads, and a note it stashed on its
own scratchpad an hour ago may as well not exist.

Pre-prompt recall makes memory arrive unasked. When a user prompt lands, the
daemon embeds it once, asks the three indexes that share that embedding space
what is near it, and puts a short list of candidates in front of the model as a
`[Recall]` system block, before the model's first move.

The block is a hint. Nothing is asserted, and the model decides whether any of
it matters.

## What the model sees

```
[Recall] Memory that may relate to what was just asked. It may not fit; ignore what does not. Each line is one entry: its id, its tags, and one line of what it says - not the entry itself. Look one up before you rely on it.
- kb-1a2b [preference, ui] Prefers dark themes in every editor
- kb-9f31 [infra, deploy] The deploy target is the lab cluster
...and 17 more entries matched less closely.
Notes on this conversation's scratchpad. Each line is one note: its key, then the start of what it says - not the whole note.
- deploy-window: Fridays after 18:00, never before
...and 2 more notes matched less closely.
Tags near this prompt: project:adelie-ai, infra, deploy, ui, preference
```

Each entry line carries the entry id, the entry's tags, and the one line that
stands for it. That line is `KnowledgeEntry::display_line()`: the stored
`summary` where there is one, and otherwise a whitespace-collapsed prefix of
`content` capped at 200 characters. An entry with no summary is **not** skipped -
the dream cycle's summary backfill fills the column a bounded number of rows per
cycle (see `docs/features/knowledge-maintenance.md`), so a store that has just
gained the column, or an entry written moments ago, still has none, and a block
that skipped them would show nothing. An entry whose line comes out empty is
skipped, because it would spend a line of the budget to say nothing.

Each scratchpad line carries the note key and the start of the note, capped at
the same 200 characters an entry line gets. A note with a key and no body still
renders, as the key alone: a key is the pad's own unit of recognition, which is
the whole trade the `[Scratchpad]` index makes. A note with no key is skipped -
it names nothing the model could look up.

The two kinds of `- ` line are separated by the scratchpad label, because they
carry different authority: an entry is what the assistant chose to keep across
conversations, a note is what this conversation happens to have written down.

Tag names are listed nearest first, not alphabetically.

**The block names no tool.** Which read fetches an entry by its id is a property
of the tool set on the day the block renders, and a block naming a tool the model
cannot call is worse than one naming none: the model tries it and spends a round
on the failure. Saying what a line is - a stand-in for an entry, not the entry -
leaves the model to pick the read it actually has. Teaching the model what the
block is belongs in the standing instruction, not in the block.

**Every part of the block is bounded.** The entry id, the note key and the note
content all pass through the same one-line rule the summary does. A note key is
bounded to the same width `[Scratchpad]` uses
(`ports::scratchpad::NOTE_KEY_MAX_CHARS`), so the same key never renders whole in
one block and cut in the other, one block apart in the same prompt. The block is
line-structured and it is a system message, so a stored value carrying a newline
would forge a line - and the line above it is a block header the model is taught
to trust. An entry id and a note key are both taken from the write tool's caller
and stored as written, so nothing before this point bounds them.

Tag names are bounded by size rather than cut: a name that does not fit the
remaining width is left out. Half a tag name is a tag no row carries, and the
model is handed these names precisely so it can search on one.

## What the standing instruction says

The block names no tool; the standing instruction does. The `[Recall]` guidance
sits in the knowledge-base section of the system prompt
(`crates/core/src/prompts/sections/knowledge_base.txt`), beside the search and
tagging guidance it has to agree with. It states five things.

**Where the lines came from.** A search of the model's own memory, run against
the user's prompt before the model asked for anything. It is a hint, and nothing
in it is asserted to be true, current, or relevant.

**That a line is not the entry.** A line may be a written summary, or it may be
the opening of the content, and the line does not say which. The model reads the
entry with `builtin_knowledge_base_get`, which takes a batch of ids, so several
candidates cost one call. It never answers from a line.

**That ignoring the block is ordinary.** The block fires on every prompt, so a
set that does not fit the work is a set to drop, and dropping it costs nothing.

**What a scratchpad line is.** A note key and the start of what the note says,
under its own heading - not a knowledge entry, so `builtin_knowledge_base_get`
does not take one. `builtin_scratchpad_search` reads the note back. A note
another block is already showing is left out, so what appears there is material
nothing else in the turn is showing.

**What the tag names are for.** They are registered tags of this store, so they
are real names and not guesses - the same vocabulary `available_tags` reports
after a search, offered before the model makes one. A filter on one that returns
nothing means no entry in that scope carries the tag.

**That the block never replaces a search.** An absent block is not an empty
store: the floors are conservative, and the lookup ran against the user's prompt
rather than against the question the model would have asked. The section's
mandatory search rule is unaffected.

The same text also lives in `runtime_system_instruction.txt`, the legacy
monolith that `assembled_static_sections_match_original` byte-compares the
assembled sections against. Both files carry it or the gate fails.

## Three arms, one embedding

| Arm | Index | What it offers |
| --- | --- | --- |
| Knowledge | `knowledge_base.embedding` | The entries nearest the prompt, best first. |
| Scratchpad | `scratchpads.embedding` | This conversation's notes nearest the prompt. |
| Tag | `tag_registry.embedding` | The names of the tags nearest the prompt. |

The tag arm reads vectors the near-duplicate check already built, and writes
nothing. Its point is vocabulary: the model's first knowledge search is otherwise
a guess at what words this user's tags use, and `available_tags` answers that
only *after* a search has already come back thin.

Descriptions do not travel with the names. A tag's `description` says what the
tag means; the caller is asking what this prompt is about.

The scratchpad arm exists because `[Scratchpad]` cannot serve this need.
That index lists note keys, but it is gated on the "context is starting to drop"
signal - right for an index, which is a reminder that notes exist after the
writing message scrolled away, and wrong for recall. A note written earlier in a
short, fully-visible conversation is durable and invisible, and a prompt that is
exactly about that note would otherwise produce nothing.

The arm reads **this conversation's pad only**. The pad is per-conversation by
design; reaching across conversations is a different feature with its own
privacy question.

The query leaves out exactly one note: the reserved `goal`. That is what every
turn renders as `[Current task]`, and it is by construction the pad row nearest a
prompt about the current task, so without the exclusion the arm would spend its
first line restating the task the prompt already carries, every turn. Excluding
it in the query rather than after the read means it never occupies a slot in the
scan the "and N more" count is measured against.

Nothing else is excluded there, deliberately. A `todo` step and an
`outcome:<step>` finding are in view only while `[Plan]`'s tree still shows them
- a finding is dropped once its parent step is done, and the tree elides past its
cap - and a note that has left the tree is durable and invisible, which is the
condition this arm exists for. What the turn *actually showed* is decided at
render time instead.

## Nothing already in view

A second look at the same pad would otherwise pay twice for one memory, so the
block drops what the rest of the turn's prompt already shows:

| Dropped | Because |
| --- | --- |
| A pinned note | `[Pinned]` carries its whole content every turn. |
| A key the `[Scratchpad]` index has just listed | The index named it on this same round. |
| A step or finding `[Plan]` has just named | The tree showed it on this same round. |
| A knowledge entry a pinned note attaches | `[Pinned]` renders that entry's live content (#1104). |

Each list is what the block **showed**, not what the pad holds. That distinction
is the whole point: `[Scratchpad]` never lists an `outcome:` key, and `[Plan]`
drops a finding once its parent step is done, so a rolled-up finding appears on
neither list and the arm offers it - which is exactly its job.

Whether the index speaks is not knowable before assembly - it is gated on the
window having dropped history, and the window is not fixed until the budget pass
finishes. So the lookup hands `surfaced_blocks` its *candidates*, and the block
is rendered there, after that decision. On a short turn where the index stays
silent, no key is dropped for it - which is the case the arm exists for. The two
`[Pinned]` rules apply on every turn, because that block is ungated.

A dropped candidate is not counted in the "matched less closely" line either.
That line promises the model something it has not been given.

## Nothing stamped as external content

A subagent's final answer is mirrored onto the session pad, and when that child
had read outside the trust boundary the note is stamped with
`EXTERNAL_CONTENT_MARKER`. `builtin_scratchpad_search` is classified
`Declared(ExternalContentMarker)` exactly so that reading such a note back
taints the turn and closes the tool gate.

This block has no tool call in it, so no `observe_result` runs and nothing would
close. The scratchpad arm therefore **drops a stamped note outright**. Without
that, possibly-injected text would land in a system message, ahead of the user
prompt, with the Egress, Mutate and Execution tiers all still open - and with no
model choice and no attacker step needed, on every turn where the note happened
to rank near the prompt.

Dropping rather than tainting, because the note lives on the pad indefinitely:
closing the gate whenever it ranked near would degrade the conversation
permanently. The parent still reaches that answer through
`get_subagent_status`, which taints correctly.

## What bounds the cost

**A relevance floor, not a top-k.** Each arm drops a candidate that is not near
enough, rather than padding the list out to fill the budget. A prompt with
nothing near it - "thanks", "run the tests" - emits no block at all. The floors
are cosine-distance ceilings, in `crates/core/src/recall.rs`. They are
deliberately conservative starting points rather than measured values: a block
that stays quiet costs nothing, and a block full of unrelated memory can pull the
model off the ask.

**A line budget.** Eight entry lines, five scratchpad lines and five tag names.
Every part is capped - 64 characters of id, 120 of an entry's tags, 200 of
summary, 64 of a note key, 200 of a note's content, and 240 for the whole tag
line - so the block cannot exceed about 4800 characters whatever the store and
the pad hold. Real entries carry a short id and a few tags, and real notes are
short and distilled, which puts the usual block near 300 tokens.

**One round.** Every other per-turn block re-renders each round, because each is
answering "is this still in view?". `[Recall]` answers "what might this prompt be
about?", and the user prompt asks that once. Repeating it across twenty tool
rounds would spend thousands of tokens on an answer the model has already taken
or ignored.

## Saying what did not fit

A model that sees eight entries cannot tell whether the store holds exactly eight
relevant things or four hundred, and those call for different next moves - accept
the list, or go search properly. So the block ends with a count of what cleared
the floor and did not fit.

That count means something only because the floor defines it. "How many matched"
is not a defined quantity over a hybrid search, where every embedded row scores
non-zero against any query - the same trap that produced `scope_size` instead of
a match count in the search tool.

Each arm counts its own, and says which. The knowledge lookup reads at most 50
rows and the scratchpad lookup at most 25 - fewer, because it reads one
conversation's pad rather than the whole store, so the tail it would be counting
is short. When a scan fills up *and* every row it read cleared the floor, that
arm's count is a lower bound and says so: "and 42 or more entries matched less
closely." When the scan read past the floor, the count is exact and carries no
hedge. When nothing was dropped, there is no line.

## Failure

Recall never fails a turn.

The embedding call is bounded at five seconds, the same ceiling the
knowledge-base search tool applies. On timeout or an embedding error the
knowledge and scratchpad arms degrade to full-text search, and the tag arm goes
quiet - the registry carries no full-text index to fall back to. The degradation
is logged once, not once per arm.

The whole lookup carries a second ceiling of ten seconds. The embedding timeout
bounds only the embedding; the database round trips around it are bounded by the
connection pool acquire timeout, which is measured in tens of seconds. Recall
runs before every turn's first round, so a saturated pool would otherwise hold
each turn far longer than the embedding timeout suggests.

The degraded search asks for **any** of the prompt's terms, not all of them. The
ordinary `search_text` joins every lexeme with `AND`, which is right for a
model-authored query of two or three words and wrong for a whole user sentence:
"where does the registry live?" becomes `registri & live`, and an entry saying
"the registry is on the storage host" never says "live". The fallback would then
answer nothing at exactly the moment it exists to answer something. Ranking still
puts the entry carrying more of the terms first.

Over the full-text path there is no distance to compare, and no floor is applied:
a row that carries none of the prompt's terms is never returned, which is a floor
of its own.

If the degraded read fails as well, the block is omitted and the turn proceeds.

An arm that fails outright is a narrower loss. The scratchpad arm reads a
different table from the other two, so it can fail on its own, and when it does
it costs its own lines and nothing else: the knowledge and tag arms still
render. The knowledge arm gets no such treatment, because a knowledge arm that
cannot read is the block's whole point failing - losing the pad lines is a
smaller loss than losing the block.

## Multi-tenancy

Every query carries an explicit `WHERE user_id` predicate. Row-level security is
a non-FORCE backstop that the table owner bypasses, so the predicate is the
guard. The two scratchpad queries carry a `conversation_id` predicate beside it,
and the caller's `owner_todo` read snapshot, so a subagent turn sees the same pad
its other reads see. `crates/storage/tests/recall_candidates.rs` holds all of
them to it, and runs under `just test-db`.

## Configuration

```toml
[recall]
enabled = true   # the default
```

It also stays off on its own when there is no knowledge store or no embedding
backend, and the daemon says which of the three reasons applies at startup.

Turning it off restores exactly the behaviour that preceded the feature: the
assistant reaches its knowledge base only when it decides to search.

## Known limits

**The floors are untuned.** Both are conservative starting points chosen to keep
the block quiet, not values measured against a real store. Widening them is the
safe direction.

**The scan reads whole rows.** Fifty entries are read to render eight lines,
because the count of what did not fit has to be a count. The row count is
bounded; the bytes those rows carry are not, so a store of unusually long entries
pays more per prompt than a store of one-liners.

**It fires on every turn, including agent and subagent runs.** Any turn that goes
through `send_prompt` gets a lookup, so a spawned agent working from a
machine-written brief pays one embedding and three reads as well.

**A pin the `[Pinned]` byte budget cut short is still suppressed.** The knowledge
arm drops the attachments the turn *resolved*, which is a superset of what
`[Pinned]` had room to print. On that rare turn the block says in its own words
that pins did not fit, so the model is not left believing the fact is absent.

**Cancellation waits on the lookup.** A turn cancelled while the lookup is in
flight still waits for it, bounded by the ten-second whole-lookup ceiling.

**The scratchpad arm scans the whole pad, every turn.** The vector query unnests
each note's chunks and groups on the row, so no vector index applies and the
`LIMIT` bites after the aggregate. One conversation's rows bound it, and a pad
only grows, so an old conversation pays more per prompt than a new one. This
shape previously ran only when the model called `builtin_scratchpad_search`. Not
measured.

## Where the code is

| Piece | Path |
| --- | --- |
| Floors, caps, and the block text | `crates/core/src/recall.rs` |
| The standing guidance for the block | `crates/core/src/prompts/sections/knowledge_base.txt` |
| The port the daemon fills | `crates/core/src/ports/recall.rs` |
| Looked up once per turn | `ConversationHandler::recall_lookup`, `crates/core/src/service.rs` |
| What the other blocks showed | `planning::listed_scratchpad_keys` and `planning::plan_note_keys` |
| Rendered on the first round | `surfaced_blocks`, `crates/core/src/context/mod.rs` |
| Embedding, every query, degradation | `crates/daemon/src/recall.rs` |
| The knowledge query | `PgKnowledgeBaseStore::nearest_by_embedding`, `crates/storage/src/knowledge.rs` |
| Its degraded form | `PgKnowledgeBaseStore::search_text_any_term`, same file |
| The scratchpad query | `PgScratchpadStore::nearest_by_embedding`, `crates/storage/src/scratchpad.rs` |
| Its degraded form | `PgScratchpadStore::search_text_any_term`, same file |
| The tag query | `tag_registry::nearest_tags`, `crates/storage/src/tag_registry.rs` |
