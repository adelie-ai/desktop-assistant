# Pre-prompt recall: the `[Recall]` block

The assistant reaches its knowledge base only when it decides to. It has to
notice that a search might help, choose a query, and spend a tool round. When it
does not notice, the store is memory nobody reads.

Pre-prompt recall makes memory arrive unasked. When a user prompt lands, the
daemon embeds it once, asks the two indexes that share that embedding space what
is near it, and puts a short list of candidates in front of the model as a
`[Recall]` system block, before the model's first move.

The block is a hint. Nothing is asserted, and the model decides whether any of
it matters.

## What the model sees

```
[Recall] Memory that may relate to what was just asked. It may not fit; ignore what does not. Each line is one entry: its id, its tags, and one line of what it says - not the entry itself. Look one up before you rely on it.
- kb-1a2b [preference, ui] Prefers dark themes in every editor
- kb-9f31 [infra, deploy] The deploy target is the lab cluster
...and 17 more entries matched less closely.
Tags near this prompt: project:adelie-ai, infra, deploy, ui, preference
```

Each entry line carries the entry id, the entry's tags, and the one line that
stands for it. That line is `KnowledgeEntry::display_line()`: the stored
`summary` where there is one, and otherwise a whitespace-collapsed prefix of
`content` capped at 200 characters. An entry with no summary is **not** skipped -
until the dream cycle has filled the column in, almost every entry has none, so a
block that skipped them would show nothing. An entry whose line comes out empty
is skipped, because it would spend a line of the budget to say nothing.

Tag names are listed nearest first, not alphabetically.

**The block names no tool.** Which read fetches an entry by its id is a property
of the tool set on the day the block renders, and a block naming a tool the model
cannot call is worse than one naming none: the model tries it and spends a round
on the failure. Saying what a line is - a stand-in for an entry, not the entry -
leaves the model to pick the read it actually has. Teaching the model what the
block is belongs in the standing instruction, not in the block.

**Every part of the block is bounded.** The id passes through the same one-line
rule the summary does. The block is line-structured and it is a system message,
so a stored value carrying a newline would forge a line - and the line above it
is a block header the model is taught to trust. An entry id is taken from the
write tool's caller and stored as written, so nothing before this point bounds
it.

Tag names are bounded by size rather than cut: a name that does not fit the
remaining width is left out. Half a tag name is a tag no row carries, and the
model is handed these names precisely so it can search on one.

## Two arms, one embedding

| Arm | Index | What it offers |
| --- | --- | --- |
| Knowledge | `knowledge_base.embedding` | The entries nearest the prompt, best first. |
| Tag | `tag_registry.embedding` | The names of the tags nearest the prompt. |

The tag arm reads vectors the near-duplicate check already built, and writes
nothing. Its point is vocabulary: the model's first knowledge search is otherwise
a guess at what words this user's tags use, and `available_tags` answers that
only *after* a search has already come back thin.

Descriptions do not travel with the names. A tag's `description` says what the
tag means; the caller is asking what this prompt is about.

The scratchpad is a third index in the same space, and is a separate piece of
work.

## What bounds the cost

**A relevance floor, not a top-k.** Each arm drops a candidate that is not near
enough, rather than padding the list out to fill the budget. A prompt with
nothing near it - "thanks", "run the tests" - emits no block at all. The floors
are cosine-distance ceilings, in `crates/core/src/recall.rs`. They are
deliberately conservative starting points rather than measured values: a block
that stays quiet costs nothing, and a block full of unrelated memory can pull the
model off the ask.

**A line budget.** Eight entry lines and five tag names. Every part is capped -
64 characters of id, 120 of an entry's tags, 200 of summary, and 240 for the
whole tag line - so the block cannot exceed about 3400 characters whatever the
store holds. Real entries carry a short id and a few tags, which puts the usual
block near 300 tokens.

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

The lookup reads at most 50 rows. When the scan fills up *and* every row it read
cleared the floor, the count is a lower bound and says so: "and 42 or more
entries matched less closely." When the scan read past the floor, the count is
exact and carries no hedge. When nothing was dropped, there is no line.

## Failure

Recall never fails a turn.

The embedding call is bounded at five seconds, the same ceiling the
knowledge-base search tool applies. On timeout or an embedding error the
knowledge arm degrades to full-text search, and the tag arm goes quiet - the
registry carries no full-text index to fall back to. The degradation is logged
once, not once per arm.

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

## Multi-tenancy

Both queries carry an explicit `WHERE user_id` predicate. Row-level security is a
non-FORCE backstop that the table owner bypasses, so the predicate is the guard.
`crates/storage/tests/recall_candidates.rs` holds both to it, and runs under
`just test-db`.

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
machine-written brief pays one embedding and two reads as well.

**Cancellation waits on the lookup.** A turn cancelled while the lookup is in
flight still waits for it, bounded by the ten-second whole-lookup ceiling.

## Where the code is

| Piece | Path |
| --- | --- |
| Floors, caps, and the block text | `crates/core/src/recall.rs` |
| The port the daemon fills | `crates/core/src/ports/recall.rs` |
| Produced once per turn | `ConversationHandler::render_recall_surface`, `crates/core/src/service.rs` |
| Rendered on the first round | `surfaced_blocks`, `crates/core/src/context/mod.rs` |
| Embedding, both queries, degradation | `crates/daemon/src/recall.rs` |
| The knowledge query | `PgKnowledgeBaseStore::nearest_by_embedding`, `crates/storage/src/knowledge.rs` |
| The degraded query | `PgKnowledgeBaseStore::search_text_any_term`, same file |
| The tag query | `tag_registry::nearest_tags`, `crates/storage/src/tag_registry.rs` |
