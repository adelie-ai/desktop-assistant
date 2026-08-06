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
[Recall] Memory that may relate to what was just asked. It may not fit; ignore what does not. To read one in full, search its wording with builtin_knowledge_base_search.
- kb-1a2b [preference, ui] Prefers dark themes in every editor
- kb-9f31 [infra, deploy] The deploy target is the lab cluster
...and 17 more entries matched less closely.
Tags near this prompt: deploy, infra, preference, project:adelie-ai, ui
```

Each entry line carries the entry id, the entry's tags, and the one line that
stands for it. That line is `KnowledgeEntry::display_line()`: the stored
`summary` where there is one, and otherwise a whitespace-collapsed prefix of
`content` capped at 200 characters. An entry with no summary is **not** skipped -
until the dream cycle has filled the column in, almost every entry has none, so a
block that skipped them would show nothing.

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

**A line budget.** Eight entry lines and five tag names, each entry line bounded
at 200 characters, which puts the whole block near 300 tokens.

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

Over the full-text path there is no distance to compare, and no floor is applied:
`tsv @@ query` is itself a binary relevance test, so a row that does not carry the
prompt's terms is never returned.

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

## Where the code is

| Piece | Path |
| --- | --- |
| Floors, caps, and the block text | `crates/core/src/recall.rs` |
| The port the daemon fills | `crates/core/src/ports/recall.rs` |
| Produced once per turn | `ConversationHandler::render_recall_surface`, `crates/core/src/service.rs` |
| Rendered on the first round | `surfaced_blocks`, `crates/core/src/context/mod.rs` |
| Embedding, both queries, degradation | `crates/daemon/src/recall.rs` |
| The knowledge query | `PgKnowledgeBaseStore::nearest_by_embedding`, `crates/storage/src/knowledge.rs` |
| The tag query | `tag_registry::nearest_tags`, `crates/storage/src/tag_registry.rs` |
