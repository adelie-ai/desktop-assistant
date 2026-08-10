# Knowledge-base search: what the response tells the model

The system prompt makes `builtin_knowledge_base_search` mandatory before the
assistant asks the user for any fact. A search that returns only `results`
leaves the model unable to tell an empty answer caused by a tag no entry carries
from an empty answer caused by a store that holds nothing, so it guesses tag
filters and each bad guess costs a turn.

The response therefore reports what was searched, not only what was found.

## Response fields

```json
{"ok": true,
 "results": [{"id": "...", "content": "...", "summary": "...", "tags": ["..."],
              "metadata": {}, "updated_at": "..."}],
 "returned": 10,
 "truncated": true,
 "message": "results were truncated; narrow with a more specific `query`, ...",
 "scope_size": "MANY",
 "available_tags": ["project:adelie-ai", "topic:weather", "preference"]}
```

| Field | Meaning |
| ----- | ------- |
| `results` | The matched entries, best match first. Each carries a `summary`: one line condensing what that entry says, so a caller can judge a hit without reading the whole `content`. It is `null` for an entry that has none: one stored before the field existed, one whose write named no summary, or one whose summary was cleared. |
| `returned` | How many entries are in `results`. Same name `builtin_scratchpad_search` uses. |
| `truncated` | Present, and `true`, only when the page filled up (`returned` reached `limit`) **and** the scope is larger than the page. A full page under `FEW` carries neither it nor `message`, because `FEW` already means the page holds the whole scope. It always travels with `message`, which says how to narrow. Its absence is the claim that nothing was left behind. |
| `scope_size` | `NONE`, `FEW`, `MANY`, or `UNKNOWN`. See below. |
| `available_tags` | Tag names carried by entries in the scope, most frequent first with the tag name breaking ties. No counts. At most 50. Empty when `scope_size` is `UNKNOWN`. |

## Where a summary comes from

`builtin_knowledge_base_write` takes a `summary` argument, in the single form and
inside each object of the batch `entries` form. The model writes it, one line,
saying what the entry says rather than naming its topic — because that line is
what stands in for the entry wherever entries are listed rather than read, and a
topic label tells a reader nothing it can act on.

The argument follows the same rule `tags` follows:

| The write sends | What happens |
| --- | --- |
| nothing, or `null` | The stored summary is kept. A create then stores none. |
| a string | Collapsed to one physical line, then cut to `SUMMARY_MAX_CHARS` (200) and stored. |
| an empty string, or one that is only whitespace | The stored summary is cleared, and the entry reads back with no summary again. |
| anything else | The write is refused. |

The collapse comes before the cut, so a loosely-formatted line is not cut far
shorter than a dense one saying the same thing — the rule and its reason live in
`desktop_assistant_protocol::one_line`. Whitespace-only counts as empty for the
same reason: the line is normalized first, and what is empty after that is empty.

Two boundaries hold, and both are deliberate:

- **A write with no summary is not refused.** Refusing it would lose the fact to
  gain a one-liner, which is the wrong trade for a memory store. Every read path
  reports the missing summary honestly as `null`, and a reader listing that entry
  falls back to the start of its content.
- **An over-long summary is cut, not refused.** A model that answers "one line"
  with a paragraph loses the tail of the line, never the write.

The write response reports the summary actually stored, next to the tags it
reports for the same reason: the boundary rewrites both, and a caller that cannot
see the stored value believes the entry carries what it sent.

"The write is refused" is per entry, not per call. A batch whose third entry is
malformed has already stored the first two, and the error carries none of their
ids — the shape #1113 tracks, which this argument gives one more way to reach and
does not otherwise change.

Cleared means absent, not empty: the store maps an empty summary to `NULL`, on
both halves of the upsert. An empty string would be a third state nothing wants —
a render site would print a blank row instead of falling back to the content, and
a pass over the entries that have none would look for them with
`WHERE summary IS NULL`.

Two consumers of the field are built. The `[Recall]` block offers candidate
entries to the model before its first move, one line per entry, and that line is
the summary (#1100). The dream cycle's summary backfill writes the line for
entries that carry none - every entry stored before the argument existed, and any
later write that named no summary - and writes it again when the body changes
after it (#1099); see `docs/features/knowledge-maintenance.md`. That pass is
bounded per cycle, so an entry written moments ago, or one in a store still
draining its backlog, has none yet, and a reader falls back to the body.

Beyond those two, a summary travels on every read: the knowledge tools report it
beside the content, and it reaches each client on the wire. What a given client's
knowledge browser does with it is that repository's own work; `display_line` on
the domain and wire types is the shared rule for it.

## Scope, not match count

`scope_size` and `available_tags` both describe the **scope**: the entries that
pass the caller's `tags` and `exclude_tags` filters. Neither describes the
entries that matched the query.

The number of query matches cannot be computed for this tool. The search is
hybrid (`crates/storage/src/knowledge_search.rs`): the full-text arm is
query-scoped (`tsv @@ query`), but the vector arm is not, because a cosine
distance is defined for every embedded row. "Entries matching the query" is
therefore every embedded row plus the full-text hits, which is the whole store.
Reporting that as a match count would state a falsehood.

## What decides the order (#1167)

`results` is ordered by the **activation score**, the same score the `[Recall]`
block ranks by, so a person cannot get one ordering from the tool and another
from the block.

The two arms admit and the score ranks:

- The **vector arm** measures every in-scope row this query can be compared
  with, states the store's own median and median absolute deviation over those
  distances, and admits the nearest of them. Each admitted row's semantic term
  is how many of the store's own deviations below its median this query put it
  - never a raw distance, which means nothing across a store or an embedding
  model. Added to it: what the use log knows about the entry, and what the
  entry's own text says about how salient it is.
- The **full-text arm** admits rows the vector arm cannot compare at all: one
  written since the last embedding backfill, or one still stamped with a
  superseded model. Such a row carries no distance, so it carries no semantic
  term and no score. It keeps the order the database ranked it in and follows
  the rows that were measured.

The spread is measured in the pass that ranks and is never cached: the median
and the deviation are statistics of the distances from *this* query's point, so
a query in a dense region of the store has a different distribution from one in
a sparse region. A store too small to state one is read by the same stated
estimate the block falls back to.

**What this costs, stated rather than left to be found.** On a store whose rows
are embedded, the full-text arm no longer decides any line of a full page - it
fills the page only where the vector arm returned fewer rows than were asked
for. A query whose whole signal is lexical (an identifier, a serial number, a
quoted phrase an embedding represents poorly) therefore lost the ranking help
the previous reciprocal-rank fusion gave it. Issue #1239 tracks the activation
score's own full-text-rank term, which is what gives that back without
reintroducing a fused rank.

The four values:

- `NONE` - no entry passes the filters the caller supplied. This says nothing
  about the store as a whole: dropping the filters may well find plenty.
- `FEW` - the scope is no larger than this page, so a plain listing would show
  all of it.
- `MANY` - the scope holds more entries than this page could show.
- `UNKNOWN` - the scope was not measured this time. Treat it as no information
  about the store, never as an empty store, and judge the page on `results`
  alone.

`FEW` does not mean the caller has seen everything. The page holds what matched
the query; the scope is what passed the filters. A query that matched nothing
still reports `FEW` when the scope is small, and the entry the caller wanted may
be sitting in that scope unmatched.

It is a bucket rather than a number because the count behind it comes from a
capped sample (below), and a raw figure invites the reader to trust a number
that is only exact under the cap. For the same reason `available_tags` carries
no counts: the ordering survives sampling, the counts would need a caveat.

## The tag census

One extra aggregate runs per search, over the same normalized filters the search
itself used:

```sql
WITH scope AS (
    SELECT tags FROM knowledge_base
    WHERE user_id = $1 AND deleted_at IS NULL
      AND ($2::text[] IS NULL OR tags && $2)
      AND ($3::text[] IS NULL OR NOT (tags && $3))
    ORDER BY created_at DESC, id DESC
    LIMIT 1000
),
census AS (
    SELECT t.tag, count(*) AS n
    FROM scope, unnest(scope.tags) AS t(tag)
    GROUP BY t.tag
    ORDER BY n DESC, t.tag
    LIMIT 50
)
SELECT (SELECT count(*) FROM scope) AS scope_count,
       COALESCE((SELECT array_agg(tag ORDER BY n DESC, tag) FROM census),
                ARRAY[]::text[]) AS available_tags
```

Three properties are load-bearing:

1. **`WHERE user_id`.** Tag names carry project and person names, so an
   unscoped census is a disclosure, not just a wrong number.
2. **`ORDER BY created_at DESC, id DESC`.** The `created_at` leg rides
   `knowledge_base_user_id_created_at_idx` (migration 016), so the rows arrive
   newest first without a sort. The `id` leg makes that order **total**.
   `created_at` alone is not: rows sharing one timestamp are cut apart by their
   physical position, which moves after any `VACUUM` or update, so two
   identical searches would report different tags and the model would see the
   vocabulary churn for no reason it can act on. `id` is unique, so it settles
   every tie. Postgres keeps the index early-stop and sorts each timestamp
   group incrementally.
3. **The 1000-row cap.** A tail guardrail for a large multi-tenant store, not an
   optimisation of the common path. A personal knowledge base never reaches it.
   The tool schema states the cap, because a tag carried only by older entries
   can be missing from `available_tags`.

A sample that reached the cap says only "at least 1000", so it always classifies
as `MANY` - answering `FEW` there would claim the whole scope fit in a page the
caller may have sized above the cap.

### What the cap does and does not bound

The cap bounds how many rows reach the aggregate. It does not bound how many
rows the read touches. `LIMIT` stops after 1000 rows **pass the filters**, so
the read is bounded by how many in-scope entries the user holds. A selective
`exclude_tags` that removes most of the recent entries therefore reads further
back, up to the whole of that user's index.

Measured on a 200k-row table, such a filter read to the end: `Rows Removed by
Filter: 199800`, 7595 buffers, about 68 ms warm. A filter that keeps most rows
stops early, as the common path does.

Sampling 1000 rows first and filtering afterwards would bound the read, and is
the wrong trade. It would report `NONE` for a scope that is merely old, which
is the exact falsehood this feature exists to remove. The read stays honest,
and the search survives a slow census because the census is best-effort.

### A census failure costs the measurement, not the search

The census is one extra statement, issued after the search has already returned
its entries. `PgKnowledgeBaseStore::search` therefore treats it as best-effort:
on error it logs once at `warn` and returns the entries anyway, with
`scope_size` `UNKNOWN` and an empty `available_tags`.

It reports `UNKNOWN` and never `NONE`. `NONE` is the positive claim that no
entry passes the caller's filters, so reporting it for a census that did not run
would tell the model the store is empty when the store may hold everything the
model asked for.

`UNKNOWN` also does not suppress `truncated` the way `FEW` does. `FEW` proves
the page holds the whole scope; `UNKNOWN` proves nothing, so a full page under
it is still evidence that entries were left behind.

## The degraded path reports the same fields

When the embedding backend times out the search falls back to full-text only
(`crates/storage/src/knowledge.rs`, the empty-embedding branch). That costs
semantic recall. It does not change the response shape: the census runs on both
branches, so a caller never has to handle a degraded contract.

## What the prompt tells the model to do with them

A field the model is never told to read is a field it does not use, so the
knowledge-base prompt section states the procedure:

1. Search with a natural-language question and no tags, then filter on a tag
   from `available_tags`. Never invent one. The standing advice to filter on the
   narrowest tag carries a condition in its own sentence - only once
   `available_tags` has shown the tag exists - so the model cannot read it as
   licence to filter on a guess, whichever line it reaches first.
2. When no tag fits, sweep with `builtin_knowledge_base_list` and its
   `next_cursor`, bounded at three pages of fifty.
3. When the sweep finds the entry, re-tag it, preferring a tag that
   `available_tags` already reports, and carrying the entry's existing tags
   forward.

Two of those are worth stating plainly, because both invite a wrong instruction.

**A larger search `limit` does find more entries, and is still the wrong
retry.** It finds more: `search_hybrid` admits `limit * 2` rows from the vector
arm, so that activation ranking has rows to lift, and `limit` from the full-text
arm, so a bigger limit really does surface entries a smaller one truncated
away. It is the wrong
move for a different reason - the model cannot tell how far down the ranking the
entry sits, so the retry is a guess that costs an embedding round-trip and may
still miss. A sweep is bounded and it reports what has already been read. The
prompt gives that reason rather than the tidier falsehood.

**A re-tag replaces the whole tag list.** `build_write_entry` uses the supplied
`tags` array verbatim, and the upsert is `SET tags = EXCLUDED.tags`. A model
told only to "add the missing facet" sends one tag and destroys the entry's KIND
tag and its project scope - so the prompt says the tags sent replace the old
ones, and to carry the existing ones forward.

Step 3 needs the tag-registry dedup gate on `builtin_knowledge_base_write`.
Without it, an instruction to add tags splits the vocabulary faster than the
census can report it.

## Reading an entry by id

Search cannot answer an id. It matches an entry's **content**, so an id finds an
entry only when that entry happens to mention it, and `builtin_knowledge_base_list`
filters by tag and source but never by id. `builtin_knowledge_base_get` is the
read that answers one:

```json
{"ok": true,
 "entries": [{"id": "...", "content": "...", "summary": "...", "tags": ["..."],
              "metadata": {}, "source": "explicit",
              "created_at": "...", "updated_at": "..."}],
 "returned": 1,
 "not_found": ["..."],
 "truncated": true,
 "message": "not every id was answered; ask for at most 64 ids at a time, ..."}
```

A row whose content had to be cut carries one extra field:

```json
{"id": "...", "content": "the start of a very long entry", "content_truncated": true,
 "summary": "...", "tags": ["..."], "metadata": {}, "source": "...",
 "created_at": "...", "updated_at": "..."}
```

It takes a batch (`ids`, or `id` for one), because the `[Recall]` block offers
several candidates and the model often wants two or three of them. Entries come
back in the order the ids were asked for, and a repeated id is read once.

Three rules matter more than the shape.

**A miss is a normal outcome.** An id that does not resolve is named in
`not_found` and the rest of the batch still returns. A stale reference is a
reference worth dropping, not an error (base rule 8.2).

**Every miss reads the same.** The store scopes the read by `user_id` and hides
retired rows, so another user's id, a retired id, and an id that never existed
are one case. The response says nothing about which, and carries no per-id
reason at all. Row-level security is a non-FORCE backstop that the table owner
bypasses, so the `user_id` predicate in `get_many` is the real guard;
`crates/storage/tests/knowledge_get_many.rs` holds it against a real database.

**Every bound reports itself.** At most 64 ids per call
(`KNOWLEDGE_GET_MAX_IDS`), and the response carries the same byte budget the
scratchpad read uses, so a batch of long entries cannot spend the whole context.
Any bound that bites sets `truncated` and a `message`. An id in neither `entries`
nor `not_found` was left out by the budget, not lost.

The third bound is the one the scratchpad read does not need. A note is capped at
`MAX_NOTE_BYTES` on write, so a single note always fits the response budget; a
knowledge entry has no write-side length cap at all. So the first row of a
response is not admitted on trust: an entry that alone overruns the budget
arrives with its `content` cut and `content_truncated: true` on the row. Leaving
it out instead would make a long entry unreadable by any call, and letting it
through whole would hand it to the generic 256 KiB tool-result cap
(`crates/core/src/context/mod.rs`), which cuts raw bytes and lands mid-JSON. A
later row is never cut - it waits for its own call, and the caller still holds
its id.

**The budget is not a guarantee about the whole row, and the gap is worth
stating.** Only `content` is cut. An entry whose `tags` or `metadata` alone
overrun the budget still travels whole and still reaches the generic 256 KiB
cap - the client-facing `CreateKnowledgeEntry` accepts any `metadata` value, and
neither the column nor the write path bounds it. Cutting those fields would
change what the entry means, so this read does not. The real fix is a size cap
on the write path, where one bound would hold all four reads; `content` is cut
here because it is the field a model writes and the one that grows.

## Where things live

| Concern | Location |
| ------- | -------- |
| Page type, `ScopeSize`, the two caps | `crates/core/src/ports/knowledge.rs` |
| The census SQL and both search arms | `crates/storage/src/knowledge.rs` |
| Tool response and schema | `crates/mcp-client/src/builtin.rs` |
| Census behaviour under a real database | `crates/storage/tests/knowledge_tag_census.rs` |
| The summary's write rules under a real database | `crates/storage/tests/knowledge_summary.rs` |
| Batch read by id, scoping and retirement | `crates/storage/tests/knowledge_get_many.rs` |
| Prompt guidance that consumes these fields | `crates/core/src/prompts/sections/knowledge_base.txt` |
