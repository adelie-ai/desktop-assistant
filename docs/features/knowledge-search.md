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
 "results": [{"id": "...", "content": "...", "tags": ["..."], "metadata": {}, "updated_at": "..."}],
 "returned": 10,
 "truncated": true,
 "message": "results were truncated; narrow with a more specific `query`, ...",
 "scope_size": "MANY",
 "available_tags": ["project:adelie-ai", "topic:weather", "preference"]}
```

| Field | Meaning |
| ----- | ------- |
| `results` | The matched entries, best match first. |
| `returned` | How many entries are in `results`. Same name `builtin_scratchpad_search` uses. |
| `truncated` | Present, and `true`, only when the page filled up (`returned` reached `limit`) **and** the scope is larger than the page. A full page under `FEW` carries neither it nor `message`, because `FEW` already means the page holds the whole scope. It always travels with `message`, which says how to narrow. Its absence is the claim that nothing was left behind. |
| `scope_size` | `NONE`, `FEW`, `MANY`, or `UNKNOWN`. See below. |
| `available_tags` | Tag names carried by entries in the scope, most frequent first with the tag name breaking ties. No counts. At most 50. Empty when `scope_size` is `UNKNOWN`. |

## Scope, not match count

`scope_size` and `available_tags` both describe the **scope**: the entries that
pass the caller's `tags` and `exclude_tags` filters. Neither describes the
entries that matched the query.

The number of query matches cannot be computed for this tool. The search is
hybrid RRF (`crates/storage/src/knowledge.rs`): the full-text arm is
query-scoped (`tsv @@ query`), but the vector arm is not, because a cosine
distance is defined for every embedded row. "Entries matching the query" is
therefore every embedded row plus the full-text hits, which is the whole store.
Reporting that as a match count would state a falsehood.

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

## Where things live

| Concern | Location |
| ------- | -------- |
| Page type, `ScopeSize`, the two caps | `crates/core/src/ports/knowledge.rs` |
| The census SQL and both search arms | `crates/storage/src/knowledge.rs` |
| Tool response and schema | `crates/mcp-client/src/builtin.rs` |
| Census behaviour under a real database | `crates/storage/tests/knowledge_tag_census.rs` |
