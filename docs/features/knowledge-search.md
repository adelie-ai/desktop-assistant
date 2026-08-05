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
| `truncated` | Present, and `true`, only when the page filled up (`returned` reached `limit`). It always travels with `message`, which says how to narrow. Its absence is the claim that nothing was left behind. |
| `scope_size` | `NONE`, `FEW`, or `MANY`. See below. |
| `available_tags` | Tag names carried by entries in the scope, most frequent first with the tag name breaking ties. No counts. At most 50. |

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

The three values:

- `NONE` - the scope holds no entries. No other filter would have done better.
- `FEW` - every entry in the scope fit in this page, so narrowing gains nothing.
- `MANY` - anything else.

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
    ORDER BY created_at DESC
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
2. **`ORDER BY created_at DESC`.** It rides
   `knowledge_base_user_id_created_at_idx` (migration 016), so the read stops
   after 1000 rows instead of scanning the table. It also makes the sample
   stable: a bare `LIMIT` takes rows in heap order, which moves after any
   `VACUUM` or update, so two identical searches would report different tags.
3. **The 1000-row cap.** A tail guardrail for a large multi-tenant store, not an
   optimisation of the common path. A personal knowledge base never reaches it.
   The tool schema states the cap, because a tag carried only by older entries
   can be missing from `available_tags`.

A sample that reached the cap says only "at least 1000", so it always classifies
as `MANY` - answering `FEW` there would claim the whole scope fit in a page the
caller may have sized above the cap.

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
