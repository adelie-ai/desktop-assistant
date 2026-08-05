# The tag vocabulary in front of knowledge-base writes

Knowledge-base reads filter tags by exact array overlap. `topic:weather`,
`topic:forecast` and `topic:weather-forecast` are therefore three tags that
never match one another, and the same intent fragments across them. The
assistant mints those variants itself: it writes the tag that fits the sentence
in front of it, not the tag it used last month.

`builtin_knowledge_base_write` puts every tag the model supplies through the
formal tag vocabulary (the `tag_registry` table, issue #108) before it stores
the entry. A tag the vocabulary considers the same concept as one already in
use is stored under the existing name.

## What the model sends

```json
{"content": "Rain is expected on Tuesday.",
 "tags": ["memory", "topic:forecast"],
 "new_tag_descriptions": {"topic:forecast": "Weather predictions for coming days"}}
```

`tags` stays a plain array of strings. The descriptions travel in a sibling map
keyed by tag name, not as objects inside `tags`: a tool schema that accepts
either a string or an object at the same position is a union, and a union in a
tool schema has broken whole model turns in this fleet before.

| Field | Meaning |
| ----- | ------- |
| `tags` | The tags to store, as today. |
| `new_tag_descriptions` | Optional map from a tag name to a one-line description of what that tag means. |

A description matters only for a tag the vocabulary does not already hold. An
existing tag matches on its name and answers before any embedding happens, so
its entry in the map is read by nothing and costs nothing.

The description is what makes the check work. The vocabulary decides that two
tags are the same concept by comparing embeddings, and a short facet tag such as
`topic:forecast` carries almost no signal on its own. `"topic:forecast: Weather
predictions for coming days"` sits close to `"topic:weather: Forecasts, rain,
and temperature"`; the two bare names do not.

Both the single-entry form and each object in the `entries` batch accept the
field. A batch ignores every top-level field, `new_tag_descriptions` included,
so a batched write describes its new tags inside each entry.

## What the write reports back

```json
{"ok": true, "count": 1,
 "entries": [{"id": "...", "tags": ["memory", "topic:weather"],
              "created_at": "...", "updated_at": "..."}]}
```

`tags` reports what was stored, which is not always what was sent. Without it a
model that wrote `topic:forecast` would go on believing the entry carries that
tag, search for it later, and find nothing - the exact failure the vocabulary
exists to prevent.

## Which writes are gated

Both paths that set tags:

- creating an entry (`content` plus `tags`);
- re-tagging an existing entry (`id` plus `tags`, no `content`).

A content-only update re-registers nothing. Its tags come from the stored entry,
so they are already in the vocabulary.

## Cost

One embedding per genuinely new tag, not one per write. A tag already in the
vocabulary is answered by a name lookup.

## When it is off

The vocabulary is a capability, not a requirement. It needs the database that
holds it and an embedding backend that can recognise a near duplicate. Three
states, and none of them fails a write:

| State | Behaviour |
| ----- | --------- |
| No database, or no embedding backend | The gate is not wired. Tags are stored as written, exactly as before. The daemon says so once at startup. |
| Wired, but the embedding backend fails or times out | The vocabulary is not consulted again for the rest of that write. Its remaining tags are stored as written. The daemon logs once for the write, not once per tag. |
| Wired and answering | Tags are resolved against the vocabulary. |

Losing a user's memory because an optional backend was unreachable would be a
far worse outcome than a duplicate tag.

## The time it can take

Two ceilings bound what a person waits for.

One embedding call is bounded at 5 seconds, matching the query-embedding timeout
the built-in tools already apply. A hung backend therefore costs one timeout,
not one per tag: the timeout is a failure, and the first failure stops the
vocabulary being consulted for the rest of that write.

One write call may spend 15 seconds in total consulting the vocabulary, counted
across every entry in the call. A backend that answers slowly raises no error,
so nothing else would stop it, and the caller chooses how many tags one write
carries. When the budget runs out the remaining tags are stored as written.


## What it does not do

It does not repair tags already stored. Registry rows written before the
normaliser was shared between the two paths carry mangled names
(`projectadelie-ai` for `project:adelie-ai`); those rows simply miss the
exact-name lookup and a correctly-named row is created beside them. Repairing
them is tracked separately.

It does not rename tags on existing entries. A redirect applies to the write in
front of it.
