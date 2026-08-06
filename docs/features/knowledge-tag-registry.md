# The tag vocabulary in front of knowledge-base writes

Knowledge-base reads filter tags by exact array overlap. `topic:weather`,
`topic:forecast` and `topic:weather-forecast` are therefore three tags that
never match one another, and the same intent fragments across them. The
assistant mints those variants itself: it writes the tag that fits the sentence
in front of it, not the tag it used last month.

`builtin_knowledge_base_write` puts every tag the model supplies through the
formal tag vocabulary (the `tag_registry` table, issue #108) before it stores
the entry. A tag the vocabulary considers the same concept as one it already
holds is stored under the held name.

## What a tag is checked against

The vocabulary is the set of tags the registry has registered. It is not the set
of tags the knowledge base's entries carry, and the two are not the same set.

Nothing seeds the registry from `knowledge_base.tags`, and the only other writer
is dreaming extraction, which is off by default (`dreaming_enabled`). So on a
default install the registry is empty on the day this gate turns on, and it fills
up one tag at a time as writes go through it. The first write of `topic:forecast`
has nothing to match against and registers it; a later `topic:weather` is then
matched against it.

That is a real limit, not a description of the intended end state. Seeding the
registry from the tags entries already carry is #1094.

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

A description is matched to its tag on the normalised name, on both sides. The
model writes each field in whatever shape reads well, so `"Topic: Embeddings"`
as a key beside `"topic:embeddings"` in `tags` still finds its description, and
so does the reverse.

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

### A write that was not checked says so

```json
{"ok": true, "count": 1, "tag_check": "UNKNOWN",
 "entries": [{"id": "...", "tags": ["memory", "topic:forecast"], "...": "..."}]}
```

`tag_check` is present only when at least one tag on that write went to the
store without the vocabulary answering for it. Its one value is `UNKNOWN`, and
it means what `scope_size: UNKNOWN` means on `builtin_knowledge_base_search`
(see [knowledge-search.md](knowledge-search.md)): this was not measured. Across
the two tools, not measured never reads as measured.

Without the field a degraded write answers byte-identically to a checked one, so
the model reads its own wording back as established vocabulary - the failure the
gate exists to prevent, arriving through the gate itself.

## Which writes are gated

Both paths that set tags:

- creating an entry (`content` plus `tags`);
- re-tagging an existing entry (`id` plus `tags`, with or without `content`).

A write that carries no `tags` field registers nothing, because it proposes
nothing. The gate acts on the tags a write asks to store, not on the tags an
entry already holds, so a content update by `id` keeps the stored tags and
consults the vocabulary for none of them.

That is deliberate, and the reason is not that those tags are known to be
registered - most are not, per "What a tag is checked against" above. It is that
a redirect must not reach a tag the write never mentioned. Re-proposing the
stored tags would let the vocabulary rename them on a write whose author only
changed the text, and the caller would have no way to see it. The gap that
leaves - a tag carried by an existing entry that the vocabulary cannot match
against - belongs to #1094, which seeds the registry from the tags entries
already carry.

## Cost

One embedding per tag the registry does not already hold, on each write that
proposes it.

A tag that is *created* costs one embedding once: the next write of the same
name matches on the name and costs nothing. A tag that is *redirected* costs one
embedding every time, because the redirect records no alias - the proposed name
is never registered, so the next write of `topic:forecast` misses the name
lookup again and embeds again. Recording the alias is #1095.

## When it is off

The vocabulary is a capability, not a requirement. It needs the database that
holds it and an embedding backend that can recognise a near duplicate. Three
states, and none of them fails a write:

| State | Behaviour |
| ----- | --------- |
| No database, or no embedding backend | The gate is not wired. Tags are stored as written, exactly as before. The daemon says so once at startup, and every write with tags reports `tag_check: UNKNOWN`. |
| Wired, but the vocabulary fails or runs out of budget | It is not consulted again for the rest of that write. Its remaining tags are stored as written, the write reports `tag_check: UNKNOWN`, and the daemon logs once for the write, not once per tag. |
| Wired and answering | Tags are resolved against the vocabulary, and `tag_check` is absent. |

Losing a user's memory because an optional backend was unreachable would be a
far worse outcome than a duplicate tag.

## The time it can take

Three ceilings bound what a person waits for.

One embedding call is bounded at 5 seconds, matching the query-embedding timeout
the built-in tools already apply.

One whole consultation is bounded at 10 seconds. A consultation is not only its
embedding: it reads the vocabulary, embeds, searches for a near neighbour, and
registers. Each of those database round trips is bounded only by the connection
pool's acquire timeout, tens of seconds apiece, so bounding the embedding alone
left a saturated pool free to hold a live turn far longer. A consultation that
hits the ceiling is a failure, and the first failure stops the vocabulary being
consulted for the rest of that write - so a hung backend costs one ceiling for
the whole write, not one per tag.

The vocabulary may spend 15 seconds inside one write call, added up across every
entry. A backend that answers slowly raises no error, so nothing else would stop
it, and the caller chooses how many tags one write carries. Once the budget is
spent the remaining tags are stored as written.

The budget counts time spent consulting, not elapsed time. Reading an existing
entry and storing each entry are not consultations, so a batch with slow stores
keeps its full check.

The budget gates the start of a consultation, not its end, so one already in
flight finishes. The vocabulary's whole share of a write is therefore the budget
plus at most one consultation ceiling: 25 seconds.

## What it does not do

It does not repair the registry rows written before the normaliser was shared
between the two paths. Those rows carry mangled names - `projectadelie-ai` or
`project-adelie-ai`, depending on the raw string the model used at the time, for
what is now `project:adelie-ai`.

Nothing here guards against one. A mangled row misses the exact-name lookup, so
a correctly-named row is created beside it, but it is a close neighbour of the
correct name in embedding space, so it can still win the nearest-neighbour
search and capture a redirect. There is no cheap test that tells a mangled key
from a legitimate one: every one of them is in normalised form, so comparing a
candidate against the normaliser passes them all, and the rule that would catch
them - refusing a candidate that no knowledge-base row carries - needs an
evidence query that does not belong on the write path under this budget.

Repairing those rows is #1089, which already needs that evidence query, and it
is what closes this.

It does not rename tags on existing entries. A redirect applies to the write in
front of it.
