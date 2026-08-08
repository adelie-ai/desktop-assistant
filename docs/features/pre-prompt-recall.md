# Pre-prompt recall: the `[Recall]` block

The assistant reaches its memory only when it decides to. It has to notice that
a search might help, choose a query, and spend a tool round. When it does not
notice, the knowledge base is a store nobody reads, and a note it stashed on its
own scratchpad an hour ago may as well not exist.

Pre-prompt recall makes memory arrive unasked. When a user prompt lands, the
daemon embeds it once, asks every index that shares that embedding space what is
near it, and puts a short list of candidates in front of the model as a
`[Recall]` system block, before the model's first move.

The block is a hint. Nothing is asserted, and the model decides whether any of
it matters.

## What the model sees

```
[Recall] Memory that may relate to what was just asked. It may not fit; ignore what does not. Each line is one entry: its id, its tags, and one line of what it says - not the entry itself. Look one up before you rely on it.
- kb-1a2b [preference, ui] Prefers dark themes in every editor
- kb-9f31 [infra, deploy] The deploy target is the lab cluster
...and 17 more entries also matched.
Notes on this conversation's scratchpad. Each line is one note: its key, then the start of what it says - not the whole note.
- deploy-window: Fridays after 18:00, never before
...and 2 more notes also matched.
Tags the entries above carry: infra, deploy, ui, preference
Procedures on file that may fit this situation. Each line is one skill: its name, then what it is for - not the procedure itself. None of these is chosen for you; check that one fits before you follow it.
- deploy-the-lab: Roll a new image out to the cluster and watch it settle.
- rotate-a-key [files missing]: Replace a credential and update every consumer.
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

Each skill line carries the skill's name - the handle it is fetched by - and its
own "when to use" description, capped at the same 200 characters. **The body
never appears.** A skill body is a whole playbook, and the arm's economy is that
recognition costs less than recall: a line says a procedure exists, and the model
reads it only if it decides to. A skill whose files have left disk carries
`[files missing]`: the body still reads and the steps are still good, but the
skill's directory is gone, so any script it bundles cannot be run.

The three kinds of `- ` line are separated by their labels, because they carry
different authority: an entry is what the assistant chose to keep across
conversations, a note is what this conversation happens to have written down, and
a skill is something to *do*.

The tag names are the tags the entries above them carry, most-carried first.
They are not searched for; they light up from the entries that surfaced.

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

**What a skill line is.** A procedure on file that may fit the situation, under
its own heading - a name and one line of what it is for, never the procedure. It
is not a recommendation: nothing chose it, and standing near the prompt is not
the same as applying to the work. The model reads it with `builtin_skill_get` and
checks it against the work in hand before following any of it, because a fact
that does not fit costs a few tokens to ignore and a procedure that does not fit
gets carried out. Only skills a person has approved appear, so an absent skill is
not evidence that the library has nothing.

**What the tag names are for.** They are the tags the entries above them carry,
so they are real names of this store and not guesses - the same vocabulary
`available_tags` reports after a search, offered before the model makes one. A
search or a filter on one reaches the entries the block had no room for. A filter
that returns nothing means no entry in that scope carries the tag.

**That the block never replaces a search.** An absent block is not an empty
store: the bar is conservative, and the lookup ran against the user's prompt
rather than against the question the model would have asked. The section's
mandatory search rule is unaffected.

The same text also lives in `runtime_system_instruction.txt`, the legacy
monolith that `assembled_static_sections_match_original` byte-compares the
assembled sections against. Both files carry it or the gate fails.

## The arms, one embedding, and a vocabulary that lights up from one of them

| Arm | Index | What it offers |
| --- | --- | --- |
| Knowledge | `knowledge_base.embedding` | The entries nearest the prompt, best first. |
| Scratchpad | `scratchpads.embedding` | This conversation's notes nearest the prompt. |
| Tags | none | The tags the surfaced entries carry, most-carried first. |
| Skills | `skill_index.embedding` | The approved procedures nearest the prompt, best first. |

The point of the tag names is vocabulary: the model's first knowledge search is
otherwise a guess at what words this user's tags use, and `available_tags`
answers that only *after* a search has already come back thin.

**The names are derived, not searched for.** A direct search of the tag registry
was measured and does not discriminate at any threshold: a registry row embeds
`"<name>: <description>"`, which is a label, and a prompt is a question, so the
distance between them measures style as much as subject. Acknowledgements sat
0.35 to 0.42 from the nearest registry row and real hits sat 0.22 to 0.44, so two
of four hits were further away than every acknowledgement. The registry's own
near-duplicate check works at 0.10 because it compares a label with a label.

Reading the names off the entries that surfaced is spreading activation from a
hit rather than a weak direct match, and three properties follow. It cannot fire
when the entry arm is silent. It costs no second query and no second embedding
comparison. And a name appears because it describes something the prompt actually
reached.

Only the entries the block **showed**. An entry the width dropped is not in front
of the model, so its tags did not light up.

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

## The skill arm: procedural memory

The three arms above are all **declarative** memory - what is true, and what
happened. A skill is **procedural**: how to do a thing. It is a different kind of
memory, and it wants a different cue.

Before this arm, a skill was reachable only through `builtin_skill_search`, which
is free recall and has the failure mode free recall always has: the model must
first suspect that a relevant skill exists. If it does not think to look, the
skill is invisible, however good it is and however many times it has helped.
`[Recall]` had already solved that for facts - recognition instead of free
recall - and skills got none of it.

**Procedural memory is cued by the situation, not by the query.** Nobody
retrieves how to ride a bicycle by searching their memory for it; the bicycle
cues it. "Deploy this" is a weak query and a strong situation. So the
prompt-cued arm here is the first half of the answer and not the whole of it -
the situation signal widens the cue later, and the arm reads it when it exists.

**Only a skill somebody wrote on this machine and approved is offered**, and
the two conditions answer different questions.

Approval (`approved_at`) records that a person agreed the procedure may be
followed. It is deliberately a separate axis from `trust_tier`, which records
where the skill came from: a skill Adele wrote for herself is
`TrustTier::Local`, the most trusted provenance the catalog has, and must still
not be followed until somebody says so.

An unapproved skill is **excluded**, not marked, and the reason is specific
rather than general. `builtin_skill_get` refuses an unapproved skill's body, so a
line offering one is a line the model can only fail on. Worse, it would accrue an
offer every turn it ranked near a prompt and could never accrue an open, because
the open is recorded by the read that is being refused - and "surfaced often,
never opened" is precisely the profile ranking treats as evidence to retire a
skill. Marking would therefore poison the use signal for exactly the skills the
assistant authored. The exclusion happens inside the scan, so an unapproved row
is absent from the spread as well as from the candidates.

What tells anyone those skills exist, then. `builtin_skill_search` says so in
words - "N matching skill(s) are awaiting approval and are not shown" - whenever
the model does search, and the catalog's `approved_at` column is indexed for a
browse surface to filter on. The block spends nothing on it, because it renders
on every prompt and a standing line about a procedure nobody can approve yet is a
nag with no resolution.

**A skill from outside this machine is excluded too**, and this one is about
provenance rather than consent. A skill installed from a repository or a
`.well-known` source carries a description its author wrote, and the platform
already rules that such text is third-party content: `builtin_skill_search`
returns the same field and is classified `Declared(SkillTrustTier)`, so a
non-local hit taints the turn and closes the tool gate. This block has no tool
call in it, so nothing would taint - the text would land in a system message,
ahead of the user prompt, with the Egress, Mutate and Execution tiers all still
open, and with no model choice and no attacker step needed.

Dropping is the answer rather than tainting, for exactly the reason the
scratchpad arm drops a note stamped as external: a catalog row lives
indefinitely, and closing the gate whenever one happened to rank near the prompt
would degrade the conversation permanently. An installed skill stays reachable
through `builtin_skill_search`, which taints correctly - the same shape as a
subagent's external answer, which the block never carries and
`get_subagent_status` does.

**The cost, stated plainly.** A library that is mostly installed rather than
written locally gets little from this arm today. Widening it needs either a way
for the block to taint the turn it opens, or a person's judgement on the
description itself - neither of which belongs in this arm.

**A skill whose files are gone is marked, not excluded**, and the asymmetry is
the same test applied twice: can the model act on the line? It can. The catalog
is cumulative, so the body still reads and `builtin_skill_get` still returns it.
Only the bundled scripts are unreachable, and `[files missing]` says so.

**The arm calibrates against its own dispersion.** A skill row embeds a name, a
short "when to use" line and a playbook body; a knowledge row embeds a fact, and
a pad row a telegraphic note. The three put their distances in different places,
so one bar means the same in all three only when each is read against its own
spread. The skill catalog is the case that makes the rule visible: it is small,
and its rows are shaped unlike anything else the block reads.

**One name, one line, and it is the line a fetch would open.** The catalog can
hold a global skill and a user's own under one name. Two lines for one openable
procedure would be two the model cannot tell apart, and a line describing a
procedure other than the one `builtin_skill_get` hands back would be worse - the
model briefed on one method and given another's steps. So the scan applies that
tool's own resolution: the user's own row while it is usable (on disk and
approved), else the global one, else the user's own tombstone.

Making the two agree also closed a defect in the tool. It used to prefer the
caller's own row whenever the files were on disk, so an **unapproved** personal
row shadowed a live global skill of the same name for every later fetch - and
`promote_plan_to_skill` writes unapproved personal rows under a name the
assistant chose. The fallback now reaches past an unusable personal row for
either reason, on the argument the tombstone case already carried.

The order of the passes is the whole design. A name resolves over every
approved row it has - including one this query would not have matched, since a
row the embedding backfill has not reached yet is the ordinary case. The trust
rule applies to the row that resolution landed on, so a non-local row shadowing
a local one drops the name outright instead of letting the block offer a line
the fetch will not return. Only then is the spread measured, over the set the
arm can actually draw from: a handful of local skills inside a large installed
library would otherwise be graded against the installed library's geometry, and
the minimum-sample rule would count those rows too.

Reaching past the caller's own unapproved draft also says so. The draft is one
the assistant wrote under a name it chose, so nothing else would mention it -
`builtin_skill_get` returns the shared skill and names the draft it passed
over.

**Offers and opens are recorded.** The block's own skill lines are written to the
skill use log as an offer, and a `builtin_skill_get` that hands the body back
records an open against a standing offer. A skill surfaced often and opened never
is therefore visible as such, and a skill opened repeatedly outranks a nearer one
nothing has read. The log has its own tables (`skill_use_stats`, `skill_offers`)
rather than the knowledge log's, whose foreign key to `knowledge_base(id)` a
skill has no row for. It records two acts and not three: no tool sets a mark on a
skill, so there is no marks table waiting for a writer.

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

A dropped candidate is not counted in the "also matched" line either.
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

**One dimensionless bar, and the width is what comes out.** Each arm drops a
candidate that does not stand out from its own source, rather than padding the
list out to fill the budget. A prompt with nothing that stands out emits no block
at all.

The bar is not a distance. A cosine distance means nothing on its own: what
counts as near depends on the embedding model, on how much text a row holds, and
on how wide the store's subject matter is, so a value fitted to one deployment
says nothing about the next. Each candidate is read as **how far below its
source's own median it sits, counted in that source's median absolute
deviation** - and one dimensionless bar decides. That is `RECALL_BAR` in
`crates/core/src/recall.rs`.

The width is then an output rather than an input. A prompt with no cue clears
nothing, a weak cue clears a line or two, and a strong cue clears a dozen. The
block is wide exactly when there is something to be wide about, and the
configured width is a safety cap on the worst case rather than the mechanism.

**One activation score decides the order.** The bar says which candidates are
offered; activation says in what order, and it is one function rather than a
blend of multipliers:

```text
A_i = semantic + reinforcement + situation + salience
```

`semantic` is the same dimensionless deviation count the bar reads - never a raw
distance, so a source added later joins on the same scale without refitting
anything. `situation` is how well the entry's own record matches the situation
the prompt arrived in, and `salience` is how much of what makes a fact worth
keeping the entry carries; both have their own section below. `reinforcement` is
what the use log knows
(`docs/features/knowledge-use-log.md`), read as
`use_lift * ln(1 + S / S_ref)` and carrying `S`'s sign, where `S` is the ACT-R
base-level sum over the entry's opens and marks and `S_ref` is the sum one use a
day old produces.

**Dividing by a reference use is what makes the term dimensionless**, and it is
the same property the semantic half gets from reading a distance against its
source's own spread. Every term of a base-level sum is an age raised to a
negative power, so the whole sum scales with whatever unit the ages are counted
in - the same history is about three hundred times larger stated in days than in
seconds. Adding one straight to a deviation count would let that choice decide
how much the use log is worth. The ratio cancels it, and leaves one number to
argue about: `use_lift`.

Four consequences:

- **An entry nothing has used contributes nothing**, so a store with no history
  renders exactly the block it rendered before this existed. The sort is stable,
  so distance breaks every tie.
- **One use a day old is worth `use_lift * ln 2`**, which at the shipped
  `use_lift` of 0.5 is about a third of a deviation - enough to settle a
  near-tie, since adjacent rows of a real store sit a few tenths apart.
- **Doubling the accumulated sum buys at most `use_lift * ln 2`**, however large
  that sum already is. That is the cap the use log asks for, as a property of
  the function rather than a clamp: it is what stops the retrieve-mark-retrieve
  loop compounding. It is a statement about the sum and not about the number of
  uses - one more use raises the sum by over a factor of two when it is far more
  recent than everything before it, and that is recency, which this score is
  meant to carry.
- **The whole term stays under three deviations** over any history a store
  produces, because it is a logarithm. That is a bound and nothing more: history
  cannot run away with the ranking, and a lead wider than three deviations cannot
  be closed by any history at all.

It does not follow that the best semantic match always keeps the top line, and
it is worth being exact about when it does not. The bar is 6.8 by construction,
so the weakest candidate in any block sits there, and the measured prompts put a
real hit between 7.3 and 11.4 deviations. The best match therefore leads the bar
by anywhere from half a deviation to 4.6, and only the wide end of that range is
out of reach of a large history. Ten opens inside the last half hour are worth
about two and a half deviations, which is enough to lead a best match sitting at
7.3.

**The narrow end is the design working.** A best match half a deviation above the
bar means the prompt named nothing the store really holds - and a weakly cued
prompt is exactly the condition under which what has been used recently should
lead. An entry the assistant has been reading all morning taking the top line on
a prompt that brushes it is the behaviour this score was chosen for, and it is
the reason for base-level activation rather than a tiebreak bolted onto distance.
When the prompt does name something the store holds, the semantic term is several
deviations clear and the ceiling keeps that line where it belongs. The use log's
caution is about the second case, and the second case is what the ceiling
protects.
- **A standing negative mark subtracts**, so an entry that was opened and found
  wrong ends below one nobody has ever opened.

The score is `crates/core/src/domain/activation.rs`. Its weights are a struct
rather than constants, so a deployment that has kept a use log can fit its own.

**A lexical match is not ranked by activation.** It carries no distance, so it
has no semantic term - and scoring it on the use log alone would order a
degraded block by what has been opened most and throw away how well each row
matched. That is the embedding backend being unreachable, so it is the worst
moment to make the ranking worse. Such a candidate keeps the order the database
gave it, and one lookup uses one mode, so the two kinds are never in one list.

**Reading the use log costs the ranking at most.** The ids exist only once the
scan has answered, so it is one batched read after it, bounded at half a second.
A read that fails or overruns leaves every candidate on its semantic signal
alone - which is how they all ranked before the log existed - and says so once in
the journal. It states the same half second to the database as its own
`statement_timeout`, for the reason the recall scan does: giving up here has to
stop the backend working, or a slow database accumulates abandoned reads at the
rate turns arrive. One debug line per lookup also states how many of the candidates the
log had anything to say about, so an operator can tell whether reinforcement is
deciding anything at all.

**Measured over the source, never over the candidates.** The lookup shows only
the nearest rows, so their spread is the near tail's and not the store's, and
normalizing inside a truncated set inflates every score.
`PgKnowledgeBaseStore::nearest_by_embedding` measures the median and the median
absolute deviation over every row the scan could reach.

## The situation as a cue

Retrieval is keyed on prompt text, which for a life assistant is the weakest cue
available: people describe life events vaguely, and rarely in the words an entry
was written with. Encoding specificity says recall depends on the overlap
between the cue present when a memory was written and the cue present when it is
sought - which is why walking into a room brings back what was thought there.
The situation is the strongest cue the system holds, and it costs nothing to
collect, because the system already knows it without asking.

**What a situation is.** Three fields today, each optional and each read only
where its source is connected: the host the client reported (#549), and the part
of the local day and the day of the local week, both computed in the person's
own time zone. A field with no source contributes nothing rather than a
placeholder, and the two clock fields are gated on knowing the zone - a time of
day computed in the daemon's zone is a wrong value rather than a missing one, and
a wrong value costs every entry that recorded the same instant honestly. Adding
a dimension is a variant on `SituationField` and one arm in `Situation::observe`.

**When it is recorded.** An entry written by the model inside a turn acquires
the situation it was written in, and every entry adds to its record each time it
proves useful somewhere new - #238's accumulation rule. An entry the dream cycle
extracts has no client context to read, because no client is present when the
cycle runs, so it carries no situation until the first time it is reused. The
reuse write runs against the ids the open transaction counted, so an entry that
was not standing offered accumulates nothing, exactly as it accrues no open.
Neither write touches `knowledge_base`: an entry that had to be rewritten to
learn where it is useful would restate its own content, move its `updated_at`,
and put itself back in the embedding backfill queue.

**Presence is the match, not how often.** The record holds how many times each
value has been seen, and no ranking rule reads it. The use log already measures
how much an entry has been used, so weighting the match by a count would put that
signal into `A_i` twice - and it would leave the loop open, because an entry that
ranks up in a situation gets opened there, which records the situation. A binary
match closes that after one step: recording a value the record already holds
changes nothing at all.

**Each cue value is weighted by what it separates.** A plain overlap fraction has
a defect that only shows on a real store. Most deployments have one host, so
every entry carries it, every prompt matches it, and the term becomes a constant
added to every entry that happens to have a record - reordering nothing among
them and sinking every entry written before the feature shipped. The cue would be
measuring when the code landed. So each value is weighted by its own
self-information over the store, `ln(population / fan)`:

- A value every entry carries is worth **zero**. Your only host tells nobody
  anything.
- A value one entry carries is worth `ln(population)`, the most the store can
  offer.
- A value **no** entry carries is worth zero as well, and for the same reason: a
  field on which no candidate can match separates nobody. Without that rule, a
  cue value the store has never met would be maximally informative and would
  silence the fields that did match.

**Both counts are per field.** Which fields an observation can read depends on
the client that made it, so a store's coverage is uneven: a host may sit on a
third of the entries while the weekday sits on all of them. Divided by one
store-wide count, the only host in a store would come out informative merely
because two thirds of the entries record no host at all - the very error the
weight exists to prevent, displaced from "before or after the feature shipped"
onto "which client wrote it".

This is Anderson's fan effect, arriving as the definition of the weight rather
than as a correction bolted onto one.

**The term is a ratio, so it cannot grow with the fields connected.** Coverage is
the information the cue carries on the fields this entry could have matched,
divided by the information it carries on the fields it did:

```text
coverage = sum(information of matched fields) / sum(information of comparable fields)
```

A field the entry has never been seen with is in neither half, so a missing field
neither matches nor penalises. That choice has a price worth stating: an entry
that knows one thing about itself and gets it right reaches full coverage on less
evidence than one that knows three. The alternative - dividing by the whole cue -
scores "we do not know" the same as "we know, and it was somewhere else", and
conflating an unknown with a mismatch is the worse error for a store whose older
half was written before any of this was recorded. A field the entry knows and
disagrees on is in the denominator only, so a mismatch forfeits the lift rather
than subtracting from the score.

**The bound is a scale, not a fit.** A full match is worth exactly what one use
at the reference age is worth, computed from the reinforcement term rather than
restated. It introduces no coefficient of its own, it carries no unit of its own
- it works out to `use_lift * ln 2` whatever the decay exponent and whatever unit
the use log's ages are counted in - and a deployment that fits `use_lift` from
its own log moves both terms together. At the shipped weights that is about a
third of a deviation: a ninth of the reinforcement ceiling, so the situation can
settle a near-tie and can never overturn a semantic lead. Its influence is
largest where the admitted band is narrowest, which is the weakly cued prompt.

**It ranks and never admits.** The cue is read after the bar, over the set the
bar admitted, so it permutes the block and cannot change its membership. That is
what keeps the "and N more entries also matched" hedge true.

**A field too thinly recorded to measure is weighted at zero.** A fan over a
handful of entries is noise, and noise in the weight makes the ratio
meaningless, so a field recorded on fewer than `SITUATION_MIN_POPULATION`
entries contributes nothing while the fields beside it keep what they are worth.
A cue whose every field is below the floor is no cue at all, and every entry
then ranks the way it ranked before this existed. The same holds for a
deployment with nothing connected - where the read is skipped rather than run
and discarded - and for a read that fails or overruns its half-second ceiling,
where the block loses the order, never the lines, and says so once in the
journal.

**A value a client chooses never becomes a value the database refuses.** The
host is self-reported and nothing upstream bounds it - `sanitize_client_field`
runs at the prompt renderer, not at the transport - and it becomes part of a
primary key and a btree index. So every situation value is trimmed, lowercased,
stripped of control characters and cut to `MAX_SITUATION_VALUE_CHARS` before it
is stored, the trade a mark's reason already makes. The cleaner is idempotent,
because a value read back out of the store passes through it again. Lowercasing
earns its place on its own: one machine answering `Workshop` to one client and
`workshop` to another would hold two values, halve its own fan, and match
neither prompt in full.

**Recording the situation of a reuse cannot cost the reuse.** The write runs
after the transaction that counts the open has committed, in its own
transaction, and its failure becomes a warning. Inside that transaction an
unmigrated database or a missing grant would roll back the open and the counter
with it, taking out the strongest signal in the use log for as long as the cause
lasted, and saying so only in a log line. What is given up is atomicity between
the two, which costs nothing: the record is idempotent by key, so the next reuse
in the same situation records what this one missed.

The rule is `crates/core/src/domain/situation.rs`; the table is
`knowledge_situation`, created by `047_knowledge_situation.sql` and bounded per
entry per field with the least recently seen value evicted first.

## Salience

Some facts matter more than others, and the software analogues of that are
cheap. Five signals, none of which needs a model call:

| signal | read from |
| --- | --- |
| a person deliberately promoted the entry | the `source` column, where it is `explicit` |
| the entry names a date something is wanted by | phrases in its text |
| the entry is about money | phrases in its text |
| the entry is about health | phrases in its text |
| the entry records a promise made to somebody else | phrases in its text |

**Read at scoring time, never stored.** The reading is taken from the entry's own
body, summary, tags and provenance, all of which the recall scan already selects,
so there is no column, no write and no extra query. A detector added later
applies to every entry ever written rather than only to the ones written after
it, and an entry rewritten by consolidation is re-read. It also settles the "term,
never a gate" rule by construction: salience has no write path to consult, so a
low-salience fact is stored exactly as any other.

**The signals divide one fixed lift rather than each adding one.** The term is a
ratio - of the salience information this build can detect, how much does this
entry carry - so it cannot grow with how many signals a deployment happens to be
able to detect. A sixth signal takes from the five. A detector that never fires
on a store scales every entry's share by the same factor and therefore reorders
nothing, which is what the English-only phrase lists cost a store in another
language: the ranking it always had.

**The signals are not weighted equally, and the weighting is not a new number.**
A person asking for something to be kept is stronger evidence than a body of text
mentioning money, and the two are separated by who said it - priced by the ratio
the use log already declares between a person's mark and the model's.

**The bound is the same scale the situation gets:** a full reading is worth
exactly what one use at the reference age is worth, about a third of a deviation.
A mark in the use log records something that happened and a salience signal reads
what text means, so a reading is bounded by one recorded use and never outweighs
it.

**It ranks and never admits**, on the same terms as the situation: it is applied
after the bar, over the set the bar admitted, so the "and N more entries also
matched" hedge stays true.

**Two of the five signals #1127 names are deliberately absent.** A correction of
something the assistant said is already recorded as a negative mark in the use
log, which the reinforcement term reads and the daily pass reads again as its
contradiction term - detecting it a third time from text would count one fact
three times. Repetition across separate conversations has nothing to read,
because recurrence today writes a second entry rather than reinforcing the first;
it arrives with extraction-time matching.

The rule is `crates/core/src/domain/salience.rs`. A skill candidate answers no
salience at all: every signal is read off a knowledge entry, and a skill holds
none of those.

**One scan states both.** The candidates and the spread are functions of the same
query vector: the spread says what a distance from this store is worth, and the
candidates are the distances it grades. An answer held from an earlier prompt
would grade this one against a geometry nothing here saw, and the margin has no
room for that - about 0.4 deviations between the two classes, which is a few
hundredths of cosine distance. Computing them together costs one pass rather than
two: the scan is what the query spends its time on, the pass that measures the
spread reads one distance per row and no content, and only the rows the block may
show are read whole.

The scan states a `statement_timeout` of four seconds, so the ceiling the caller
keeps is a ceiling the database keeps too. Abandoning a query stops the daemon
waiting and leaves the backend scanning, and recall runs before every turn.

**Which sources measured, and which did not, is logged.** A source that states no
spread is read by the stated estimate, which admits at a fixed cosine distance -
the mechanism this design exists to remove, and it must not apply in silence. One
debug line per lookup names each source and which of the two applied, and carries
nothing of what either source holds.

Before a source can measure its own, the block reads it by a stated estimate,
which is deliberately narrow. A measurement is refused, and the estimate stands,
when the sample is under 20 rows, when a value is not a number, or when the
spread is under two percent of the median - a store of near-identical rows would
otherwise report almost no spread and put half of itself past any bar.

**A token budget, and a line budget derived from it.** The whole block is
allowed 2,560 tokens in the worst case. That is two percent of a
128,000-token window - the smaller of the two window sizes the models this
daemon drives usually carry - spent once per turn, on the first round only, on a
hint the model is told it may ignore.

Every part of every line is capped: 64 of id, 120 of an entry's tags, 200 of
summary, 128 of a note key, 200 of a note's content, 64 of a skill name, 200 of a
skill's description, and 240 for the whole tag line. So a worst-case line costs a
known size, and the width is arithmetic rather than a choice:

| Part | Worst case |
| --- | --- |
| One knowledge line | `"\n- " + id + " [" + tags + "] " + summary` = 391 bytes |
| One skill line | `"- " + name + marker + ": " + description` = 284 bytes |
| The fixed part | prefix, header, entry hint, five pad lines and their label, the tag line, three skill lines and their label, and all three "did not fit" lines = 3,519 bytes |
| The budget | 2,560 tokens at four bytes a token = 10,240 bytes |
| **The width the budget buys** | **(10,240 - 3,519) / 391 = 17 lines** |

**A new arm is paid for out of the quotient, never out of the budget.** The skill
arm took the block from twenty knowledge lines to seventeen. What a turn pays for
the block is the number the model's context notices, so it is the number held
fixed; the width is derived from it, and a width that moves is the mechanism
working. Raising the budget to keep the old width would have made the block cost
more on every prompt to spare an arithmetic result.

**The shipped width is the width the budget pays for.** The bar decides how many
lines render, which leaves this number protecting the token budget and nothing
else - so the budget's own figure is the value it takes. A lower one would be a
second, unstated policy about width. A deployment on a model with a small context
window states a narrower one with `max_entries`.

**Bytes, not characters, and the difference matters.** The token estimate the
context budget uses is `bytes / 4`, and a character is one to four bytes.
Nothing restricts an entry id, a tag, or a model-written summary to ASCII, so a
line inside its character bounds can carry four times the size the block is
charged for. Each rendered line is therefore cut to its byte bound, on a
character boundary, and a line that lost anything ends in `...` the same way a
truncated value does. An all-ASCII line is already inside the bound and passes
through untouched, so the usual line is unchanged; a line in another script
shows fewer characters for the same cost.

Real entries carry a short id and a few tags, so a real line costs about a
quarter of the worst case. The width rests on the bound rather than on the
average, because only the bound is a promise. `crates/core/src/recall.rs` pins
every number with a test - the worst case in ASCII, the worst case in a
four-byte script, and the usual case, each at the shipped width and at a
narrower one - so a later change to the line format cannot inflate the block in
silence.

Five scratchpad lines, five tag names, and three skill lines. Those are not what
the budget buys: the pad holds one conversation's notes, the tag line hands over
a vocabulary rather than listing one, and a skill line is the strongest nudge the
block can make - a long list of procedures reads as a plan rather than as a hint.
All three are safety caps on the worst case; the bar still decides how many
render.

**One round.** Every other per-turn block re-renders each round, because each is
answering "is this still in view?". `[Recall]` answers "what might this prompt be
about?", and the user prompt asks that once. Repeating it across twenty tool
rounds would spend thousands of tokens on an answer the model has already taken
or ignored.

## Saying what did not fit

A model that sees the lines the block had room for cannot tell whether the store
holds exactly that many relevant things or four hundred, and those call for
different next moves - accept the list, or go search properly. So the block ends
with a count of what cleared the bar and did not fit.

That count means something only because the bar defines it. "How many matched"
is not a defined quantity over a hybrid search, where every embedded row scores
non-zero against any query - the same trap that produced `scope_size` instead of
a match count in the search tool.

Each arm counts its own, and says which. The knowledge lookup reads at most 50
rows; the scratchpad and skill lookups read at most 25 each - fewer, because one
reads a single conversation's pad and the other a catalog of tens of rows, so the
tail either would be counting is short. When a scan fills up *and* every row it
read cleared the bar, that arm's count is a lower bound and says so: "and 42 or
more entries also matched." When the scan read past the bar, the count is exact
and carries no hedge. When nothing was dropped, there is no line.

**The line says "also matched", and neither ranks nor asserts.** It must not
rank: distance decided the order until activation ranking landed, so an entry
that did not fit may now have matched more closely than one that did, and the
scratchpad arm, which activation does not rank, would then be described one way
and the entry arm another. It must not assert: the standing guidance beside this
block says nothing in it is asserted to be true, current, or relevant, so a line
calling the remainder relevant would contradict the block it closes. What is
left is the fact the count is made of - these rows cleared the bar, and there was
no room for them - which is true of every arm, and of a capped scan.

Activation ranking does not touch the count or the hedge, and the reason is worth
stating because it is easy to assume otherwise. The rule the hedge rests on is
about **admission**: rows arrive nearest-first, the bar is a test on distance, so
the rows it admits are a prefix - and a scan that read past the bar therefore
knows there is nothing better beyond it. Activation reorders that admitted set
and never changes its membership, so the count of it, and whether that count is
exact, are the numbers they were. What ranking does change is which admitted rows
*fit*: on a capped scan the lines are the best of what was read rather than the
best that exists, which is what the hedge already says.

## Failure

Recall never fails a turn.

The embedding call is bounded at five seconds, the same ceiling the
knowledge-base search tool applies. On timeout or an embedding error every arm
degrades to full-text search, and no dispersion is measured - a full-text row
carries no distance to read against a spread. The degradation is logged once, not
once per arm. The degraded skill read applies the same approval rule as the
scan: a backend outage must not turn into an offer of something nobody approved.

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

Over the full-text path there is no distance to compare, and no bar is applied: a
row that carries none of the prompt's terms is never returned, which is a floor
of its own.

If the degraded read fails as well, the block is omitted and the turn proceeds.

An arm that fails outright is a narrower loss. The scratchpad arm reads a
different table from the knowledge arm, so it can fail on its own, and when it
does it costs its own lines and nothing else: the knowledge arm still renders.
The knowledge arm gets no such treatment, because a knowledge arm that cannot
read is the block's whole point failing - losing the pad lines is a smaller loss
than losing the block.

A dispersion measurement that fails costs the block its unit and nothing else:
the candidates still travel, and the block reads them against the stated
estimate. The failure is logged and the estimate is not held, so the next turn
measures again.

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
# max_entries = 20   # absent means 20, the width the token budget pays for
```

It also stays off on its own when there is no knowledge store or no embedding
backend, and the daemon says which of the three reasons applies at startup.

`max_entries` states the most knowledge lines the block may show. It is a safety
cap rather than a target: the bar decides how many lines a prompt actually
renders, and most prompts render fewer. Lower it on a model with a small context
window. The value is held to what the block can honestly render - at least one
line, and never more than the 50 rows the lookup reads, because a block that
showed more would count a tail it never saw.

Both settings are read once, when the conversation handler is built, so an edit
needs a restart. A live config reload reports `[recall]` as a restart-required
area rather than reporting no change, and the daemon logs the width it wired at
startup.

Turning it off restores exactly the behaviour that preceded the feature: the
assistant reaches its knowledge base only when it decides to search.

## Known limits

**The bar was measured against one store.** It separated cleanly over thirteen
prompts on one store with one embedding model: the strongest candidate for a
prompt with no cue reached 6.4 deviations, and the weakest candidate for a prompt
with a real cue reached 7.3, so the margin between the two classes is about
fifteen percent. The bar is dimensionless, so it has a far better claim to
carrying across stores and models than a raw cosine ceiling does, and that claim
is untested on a second store.

**The stated estimate is a stand-in, not a measurement.** A store under 20
comparable rows has no measurable geometry, so the block reads it by a fixed
median and spread until it has one. Those two numbers are the one place a
distance is still stated by hand.

**The pad is read by the stated estimate too, and its admission tightened.** One
conversation's scratchpad rarely holds enough rows for a median absolute
deviation over it to be a measurement rather than noise, and the pad read is
already the block's most expensive query, so no second pass measures it. The
estimate is the knowledge store's, and a note embeds `"<key> <content>"`, which
is terser than an entry's body - so the two are not the same distribution. In
cosine terms the pad now admits a note about a third nearer than it did. The
direction is quiet rather than loud, which is the safe one, but it is a change
and not a measurement. #1146 covers measuring the pad where it is large enough
to state its own.

**The spread costs two sorts over the scanned distances.** They are sorts of one
double per row, on rows the scan has already computed, so they are small beside
the vector arithmetic that dominates - but they are not free, and #859 is what
makes the scan itself structural past roughly ten thousand entries. Holding the
measurement between turns is the obvious saving and it is deliberately not taken:
the statistic is a function of the prompt, so a held one grades the wrong
geometry.

**The scan reads whole rows for the ones it may show.** Fifty entries are ranked
to render a handful of lines, because the count of what did not fit has to be a
count. The row count is bounded; the bytes those rows carry are not, so a store
of unusually long entries pays more per prompt than a store of one-liners.
Measured against a populated store, the whole query costs a few milliseconds.

**It fires on every turn, including agent and subagent runs.** Any turn that goes
through `send_prompt` gets a lookup, so a spawned agent working from a
machine-written brief pays one embedding and three reads as well.

**The skill scan is a scan.** `skill_index.embedding` is a `vector[]` column and
takes no ANN index, the same structural limit the knowledge base has, so the arm
unnests every approved row's chunks on every turn. A skill catalog holds tens of
rows where a knowledge base holds thousands, so this is the cheapest of the three
reads today - and it grows with the catalog. Not measured.

**The skill use log is read on every turn that finds a candidate**, on the same
terms as the knowledge one: a batched primary-key read after the scan, bounded at
half a second, whose failure costs the order of the skill lines and never the
lines.

**A pin the `[Pinned]` byte budget cut short is still suppressed.** The knowledge
arm drops the attachments the turn *resolved*, which is a superset of what
`[Pinned]` had room to print. On that rare turn the block says in its own words
that pins did not fit, so the model is not left believing the fact is absent.

**Cancellation waits on the lookup.** A turn cancelled while the lookup is in
flight still waits for it, bounded by the ten-second whole-lookup ceiling. The
use-log read added half a second to what that ceiling has to cover, so the slack
it keeps for what nothing bounds - pool acquisition above all - is now half a
second rather than one.

**The use log is read on every turn, including the ones with no cue.** The bar is
the core's decision and the read is the adapter's, so the adapter cannot know
that nothing will clear. On a prompt that surfaces nothing, two primary-key
lookups over at most fifty ids are made and every row is discarded.

**No client can write a person's mark.** The use log has no wire surface, and the
only mark any code path writes is the model's. So `person_mark`, the largest
coefficient the score carries, is unreachable today, and a person has no way to
say "this entry was useful" or to ask why a line is at the top.

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
| The activation score | `crates/core/src/domain/activation.rs` |
| The base-level sum it reads | `KnowledgeUseRecord::use_sum`, `crates/core/src/domain/knowledge_use.rs` |
| The situation cue, and what it is worth | `crates/core/src/domain/situation.rs` |
| The salience signals, and what a reading is worth | `crates/core/src/domain/salience.rs` |
| The standing guidance for the block | `crates/core/src/prompts/sections/knowledge_base.txt` |
| The port the daemon fills | `crates/core/src/ports/recall.rs` |
| Looked up once per turn | `ConversationHandler::recall_lookup`, `crates/core/src/service.rs` |
| What the other blocks showed | `planning::listed_scratchpad_keys` and `planning::plan_note_keys` |
| Rendered on the first round | `surfaced_blocks`, `crates/core/src/context/mod.rs` |
| Embedding, every query, degradation | `crates/daemon/src/recall.rs` |
| The bounded use-log read | `recall::use_records`, same file |
| The bounded situation read | `recall::situation_signal`, same file |
| The situation writes and the fan count | `crates/storage/src/knowledge_use.rs` |
| The knowledge query, and the spread it states | `PgKnowledgeBaseStore::nearest_by_embedding`, `crates/storage/src/knowledge.rs` |
| Its degraded form | `PgKnowledgeBaseStore::search_text_any_term`, same file |
| The scratchpad query | `PgScratchpadStore::nearest_by_embedding`, `crates/storage/src/scratchpad.rs` |
| Its degraded form | `PgScratchpadStore::search_text_any_term`, same file |
| The skill query, and the spread it states | `PgSkillIndexStore::nearest_by_embedding`, `crates/storage/src/skill_index.rs` |
| Its degraded form | `PgSkillIndexStore::search_text_any_term`, same file |
| The skill use log | `crates/core/src/ports/skill_use.rs`, `crates/storage/src/skill_use.rs` |
| Where a skill open is recorded | `BuiltinToolService::record_skill_open`, `crates/mcp-client/src/builtin.rs` |
| The tag names | `recall::carried_tags`, `crates/core/src/recall.rs` |
