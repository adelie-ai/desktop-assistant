# Negative memory

Adele learns from being burned. An action that produced a bad outcome comes
back before the same action is taken again.

This is a separate system from the knowledge base, and it has to be, because
copying the factual policy onto it gets it wrong in five ways. The whole rule
lives in `crates/core/src/domain/negative_memory.rs`; the store is
`crates/storage/src/negative_memory.rs` over migration `049_negative_memory.sql`;
the two seams that use it are in the tool dispatch loop of
`crates/core/src/service.rs`.

## What a burn is

Three things, never a bare proposition.

| part | what it holds |
| --- | --- |
| action | the tool that went badly |
| context | the facets it went badly with |
| outcome | what went wrong, in the words of whatever recorded it |

Two things are matched, and only the second ever moves.

- **The act**: the tool's name, and a digest of the call's arguments exactly as
  they were made. This is the burn's identity, and it never widens. A call whose
  arguments differ in any way - one value changed, one argument added, one left
  out - is a different act and a different lesson.
- **The circumstance**: the situation facets - the host, the part of the day,
  the weekday - described in [pre-prompt recall](pre-prompt-recall.md). These
  are what a second occurrence drops.

That is stricter than it has to be, deliberately. A burn that fails to fire
costs one repeat of a mistake. A burn that fires on the wrong act costs the
assistant's usefulness in a way nobody can see, and that is the whole argument
of this feature.

The burn also *records* the arguments it can hold whole - short scalars, free of
control characters, up to a cap, kept in name order - so a warning can say which
call it is about. That record is for reading, never for matching. The digest has
no such limit and hashes every argument at full length, which is what stops two
long commands in one directory folding into one lesson.

## Strength and scope move in opposite directions

This is the idea the rest falls out of.

**Strength is full on the first bad outcome.** A fact earns strength by
reinforcement; a burn does not have that time, because the second occurrence is
the one the memory exists to prevent. Nothing raises a burn above full. Strength
only ever falls: it halves every two weeks without a repeat, and a burn under a
quarter of full - four weeks - stops interrupting anything. It stays readable
for another four weeks after that, then the next write drops it.

**Scope is as narrow as the evidence allows.** A fresh burn requires every
situation facet it was observed in, on top of the act itself, so it fires only
on a repeat of exactly what went wrong. It widens only when the same act fails
again somewhere else: the situation facets the second occurrence *observed with
a different value* are dropped, because the failure happened without them, so
they were not the cause. Two failures in the same situation widen nothing, and
neither does a repeat that observed no situation at all - absence is not
disagreement, and reading it as disagreement would let one headless retry strip
a burn's whole circumstance.

Over-generalization is the failure mode this shape exists to prevent. An
assistant that turns "this failed once" into "never do this" becomes uselessly
cautious, and the caution is invisible - it presents as reticence rather than as
an error. A narrow burn that rarely fires is the safe mistake.

## Where it fires, and what the model sees

At the decision point, which is a tool call, between the provenance gate and the
call itself. A burn recalled afterwards taught nothing, so this is not part of
the `[Recall]` block and is not keyed on the prompt.

A held call does not run. The model reads the lesson in place of the tool
result:

```
This call has not run. The same call went badly before, and what follows is a
candidate warning, not a refusal.

- Last 3 days ago, 2 times: rm -rf failed: build is a mount point
  Called with: command=rm -rf build, cwd=/srv/app
  It went badly at: host=workshop, time_of_day=morning, weekday=thursday

Decide whether the cause still applies. If it does not - the fault is fixed, the
interface changed, or you mean something different this time - make the same
call again and it will run. If it does, take another way.
```

It is a candidate rather than an instruction, in the same terms a surfaced
procedure is (see [the skill library](skill-library.md)), and the mechanism says
so as well as the wording: the identity is marked met for the rest of the turn,
so making the same call again runs it. That is also what stops the warning
becoming a loop, and it is why a call that just failed is not held again inside
the same turn.

An identity takes effect from the *next round*, not from the next call. A model
may emit the same call twice in one response, and both copies are held: the
model has read nothing between them, so letting the second through would run the
act before the warning had been read at all.

The turn reads its live burns once, before its first round, and matches them in
memory. A read per tool call would put a database round trip in front of every
one.

## What a person sees, and what they can do about it

Over-generalization presents as reticence rather than as an error, so a burn
that is wrong is invisible by construction. Three things make it visible.

**The held call says which lesson held it.** A held call reports on the activity
feed as a finished tool call with `ok: false` and an output of the form
`held by a stored lesson (<id>): <what went wrong>`. The warning above goes to
the model in place of the tool result and never reaches a screen, so without
this line the reticence has no name and nothing to look up.

**Three commands, all tenant-level.** `list_negative_memories` gives the act,
the recorded arguments, the circumstances still required, the strength now and
the date it goes quiet. `get_negative_memory` gives one in full.
`clear_negative_memory` overrules one. `docs/WEBSOCKET_API.md` carries the wire
shapes. They are gated on a database, like the rest of the feature, and report
`unsupported` rather than an empty list without one - a person asking why the
assistant will not act would read an empty list as "nothing is holding it".

Strength and holding are two different questions, and the surface answers the
second. A burn keeps its confirmation stamp when it is cleared, because the
record is overlaid rather than changed, so a burn cleared a second ago still
reads at full strength and holds nothing. `firing` asks the question the
dispatch loop asks, and the expiry is in the future only where `firing` is
true.

**A widened burn shows how it widened.** A situation facet a second occurrence
dropped keeps its row and gains a `dropped_at` stamp, and
`get_negative_memory` returns those beside the scope. Widening is the only
mechanism that over-generalizes a burn, so this is the trace it leaves: an empty
circumstance list beside a long list of dropped ones is exactly what a burn that
now fires everywhere looks like. Every scope read filters `dropped_at IS NULL`,
so keeping the row changes nothing about what a burn requires.

Clearing is a person's judgement about the assistant's own memory, and it stays
out of the model's reach. No tool the model is offered mentions negative memory;
the one SQL door takes `SELECT` and nothing else on the read path and refuses
the table outright on the write path; and the clear is a command, which arrives
from an authenticated client connection rather than from a turn.

## Extinction is an overlay

A burn that stops applying is not deleted. The same call succeeding writes a
`correction` row carrying the burn's own action and scope, and the burn's
`superseded_by` names it. The burn keeps every column it had, and
`history(action)` reads both.

A person clearing a burn writes that same overlay, by the same store call. The
note is the only thing that differs: a person who gives no reason gets the
daemon's own line saying a person cleared it, so a later reader can tell a
judgement from an observed success. A cleared burn stops firing, stops being
listed, and stays readable by id - which is the property a delete path would
have cost.

Only the burns a success would actually have fired are extinguished: a success
elsewhere says nothing about a lesson whose context still holds. That covers
lessons this turn wrote as well as ones it read, so a tool that fails once and
works on the retry does not leave a lesson its own retry disproved.

One trial extinguishes, where nature would want several safe exposures. The
asymmetry is deliberate. The dangerous failure here is an assistant that stays
cautious after the cause is gone, so the correction is the quick half.

## Three negatives, which are not the same thing

|  | says | acts on | retrieved |
| --- | --- | --- | --- |
| negative mark | this entry was retrieved and was useless | the store, as prune evidence | never |
| a refuted claim | this claim is untrue | content: a negative fact | when the query is *about* the subject |
| negative memory | this action in this context went badly | content: a negative procedure | at the decision point, before acting |

The negative mark is the [knowledge use log](knowledge-use-log.md)'s, set by
`builtin_knowledge_base_mark` and keyed on a knowledge entry. Marking an entry
writes no burn, and recording a burn marks no entry: the two answer different
questions about different objects.

## Operating notes

- **Gated on a database.** With no Postgres the store is unwired and the
  dispatch loop behaves exactly as it did before this existed.
- **Never fails a turn.** An unreadable store costs the turn its lessons and
  nothing else. Both writes run off the turn's path.
- **The writer is the reaper.** There is no sweep. Recording a bad outcome first
  drops this user's rows that nothing has confirmed past the forget horizon, and
  the foreign key takes their facets with them. A burn and the correction over
  it are one unit and go together, so a reap never leaves a correction naming a
  lesson that is gone.
- **A skewed clock silences rather than entrenches.** The confirmation stamp is
  written by the database and read against the daemon's clock. A little skew
  means nothing. A stamp more than an hour ahead is not believed: read as fresh
  it would sit at full strength for as long as the skew lasted, interrupting
  work and unable to be silenced or reaped, so instead it scores zero, stays
  quiet, and is reaped on the next write.
- **Nothing an outside party wrote is replayed.** A burn is read back in
  another conversation, at the moment the model is deciding whether to act -
  the last place an outside party's sentence should reach. Two channels get
  there. A tool's error text can be a remote server's own words, and the
  arguments are written by the model, which may be quoting a page it has just
  read. Once a turn has read content from outside the trust boundary, a burn it
  records keeps the lesson and drops both: the outcome becomes a fixed line
  saying why, and the recorded arguments are left out. The act is the digest, so
  the lesson still fires on exactly the call it was about; only what a warning
  can say about it is lost.
- **The daemon carries the surface; a client has to render it.** The three
  commands and the hold notice exist on the wire. Until a client shows them, a
  person still meets a held call as an assistant that would not act - the
  reason is now readable, but nothing on screen reads it.
- **A clear does not reach a turn already running.** The turn reads its live
  burns once, before its first round, so a burn cleared while a turn is in
  flight still holds that turn's calls; it holds nothing from the next turn on.
  An accepted cost, not an oversight: the alternative is a database round trip
  in front of every tool call, which is exactly what the single batched read
  exists to avoid. What would remove it is a per-turn invalidation signal, and
  a window of one turn does not pay for one.
- **A cleared burn is not resurrected by a later failure.** The act failing
  again writes a fresh lesson at full strength beside the cleared one. That is
  new evidence and it should count; what it must not do is confirm the cleared
  row back to life, and the confirm path cannot, because it only ever selects a
  row whose `superseded_by` is null.
- **Personal data.** Both tables carry `user_id`, enable their own row-level
  security, and are registered in `PERSONAL_DATA_TABLES`. What a person's
  assistant tried and how it failed is as personal as the work it was doing.
