# Repeated tool calls

Nothing in the turn loop used to notice that the model had already made a given
tool call. An identical `(tool, arguments)` pair ran again, returned the same
bytes again, and those bytes were appended to the context again.

On its own that is waste. With context eviction it is a loop with an engine:

1. The model calls a tool that returns a large result.
2. The result is large enough that the context must evict something.
3. What gets evicted is an earlier tool result, often the one still needed.
4. The model no longer sees it, so it calls the same tool with the same
   arguments.
5. Return to step 2.

The loop is stable. It does not converge and it does not fail; it ends when the
round cap fires or the model happens to answer. One measured turn spent about
fifty rounds and seventy-odd tool calls on a two-number question, fetching over
a megabyte against a much smaller context, and produced a correct answer - so it
appears in no failure metric.

## The key

Every dispatched call is keyed on three things:

- **The connection that runs it**, by its own label. This is the opposite choice
  from the negative-memory key, which strips the location root deliberately
  because a lesson about a tool is portable across machines. This key asks
  whether THIS call was already made, and reading a path on the daemon says
  nothing about the same path on the user's own machine. Merging them could
  serve one host's bytes as the other's.
- **The provider's own name** beneath that connection.
- **The arguments**, parsed and re-serialized. That re-serialization is the
  normalization: `serde_json::Map` is a `BTreeMap` in this workspace, so object
  keys come out sorted and insignificant whitespace is dropped. `{"b":2,"a":1}`
  and `{ "a" : 1, "b" : 2 }` are one key.

The key holds digests rather than the argument text, and the ledger holds a
digest rather than the result text, because both are already in the transcript.

## Two answers, and only one of them withholds work

**A result that repeats the one before it is not appended again.** When a call
runs and returns exactly what that key returned on its previous run, the turn
appends a pointer to the message already holding those bytes. The tool ran, so
nothing here can be stale. This is what breaks the loop above, and it applies
whether or not anything is ever suppressed.

It compares against the previous result, not against every result the key has
produced, so a tool alternating between two answers stores both every time.

**Suppression is an execution saving on top.** Two matching runs make a key
suppressible; from there some calls are answered from the transcript without
running the tool. A suppressed call still spends its round, so what it saves is
the execution.

## The backoff

Suppression can be stale, so it is bounded. Each key carries a suppression
counter and a threshold:

- Each suppressed call increments the counter.
- When the counter reaches the threshold the call runs, the counter resets, and
  the threshold doubles - from two, up to a ceiling of sixteen.
- Any run whose result differs from the previous one clears the suppressible
  state outright, back to needing two matching runs.

Twenty-one identical calls run the tool on calls 1, 2, 5, 10 and 19. A key
called on every round of a full turn is re-checked at least ten times.

Without the ceiling the property would hold only in the limit: unbounded
doubling puts the ninth run at call 263, past the round budget, so a key asked
for often enough would never be re-checked inside the turn.

## The size floor

Neither part applies to a result under 512 bytes.

The pointer renders to about 300 bytes and the "not run" notice to about 500, so
a short result replaced by one makes the context bigger - the one thing a rule
that exists to stop the context refilling may not do. Short results are also
where a stale answer costs most: a poll's `{"status":"running"}` is a few dozen
bytes. The floor is the safe direction on both counts, and `core::planning`'s
eviction holds the same line at the same size for the same reason.

## What is never suppressed

- **`builtin_tool_search`**, because the turn loop parses its result to activate
  the tools it found. A search answered from the transcript would return the
  right text and activate nothing, leaving the model calling a tool the next
  round no longer advertises.
- **`spawn_subagent`**, because it creates something. Its detached form returns
  a fresh child id and can never repeat its own bytes, but `wait` defaults to
  true and the blocking form returns the child's answer verbatim, so two spawns
  of one prompt that agree could otherwise make the key suppressible.

A repeated result still becomes a pointer for both, so only the execution saving
is given up.

## What the model reads

Three results, worded so they cannot be confused:

| What happened | What the model gets |
|---|---|
| The tool ran and returned something new | the bytes |
| The tool ran and returned what it returned before | a pointer saying the call ran and the bytes are current |
| The tool did not run | a pointer saying so, and that the result may be out of date |

Both pointers name the message and `builtin_transcript_get`, which reads it back
without running the tool again.

## What the rule does not hold

- **A side effect with no trace in the output.** A call that appends a line and
  answers `""` looks exactly like one that reads and answers `""`. An empty
  success renders to a 49-byte marker, far under the floor, so a side effect
  that says nothing is never suppressible. What remains is a call that changes
  something and returns half a kilobyte of unchanging text - where the text is
  itself evidence that nothing observable moved - and the backoff bounds that.
- **An error repaired mid-turn.** An error is recorded like any other output, so
  a server answering identically twice while it restarts is suppressed for a
  bounded run of calls after the cause is fixed.

## Scope

The ledger lives for exactly one turn. A new turn starts clean.
