# Turn capture

Every turn leaves a record on its conversation's scratchpad, written by the
daemon and not by the model.

## Why the harness writes it

Asking the model to record what mattered asks it to notice, mid-task and under
time pressure, that something was worth keeping - and gives it no feedback when
it fails. A turn that forgot to record the decision it just took looks exactly
like a turn in which no decision was taken.

So capture belongs to the harness. It runs on every turn, including a turn that
opened no step, called no tool, or ended in an error.

## What it keeps

One `turn`-typed note per turn, keyed `turn:<id of the message that opened the
turn>`:

```
Asked: from now on deploy with the kustomization, never with a raw apply

Answered: understood - I will use deploy/kustomization.yaml from here on

Ran:
- read_file -> answered (1204 bytes), read it with builtin_transcript_get message_id="0198…"
- terminal_run -> declined (312 bytes), read it with builtin_transcript_get message_id="0198…"
```

The key is derived from the turn, so re-running the capture writes the same row
rather than a second one.

## What it does not keep, and why

**A tool's result bytes and a tool call's arguments never reach the note.** The
note is durable and a later, clean turn reads it back - through
`builtin_scratchpad_search` and through the pad arm of `[Recall]`. A result
payload is the clearest case of outside content there is, and a call's arguments
are text the model wrote after it may have read one. Putting either in the note
would carry outside influence past the gate that exists to bound it.

Nothing is lost. The transcript holds every byte, and the note names the message
id each result is one `builtin_transcript_get` away at.

What the note does hold is the conversation's own two voices - the user's prompt
and the assistant's closing text - plus tool names, result sizes and
daemon-minted message ids. A tool name is bounded before it is stored, because
it is model-supplied text like any other.

## The stamp

**The note carries the writing turn's provenance.** The user's prompt is the
user's and is never outside content, but the assistant's closing text is another
matter: a turn that read a page routinely quotes it. So a note written by a turn
that took in externally-controlled bytes is stamped, and the two paths that read
it back account for that - `builtin_scratchpad_search` marks it, which folds
into the reading turn's own provenance, and the `[Recall]` pad arm drops it at
the strict level.

The rolling context summary carries the same assistant text unstamped. It is not
the precedent this follows: it is not embedded, not retrieved by relevance to a
later prompt, and not returned by a tool whose grading contract is the marker.

With the operator's `hard_withhold` setting on, a tainted turn's note keeps what
the USER said and replaces what the TURN derived with the placeholder. The
user's words are never outside content, and destroying them would defeat the one
thing the capture exists to keep.

## Where it shows up

- **Not** in the `[Scratchpad]` index or the `[Working state]` count. Both list
  `note`-typed keys, so a long conversation's captures never crowd out the notes
  a person or the model wrote on purpose.
- **Yes** in `builtin_scratchpad_search`, and in the pad arm of `[Recall]`.
  That is the point: the transcript makes a past turn findable by position, and
  this makes it findable by relevance.

## Cost, and what fails

One scratchpad write per turn, and the embedding that write already does for
every note. No model call.

A capture that cannot be written logs a warning and the turn is unaffected. The
transcript already holds every byte the capture restates, so failing the turn
over it would trade the answer the user is waiting for against a convenience.

The capture runs after the turn's work. The answer streams to the user chunk by
chunk while the turn runs, so on the ordinary path nothing here sits between a
person and the reply they are reading.

**Two exits do not stream.** A turn that ends in a provider error, and one the
user cancels, hand their text back as a return value rather than through the
chunk callback, and the capture is awaited before that return - so those paths
pay the write, and the embedding with it, before the message appears. The cancel
path also holds the conversation's turn lock while it does, so a queued prompt
waits with it.

That is a stated cost, not a property. Detaching the write would take the
capture out of the turn's own consistency and make its failure invisible on
exactly the exits most likely to carry an interrupted decision.
