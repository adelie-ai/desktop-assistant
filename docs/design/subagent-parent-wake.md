# Subagent parent-wake: event-driven re-engagement on completion

Status: in progress · Epic: #117 (multi-agent) · Related: #607/#608 (result
pad-handoff), #551 (task dependencies), #578 (park-and-notify)

## Problem

`spawn_subagent { wait: false }` returns the child's task id immediately and the
parent's turn then *ends*. Nothing is running to notice when the child finishes,
so `get_subagent_status` polling can only happen if the parent is somehow driven
again — and it isn't, because the turn is over. In practice the user has to send
a message to re-engage the parent conversation before it looks at any results.

Observed on the prod daemon (`adele-prod`, 2026-07-24): a parent fanned out four
price-research subagents (`wait:false`), wrote their ids to the scratchpad, and
only polled `get_subagent_status` ~5 minutes later — after the operator prodded
it. Fully unattended, it would never have looked.

## Goal

When a subagent wraps up, the daemon re-engages the parent **without polling**:

1. Per-child: as each child returns, wake the parent so it can evaluate that
   child's output and act on it — without waiting for the others.
2. Holistic: once the last child finishes, wake the parent to review the whole
   result set against the original request.

The wake does **not** inline the child's answer. Each child already writes its
final answer to the *session* scratchpad as a `result` note under its own
`owner_todo` (#607). The wake message just tells the parent a child finished and
hands it the concrete scratchpad reference(s) — the parent decides what to do.

## Mechanism

### 1. Registry signal (slice 1, `crates/application`)

`BackgroundTaskRegistry::finalize` is the single terminal-state chokepoint for
every task. When the finalized task is a `TaskKind::Subagent`, the registry
invokes a late-set `SubagentCompletionObserver` with a `SubagentCompletion`:

| field                    | use                                                        |
|--------------------------|------------------------------------------------------------|
| `user_id`                | tenant scope for the wake turn                             |
| `parent_task_id`         | which parent fanned this out                               |
| `session_conversation_id`| the conversation to wake + where the result note lives     |
| `child_conversation_id`  | the child's own transcript (for drill-in)                  |
| `child_task_id`          | `get_subagent_status(task_id)` handle                      |
| `child_name`             | human label for the wake message                           |
| `owner_todo`             | scratchpad namespace of the child's `result` note          |
| `status`                 | completed / failed / cancelled                             |
| `siblings_remaining`     | non-terminal sibling subagents under the same parent       |
| `notify_parent`          | whether the parent is owed a wake at all (see below)       |

`siblings_remaining == 0` is the cue for the holistic pass. Fires for failed and
cancelled children too, so the parent never waits forever on a child that died.
The observer is a late-set slot (like `SubagentAwareToolExecutor`'s conversation
`Weak`) because the coordinator that consumes it is built after the registry.

`notify_parent` carries the **dispatch mode**, recorded on `TaskKind::Subagent`
at spawn as `!wait`. Only a detached `spawn_subagent { wait: false }` leaves the
parent uninformed; a blocking spawn (`wait: true`, the default) returns the
child's answer straight into the still-running parent turn. The signal fires
either way — the registry reports the fact, the coordinator decides. Persisted
`kind_json` rows written before the field default to `false`, so an old row can
never resurrect as an autonomous turn.

### 2. Parent-wake coordinator (slice 2, daemon)

Implements the observer. Drives an autonomous wake turn on the session
conversation through the normal send-message path, with a fanout-only sink (no
originating caller) and a synthetic `request_id`, so the turn renders live in
whatever client is viewing that conversation (`FanOutSink`, #1).

Correctness constraints:

- **One turn per conversation.** Never launch a wake turn while a turn (a live
  foreground turn, or a prior wake turn) is running on that conversation. State
  per session conversation: `{ running, pending: Vec<SubagentCompletion> }`.
- **Coalesce.** First completion triggers a wake immediately; completions that
  arrive while a wake turn is running batch into `pending` and are drained by
  the next wake turn. A burst of N near-simultaneous finishers yields at most
  one extra wake turn, not N.
- **Bound autonomy.** Only wake when the parent task is a top-level
  `Conversation` (the conversation the user actually sees). A nested subagent
  finishing does not spin up an autonomous turn on a hidden conversation.
- **Detached children only.** Drop completions whose `notify_parent` is
  `false`. The parent blocked on those and already delivered their results, so
  a wake would queue a second turn behind the parent's own turn on the
  per-conversation lock and then ask it to consolidate an answer it has already
  given.

### 3. Wake message (slice 3)

Injected prompt names the finished child(ren), their status, the scratchpad
reference(s) (`result` note under `owner_todo`, and `get_subagent_status(id)`),
and the remaining count. On the holistic pass (`siblings_remaining == 0`) it
lists every child's reference and asks for the consolidated answer to the
original request. It states the results are already saved and lets the parent
decide — no forced action.

### 4. Kill switch (slice 4)

Default-on. `daemon.toml [subagents] wake_parent = false` disables globally.

## Testing

Spec-driven per slice. Slice 1 lives entirely in `crates/application` and is
unit/integration-testable with no LLM: assert the observer fires with the right
payload for completed/failed/cancelled subagents, does *not* fire for
`Conversation`/`Standalone`/`Maintenance` tasks, computes `siblings_remaining`
correctly, echoes the dispatch mode as `notify_parent`, and is a safe no-op when
unset. Slice 2 additionally asserts the negative: a default (`wait: true`)
spawn drives no wake turn at all, while a `wait: false` spawn drives exactly
one.
