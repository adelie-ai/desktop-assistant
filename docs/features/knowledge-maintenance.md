# Knowledge maintenance (the "dream cycle") + live panel sync

The knowledge base is maintained by five background passes. Historically they
ran only on daemon timers; they can now also be **triggered on demand** from the
knowledge panels in every GUI (GTK, TUI, KDE KCM), and the panels **update live**
as entries change. This page documents the passes, the trash lifecycle, the
on-demand trigger path, the event-broadcast chain that drives live refresh, the
concurrency model, and the cancellation story — so none of it has to be
re-derived from the code.

## The five passes

All live in `crates/storage/src/dreaming/` + `crates/storage/src/embedding_backfill.rs`:

| Pass | Entry point | What it does | Cadence |
| ---- | ----------- | ------------ | ------- |
| **Extraction** | `run_dreaming_scan` | Scans conversations past their watermark, asks an LLM to extract durable facts, writes them (+ archival of long-quiet conversations). | frequent (hourly) |
| **Summary backfill** | `run_dreaming_scan` | Writes the one-line `summary` for entries that have none, and rewrites one whose body changed after it was written. Batched, capped per cycle, and never touches `content`. | frequent (hourly) |
| **Consolidation** | `run_consolidation_scan` | Loads a user's whole active KB and recomputes it holistically (prune / merge / tighten) with a stronger model, applied transactionally with soft-delete and bounded by the rules below. | slow (daily) |
| **Embedding recompute** | `backfill_knowledge_embeddings` | Re-embeds rows. The periodic backfill only touches NULL/stale/model-mismatched rows; the **force** path (`invalidate_all_knowledge_embeddings` → backfill) re-embeds everything. | periodic + on-demand |
| **Trash sweep** | `sweep_expired_trash` | Frees soft-deleted entries past their retention window. No LLM, no embeddings — a single indexed DELETE per user. | frequent (hourly) |

Embedding model changes are handled automatically: each row stamps its
`embedding_model`, and the periodic backfill re-embeds rows whose stamp ≠ the
current model (`invalidate_stale_embeddings`). The **Recalculate Embeddings**
button is the force escape hatch for out-of-band cases (rows edited by raw SQL,
corrupted vectors).

Correctness across a model change does not rest on that sweep. Four of the five
vector searches - knowledge, skills, the scratchpad, and the tag-registry dedup
check - also filter on `embedding_model`, admitting only rows embedded by the
model that produced the query vector, because pgvector answers a comparison
across vector dimensions with an error rather than a miss. Those tables
therefore stay queryable while they hold two models' vectors at once:
mid-reindex, after a failed sweep, or across a live backend swap. The sweep is
what shortens that degraded window, not what prevents the error.

The fifth, tool search over `tool_definitions`, has no such predicate yet and
still depends on the sweep having completed (#703). It is the table most likely
to be caught mid-reindex, because `backfill_tool_embeddings` updates in place.

Two boundaries hold the degradation sane: the full-text arm
of each hybrid query is never model-scoped, so recall falls back to lexical
matching instead of vanishing; and a stamp whose digest matches the current
model is treated as the same model even when the name is spelled differently, so
a cosmetic rename costs nothing (see `crates/storage/tests/embedding_fingerprint.rs`
and `crates/storage/tests/search_embedding_model_scope.rs`).

## What the summary backfill may and may not do

An entry's `summary` is the one line it is offered back as - in a knowledge
panel row, and in the `[Recall]` block that puts candidate memory in front of
the model before it acts. The column is nullable and unenforced at the write
boundary, because refusing a write that omits a summary would lose the fact, so
most rows carry none and read back as a cut-down prefix of their body. This pass
fills them (`crates/storage/src/dreaming/summarize.rs`). Four rules bound it:

1. **The body is never rewritten.** The write statement names `summary` and its
   freshness stamp and nothing else. #694 is the standing concern that the store
   is becoming model-rewritten prose rather than accumulated evidence, and a
   summarising pass that edits the body is exactly that failure.
2. **A cycle takes at most `MAX_SUMMARIES_PER_CYCLE` (200) rows**, shared evenly
   across the users that have work, so one large backlog cannot starve a small
   one. It is a backfill, not a deadline: the leftover is logged and taken next
   cycle.
3. **Entries are asked about in batches** of up to `MAX_SUMMARY_BATCH_ROWS` (20),
   also bounded by a per-prompt character budget. One call per row is the
   expensive way to spend a backfill of hundreds of rows.
4. **A row that fails keeps no summary** and is in the worklist again next cycle.
   Nothing is stamped on failure - deliberately unlike the embedding backfill,
   which stamps a failed row to stop a tight retry loop, because re-attempting a
   summary costs one line inside a batched prompt while re-attempting an
   embedding costs a metered vector call.

### Drift, and why `summary IS NULL` is not the worklist

A summary condenses `content`, so an edit to the body leaves the stored line
describing something the entry no longer says - and a confidently wrong line is
worse than none, because a reader believes it and never opens the entry. Two
normal paths produce that state: the knowledge write path preserves a stored
summary when an update names none, and consolidation rewrites content without
touching the summary at all.

So freshness is tracked the way embeddings already track it. `summary_updated_at`
(migration 043) records when the line was written, and work is due when the stamp
is absent or older than `updated_at` - the same shape as
`embeddings_updated_at < updated_at`. The knowledge write path stamps it
alongside the summary in all three of that field's states: a supplied summary
takes the same transaction time as `updated_at` and so reads as current, a
cleared summary loses its stamp with its text, and an absent one keeps the stamp
it had, which is what makes a content-only update surface as drift.

The pass writes `summary` and `summary_updated_at` only. It leaves `updated_at`
alone on purpose: bumping it would mark the row's embedding stale and send the
whole backfilled store back through the embedding backfill for a change that
never touched the embedded text.

`updated_at` is instead a **precondition** on the write. The pass reads a row,
spends a model call, then writes, and a content write can land in that window -
which would store a line describing the body the pass read while the freshness
stamp declared it current, so nothing would ever revisit it. Matching the body's
modified time makes the write a no-op in that case; the line is discarded and the
row stays in the worklist for the next cycle.

## What consolidation may and may not do

The model sees the whole active store and returns a plan. The plan is not applied
verbatim, because the judgment behind it is formed from prose alone with no
signal about whether an entry was ever retrieved or cited. Three rules bound it
(`crates/storage/src/dreaming/consolidation.rs`):

1. **A deliberately promoted entry is never pruned.** Rows written during a live
   turn carry `source = 'explicit'` - the user asked, or Adele decided in the
   moment that a fact was worth keeping. Consolidation may rewrite or merge such
   an entry, but a proposed delete is refused and counted in the run's stats. The
   provenance follows the content: an edit keeps it, and a merge stamps the
   surviving canonical row `explicit` if any member was, so the protection cannot
   be laundered away over successive nights.
2. **Settled prose is left alone.** `review_generation` counts how many times
   consolidation has rewritten an entry. At `MAX_REVIEW_GENERATION` (2) the entry
   is settled: further edits and merges touching it are refused, because
   consolidation re-reads its own output every pass and an uncapped entry becomes
   a paraphrase of a paraphrase, drifting from what was observed toward what the
   model believes. This settles individual entries, not the store - extraction
   keeps adding generation-0 rows, scope can still be attached, and a settled
   entry stays prunable, so consolidation's own output never becomes permanent.
3. **Outright prunes are capped per run** at `MAX_DELETE_FRACTION` (0.1) of the
   active set, floor 1, with the excess dropped and a warning logged. Merges do
   not count against it: their content survives in the canonical row. The cap is
   a blast-radius bound on one night's unreviewed opinion.

## The trash: soft delete, retention, reaping

Consolidation retires an entry by stamping `deleted_at`, not by deleting the
row. A retired entry is excluded from every read path — search, list, get, the
embedding pipeline — so the tombstone behaves as if it were gone while staying
recoverable and auditable.

A write cannot land on one either. The upsert's conflict clause excludes a
retired row, so a caller that still holds the id of an entry consolidation
retired is refused and told to store the text as a new entry instead. Without
that exclusion the write would put live content into a row no read path can
reach, which the reap then frees on the tombstone's original clock — and the
caller would be told it succeeded. The write does not revive the row: that would
resurrect a duplicate a merge had absorbed, and leave `superseded_by` pointing at
the row that replaced it. Restoring a tombstone is not something any write path
does.

What happens next is a three-step lifecycle, all in
`crates/storage/src/dreaming/trash.rs`:

1. **Retention.** `[backend_tasks] knowledge_trash_retention_days` (default 30,
   the historical `SOFT_DELETE_TTL_DAYS`) is how long a tombstone is kept. `0`
   means "do not retain" — reap on the next sweep.
2. **Automatic reap.** The daemon's trash-sweep loop calls
   `sweep_expired_trash` every `knowledge_trash_sweep_interval_secs` (default
   3600; `0` disables the sweep). It iterates the users who hold tombstones and
   reaps each user's expired rows under that user's scope. A consolidation
   cycle *also* reaps inside its apply transaction, using the same configured
   retention — but that is a convenience trigger, not the only one. Before this
   split the reap lived only inside consolidation, so an instance with dreaming
   disabled accumulated tombstones forever: invisible to every read, never
   freed.
3. **Empty on demand.** `Command::EmptyKnowledgeTrash` reaps every tombstone the
   calling user owns immediately, ignoring the retention window, and replies
   with the number of rows freed (`0` for an already-empty trash — a normal
   outcome, not an error). `Command::GetKnowledgeTrashCount` reports how much is
   in the trash, since no other read path can see it.

Every one of these is scoped to a single `user_id`: one user's sweep, empty, or
count never touches another's rows. The only cross-user statement is the
sweep's "which users hold tombstones" scan, which installs a per-user scope
before deleting anything.

### What a tombstone records

Retiring a row is two very different outcomes wearing the same shape: a **merge**
relocates the content into a canonical row, a **prune** destroys it. Both used to
write nothing but `deleted_at`, so no query could tell them apart and "was this
fact relocated or thrown away?" was unanswerable. Three columns close that
(migration 038):

| Column | Merge member | Prune |
| ------ | ------------ | ----- |
| `deleted_kind` | `'merge'` | `'prune'` |
| `superseded_by` | id of the canonical row that absorbed it | NULL |
| `deleted_reason` | NULL (the model states none per member; `superseded_by` is the reason) | the model's stated reason, trimmed and clamped to `MAX_DELETE_REASON_CHARS` |

All three are NULL on tombstones written before the migration and on deletes that
did not come from consolidation. `superseded_by` is intentionally not a foreign
key: tombstones are hard-reaped once past retention, so the target can legitimately
disappear, and an FK would either block the reap or erase the audit link. A
dangling id is expected - the same contract `metadata.source_conversation_id`
already carries against archival hard-deletes.

Splitting a period's tombstones by outcome is then a plain query:

```sql
SELECT deleted_kind, COUNT(*)
FROM knowledge_base
WHERE user_id = $1 AND deleted_at IS NOT NULL
GROUP BY deleted_kind;
```

## On-demand trigger path

```
panel button ─ start_knowledge_maintenance(op) ─┐
                                                 │  Command::StartKnowledgeMaintenance { op }
GUI ── client-common AssistantCommands ──────────┤
                                                 ▼
                        DefaultAssistantApiHandler::handle_command
                                                 │  registry.spawn(TaskKind::Maintenance, body)  ← returns TaskId immediately
                                                 ▼
                        DaemonKnowledgeMaintenanceService::run_<op>(ctx.token)
                                                 │  (shared with the timer loops)
                                                 ▼
                        run_dreaming_scan / run_consolidation_scan / invalidate+backfill
```

Key points:
- **Never inline.** The command returns immediately with a `MaintenanceTaskStarted { task_id }`; the work runs as a tracked background task via `BackgroundTaskRegistry::spawn`. This matters because the dispatch loop handles non-`SendMessage` commands **serially per connection** (`crates/transport-dispatch/src/lib.rs`), so a multi-minute scan run inline would block every other command on that GUI's connection. (It is not a global lock — other connections run concurrently — and all I/O is async.)
- **One implementation, shared.** `DaemonKnowledgeMaintenanceService` (`crates/daemon/src/maintenance_service.rs`) is driven by BOTH the on-demand handler and the dreaming/consolidation timer loops in `main.rs`. A per-op `tokio::sync::Mutex` rejects a second concurrent run of the same op (timer- or button-triggered) with a clear error.
- **Surfaced as a background task.** Progress/completion ride the existing `Task*` events and the task UI; cancel it with the existing task-cancel command (`CancelBackgroundTask { id: task_id }`).
- **Total failure surfaces as a failed task.** A pass where *every* unit (conversation for extraction, user/prompt-slice for consolidation) fails its LLM call returns an error, so the task finalizes as `Failed` — not a silent `Completed` with 0 changes. A pass where the model legitimately changed nothing still completes successfully; a cancelled pass is never a failure. This closed a real gap: an unauthorized consolidation model (HTTP 401 on every call) previously looked like "consolidation did nothing." The decision lives in `dreaming::common::is_total_failure`.

## Live panel refresh: the event-broadcast chain

A maintenance pass (and any manual create/update/delete) emits `Event::KnowledgeChanged`, which fans out to all of a user's connected panels:

```
notify_knowledge_changed(user_id)
  └─ BackgroundTaskRegistry: per-user tokio broadcast::Sender<api::Event>   (crates/application/src/background_tasks.rs)
       └─ dispatch forwarder (per connection that issued SubscribeBackgroundTasks)
            ├─ WS/UDS  → map_event_to_signal → SignalEvent::KnowledgeChanged → GTK/TUI panels refetch
            └─ D-Bus bridge event_forwarder → Knowledge.EntriesChanged signal → KDE KCM refetch
```

`KnowledgeChanged` carries no payload — the change kind is intentionally not
encoded; a debounced refetch is simplest and correct for create/update/delete/
maintenance alike (mirrors `ConversationListChanged` / `ScratchpadChanged`).
During extraction the event fires per conversation, during consolidation per
user, so panels update *as the scan progresses*, not only at completion.

## Cancellation & non-blocking

The passes touch only Postgres + LLM + embeddings (no MCP) and are fully async —
nothing blocks the runtime. What needed work was **prompt cancellation**:

- `registry.cancel()` only signals the task's `CancellationToken`; the body must
  observe it. The scans now check the token at batch boundaries (per
  conversation / per user / per embedding batch) and bail.
- The maintenance service builds **cancellation-aware** LLM closures: the
  streaming callback returns `false` the moment the token is cancelled (the
  documented way to stop a stream), wrapped in `with_cancellation_token` so the
  connector also observes it at connect, and bounded by a per-call
  `tokio::time::timeout` so a hung endpoint can't wedge a pass.

**Known limitation / follow-up (not done here):** LLM connectors only observe the
cancellation token at HTTP *connect*, not mid-stream, and there is no universal
per-request timeout across all LLM/embedding/MCP calls. Making connectors poll
the token during streaming (and adding a hard-`abort()` backstop to the registry)
is a broader, cross-cutting change tracked separately.

## Where things live

| Concern | File |
| ------- | ---- |
| Command / op enum / result / event | `crates/api-model/src/lib.rs` (`StartKnowledgeMaintenance`, `MaintenanceOp`, `MaintenanceTaskStarted`, `Event::KnowledgeChanged`, `TaskKind::Maintenance`, `GetKnowledgeTrashCount`, `EmptyKnowledgeTrash`) |
| Signal projection | `crates/api-model/src/signal.rs` (`SignalEvent::KnowledgeChanged`) |
| Port | `crates/core/src/ports/inbound.rs` (`KnowledgeMaintenanceService`) |
| Scans + force-recalc | `crates/storage/src/dreaming/`, `crates/storage/src/embedding_backfill.rs` |
| Trash lifecycle (count / empty / reap / sweep) | `crates/storage/src/dreaming/trash.rs`, sweep loop in `crates/daemon/src/main.rs` |
| Retention + sweep cadence config | `crates/daemon/src/config/mod.rs` (`BackendTasksConfig::knowledge_trash_retention_days`, `knowledge_trash_sweep_interval_secs`, `trash_sweep_enabled`) |
| Handler arm + `notify_knowledge_changed` | `crates/application/src/lib.rs`, `crates/application/src/background_tasks.rs` |
| Daemon service + timer wiring | `crates/daemon/src/maintenance_service.rs`, `crates/daemon/src/main.rs` |
| D-Bus method + signal | `crates/dbus-bridge/src/adapter/knowledge.rs`, `crates/dbus-bridge/src/adapter/event_forwarder.rs` |
