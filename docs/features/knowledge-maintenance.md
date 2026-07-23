# Knowledge maintenance (the "dream cycle") + live panel sync

The knowledge base is maintained by four background passes. Historically they
ran only on daemon timers; they can now also be **triggered on demand** from the
knowledge panels in every GUI (GTK, TUI, KDE KCM), and the panels **update live**
as entries change. This page documents the passes, the trash lifecycle, the
on-demand trigger path, the event-broadcast chain that drives live refresh, the
concurrency model, and the cancellation story — so none of it has to be
re-derived from the code.

## The four passes

All live in `crates/storage/src/dreaming/` + `crates/storage/src/embedding_backfill.rs`:

| Pass | Entry point | What it does | Cadence |
| ---- | ----------- | ------------ | ------- |
| **Extraction** | `run_dreaming_scan` | Scans conversations past their watermark, asks an LLM to extract durable facts, writes them (+ archival of long-quiet conversations). | frequent (hourly) |
| **Consolidation** | `run_consolidation_scan` | Loads a user's whole active KB and recomputes it holistically (prune / merge / tighten) with a stronger model, applied transactionally with soft-delete. | slow (daily) |
| **Embedding recompute** | `backfill_knowledge_embeddings` | Re-embeds rows. The periodic backfill only touches NULL/stale/model-mismatched rows; the **force** path (`invalidate_all_knowledge_embeddings` → backfill) re-embeds everything. | periodic + on-demand |
| **Trash sweep** | `sweep_expired_trash` | Frees soft-deleted entries past their retention window. No LLM, no embeddings — a single indexed DELETE per user. | frequent (hourly) |

Embedding model changes are handled automatically: each row stamps its
`embedding_model`, and the periodic backfill re-embeds rows whose stamp ≠ the
current model (`invalidate_stale_embeddings`). The **Recalculate Embeddings**
button is the force escape hatch for out-of-band cases (rows edited by raw SQL,
corrupted vectors).

## The trash: soft delete, retention, reaping

Consolidation retires an entry by stamping `deleted_at`, not by deleting the
row. A retired entry is excluded from every read path — search, list, get, the
embedding pipeline — so the tombstone behaves as if it were gone while staying
recoverable and auditable. What happens next is a three-step lifecycle, all in
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
