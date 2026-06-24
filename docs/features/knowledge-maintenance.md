# Knowledge maintenance (the "dream cycle") + live panel sync

The knowledge base is maintained by three background passes. Historically they
ran only on daemon timers; they can now also be **triggered on demand** from the
knowledge panels in every GUI (GTK, TUI, KDE KCM), and the panels **update live**
as entries change. This page documents the passes, the on-demand trigger path,
the event-broadcast chain that drives live refresh, the concurrency model, and
the cancellation story — so none of it has to be re-derived from the code.

## The three passes

All live in `crates/storage/src/dreaming/` + `crates/storage/src/embedding_backfill.rs`:

| Pass | Entry point | What it does | Cadence |
| ---- | ----------- | ------------ | ------- |
| **Extraction** | `run_dreaming_scan` | Scans conversations past their watermark, asks an LLM to extract durable facts, writes them (+ archival of long-quiet conversations). | frequent (hourly) |
| **Consolidation** | `run_consolidation_scan` | Loads a user's whole active KB and recomputes it holistically (prune / merge / tighten) with a stronger model, applied transactionally with soft-delete. | slow (daily) |
| **Embedding recompute** | `backfill_knowledge_embeddings` | Re-embeds rows. The periodic backfill only touches NULL/stale/model-mismatched rows; the **force** path (`invalidate_all_knowledge_embeddings` → backfill) re-embeds everything. | periodic + on-demand |

Embedding model changes are handled automatically: each row stamps its
`embedding_model`, and the periodic backfill re-embeds rows whose stamp ≠ the
current model (`invalidate_stale_embeddings`). The **Recalculate Embeddings**
button is the force escape hatch for out-of-band cases (rows edited by raw SQL,
corrupted vectors).

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
| Command / op enum / result / event | `crates/api-model/src/lib.rs` (`StartKnowledgeMaintenance`, `MaintenanceOp`, `MaintenanceTaskStarted`, `Event::KnowledgeChanged`, `TaskKind::Maintenance`) |
| Signal projection | `crates/api-model/src/signal.rs` (`SignalEvent::KnowledgeChanged`) |
| Port | `crates/core/src/ports/inbound.rs` (`KnowledgeMaintenanceService`) |
| Scans + force-recalc | `crates/storage/src/dreaming/`, `crates/storage/src/embedding_backfill.rs` |
| Handler arm + `notify_knowledge_changed` | `crates/application/src/lib.rs`, `crates/application/src/background_tasks.rs` |
| Daemon service + timer wiring | `crates/daemon/src/maintenance_service.rs`, `crates/daemon/src/main.rs` |
| D-Bus method + signal | `crates/dbus-bridge/src/adapter/knowledge.rs`, `crates/dbus-bridge/src/adapter/event_forwarder.rs` |
