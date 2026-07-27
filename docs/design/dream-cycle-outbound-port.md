# Dream-cycle policy: the outbound port

Status: proposed. Issue: #744 (part of #720, the 2026-07 core review). Related: #694 and
its phases #711 / #893 / #894 (memory architecture), #742 (SQLite excluded from the gate),
#882 (user_id audit does not cover storage-sqlite), #639 (the skill-index precedent),
#680 (multi-tenancy boundary)

Companion to `skill-execution-and-portability.md`, whose "Why the policy left the storage
adapters" section describes the same mistake, made once already and reversed for the skill
index. This document works out what the same reversal costs for the dream cycle, where the
answer is not the same, because consolidation deletes and the skill index does not.

## The problem, priced

`crates/storage/src/dreaming/` is 2,583 lines across 8 files and issues **21 SQL
statements**. Inside `sqlx::query*` blocks there are 237 lines - 9 percent of the module.
The other 91 percent is domain policy that happens to live in a Postgres adapter:

| file | lines | SQL call sites | what it actually is |
|---|---|---|---|
| `consolidation.rs` | 683 | 2 | the curation prompt, guard rules, delete cap, slicing |
| `extraction.rs` | 618 | 1 | the extraction prompt, response parsing, tag policy |
| `reconcile.rs` | 526 | 8 | union-find merge clustering, then the apply transaction |
| `common.rs` | 248 | 4 | transcript formatting, JSON payload extraction |
| `trash.rs` | 167 | 4 | near-pure storage, no LLM contact |
| `types.rs` | 133 | 0 | tunables and enums, no persistence at all |
| `mod.rs` | 139 | 0 | orchestration |
| `archival.rs` | 69 | 2 | two UPDATEs against `conversations` |

`extraction.rs` is 2.5 percent SQL. `consolidation.rs` is 2.7 percent. `consolidate_user`
(consolidation.rs:150-388) is the largest function in the module at 240 lines and touches
persistence at exactly two of them - `load_active_entries` at :156 and `apply_ops` at :384.

### What a second adapter is on the hook for today

`crates/storage-sqlite/DESIGN.md:126` already names the bill: *"inc3 - dreaming + db_query.
Port `crates/storage/src/dreaming/*` (raw `&PgPool` today, not behind a port)"*, with the
matching TODO at `storage-sqlite/src/lib.rs:36`. Absent a port, that reads as: reimplement
~2,600 lines, of which roughly 180 are literal prompt text
(extraction.rs:400-494 and :496-510, consolidation.rs:478-511 and :513-547) and 208 are
response parsers, in a second crate, and keep them in step by hand forever.

Six rules are worse than merely duplicated - they are currently written *as SQL*, so a
second adapter does not copy them, it re-derives them:

| rule | where it lives now |
|---|---|
| a merged cluster containing an explicit entry keeps `explicit` | `source = CASE WHEN $3 THEN 'explicit' ELSE 'consolidation' END`, reconcile.rs:256 |
| the same rule for standalone edits | `CASE WHEN source = 'explicit' THEN ...`, reconcile.rs:313-314 |
| an entry settles at the generation cap | `review_generation = LEAST(review_generation + 1, $4)`, reconcile.rs:259 and :317 |
| a deliberately-entered fact is never pruned | `AND source IS DISTINCT FROM 'explicit'`, reconcile.rs:372 |
| an already-tombstoned row is not retired twice | `AND deleted_at IS NULL`, reconcile.rs:286 and :371 |
| the first review timestamp is preserved | `reviewed_at = COALESCE(reviewed_at, NOW())`, reconcile.rs:390 |

This is precisely the failure `core/src/skill_catalog.rs:15-21` records: *"each adapter
re-implemented that policy in its own SQL and the two drifted apart"* - Postgres pruned by
name-list, SQLite deleted the scope wholesale, and identical inputs produced different
catalogs depending only on which store was configured. The difference in stakes is that a
drifted skill catalog is recoverable by rescanning, and a drifted consolidation deletes
knowledge.

### The cost paid every day, not only by a hypothetical adapter

Consolidation policy is only reachable through Postgres, so it is outside `just check` and
only runs under `just test-db` (`cargo test -p desktop-assistant-storage`, justfile:108).
`just check` verifies exactly one dreaming policy *rule* today - the plain `#[test]` at
`dreaming_db_paths.rs:399` asserting `MAX_DELETE_FRACTION == 0.1` - plus 28 helper-level
unit tests. The delete cap, the never-prune-explicit rule, the settled-entry rules and
tenant isolation are all behind a container someone has to remember to boot.

That constant has a price attached: `types.rs:66-74` records that the previous
`MAX_DELETE_FRACTION = 0.5` cost the reference instance 606 of 608 extracted facts (#694).
The rule that would have caught it is exactly the kind that does not run in the gate.

### The leak is small and precisely located

`crates/daemon/src/maintenance_service.rs` is the only file in `crates/daemon/src` that
names `PgPool` - three occurrences (the import at :26, the field at :45, the constructor
parameter at :70), consumed at four call sites (:166, :188, :212, :219), with no SQL of its
own. Outside the module, exactly six non-test lines in the workspace name
`storage::dreaming`: maintenance_service.rs:27, main.rs:1946 (`sweep_expired_trash`),
config/mod.rs:539 and :2013 (`SOFT_DELETE_TTL_DAYS` as a config default), and
knowledge.rs:256 and :260 (`trash_count` / `empty_trash` delegated out of the
`KnowledgeBaseStore` impl). The dozens of other "dreaming" hits are `[purposes.dreaming]`
config keys, not code coupling.

Two of the four call sites - `invalidate_all_knowledge_embeddings` (:212) and
`backfill_knowledge_embeddings` (:219) - are `embedding_backfill.rs`, not `dreaming/`. A
change scoped literally to `crates/storage/src/dreaming/*` will land, be correct, and leave
the daemon still naming `PgPool`. Decision 9 handles that.

## What is already right

Four things are correct and this work must not disturb them.

**The inbound port.** `KnowledgeMaintenanceService` (core/src/ports/inbound.rs:943) exposes
`run_extraction`, `run_consolidation` and `recalculate_embeddings` as `#[async_trait]`, held
by the application layer as `Arc<dyn ...>` (application/src/lib.rs:491). The application
layer is already insulated. **The break is below the adapter boundary, not above it**, and
no inbound signature changes anywhere in this work.

**The test suite.** `crates/storage/tests/dreaming_db_paths.rs` is 1,232 lines and 23 named
tests, driving the real stack through only `run_dreaming_scan`, `run_consolidation_scan` and
`update_watermark` with a canned `DreamingLlmFn` (:47-53). It is already conformance-shaped.
Its 22 DB-gated tests are the sole regression evidence for the swap phases and must run
verbatim through them - which means the storage entry-point signatures stay stable until the
swap is proven, and the suite stays under `crates/storage/tests/` so `just test-db` keeps
finding it.

**The degradation path.** `main.rs:2009-2010` builds the maintenance service only when both
a pool and an embedding client exist, types it `Option<Arc<dyn ...>>`, and warns by name at
:2132-2136, with a skip log for the trash sweep at :1973. A daemon with no database runs
fine and says why. This survives unchanged; the `Option` simply comes to hold a store rather
than a pool.

**Per-op mutual exclusion.** The extraction / consolidation / embeddings locks live in the
daemon (maintenance_service.rs:62-64), not the adapter, so no concurrency control moves.

Two smaller things are also already fine. `trash_count` and `empty_trash` are *already*
outbound-port methods (core/src/ports/knowledge.rs:77-82) with the Postgres adapter
delegating to `dreaming::trash` (knowledge.rs:255-260) - a working precedent for "the port
method lives in core, the body stays in the adapter". And RLS is no portability constraint
here: migration 029 is non-FORCE and scoped to the `adele_query` role used by
`execute_database_query`, while the daemon owns the tables (029_rls_backstop.sql:28-33).

## The hard part: the transactional boundary

Everything about the port shape is decided by one function. `reconcile::apply_ops`
(reconcile.rs:199-406) opens a single transaction at :208 and commits at :401, covering:

1. the opportunistic trash reap (:218),
2. per merge cluster: a member-provenance SELECT (:228), the canonical UPDATE (:253), the
   member soft-deletes (:282),
3. standalone content updates (:310),
4. scope-adds, each preceded by a metadata re-read (:333, :345),
5. standalone prunes (:366),
6. the `reviewed_at` batch over every reviewed id (:388).

The skill-index precedent explicitly does not transfer here, and says so in its own words:
`core/src/skill_catalog.rs:23-27` argues the reconcile needs no transaction *because nothing
is deleted*, so the worst a partial pass leaves is a stale presence flag. Consolidation
soft-deletes (reconcile.rs:282, :366) and hard-reaps (trash.rs:61). A half-applied plan
leaves a merge canonical rewritten with its members still live - duplicated content - or
members retired with no canonical carrying them forward. Copying "primitives only"
(skill_index/mod.rs:58-65) verbatim makes that reachable.

Three further properties constrain any answer:

- **`apply_ops` re-reads inside the transaction.** Cluster member `source` and `metadata`
  are read at :228-236 and `cluster_is_explicit` computed in Rust at :249-251, rather than
  trusting the snapshot loaded at consolidation.rs:428. That is load-bearing:
  `PgKnowledgeBaseStore::write` upserts with `source = COALESCE(EXCLUDED.source,
  knowledge_base.source)` (knowledge.rs:55), so a live turn can promote a row to `explicit`
  between load and apply, and the daemon's locks exclude only other maintenance passes.
- **Counts are write-derived, not plan-derived.** `stats.soft_deleted` accumulates
  `result.rows_affected()` (reconcile.rs:299, :381) deliberately, so a member already
  tombstoned by an earlier op is not counted twice (comment at :297-298). Tests assert exact
  values at dreaming_db_paths.rs:314, :441, :511 and :1008-1012.
- **`reap_expired_for_user` is generic over `PgExecutor`** (trash.rs:51-58) purely so one
  statement serves both the pool and the open transaction. That generic cannot survive a
  `dyn`-compatible trait, and `dyn` is required because conformance cases take `&dyn`.

## Options

### A. Fine-grained primitives (the skill-index shape, copied)

`update_entry`, `soft_delete_entry`, `add_scope`, `mark_reviewed`, `reap_before` - each a
port method, core sequences them.

*Cost.* Atomicity is gone. A partial consolidation is reachable and invisible to `just
check`; the failure mode is duplicated or orphaned knowledge, discovered days later. The six
SQL-encoded rules must either move to core (dropping the documented backstop at
reconcile.rs:360-364, which exists so *"no future caller of `apply_ops` can prune a
deliberately-entered fact by forgetting the filter"*) or be restated per adapter, which is
the duplication the port exists to remove.

*For SQLite.* Every primitive is a separate write, so a plan of N ops takes N write
transactions against a single-writer database, each one contending with live turns.

### B. A unit-of-work handle in `core::ports`

`begin() -> Box<dyn KnowledgeMaintenanceTx>` with the read and write methods duplicated on
the handle.

*Cost.* Roughly doubles the trait surface, and it leaks a storage concept - a transaction -
into `core::ports`, which is the thing the skill-index write-up congratulates itself on
avoiding. The current code only escapes duplicating pool-versus-transaction bodies by being
generic over `PgExecutor` (trash.rs:51-57), and a `dyn` trait cannot be generic that way, so
each adapter writes both forms by hand.

*For SQLite.* Nothing prevents core holding an open handle across an `await` on something
slow, which on a single-writer database blocks every other write in the process.

### C. A closure handed to the adapter

`store.in_transaction(|tx| async move { ... })`.

*Cost.* Requires higher-ranked lifetimes over boxed futures for an `async_trait` method -
awkward to write, worse to read, and unlike anything else in `core::ports`. It has B's
leak with worse ergonomics, and nothing structurally stops the LLM call migrating inside the
closure.

### D. One coarse plan-apply verb

Core computes a fully-resolved, typed plan; the port applies it atomically in one call;
the adapter owns the transaction and it never escapes.

*Cost.* The port is not "primitives only", and the plan type becomes the de-facto contract -
adding an operation kind is a port change. Naively it also loses the in-transaction re-read:
the provenance decision would be made from the snapshot loaded at consolidation.rs:428, so a
row promoted to `explicit` mid-pass could be pruned.

*For SQLite.* One short write transaction per user per pass. This is the shape that respects
single-writer best, because the LLM call and all plan computation are provably outside it.

### E. Per-command compare-and-set, no transaction

Each command carries the `source` (and `updated_at`) it expects and reports whether it
applied; retries are safe because every command is idempotent.

*Cost.* Alone it does not give atomicity - a merge whose canonical UPDATE succeeds and whose
member soft-deletes fail duplicates content. But as a *property of the commands inside* D it
is exactly the missing piece: it restores apply-time freshness without a re-read round trip,
and it keeps the never-prune rule in the command rather than in each adapter's `WHERE`
clause.

## Decisions

### 1. The port is a coarse plan-apply verb: option D, with E's commands

```rust
#[async_trait::async_trait]
pub trait KnowledgeMaintenanceStore: Send + Sync {
    async fn apply_consolidation_plan(
        &self,
        user: &UserId,
        plan: &ConsolidationPlan,
        now: DateTime<Utc>,
    ) -> Result<PlanOutcome, CoreError>;
    // ... reads and single-row writes below
}
```

`ConsolidationPlan` is a value core computes before any write begins: it carries resolved
merges, updates, scope-adds, prunes, the reviewed-id set, and the retention cutoff. The
adapter applies the whole thing in one transaction or none of it. The existing
`OpBuffer` (reconcile.rs:56-190) and `SynthesizedMerge` (reconcile.rs:49-54) are already
pure values computed before `pool.begin()`, so this is not a new concept - it is the
boundary the code already has, made explicit.

D is chosen over A because a partial consolidation destroys knowledge and no gate would
catch it; over B and C because both put a transaction abstraction into `core::ports` for a
single call site, and B additionally forces every method to exist twice; over E alone
because E does not make a merge atomic.

### 2. Every command carries a compare-and-set expectation; outcomes come back per command

Each write command names the `source` it expects to find. `PlanOutcome` reports, per
command, whether it applied - so `ConsolidationStats.soft_deleted` stays derived from what
was *written*, not from what was planned, and the deliberate no-double-count behaviour at
reconcile.rs:297-298 survives with the exact values dreaming_db_paths.rs:314, :441, :511 and
:1008-1012 assert.

This is what buys back D's lost re-read. The current code mixes both styles already - prunes
use an apply-time predicate (reconcile.rs:372) while merges use read-then-decide
(reconcile.rs:249-251) - so making the expectation explicit in the command unifies them
rather than inventing a mechanism.

### 3. The SQL backstops stay, and become conformance cases

The never-prune predicate (reconcile.rs:372) and the generation clamp (reconcile.rs:259,
:317) are enforced twice on purpose today: a core-side filter (consolidation.rs:239-245,
:68-70) *and* a SQL refusal. Moving the guards to core and deleting the backstops would
remove a documented defense-in-depth property and every current test would stay green,
because the planner filters first so the SQL clause is never the thing that saves the row.

The port contract therefore *requires* the backstop: an adapter must refuse to prune a row
whose stored `source` is `explicit`, and must clamp `review_generation` at the supplied cap,
regardless of what the plan says. The conformance suite asserts it directly by handing the
store a plan that violates both - the one thing today's suite cannot do, because the planner
is in the way.

### 4. Tenancy is an explicit parameter, not a task-local

Every port method takes `user: &UserId`. Today tenancy is resolved from `current_user_id()`
at SQL-composition time in nine places (common.rs:74, :105, :123; consolidation.rs:419;
reconcile.rs:205; trash.rs:43, :130, :143; extraction.rs:370), so the multi-tenant boundary
depends on whichever side happened to install `with_user_id` (extraction.rs:79,
consolidation.rs:104, trash.rs:95). Moving the per-user loop into core without settling this
lands every query in the sentinel partition - fail-open, in a codebase whose stated posture
(`multi-tenancy-boundary.md`) is fail-closed.

Adapters may still install the task-local internally as belt and braces; the *contract* is
the parameter. The conformance suite carries cross-tenant cases, which the skill-index
contract does not have and cannot have, the skill index being host-global by design
(skill_index/mod.rs:2-8). A SQLite adapter that passed the skill-index-shaped contract could
still leak across tenants; this one cannot.

### 5. `now` and cutoffs are injected

Following `reconcile_scan`, which takes `now: DateTime<Utc>` explicitly so tests are
deterministic (skill_catalog.rs:62-67), all 16 raw `NOW()` uses become supplied instants and
`NOW() - make_interval(days => $1)` (archival.rs:47, :60; trash.rs:65) becomes a bound
cutoff.

Three things fall out. `make_interval` is the only construct in the module with no
documented SQLite translation, and it disappears. Retention, archival-window and
`reviewed_at` behaviour become deterministic, so dreaming_db_paths.rs:343 and :1183 can have
adapter-free counterparts. And `MAX_RETENTION_DAYS = 365_000` (trash.rs:29-35), which exists
only because Postgres errors on an out-of-range interval, stops being a policy constant and
becomes an adapter detail - or disappears.

### 6. Two ports, because there are two aggregates

Archival updates `conversations`, not `knowledge_base` (archival.rs:43-67). It gets one
method on the existing `ConversationStore` port (core/src/ports/store.rs) -
`archive_quiet_conversations(cutoff) -> usize` - rather than riding along on a
knowledge-maintenance port, which would otherwise own two unrelated aggregates for
historical reasons. `run_archival_phase`'s unreachable user-scoped branch (archival.rs:54-68
is never reached, because mod.rs:74-83 calls it with no scope installed) is ported as the
all-users sweep only, and the `desktop_assistant_auth_jwt::DEFAULT_USER_ID` import
(archival.rs:25) leaves the storage module with it.

The trash reap and sweep go on `KnowledgeMaintenanceStore`. `trash_count` and `empty_trash`
stay exactly where they are on `KnowledgeBaseStore`.

### 7. The port is `#[async_trait]`, and the contract is executable with a reference implementation

`SkillIndexStore` is `#[async_trait]` and dyn-compatible; the adjacent `KnowledgeBaseStore`
uses `impl Future` and is not (core/src/ports/knowledge.rs:9-83). The new port follows
`SkillIndexStore`, because conformance cases take `&dyn` and the daemon holds the store as
`Arc<dyn ...>`.

The contract lives at `core::ports::knowledge_maintenance::conformance`, gated by core's
existing `test-support` feature (Cargo.toml:26-29), invoked from each adapter's suite by the
same per-suite `conformance_tests!` macro shape used at storage/tests/skill_index.rs:152-178
and storage-sqlite/tests/skill_index.rs:22-44.

**An in-memory reference implementation in core is mandatory, not optional.** In the
skill-index precedent the always-on coverage comes entirely from `InMemorySkillIndex`
(skill_index/mod.rs:147-243), whose own doc says why: the Postgres suite pass-skips without
`TEST_DATABASE_URL`. The SQLite arm contributes nothing to the gate, because
`storage-sqlite` is default-off and its test files are `#![cfg(feature = "sqlite")]` with no
recipe passing `--features sqlite` (#742). A conformance suite wired only into the two
database adapters would gate exactly nothing. The reference implementation must model
soft-delete, review generations, provenance and watermarks - that is real work, and it is
the work that turns the delete cap and the never-prune rule from `just test-db` into
`just check`.

Explicitly outside the contract, following the precedent's handling of ranking
(conformance.rs:14-18): **tag-dedup similarity**. An adapter with no vector search answers
"no near match", which is already `create_or_match_tag`'s normal `Created` path
(tag_registry.rs:189-197). The consequence is honest and stated - that adapter's tag
vocabulary drifts wider - and one conformance case asserts the degraded path still produces
a valid tag rather than an error.

### 8. Tag resolution splits at the threshold

`create_or_match_tag` (tag_registry.rs:136-226) is an embedding call, a pgvector `<=>`
nearest-neighbour query and an INSERT, gated by `TAG_DEDUP_DISTANCE_THRESHOLD = 0.10`
(tag_registry.rs:26) compared *inside* the adapter at :190. Extraction cannot be ported
without it, because `write_extracted_fact` hands the pool straight to it
(extraction.rs:339).

The port exposes `nearest_tag_by_embedding(user, embedding) -> Option<(TagRecord, f32)>`
plus `list_active_tags` and `insert_tag`; the threshold moves to core and core decides. That
keeps the policy constant in one place and lets a vectorless adapter return `None` honestly.
Tag embeddings are one scalar vector per row, not the dimensionless per-model `vector[]`
chunk array that is the central unsolved problem of storage-sqlite inc2
(DESIGN.md:131-140), so this does not drag inc2 onto the critical path.

### 9. `embedding_backfill` gets port methods, not a port

`recalculate_embeddings` (maintenance_service.rs:200-222) calls two `&PgPool` functions in
`embedding_backfill.rs` (761 lines). Porting its internals triples the diff for no
portability gain that this issue is about.

Instead the two entry points become methods on the existing `KnowledgeBaseStore` port, and
`PgKnowledgeBaseStore` delegates to `embedding_backfill` - precisely the pattern already in
that file, where `trash_count` and `empty_trash` delegate to `dreaming::trash`
(knowledge.rs:255-260). The daemon's `PgPool` field goes; the module stays where it is.
`BATCH_SIZE = 32` (embedding_backfill.rs:25) is an sqlx paging shape and stays with it.

## Where the constants, tunables and prompts land

**To `core`, verbatim, as domain policy.** `MAX_DELETE_FRACTION` (types.rs:75, with its
#694 incident note intact), `MAX_DELETE_REASON_CHARS` (:82), `MAX_MESSAGE_CHARS` (:29),
`MAX_HOLISTIC_PROMPT_CHARS` (:61), the function-local `PER_ENTRY_OVERHEAD = 200`
(consolidation.rs:456), and `SOFT_DELETE_TTL_DAYS` (:55). Five of the eight are crate-private
today (`mod.rs:33-37` re-exports only five of the eight), so most of this is invisible
outside the crate.

`SOFT_DELETE_TTL_DAYS` is the one the daemon reaches across the boundary for
(config/mod.rs:539, asserted at :2013). It stays a *default* that `[backend_tasks]
knowledge_trash_retention_days` overrides, and the configured value keeps threading daemon
config -> `soft_delete_retention_days` (maintenance_service.rs:56) -> the pass -> the reap
byte for byte. The config default simply points at core instead of at the adapter. It must
not be replaced with a literal `30`, or two independent thirties start drifting.

**To `core`, as parameters rather than constants.** `MAX_REVIEW_GENERATION` (:50) is both a
Rust guard (`is_settled`, consolidation.rs:68-70) and a SQL clamp bound at reconcile.rs:259
and :317; `MAX_CONVERSATIONS_PER_SCAN` (:32) reaches the database as `LIMIT $1`
(common.rs:58). Both become arguments core passes to the port. Leaving either in the adapter
is exactly the "each backend re-picks 10" drift the port exists to prevent.

**To `core::domain`, as types.** `KbScope` and `KbMetadata` (kb_metadata.rs, 149 lines,
pure serde, no sqlx) are used only by three dreaming files (extraction.rs:31,
consolidation.rs:43, reconcile.rs:21) and already appear on the would-be port surface via
`ProposedOp::AddScope`. `KbDeleteKind` (types.rs:97-116) moves with them; its `as_str`
spelling stays pinned to migration 038's CHECK constraint, and that doc comment moves too.

`SOURCE_EXPLICIT = "explicit"` (:89) becomes a `KbSource` enum in `core::domain`, with the
adapter owning the spelling map exactly as `KbDeleteKind::as_str` already does. The same
vocabulary value is currently declared independently four times - types.rs:89,
daemon/knowledge_service.rs:25, a bare literal at mcp-client/builtin.rs:1064, and twice in
SQL (reconcile.rs:256, :372) - against a column migration 026 documents as a closed set of
three values, carried across the existing port as an untyped `Option<String>`
(core/src/domain/knowledge.rs:13-18).

**To `core`, unchanged, as prompt text.** All four builders move: `build_system_prompt`
(consolidation.rs:478), `build_user_prompt` (:513), `build_extraction_system_prompt`
(extraction.rs:400) and `build_extraction_user_prompt` (:496). They are pure functions of
their inputs with zero storage coupling, and core already has a home for prompt text
(`core/src/prompts/mod.rs:267-274` uses `include_str!` over `sections/*.txt`).

One asymmetry must survive the move intact: the consolidation prompt **does not state** the
three hard rules. Never-prune-explicit, the settled-entry rules and the delete cap are
enforced only after the model replies (consolidation.rs:239-245, :266-270, :354-363), which
is why `protected_from_delete` and `settled_unchanged` exist as reported stats
(types.rs:125-132). That is deliberate and load-bearing - the model is not asked to
self-police, it is checked. Prompt, parser, planner and typed plan therefore move together;
splitting them across the boundary lets the declared op schema (`RawOp`,
consolidation.rs:551-578) drift from the applier with nothing to catch it.

**Staying in the adapter.** `MAX_RETENTION_DAYS` (trash.rs:29-35, a Postgres interval-range
artifact, and see decision 5), `BATCH_SIZE` (embedding_backfill.rs:25), and the four
deliberately cross-user statements (common.rs:44, consolidation.rs:407, trash.rs:161,
archival.rs:47) whose allowlist entries the static audit keys on by path.

**Staying in the daemon.** `MAINTENANCE_CALL_TIMEOUT = 120s` (maintenance_service.rs:39),
the retry count `RetryingLlmClient::new(c, 3)` (main.rs:2049, :2059), and the 30s/60s/120s
startup delays (main.rs:1938, :2097, :2166). None of these are storage or domain concerns.

**The LLM boundary.** `DreamingLlmFn` (types.rs:15-23) is a boxed string-in/string-out
closure whose stated rationale is *"so the daemon can plug in any backend"*. Core already
owns `LlmClient` and `ReasoningConfig` (core/src/ports/llm.rs) and `EmbeddingClient`, so
once the policy lives in core the closure buys nothing: it becomes `&dyn LlmClient`, and the
cancellation-plus-timeout wrapper the daemon builds at maintenance_service.rs:103-133 becomes
a daemon-side `LlmClient` decorator. Moving the policy needs **no new core dependency** -
`async-trait`, `chrono`, `tokio-util`, `serde_json`, `uuid` v7 and `tracing` are all already
there, and core has no `sqlx`.

## Landing sequence

Seven phases. Each is green at every commit, and **no phase needs a migration** - every
column the work touches (`source`, `review_generation`, `reviewed_at`, `deleted_at`,
`deleted_kind`, `deleted_reason`, `superseded_by`) exists as of 038, the highest ordinal on
main. Nobody should reserve 039 for this.

Phases 1-3 are behaviour-preserving by construction. Phases 4-6 are behaviour-preserving by
intent, and the 1,232-line DB suite is their only evidence - which runs under `just test-db`,
**not** `just check`, so a pull request for any of them that claims only `just check` has not
verified itself.

**Phase 1 - move the pure policy (behaviour-preserving, no port).**
Prompts, parsers, `OpBuffer` and its union-find, `slice_entries`, `clamp_delete_reason`,
`extract_json_payload`, `is_total_failure`, and the constants move to `crates/core` with
their 28 in-file unit tests. `storage::dreaming` calls into core and keeps `pub use` shims
for the five exported constants so `dreaming_db_paths.rs:33-37` does not churn in the same
diff. Golden-string tests for all four prompt builders land *in this commit* - the extraction
system prompt interpolates the live tag registry (extraction.rs:452-474), so a whitespace or
join change silently alters model behaviour with nothing failing. Acceptance: the four
rendered prompts are byte-identical, and the moved unit tests run under `just check`.

**Phase 2 - domain types (behaviour-preserving).**
`KbScope`, `KbMetadata` and `KbDeleteKind` move to `core::domain`; `KbSource` is introduced
and the four independent declarations collapse onto it. A new `KbReviewEntry` maintenance
projection is added rather than widening `KnowledgeEntry` - the shared type carries only
id/content/tags/metadata/created_at/updated_at/source, its timestamps are `String`, and
consolidation needs `review_generation`, `deleted_at`, `deleted_kind`, `superseded_by` and
`reviewed_at`. Widening the shared type would change every `KnowledgeBaseStore` read
projection for the benefit of one caller.

**Phase 3 - define the port, the contract and the reference implementation (additive).**
`core::ports::knowledge_maintenance` with `KnowledgeMaintenanceStore`, `ConsolidationPlan`
and `PlanOutcome`; `conformance` behind `test-support`; an in-memory reference
implementation; `PgKnowledgeMaintenanceStore` in `crates/storage` implementing it by moving
the existing statements. Nothing is switched over - the free functions still run - so this
phase cannot regress behaviour. Acceptance: the contract passes against both the reference
implementation (under `just check`) and Postgres (under `just test-db`), including the two
cases today's suite cannot express - a plan that names an `explicit` row for pruning is
refused by the store, and a plan that would push `review_generation` past the cap is clamped.

**Phase 4 - switch consolidation onto the port.**
`core::knowledge_maintenance::run_consolidation_pass` drives `&dyn
KnowledgeMaintenanceStore`; `storage::dreaming::run_consolidation_scan` becomes a shim that
constructs the Postgres store and delegates, so 18 of the 22 DB-gated tests run verbatim as
equivalence evidence. Acceptance: `just test-db` green with the suite unmodified, and
`stats.soft_deleted` still write-derived.

**Phase 5 - switch extraction and tag resolution onto the port.**
Tag resolution splits per decision 8; `run_dreaming_scan` becomes a shim the same way. The
retry semantics currently encoded as *which persistence call is skipped*
(process_one_conversation_for_extraction, extraction.rs:129-193: the watermark advances at
:150 and :191 and deliberately does not on the LLM-failure path at :159-168) become explicit
in core and get a named test, because they are invisible policy today.

**Phase 6 - archival and the trash sweep.**
`archive_quiet_conversations(cutoff)` onto `ConversationStore`; the sweep and reap onto
`KnowledgeMaintenanceStore`; `main.rs:1946` stops calling a storage free function.

**Phase 7 - daemon cutover (the phase that closes #744).**
`maintenance_service.rs` takes `Arc<dyn KnowledgeMaintenanceStore>` instead of `PgPool`; the
two `embedding_backfill` calls move onto `KnowledgeBaseStore` per decision 9;
`config/mod.rs:539` sources `SOFT_DELETE_TTL_DAYS` from core; the storage shims are deleted;
`docs/features/knowledge-maintenance.md` (:13, :53, :81, :203, :204),
`storage-sqlite/DESIGN.md` inc3 and `storage-sqlite/src/lib.rs:36` are corrected.
Acceptance: `rg PgPool crates/daemon/src` returns nothing.

**The audit allowlist moves with the SQL, in whichever phase moves it.**
`crates/storage/tests/audit_user_id_scoping.rs` allowlists three dreaming files *by path*
(:121, :133, :140) and its `is_allowed` (:431-437) only matches paths - it never fails on an
unused entry, so a stale one rots silently. One already has: :142 justifies the exemption by
naming `load_entries_needing_review_by_user`, which no longer exists; the actual cross-user
query is `load_user_ids_with_active_entries` (consolidation.rs:406). The four cross-user
statements must stay under `crates/storage/src/`, because the audit walks only that tree; a
port method whose body left the tree would take its scan out of audit range entirely. Fix
the stale rationale in the phase that touches the file.

## Interlocks

**#694 and its phases are the real collision, and it is head-on.** #711 edits
`build_user_prompt` and `build_system_prompt` at exactly consolidation.rs:478 and :513. #893
turns deletion into a disposition and needs new columns, hence a migration. #894 states it
*"supersedes the holistic design that #711 and #893 currently assume"* - it rewrites
`consolidate_user`, `slice_entries` and the prompts wholesale, replacing the nightly
whole-store pass with an incremental, similarity-cross-checked one. None of #694, #711,
#744, #893 or #894 is assigned, so the order is a free choice.

**Recommended order: #744 phases 1-3 first, then #694's phases, then #744 phases 4-7.**

Phases 1-3 are behaviour-preserving and additive, so they do not fight #894's design; they
give it somewhere to be written. #894's core claim is arithmetic - that clustered,
deduplicated retrieval costs 30-40K tokens on a busy day and near zero on a quiet one,
against today's fixed 160.2K characters - and arithmetic that decides how much of a user's
knowledge base gets shipped to a model should be unit-tested in the gate, not exercised
against a container. Phase 1 alone converts #711 from a Postgres-only prompt edit into a
core change with a golden test. Running the other order means #744 moves code #894 is about
to delete, and #894 is written where it cannot be cheaply tested.

Phases 4-7 should follow #893's migration rather than race it, since #893 widens
`deleted_kind` / `deleted_reason` / `superseded_by` to live rows and both would otherwise be
editing the same apply path.

**Migration ordinals.** #744 needs none at any phase. #730 (schema-version table and
migration lock) and #893 both want the next ordinal; they should coordinate with each other,
not with this.

**No file overlap elsewhere.** #721, #722, #738 and #740 all live in
`crates/storage/src/database.rs` (the `db_query` tool); #730 is `crates/storage/src/pool.rs`
plus a migration. None touches `dreaming/`. #738/#740 will edit `PERSONAL_DATA_TABLES`
(database.rs:67-87), which `audit_user_id_scoping.rs` also consumes - the same file as the
allowlist edits above, but a textually separate region.

**#742 is a hard dependency for the *claim*, not for the work.** Until the SQLite adapter
compiles in the gate, "both adapters run the contract" is aspirational. Decision 7's
in-memory reference implementation is what makes the contract enforced today; #742 is what
makes it enforced for SQLite. State the distinction rather than overclaiming.

**#882** notes the user_id audit does not cover `storage-sqlite` at all. Decision 4's
cross-tenant conformance cases partially compensate - they test behaviour rather than
scanning source - but they do not replace the audit.

## What this does not do

- **It does not implement anything in `crates/storage-sqlite`.** That crate has no
  `knowledge_base`, `tag_registry` or `dreaming_watermarks` table at all, and only two
  migrations. The schema, the migration-ordinal coordination and the `deleted_kind` CHECK
  contract are the real cost of a second adapter and none of it is visible in this port's
  diff. Worth recording, though: the dreaming SQL uses no vector and no full-text
  construct, so the dreaming half of inc3 does **not** depend on inc2 - which inverts the
  ordering at `storage-sqlite/DESIGN.md:123-128`.
- **It does not change dream-cycle behaviour, prompts, or any tunable's value.** Not the
  delete cap, not the generation cap, not the retention default. A behaviour change hiding
  in a move is the main way this goes wrong.
- **It does not turn any constant into a config knob.** `MAX_DELETE_FRACTION` is the
  strongest candidate given #694, and it stays a compile-time constant here.
- **It does not fix two things that look like bugs and may not be touched in a
  behaviour-preserving move.** `slice_entries` measures `content.len()` in *bytes* against a
  constant named `..._CHARS` (consolidation.rs:462-464), and `run_archival_phase` branches
  on the `DEFAULT_USER_ID` sentinel (archival.rs:38). Both need their own tracker entries.
- **It does not fix the `[backend_tasks]` reload gap.** `plan_reload` classifies a
  `[backend_tasks]` diff as hot-appliable (reload.rs:119-125), `RestartArea`
  (reload.rs:41-58) has no `BackendTasks` variant, and the maintenance service is built once
  with its values baked in (main.rs:2071-2083) with the timer cadences captured by value at
  spawn (main.rs:1925-1926, :1993-2000, :2150-2153). So a `SetBackendTasksSettings` write
  changes nothing and reports nothing. It is named here only so a refactor in this area does
  not silently make it better or worse without a test. It needs its own issue.
- **It does not port `embedding_backfill.rs`'s internals** - only its two entry points, per
  decision 9.
- **It does not widen `just check`.** #742 owns that.

## Open questions

- **Sequencing against #694.** The recommendation above is a recommendation. If #894 is
  starting imminently, landing phases 1-3 first is still the cheaper order, but that is the
  maintainer's call and it should be made once, in writing, on both issues.
- **Tag-dedup degradation on a vectorless adapter.** Decision 8 says such an adapter answers
  "no near match" and its vocabulary drifts wider. Brute-force cosine over the per-user tag
  vocabulary in core is genuinely feasible here and nowhere else in the knowledge base, since
  tag embeddings are one scalar vector per row. It would preserve behaviour at the cost of
  putting similarity math in core. Worth doing only if vocabulary parity turns out to matter.
- **`MAX_HOLISTIC_PROMPT_CHARS`.** Its own doc comment reasons in tokens against a model's
  context window, and the daemon already resolves that model at main.rs:2026-2037, so it
  could be provider-supplied rather than a fixed 200K. #894 may delete the question outright
  by removing the whole-store prompt; decide after #894, not before.
- **Whether the dreaming prompts join the golden assembled-prompt snapshot**
  (prompts/mod.rs:306-313) or carry their own golden tests. Phase 1 assumes the latter;
  joining the existing snapshot makes it noisier but keeps one guard.
- **Two DB-gated tests become redundant on extraction.** dreaming_db_paths.rs:751 and :787
  restate consolidation.rs:657 and :667 (blank reason to `None`, char-boundary clamp). Their
  only residual database value is that `Option::None` binds to SQL NULL rather than an empty
  string. Delete them, or downgrade them to a one-line conformance assertion? Stating the
  answer up front keeps the deletion from reading as lost coverage in review.
