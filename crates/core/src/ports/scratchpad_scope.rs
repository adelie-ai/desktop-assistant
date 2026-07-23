//! Per-subagent-turn scratchpad scope (task-locals).
//!
//! A subagent works in its own child conversation for reasoning/history, but
//! reads and writes the **session-global** scratchpad — the top-level
//! session's pad — with its entries namespaced by an `owner_todo` materialized
//! path (e.g. `"1.1"`). These task-locals carry that scope into the builtin
//! scratchpad tool and the scratchpad store without growing any port
//! signature, exactly mirroring the [`crate::ports::auth`] `UserId` and
//! [`crate::ports::conversation_ctx`] `ConversationId` precedents.
//!
//! Four slots, installed together by [`with_subagent_scope`] around the child
//! turn body (moved in as plain data, re-installed inside the spawned body —
//! never read across a `tokio::spawn`):
//!
//! - [`SCRATCHPAD_SCOPE`] — the conversation whose pad the scratchpad tools
//!   operate on (the SESSION/root conversation), distinct from the
//!   [`crate::ports::conversation_ctx`] `ConversationId`, which stays the child
//!   conversation (history / LLM / KB provenance).
//! - [`SCRATCHPAD_OWNER_TODO`] — the materialized-path namespace the running
//!   agent writes under and is confined to. Root sentinel `""` / unset =
//!   top-level.
//! - [`SCRATCHPAD_VISIBLE_BEFORE`] — the spawn snapshot cut: a canonical
//!   lowercase-hyphenated UUIDv7 string (the child's `spawn_marker`). Bound as
//!   TEXT because `scratchpads.id` is TEXT and UUIDv7 canonical strings are
//!   time-ordered. Its presence gates whether the snapshot predicate applies.
//! - [`SCRATCHPAD_ANCESTORS`] — the frozen ancestor-namespace chain (each
//!   ancestor *agent*'s `owner_todo`, e.g. `["", "1.1"]` for a child under
//!   subagent `1.1`). The snapshot read admits pre-marker rows ONLY from these
//!   namespaces, so a concurrent sibling/cousin's in-flight notes are never
//!   visible even though their ids may be `< marker` (#287 critic finding 1).
//!
//! When no scope is installed (top-level turns, background workers, tests) the
//! `current_*` accessors return `None`, and every consumer falls back to its
//! pre-subagent behavior byte-for-byte: the scratchpad tools use the current
//! conversation, the store stamps `owner_todo = ""` and applies no snapshot
//! predicate.

use crate::domain::ConversationId;

tokio::task_local! {
    /// The conversation whose scratchpad the current agent operates on — the
    /// SESSION/root pad for a subagent. See module docs.
    static SCRATCHPAD_SCOPE: ConversationId;
    /// The `owner_todo` materialized-path namespace the current agent writes
    /// under and is confined to. Root `""` when unset.
    static SCRATCHPAD_OWNER_TODO: String;
    /// The spawn snapshot cut (canonical UUIDv7 string). Its `Some`/`None`
    /// presence gates whether reads apply the snapshot predicate.
    static SCRATCHPAD_VISIBLE_BEFORE: String;
    /// The frozen ancestor-namespace chain the snapshot read may draw
    /// pre-marker context from. Excludes concurrent siblings/cousins.
    static SCRATCHPAD_ANCESTORS: Vec<String>;
}

/// The full scratchpad scope a spawn installs around a child turn. Frozen at
/// spawn and moved into the spawned body as plain data, then re-installed
/// there via [`with_subagent_scope`] — a task-local never propagates across a
/// `tokio::spawn`, so the scope must be re-established inside the spawned
/// future, not read from the parent's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentScope {
    /// The session/root conversation whose pad this subagent shares.
    pub session_conversation_id: ConversationId,
    /// This subagent's own namespace (its pinned materialized path).
    pub owner_todo: String,
    /// The spawn snapshot cut (canonical UUIDv7 string).
    pub visible_before: String,
    /// The ancestor-namespace chain (each ancestor agent's `owner_todo`).
    pub ancestors: Vec<String>,
}

/// Run `fut` with all four scratchpad-scope slots installed together. Nesting
/// the four `.scope()` calls in one helper is deliberate: a spawn can never
/// install a partial scope (e.g. an `owner_todo` without its matching
/// `visible_before`), which would corrupt confinement or the snapshot.
pub async fn with_subagent_scope<F, T>(scope: SubagentScope, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let f = SCRATCHPAD_ANCESTORS.scope(scope.ancestors, fut);
    let f = SCRATCHPAD_VISIBLE_BEFORE.scope(scope.visible_before, f);
    let f = SCRATCHPAD_OWNER_TODO.scope(scope.owner_todo, f);
    SCRATCHPAD_SCOPE
        .scope(scope.session_conversation_id, f)
        .await
}

/// The session/root conversation the current scratchpad scope redirects to, or
/// `None` outside any subagent scope (use the current conversation then).
pub fn current_scratchpad_scope() -> Option<ConversationId> {
    SCRATCHPAD_SCOPE.try_with(|c| c.clone()).ok()
}

/// The current `owner_todo` namespace, or `None` outside any scope (stamp
/// `""` / apply no confinement then).
pub fn current_owner_todo() -> Option<String> {
    SCRATCHPAD_OWNER_TODO.try_with(|s| s.clone()).ok()
}

/// The current spawn snapshot cut, or `None` outside any scope (read the pad
/// unbounded then).
pub fn current_visible_before() -> Option<String> {
    SCRATCHPAD_VISIBLE_BEFORE.try_with(|s| s.clone()).ok()
}

/// The current ancestor-namespace chain, or `None` outside any scope.
pub fn current_ancestors() -> Option<Vec<String>> {
    SCRATCHPAD_ANCESTORS.try_with(|a| a.clone()).ok()
}

/// LLM-visible tool name for spawning a subagent.
///
/// Defined here in `core` -- not in the `application` crate that *implements*
/// the tool -- so the `core::service` dispatch loop can intercept the call by
/// name to mint the child's [`SubagentScope`] from the loop-owned `StepStack`,
/// without `core` depending on `application` (which would invert the layering).
/// `application::subagent_tools` re-exports this as its public tool name.
pub const SPAWN_SUBAGENT_TOOL: &str = "spawn_subagent";

tokio::task_local! {
    /// The scope the dispatch loop has computed for the subagent that a
    /// `spawn_subagent` call is about to create. Installed by the loop AROUND
    /// that tool's execution (via [`with_pending_child_scope`]) and read once by
    /// the spawn-tool body (via [`current_pending_child_scope`]) to build the
    /// child's [`SubagentScope`].
    ///
    /// Distinct from [`SCRATCHPAD_SCOPE`]/[`SCRATCHPAD_OWNER_TODO`] etc., which
    /// describe the RUNNING agent's own scope; this describes the CHILD-to-be's.
    /// The loop owns the `StepStack` (so it can `fan_out` a fresh, cascade-
    /// anchored namespace key) but the spawn tool runs inside the `ToolExecutor`
    /// with no `StepStack` handle -- this task-local bridges the two.
    static PENDING_CHILD_SCOPE: SubagentScope;
}

/// Run `fut` (a `spawn_subagent` tool execution) with `scope` installed as the
/// scope its child should adopt. See [`PENDING_CHILD_SCOPE`].
pub async fn with_pending_child_scope<F, T>(scope: SubagentScope, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    PENDING_CHILD_SCOPE.scope(scope, fut).await
}

/// The child scope the dispatch loop computed for the in-flight
/// `spawn_subagent` call, or `None` when spawning outside that loop path (e.g.
/// `SpawnStandaloneAgent` or a unit test), in which case the spawn falls back
/// to its pre-#287 behavior.
pub fn current_pending_child_scope() -> Option<SubagentScope> {
    PENDING_CHILD_SCOPE.try_with(|s| s.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(owner: &str, marker: &str, ancestors: &[&str]) -> SubagentScope {
        SubagentScope {
            session_conversation_id: ConversationId::from("sess-1"),
            owner_todo: owner.to_string(),
            visible_before: marker.to_string(),
            ancestors: ancestors.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn scope_getters_return_none_outside_scope() {
        assert!(current_scratchpad_scope().is_none());
        assert!(current_owner_todo().is_none());
        assert!(current_visible_before().is_none());
        assert!(current_ancestors().is_none());
    }

    #[tokio::test]
    async fn scope_returns_installed_conversation() {
        with_subagent_scope(scope("1.1", "m", &[""]), async {
            assert_eq!(
                current_scratchpad_scope(),
                Some(ConversationId::from("sess-1"))
            );
        })
        .await;
    }

    #[tokio::test]
    async fn owner_todo_returns_installed_path() {
        with_subagent_scope(scope("1.1", "m", &[""]), async {
            assert_eq!(current_owner_todo().as_deref(), Some("1.1"));
        })
        .await;
    }

    #[tokio::test]
    async fn visible_before_returns_installed_marker() {
        with_subagent_scope(scope("1.1", "marker-xyz", &[""]), async {
            assert_eq!(current_visible_before().as_deref(), Some("marker-xyz"));
        })
        .await;
    }

    #[tokio::test]
    async fn ancestors_returns_installed_chain() {
        // Child under subagent "1.1": its ancestor namespaces are root + "1.1".
        with_subagent_scope(scope("1.1.2", "m", &["", "1.1"]), async {
            assert_eq!(
                current_ancestors(),
                Some(vec!["".to_string(), "1.1".to_string()])
            );
        })
        .await;
    }

    #[tokio::test]
    async fn with_subagent_scope_installs_all_at_once() {
        with_subagent_scope(scope("2.3", "mk", &["", "2"]), async {
            assert!(current_scratchpad_scope().is_some());
            assert_eq!(current_owner_todo().as_deref(), Some("2.3"));
            assert_eq!(current_visible_before().as_deref(), Some("mk"));
            assert_eq!(current_ancestors().map(|a| a.len()), Some(2));
        })
        .await;
    }

    #[tokio::test]
    async fn nested_owner_todo_overrides_then_restores() {
        with_subagent_scope(scope("1.1", "m1", &[""]), async {
            assert_eq!(current_owner_todo().as_deref(), Some("1.1"));
            // A nested spawn (grandchild) installs a deeper scope; on return the
            // outer scope is restored — task-local scoping is a stack.
            with_subagent_scope(scope("1.1.1", "m2", &["", "1.1"]), async {
                assert_eq!(current_owner_todo().as_deref(), Some("1.1.1"));
                assert_eq!(current_visible_before().as_deref(), Some("m2"));
            })
            .await;
            assert_eq!(current_owner_todo().as_deref(), Some("1.1"));
            assert_eq!(current_visible_before().as_deref(), Some("m1"));
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn locals_do_not_cross_tokio_spawn() {
        // Pins the contract the spawn slice depends on: a task-local does NOT
        // propagate into a tokio::spawn'd task, so the scope must be
        // re-installed inside the spawned body (moved in as data), never read
        // from the parent. If this ever "passed through", the child would
        // silently inherit the parent's namespace.
        with_subagent_scope(scope("1.1", "m", &[""]), async {
            let inner = tokio::spawn(async { current_owner_todo() })
                .await
                .expect("join");
            assert!(
                inner.is_none(),
                "task-local must not cross tokio::spawn; got {inner:?}"
            );
        })
        .await;
    }
}
