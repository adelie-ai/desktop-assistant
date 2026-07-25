//! Acceptance tests for the subagent parent-wake *signal* (slice 1 of the
//! parent-wake feature; see `docs/design/subagent-parent-wake.md`).
//!
//! Written before the implementation (TDD). They describe the business
//! outcome: when a `TaskKind::Subagent` task reaches a terminal state, the
//! registry invokes a late-set observer with a `SubagentCompletion` carrying
//! everything a parent-wake coordinator needs — the child's task id, name,
//! session/child conversation ids, the `owner_todo` scratchpad reference, the
//! terminal status, and how many sibling subagents are still running. The
//! signal fires for completed / failed / cancelled children, never for
//! non-subagent tasks, and is a safe no-op when no observer is set.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::UserId;
use desktop_assistant_application::background_tasks::{
    BackgroundTaskRegistry, SpawnMeta, SubagentCompletion, SubagentCompletionObserver,
};
use tokio::time::timeout;

/// Shared capture buffer the test observer pushes completions into.
type Seen = Arc<Mutex<Vec<SubagentCompletion>>>;

fn unique_user(label: &str) -> UserId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    UserId::new(format!("user-{label}-{n}"))
}

fn subagent_kind(parent: &str, name: &str, child_conv: &str, session: &str) -> api::TaskKind {
    api::TaskKind::Subagent {
        parent_task_id: api::TaskId(parent.into()),
        conversation_id: child_conv.into(),
        name: name.into(),
        session_conversation_id: session.into(),
    }
}

fn capture() -> (Seen, SubagentCompletionObserver) {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let observer: SubagentCompletionObserver =
        Arc::new(move |c| sink.lock().expect("seen poisoned").push(c));
    (seen, observer)
}

async fn wait_until<F: FnMut() -> bool>(mut pred: F, label: &str) {
    for _ in 0..300 {
        if pred() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("predicate '{label}' never became true within timeout");
}

/// A terminal `Subagent` task fires the observer once, carrying the full
/// payload a parent-wake coordinator needs — including the `owner_todo`
/// scratchpad reference and a `siblings_remaining` of 0 when it is the only
/// child under its parent.
#[tokio::test]
async fn subagent_completion_fires_with_full_payload() {
    let registry = BackgroundTaskRegistry::new();
    let (seen, observer) = capture();
    registry.set_subagent_observer(observer);

    let user = unique_user("payload");
    let kind = subagent_kind("parent-1", "Walmart prices", "child-conv-1", "sess-1");
    let child_id = registry.spawn_with_meta(
        user.clone(),
        kind,
        "Subagent: Walmart prices".into(),
        SpawnMeta {
            owner_todo: "todo-walmart".into(),
            spawn_marker: None,
        },
        |_ctx| async { Ok(()) },
    );

    wait_until(
        || seen.lock().expect("seen poisoned").len() == 1,
        "observer fired once",
    )
    .await;

    let done = seen.lock().expect("seen poisoned")[0].clone();
    assert_eq!(done.child_task_id, child_id);
    assert_eq!(done.parent_task_id, api::TaskId("parent-1".into()));
    assert_eq!(done.child_name, "Walmart prices");
    assert_eq!(done.child_conversation_id, "child-conv-1");
    assert_eq!(done.session_conversation_id, "sess-1");
    assert_eq!(done.owner_todo, "todo-walmart");
    assert_eq!(done.status, api::TaskStatus::Completed);
    assert_eq!(done.user_id, user);
    assert_eq!(
        done.siblings_remaining, 0,
        "the only child under its parent has no siblings remaining"
    );
}

/// A non-subagent task (a foreground `Conversation` turn) must NOT fire the
/// subagent-completion observer — the signal is subagent-specific.
#[tokio::test]
async fn conversation_completion_does_not_fire() {
    let registry = BackgroundTaskRegistry::new();
    let (seen, observer) = capture();
    registry.set_subagent_observer(observer);

    let user = unique_user("conv");
    let mut events = registry.subscribe(&user);
    let id = registry.spawn(
        user.clone(),
        api::TaskKind::Conversation {
            conversation_id: "conv-x".into(),
        },
        "Conversation: conv-x".into(),
        |_ctx| async { Ok(()) },
    );

    // Wait for the terminal broadcast so we know finalize ran.
    wait_for_completion(&mut events, &id).await;
    // Small grace: the observer, if it (wrongly) fired, does so within finalize
    // before the broadcast, so any erroneous push has already happened.
    assert!(
        seen.lock().expect("seen poisoned").is_empty(),
        "a Conversation task must not fire the subagent observer"
    );
}

/// `siblings_remaining` reflects how many sibling subagents under the same
/// parent are still non-terminal at the moment a child finalizes. A child that
/// finishes while a sibling is still running reports 1; the sibling, finishing
/// last, reports 0.
#[tokio::test]
async fn siblings_remaining_counts_nonterminal_siblings() {
    let registry = BackgroundTaskRegistry::new();
    let (seen, observer) = capture();
    registry.set_subagent_observer(observer);

    let user = unique_user("siblings");

    // A oneshot keeps the second child running until we release it. oneshot
    // (not Notify) so the release can't be lost to a wake-before-park race: the
    // value is buffered and the body's `.await` returns even if `send` ran first.
    let (release, released) = tokio::sync::oneshot::channel::<()>();

    let first = registry.spawn_with_meta(
        user.clone(),
        subagent_kind("parent-2", "first", "c1", "sess-2"),
        "Subagent: first".into(),
        SpawnMeta {
            owner_todo: "todo-first".into(),
            spawn_marker: None,
        },
        |_ctx| async { Ok(()) },
    );
    let _second = registry.spawn_with_meta(
        user.clone(),
        subagent_kind("parent-2", "second", "c2", "sess-2"),
        "Subagent: second".into(),
        SpawnMeta {
            owner_todo: "todo-second".into(),
            spawn_marker: None,
        },
        move |_ctx| async move {
            let _ = released.await;
            Ok(())
        },
    );

    // `first` finalizes while `second` is still blocked → 1 sibling remaining.
    wait_until(
        || {
            seen.lock()
                .expect("seen poisoned")
                .iter()
                .any(|c| c.child_task_id == first)
        },
        "first child fired",
    )
    .await;
    let first_done = seen
        .lock()
        .expect("seen poisoned")
        .iter()
        .find(|c| c.child_task_id == first)
        .cloned()
        .expect("first completion present");
    assert_eq!(
        first_done.siblings_remaining, 1,
        "first child finished while its sibling was still running"
    );

    // Release the second; finishing last, it reports 0 siblings remaining.
    let _ = release.send(());
    wait_until(
        || seen.lock().expect("seen poisoned").len() == 2,
        "second child fired",
    )
    .await;
    let second_done = seen
        .lock()
        .expect("seen poisoned")
        .iter()
        .find(|c| c.child_name == "second")
        .cloned()
        .expect("second completion present");
    assert_eq!(
        second_done.siblings_remaining, 0,
        "the last child to finish has no siblings remaining"
    );
}

/// A subagent whose body returns `Err` fires the observer with `Failed`, so the
/// parent is woken to react rather than waiting forever on a dead child.
#[tokio::test]
async fn failed_subagent_fires_with_failed_status() {
    let registry = BackgroundTaskRegistry::new();
    let (seen, observer) = capture();
    registry.set_subagent_observer(observer);

    let user = unique_user("failed");
    registry.spawn_with_meta(
        user.clone(),
        subagent_kind("parent-3", "boom", "c3", "sess-3"),
        "Subagent: boom".into(),
        SpawnMeta::default(),
        |_ctx| async { Err(anyhow::anyhow!("child exploded")) },
    );

    wait_until(
        || seen.lock().expect("seen poisoned").len() == 1,
        "failed child fired",
    )
    .await;
    let done = seen.lock().expect("seen poisoned")[0].clone();
    assert_eq!(done.status, api::TaskStatus::Failed);
}

/// A cancelled subagent fires the observer with `Cancelled`.
#[tokio::test]
async fn cancelled_subagent_fires_with_cancelled_status() {
    let registry = BackgroundTaskRegistry::new();
    let (seen, observer) = capture();
    registry.set_subagent_observer(observer);

    let user = unique_user("cancelled");
    // oneshot, not Notify: the release is buffered, so it can't be lost to a
    // wake-before-park race between `cancel` and releasing the body.
    let (release, released) = tokio::sync::oneshot::channel::<()>();
    let id = registry.spawn_with_meta(
        user.clone(),
        subagent_kind("parent-4", "slow", "c4", "sess-4"),
        "Subagent: slow".into(),
        SpawnMeta::default(),
        move |_ctx| async move {
            let _ = released.await;
            Ok(())
        },
    );

    registry
        .cancel(&user, &id)
        .expect("cancel a running subagent");
    let _ = release.send(());

    wait_until(
        || seen.lock().expect("seen poisoned").len() == 1,
        "cancelled child fired",
    )
    .await;
    let done = seen.lock().expect("seen poisoned")[0].clone();
    assert_eq!(done.status, api::TaskStatus::Cancelled);
}

/// With no observer set, a subagent finalizing is a safe no-op: the task still
/// reaches a terminal state and broadcasts `TaskCompleted` as usual.
#[tokio::test]
async fn no_observer_is_a_safe_noop() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("noobs");
    let mut events = registry.subscribe(&user);
    let id = registry.spawn_with_meta(
        user.clone(),
        subagent_kind("parent-5", "lonely", "c5", "sess-5"),
        "Subagent: lonely".into(),
        SpawnMeta::default(),
        |_ctx| async { Ok(()) },
    );
    let (status, _err) = wait_for_completion(&mut events, &id).await;
    assert_eq!(status, api::TaskStatus::Completed);
}

/// Drain events for `task_id` until its terminal `TaskCompleted` arrives.
async fn wait_for_completion(
    events: &mut tokio::sync::broadcast::Receiver<api::Event>,
    task_id: &api::TaskId,
) -> (api::TaskStatus, Option<String>) {
    let want = task_id.0.clone();
    loop {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted {
                id,
                status,
                last_error,
            })) if id == want => return (status, last_error),
            Ok(Ok(_)) => continue,
            Ok(Err(e)) => panic!("broadcast recv error: {e}"),
            Err(_) => panic!("timed out waiting for completion of {want}"),
        }
    }
}
