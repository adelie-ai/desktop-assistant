//! Event-driven parent-wake: re-engage a parent conversation when its
//! subagents finish, without polling (issue #668; see
//! `docs/design/subagent-parent-wake.md`).
//!
//! [`ParentWakeCoordinator`] consumes the registry's [`SubagentCompletion`]
//! signal (slice 1) and drives an autonomous *wake turn* on the top-level
//! session conversation: one turn that tells the parent which child finished
//! and where its result is on the scratchpad, then lets the parent decide what
//! to do. Running the turn is delegated to a [`ParentWaker`] the daemon supplies
//! (over the normal fan-out send path, so the turn renders live in the client
//! that is viewing the conversation). This module owns only the *when*:
//!
//! - **Coalesce.** The first completion for a conversation starts a wake turn
//!   immediately. Completions that arrive while that turn is running are batched
//!   and drained by the *next* wake turn, so a burst of N near-simultaneous
//!   finishers costs at most one extra turn, not N.
//! - **Serialise.** At most one wake turn per conversation at a time — the
//!   coordinator awaits each [`ParentWaker::wake`] before starting the next, so
//!   two children finishing together never launch two overlapping turns on the
//!   same conversation (which would corrupt its transcript order).
//!
//! Bounding autonomy, two skips in [`ParentWakeCoordinator::on_completion`]:
//!
//! - **Detached children only.** A completion whose `notify_parent` is `false`
//!   came from a blocking `spawn_subagent { wait: true }` — the default — whose
//!   result the parent consumed inline during its own turn. Waking there would
//!   run a whole extra turn over an answer the parent already delivered.
//! - **User-visible conversations only.** The coordinator wakes the *session*
//!   conversation — the top-level one the user sees — never a hidden subagent
//!   conversation (the `session == child` skip).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use desktop_assistant_api_model as api;
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::ports::turn_interactivity::{
    TurnInteractivity, with_turn_interactivity,
};

use crate::EventSink;
use crate::background_tasks::{SubagentCompletion, SubagentCompletionObserver};

/// An [`EventSink`] that discards everything. A daemon-initiated wake turn has no
/// originating client, so its base sink is a sink to nowhere; the events that
/// matter reach viewers through the handler's fan-out (which wraps this base and
/// delivers to every connection subscribed to the conversation). Lives here,
/// with [`EventSink`], so the daemon adapter that drives wake turns need not name
/// the wire `Event` type itself.
pub struct NullEventSink;

#[async_trait]
impl EventSink for NullEventSink {
    async fn emit(&self, _event: api::Event) -> bool {
        // Always "available": there is no consumer to disconnect, and returning
        // `false` would signal the turn to abort streaming.
        true
    }
}

/// Runs a single wake turn to completion. Implemented by the daemon over the
/// fan-out send path (see `crates/daemon`); mocked in tests.
///
/// `wake` MUST return only once the turn has ended — that return is how the
/// coordinator serialises, awaiting each wake before draining the next batch.
#[async_trait]
pub trait ParentWaker: Send + Sync {
    /// Inject `prompt` as a fresh turn on `conversation_id` for `user_id` and
    /// run it to completion. Errors are the implementor's to log and swallow —
    /// a failed wake turn must not poison the coordinator.
    async fn wake(&self, user_id: UserId, conversation_id: String, prompt: String);
}

/// Per-conversation coalescing state.
#[derive(Default)]
struct PerConversation {
    /// A wake turn is currently draining this conversation.
    running: bool,
    /// Completions observed since the running turn started draining.
    pending: Vec<SubagentCompletion>,
}

/// Coalescing, serialising parent-wake coordinator. Cheap to hold behind an
/// `Arc`; the registry keeps only a `Weak` to it (via [`Self::observer`]).
pub struct ParentWakeCoordinator {
    /// The turn-runner. `Weak` because the daemon adapter it points at holds a
    /// `Weak` back to the handler that (transitively) owns this coordinator;
    /// keeping it strong would leak the graph for the process lifetime.
    waker: Weak<dyn ParentWaker>,
    /// Global kill switch (`daemon.toml [subagents] wake_parent`). When `false`
    /// the coordinator drops every completion and never wakes a parent.
    enabled: bool,
    /// Per-session-conversation coalescing state.
    state: Mutex<HashMap<String, PerConversation>>,
}

impl ParentWakeCoordinator {
    /// Build a coordinator over `waker`. `enabled` is the config kill switch.
    pub fn new(waker: Weak<dyn ParentWaker>, enabled: bool) -> Self {
        Self {
            waker,
            enabled,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// The observer to hand [`BackgroundTaskRegistry::set_subagent_observer`].
    /// Holds only a `Weak` to the coordinator, so once the daemon drops its
    /// strong `Arc` at shutdown the closure becomes an inert no-op.
    ///
    /// [`BackgroundTaskRegistry::set_subagent_observer`]: crate::background_tasks::BackgroundTaskRegistry::set_subagent_observer
    pub fn observer(self: &Arc<Self>) -> SubagentCompletionObserver {
        let weak = Arc::downgrade(self);
        Arc::new(move |completion| {
            if let Some(this) = weak.upgrade() {
                this.on_completion(completion);
            }
        })
    }

    /// Enqueue a completion and, when no wake turn is already draining this
    /// conversation, start one on a spawned task. Returns promptly — the
    /// registry invokes this synchronously from `finalize`, so it must not run
    /// the turn inline.
    fn on_completion(self: &Arc<Self>, completion: SubagentCompletion) {
        if !self.enabled {
            return;
        }
        // The parent already has this child's answer: a blocking
        // `spawn_subagent { wait: true }` (the default) returns the child's
        // final text straight into the still-running parent turn. Waking would
        // queue a second turn behind that turn's per-conversation lock and then
        // ask the parent to consolidate a result it has already delivered.
        // Parent-wake exists for the detached (`wait: false`) case, where the
        // parent's turn ends without ever seeing the answer.
        if !completion.notify_parent {
            return;
        }
        // No distinct parent to wake: a subagent whose session conversation is
        // its own conversation (no #287 session scope was installed) has no
        // separate top-level conversation to re-engage. Skip rather than run an
        // autonomous turn on a hidden subagent conversation — that both bounds
        // autonomy to user-visible conversations and avoids a pointless turn no
        // client is watching.
        if completion.session_conversation_id == completion.child_conversation_id {
            return;
        }
        let session = completion.session_conversation_id.clone();
        let start = {
            let mut state = self.state.lock().expect("parent-wake state poisoned");
            let entry = state.entry(session.clone()).or_default();
            entry.pending.push(completion);
            // Start a drain only if one isn't already running; otherwise the
            // running drain will pick this up on its next iteration.
            if entry.running {
                false
            } else {
                entry.running = true;
                true
            }
        };
        if start {
            let this = Arc::clone(self);
            tokio::spawn(async move { this.drain(session).await });
        }
    }

    /// Drain `session`'s pending completions one batch per wake turn until the
    /// queue is empty, awaiting each turn so wakes never overlap.
    async fn drain(self: Arc<Self>, session: String) {
        loop {
            // Take the current batch, or clear `running` and stop when empty —
            // both under the same lock a new completion takes, so a completion
            // arriving after an "empty" observation re-sets `running` and spawns
            // a fresh drain rather than being stranded behind a stale flag.
            let batch = {
                let mut state = self.state.lock().expect("parent-wake state poisoned");
                match state.get_mut(&session) {
                    Some(entry) if !entry.pending.is_empty() => std::mem::take(&mut entry.pending),
                    Some(entry) => {
                        entry.running = false;
                        return;
                    }
                    None => return,
                }
            };
            let Some(waker) = self.waker.upgrade() else {
                // The daemon handler is gone (shutdown). Release the flag so a
                // later incarnation isn't blocked, then stop.
                if let Some(entry) = self
                    .state
                    .lock()
                    .expect("parent-wake state poisoned")
                    .get_mut(&session)
                {
                    entry.running = false;
                }
                return;
            };
            let user_id = batch[0].user_id.clone();
            let prompt = build_wake_prompt(&batch);
            // A wake turn is one no client asked for and none is watching: it
            // runs off a drain task long after the parent's connection went
            // quiet. State that (#942) instead of leaving it to the session
            // sentinel, which would flip the answer the moment a wake ever runs
            // under a connection scope.
            with_turn_interactivity(
                TurnInteractivity::Headless,
                waker.wake(user_id, session.clone(), prompt),
            )
            .await;
        }
    }
}

/// Synthesise the wake turn's injected prompt from a batch of completions
/// (slice 3). Names each finished child, its terminal status, and the concrete
/// scratchpad reference(s) the parent can retrieve its result from — without
/// inlining the result itself. When every dispatched sibling is now done
/// (`siblings_remaining == 0` for any completion in the batch) it asks for the
/// consolidated answer; otherwise it says how many are still running and that
/// the parent needn't wait.
fn build_wake_prompt(batch: &[SubagentCompletion]) -> String {
    let n = batch.len();
    let noun = if n == 1 { "subagent" } else { "subagents" };
    let mut out = String::new();
    let _ = write!(
        out,
        "[automatic] {n} {noun} you dispatched just finished:\n\n"
    );
    for c in batch {
        let _ = writeln!(
            out,
            "- \"{}\" {}. Its result is saved in your scratchpad as a `result` \
             note under owner_todo \"{}\"; retrieve it with \
             get_subagent_status(\"{}\").",
            c.child_name,
            status_phrase(c.status),
            c.owner_todo,
            c.child_task_id.0,
        );
    }
    // `siblings_remaining` only drops as more finish, so the smallest value in
    // the batch is the freshest count; 0 means every dispatched sibling is done.
    let remaining = batch
        .iter()
        .map(|c| c.siblings_remaining)
        .min()
        .unwrap_or(0);
    out.push('\n');
    if remaining == 0 {
        out.push_str(
            "All of the subagents you dispatched have now finished. Review the \
             full set of results against the original request and produce your \
             consolidated answer. The results are already saved — decide what to \
             do with them.",
        );
    } else {
        let sib = if remaining == 1 { "is" } else { "are" };
        let _ = write!(
            out,
            "{remaining} more {sib} still running. Review the finished one(s) now \
             and act if useful — you'll be woken again as the rest complete, so \
             you don't have to wait for all of them.",
        );
    }
    out
}

/// Past-tense phrase describing a terminal status, for the wake message.
fn status_phrase(status: api::TaskStatus) -> &'static str {
    match status {
        api::TaskStatus::Completed => "completed",
        api::TaskStatus::Failed => "failed",
        api::TaskStatus::Cancelled => "was cancelled",
        // Non-terminal statuses never reach the wake path (the signal fires only
        // on terminal transitions); render them defensively rather than panic.
        api::TaskStatus::Pending => "is pending",
        api::TaskStatus::Running => "is still running",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::session::{SessionId, with_session_id};
    use desktop_assistant_core::ports::turn_interactivity::{
        TurnInteractivity, current_turn_interactivity,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::oneshot;

    fn completion(
        session: &str,
        child_conv: &str,
        name: &str,
        owner_todo: &str,
        task_id: &str,
        status: api::TaskStatus,
        siblings_remaining: usize,
    ) -> SubagentCompletion {
        SubagentCompletion {
            user_id: UserId::new("u-1".to_string()),
            parent_task_id: api::TaskId("parent".into()),
            child_conversation_id: child_conv.into(),
            session_conversation_id: session.into(),
            child_task_id: api::TaskId(task_id.into()),
            child_name: name.into(),
            owner_todo: owner_todo.into(),
            status,
            siblings_remaining,
            // These fixtures describe detached (`wait: false`) children — the
            // case parent-wake exists for. `waited` builds the other one.
            notify_parent: true,
        }
    }

    /// A completion for a child the parent blocked on (`spawn_subagent
    /// { wait: true }`, the default): its result went back inline, so nothing
    /// is owed to the parent.
    fn waited(session: &str, child_conv: &str, name: &str) -> SubagentCompletion {
        SubagentCompletion {
            notify_parent: false,
            ..completion(
                session,
                child_conv,
                name,
                "todo-x",
                "task-x",
                api::TaskStatus::Completed,
                0,
            )
        }
    }

    /// Records each wake, tracks max concurrency (to prove serialisation), and
    /// can hold its FIRST wake open on a oneshot so a test can enqueue more
    /// completions mid-turn and observe coalescing.
    #[derive(Default)]
    struct MockWaker {
        prompts: Mutex<Vec<String>>,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        hold: Mutex<Option<oneshot::Receiver<()>>>,
        /// What each wake turn observes for its interactivity (#942): the
        /// ambient answer, and the answer with a live client session installed
        /// around the read.
        interactivity: Mutex<Vec<(TurnInteractivity, TurnInteractivity)>>,
    }

    #[async_trait]
    impl ParentWaker for MockWaker {
        async fn wake(&self, _user_id: UserId, _conversation_id: String, prompt: String) {
            let ambient = current_turn_interactivity();
            let under_session = with_session_id(SessionId::new("conn-7"), async {
                current_turn_interactivity()
            })
            .await;
            self.interactivity
                .lock()
                .expect("interactivity poisoned")
                .push((ambient, under_session));
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            // The first wake awaits the hold (if installed) so the test can
            // enqueue more completions while a turn is "in progress".
            let held = self.hold.lock().expect("hold poisoned").take();
            if let Some(rx) = held {
                let _ = rx.await;
            }
            self.prompts.lock().expect("prompts poisoned").push(prompt);
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn coordinator(waker: &Arc<MockWaker>, enabled: bool) -> Arc<ParentWakeCoordinator> {
        let dynamic: Arc<dyn ParentWaker> = waker.clone();
        Arc::new(ParentWakeCoordinator::new(
            Arc::downgrade(&dynamic),
            enabled,
        ))
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

    /// A single completed child triggers exactly one wake whose prompt names the
    /// child, points at its `owner_todo` result note and `get_subagent_status`
    /// handle, and — since it was the last sibling — asks for the consolidated
    /// answer.
    #[tokio::test]
    async fn single_completion_wakes_with_scratchpad_refs_and_holistic_ask() {
        let waker = Arc::new(MockWaker::default());
        let dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "Walmart prices",
            "todo-walmart",
            "task-abc",
            api::TaskStatus::Completed,
            0,
        ));

        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 1,
            "one wake fired",
        )
        .await;
        let prompt = waker.prompts.lock().expect("poisoned")[0].clone();
        assert!(
            prompt.contains("Walmart prices"),
            "names the child: {prompt}"
        );
        assert!(
            prompt.contains("todo-walmart"),
            "cites owner_todo: {prompt}"
        );
        assert!(
            prompt.contains("get_subagent_status(\"task-abc\")"),
            "cites the status handle: {prompt}"
        );
        assert!(
            prompt.contains("consolidated answer"),
            "asks for the holistic review when none remain: {prompt}"
        );
        drop(dynamic);
    }

    /// #942: a wake turn is one nobody asked for and nobody is watching, and
    /// the coordinator states that rather than leaving it to whatever session
    /// is ambient when the drain runs.
    #[tokio::test]
    async fn a_parent_wake_turn_is_headless() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "prices",
            "todo-1",
            "task-abc",
            api::TaskStatus::Completed,
            0,
        ));

        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 1,
            "one wake fired",
        )
        .await;

        let seen = waker
            .interactivity
            .lock()
            .expect("interactivity poisoned")
            .clone();
        assert_eq!(seen.len(), 1, "exactly one wake turn ran");
        assert_eq!(
            seen[0].0,
            TurnInteractivity::Headless,
            "a wake turn has no one watching it"
        );
        assert_eq!(
            seen[0].1,
            TurnInteractivity::Headless,
            "the stated headlessness beats any session installed around the wake"
        );
    }

    /// A completed child with siblings still running produces the "N more still
    /// running, you don't have to wait" variant, not the holistic ask.
    #[tokio::test]
    async fn completion_with_siblings_remaining_says_more_running() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "Costco prices",
            "todo-costco",
            "task-def",
            api::TaskStatus::Completed,
            3,
        ));

        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 1,
            "one wake fired",
        )
        .await;
        let prompt = waker.prompts.lock().expect("poisoned")[0].clone();
        assert!(
            prompt.contains("3 more"),
            "reports the remaining count: {prompt}"
        );
        assert!(
            !prompt.contains("consolidated answer"),
            "does not ask for the holistic review while siblings run: {prompt}"
        );
    }

    /// Completions arriving while a wake turn is running are coalesced into a
    /// single follow-up turn, and wakes never overlap. Three children where two
    /// finish during the first turn ⇒ exactly two wake turns, max concurrency 1,
    /// and the second prompt carries both late children.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bursts_coalesce_into_one_followup_and_never_overlap() {
        let (release, released) = oneshot::channel::<()>();
        let waker = Arc::new(MockWaker {
            hold: Mutex::new(Some(released)),
            ..MockWaker::default()
        });
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        // First child: starts a wake that blocks on the hold.
        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "alpha",
            "todo-a",
            "task-a",
            api::TaskStatus::Completed,
            2,
        ));
        wait_until(
            || waker.in_flight.load(Ordering::SeqCst) == 1,
            "first wake is in progress",
        )
        .await;

        // Two more finish while the first wake is held — they must coalesce.
        coord.on_completion(completion(
            "sess-T",
            "child-2",
            "beta",
            "todo-b",
            "task-b",
            api::TaskStatus::Completed,
            1,
        ));
        coord.on_completion(completion(
            "sess-T",
            "child-3",
            "gamma",
            "todo-c",
            "task-c",
            api::TaskStatus::Completed,
            0,
        ));

        // Release the first wake; a single follow-up wake drains both.
        let _ = release.send(());
        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 2,
            "exactly two wake turns ran",
        )
        .await;
        // Give any (erroneous) third wake a chance to appear.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let prompts = waker.prompts.lock().expect("poisoned").clone();
        assert_eq!(prompts.len(), 2, "burst coalesced into one follow-up turn");
        assert!(
            prompts[0].contains("alpha"),
            "first turn: the initial child"
        );
        assert!(
            prompts[1].contains("beta") && prompts[1].contains("gamma"),
            "second turn carries both coalesced children: {}",
            prompts[1]
        );
        assert!(
            prompts[1].contains("consolidated answer"),
            "the coalesced batch reaching 0 remaining asks for the holistic review"
        );
        assert_eq!(
            waker.max_in_flight.load(Ordering::SeqCst),
            1,
            "wake turns never overlapped on the same conversation"
        );
    }

    /// The kill switch: a disabled coordinator drops completions and never wakes.
    #[tokio::test]
    async fn disabled_coordinator_never_wakes() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, false);

        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "alpha",
            "todo-a",
            "task-a",
            api::TaskStatus::Completed,
            0,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            waker.prompts.lock().expect("poisoned").is_empty(),
            "disabled coordinator must not wake"
        );
    }

    /// A subagent whose session conversation is its own conversation (no distinct
    /// top-level parent) is skipped — no autonomous turn on a hidden conversation.
    #[tokio::test]
    async fn no_distinct_parent_is_skipped() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        // session == child conversation.
        coord.on_completion(completion(
            "conv-self",
            "conv-self",
            "orphan",
            "todo-o",
            "task-o",
            api::TaskStatus::Completed,
            0,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            waker.prompts.lock().expect("poisoned").is_empty(),
            "a subagent with no distinct parent conversation must not wake"
        );
    }

    /// A child the parent blocked on is dropped: the parent already returned
    /// its result inline, so waking would run an extra unrequested turn on the
    /// user's own conversation.
    #[tokio::test]
    async fn waited_child_consumed_inline_is_skipped() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        coord.on_completion(waited("sess-T", "child-1", "researcher"));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            waker.prompts.lock().expect("poisoned").is_empty(),
            "a child the parent blocked on must not wake it again"
        );
    }

    /// A mixed burst: only the detached children reach the wake prompt, and the
    /// count and holistic ask are computed over those alone — a waited child
    /// never inflates "N subagents you dispatched just finished".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mixed_burst_wakes_only_for_detached_children() {
        let (release, released) = oneshot::channel::<()>();
        let waker = Arc::new(MockWaker {
            hold: Mutex::new(Some(released)),
            ..MockWaker::default()
        });
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        // A detached child starts a wake that blocks on the hold.
        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "alpha",
            "todo-a",
            "task-a",
            api::TaskStatus::Completed,
            1,
        ));
        wait_until(
            || waker.in_flight.load(Ordering::SeqCst) == 1,
            "first wake is in progress",
        )
        .await;

        // One waited and one detached child finish during that turn.
        coord.on_completion(waited("sess-T", "child-2", "beta"));
        coord.on_completion(completion(
            "sess-T",
            "child-3",
            "gamma",
            "todo-c",
            "task-c",
            api::TaskStatus::Completed,
            0,
        ));

        let _ = release.send(());
        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 2,
            "the follow-up wake ran",
        )
        .await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let prompts = waker.prompts.lock().expect("poisoned").clone();
        assert_eq!(
            prompts.len(),
            2,
            "the waited child added no turn: {prompts:?}"
        );
        assert!(
            prompts[1].contains("gamma"),
            "the follow-up carries the detached child: {}",
            prompts[1]
        );
        assert!(
            !prompts[1].contains("beta"),
            "a waited child never appears in a wake prompt: {}",
            prompts[1]
        );
        assert!(
            prompts[1].starts_with("[automatic] 1 subagent "),
            "the count covers only the children being reported: {}",
            prompts[1]
        );
    }

    /// Failed and cancelled children still wake the parent (so it never waits
    /// forever on a dead child), and the prompt states what happened.
    #[tokio::test]
    async fn failed_and_cancelled_children_wake_with_status() {
        let waker = Arc::new(MockWaker::default());
        let _dynamic: Arc<dyn ParentWaker> = waker.clone();
        let coord = coordinator(&waker, true);

        coord.on_completion(completion(
            "sess-T",
            "child-1",
            "boom",
            "todo-boom",
            "task-boom",
            api::TaskStatus::Failed,
            0,
        ));
        wait_until(
            || waker.prompts.lock().expect("poisoned").len() == 1,
            "failed child woke the parent",
        )
        .await;
        assert!(
            waker.prompts.lock().expect("poisoned")[0].contains("failed"),
            "the prompt states the child failed"
        );
    }
}
