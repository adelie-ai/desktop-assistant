//! Request-scoped tool-activity observer.
//!
//! The service dispatch loop calls tools deep inside a turn; an observer — the
//! background-task panel, a tracer, anything that wants a live feed of what a
//! turn is *doing* — needs to see each tool/MCP call and its outcome. Rather
//! than thread a sink through [`crate::ports::inbound::ConversationService`]
//! (and every implementor and test mock of the central `send_prompt` family),
//! we expose a task-local observer the caller installs around the dispatch via
//! [`with_tool_observer`] and the loop notifies via [`notify_tool_event`].
//!
//! ## Why a task-local
//!
//! This mirrors the way the dispatch loop already receives its other
//! cross-cutting tool concerns: the client-tool port ([`current_client_tools`]),
//! the conversation id ([`with_conversation_id`]), and the tool allowlist
//! ([`with_tool_allowlist`]) all reach the loop as task-locals, not as
//! parameters on `send_prompt`. A tool observer is the same family — per-turn
//! context consumed deep in the loop — so it belongs alongside them rather than
//! bloating the trait surface that every implementor and fixture must satisfy.
//! See AGENTS.md ("cross-cutting context propagates via `tokio::task_local!`").
//!
//! When the slot is unset (tests, background workers, any caller that doesn't
//! care), [`notify_tool_event`] is a no-op — the loop reports activity
//! unconditionally and the absence of an observer simply drops it.
//!
//! [`current_client_tools`]: crate::ports::client_tools::current_client_tools
//! [`with_conversation_id`]: crate::ports::conversation_ctx::with_conversation_id
//! [`with_tool_allowlist`]: crate::ports::llm::with_tool_allowlist

use std::sync::Arc;

/// One observed step in a turn's tool activity. The string fields are short,
/// already-truncated summaries produced at the dispatch site — an observer is
/// a UI/telemetry sink, not a place to re-parse arguments or results.
#[derive(Clone, Debug)]
pub enum ToolEvent {
    /// A tool is about to execute. `args` is a compact rendering of the call
    /// arguments (may be empty when there are none).
    Started { name: String, args: String },
    /// A tool finished. `ok` is false when the executor returned an error;
    /// `output` is a compact rendering of the result (or error) text.
    Finished {
        name: String,
        ok: bool,
        output: String,
    },
}

/// Sink for [`ToolEvent`]s. Cheap to clone; invoked synchronously from the
/// dispatch loop, so an implementation must not block (append-to-ring +
/// broadcast is fine; awaiting I/O is not).
pub type ToolObserver = Arc<dyn Fn(ToolEvent) + Send + Sync>;

tokio::task_local! {
    /// The observer for the current turn. Installed by the send-turn body via
    /// [`with_tool_observer`] around the dispatch; read by the service loop via
    /// [`notify_tool_event`] at each tool call.
    static TOOL_OBSERVER: ToolObserver;
}

/// Run `fut` with `observer` installed as the current tool observer. All
/// [`notify_tool_event`] calls inside the future are delivered to `observer`.
pub async fn with_tool_observer<F, T>(observer: ToolObserver, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TOOL_OBSERVER.scope(observer, fut).await
}

/// Deliver `event` to the installed observer, if any. No-op when no scope is
/// installed. Safe to call from any async context — never panics, never blocks.
pub fn notify_tool_event(event: ToolEvent) {
    let _ = TOOL_OBSERVER.try_with(|obs| obs(event));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn notify_outside_scope_is_a_silent_noop() {
        // Must not panic when no observer is installed.
        notify_tool_event(ToolEvent::Started {
            name: "x".into(),
            args: String::new(),
        });
    }

    #[tokio::test]
    async fn installed_observer_receives_events_in_order() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            Arc::new(move |e: ToolEvent| match e {
                ToolEvent::Started { name, .. } => seen.lock().unwrap().push(format!("start:{name}")),
                ToolEvent::Finished { name, ok, .. } => {
                    seen.lock().unwrap().push(format!("done:{name}:{ok}"))
                }
            }) as ToolObserver
        };
        with_tool_observer(sink, async {
            notify_tool_event(ToolEvent::Started {
                name: "search".into(),
                args: "{}".into(),
            });
            notify_tool_event(ToolEvent::Finished {
                name: "search".into(),
                ok: true,
                output: "hits".into(),
            });
        })
        .await;
        assert_eq!(*seen.lock().unwrap(), vec!["start:search", "done:search:true"]);
    }

    #[tokio::test]
    async fn slot_does_not_cross_spawn() {
        // A spawned task does not inherit the parent's task-local observer;
        // a `notify_tool_event` inside the spawn must drop silently rather than
        // reach the parent's sink. Prove it with a shared counter the sink
        // bumps on every delivery — the spawned notify must leave it at zero,
        // and a sibling notify in the parent scope must bump it to one (so the
        // sink is genuinely wired, not merely never called).
        let deliveries = Arc::new(AtomicUsize::new(0));
        let sink = {
            let deliveries = Arc::clone(&deliveries);
            Arc::new(move |_e: ToolEvent| {
                deliveries.fetch_add(1, Ordering::SeqCst);
            }) as ToolObserver
        };
        with_tool_observer(sink, async {
            tokio::spawn(async {
                notify_tool_event(ToolEvent::Started {
                    name: "x".into(),
                    args: String::new(),
                });
            })
            .await
            .unwrap();
            // The spawned task could not reach the parent sink.
            assert_eq!(
                deliveries.load(Ordering::SeqCst),
                0,
                "observer must not cross tokio::spawn"
            );

            // Sanity: a notify in the parent scope IS delivered, so the zero
            // above reflects scope isolation, not a dead sink.
            notify_tool_event(ToolEvent::Finished {
                name: "x".into(),
                ok: true,
                output: String::new(),
            });
            assert_eq!(deliveries.load(Ordering::SeqCst), 1);
        })
        .await;
    }
}
