//! Request-scoped observer for a change in what the current turn may do.
//!
//! A turn does not always keep the capabilities it started with. The
//! provenance gate ([`crate::tool_provenance`]) closes the acting tool tiers
//! once the turn has taken in content an outside party can influence, and it
//! closes them for the rest of that turn.
//!
//! A person watching needs to know, or the assistant simply looks unreliable:
//! it declines something it did a minute ago and says nothing. A *program*
//! driving the daemon needs the same fact in a form it can act on, because
//! this is an API-first platform and a caller that can only read prose has to
//! guess. So the change goes out once, as data, and the transport layer
//! renders it for both audiences.
//!
//! ## Why a task-local
//!
//! Same reason as [`crate::ports::tool_observer`]: the fact is produced deep
//! inside the dispatch loop, and threading a sink through
//! [`crate::ports::inbound::ConversationService`] would change every
//! implementor and every test fixture of the `send_prompt` family. The loop's
//! other cross-cutting concerns already arrive this way.
//!
//! ## Contract
//!
//! [`with_turn_capability_observer`] installs the slot for one future;
//! [`notify_turn_capability_change`] delivers to it and reports whether
//! anybody took the event. The report is load-bearing: a caller that installs
//! no observer (a test, a background worker, an embedder of the core service)
//! still has to tell its user something, so the turn loop falls back to a
//! plain status line. Absent capability, degraded path, never a silent drop.
//!
//! Like every `tokio::task_local!`, the slot does not cross a `tokio::spawn`.

use std::sync::Arc;

use crate::tool_provenance::ToolTier;

/// Why the turn's capabilities changed.
///
/// An enum rather than a string so a caller matches instead of parsing, and
/// so a second cause has to be handled everywhere it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnCapabilityReason {
    /// A tool result brought in content an outside party can influence, so
    /// the acting tiers closed for the rest of the turn.
    ExternalContentIngested,
}

/// One change in what the current turn may still do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnCapabilityChange {
    /// What caused the change.
    pub reason: TurnCapabilityReason,
    /// The tool tiers that are now refused for the rest of this turn.
    pub closed_tiers: Vec<ToolTier>,
    /// One line for the person watching. Composed at the dispatch site, like
    /// the summaries [`crate::ports::tool_observer::ToolEvent`] carries, so
    /// every transport shows the same words.
    pub message: String,
}

/// Sink for [`TurnCapabilityChange`]. Cheap to clone; called synchronously
/// from the dispatch loop, so an implementation must not block.
pub type TurnCapabilityObserver = Arc<dyn Fn(TurnCapabilityChange) + Send + Sync>;

tokio::task_local! {
    /// The observer for the current turn, installed by the send-turn body.
    static TURN_CAPABILITY_OBSERVER: TurnCapabilityObserver;
}

/// Run `fut` with `observer` installed for the turn.
pub async fn with_turn_capability_observer<F, T>(observer: TurnCapabilityObserver, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TURN_CAPABILITY_OBSERVER.scope(observer, fut).await
}

/// Deliver `change` to the installed observer.
///
/// Returns `false` when no observer is installed, so the caller can fall back
/// to whatever channel it does have. Never panics, never blocks.
#[must_use]
pub fn notify_turn_capability_change(change: TurnCapabilityChange) -> bool {
    TURN_CAPABILITY_OBSERVER.try_with(|obs| obs(change)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn change() -> TurnCapabilityChange {
        TurnCapabilityChange {
            reason: TurnCapabilityReason::ExternalContentIngested,
            closed_tiers: vec![ToolTier::Egress],
            message: "closed".to_string(),
        }
    }

    #[tokio::test]
    async fn an_installed_observer_takes_the_change() {
        let seen: Arc<Mutex<Vec<TurnCapabilityChange>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let observer: TurnCapabilityObserver = Arc::new(move |c| sink.lock().unwrap().push(c));

        let delivered = with_turn_capability_observer(observer, async {
            notify_turn_capability_change(change())
        })
        .await;

        assert!(delivered, "an installed observer must take the change");
        assert_eq!(seen.lock().unwrap().as_slice(), [change()]);
    }

    #[tokio::test]
    async fn no_observer_reports_the_change_was_not_taken() {
        // The turn loop reads this and falls back to a status line, so a
        // wrong answer here would lose the signal for a whole class of
        // callers.
        assert!(
            !notify_turn_capability_change(change()),
            "with no observer installed the change must report undelivered"
        );
    }
}
