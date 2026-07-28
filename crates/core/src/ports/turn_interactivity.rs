//! Request-scoped turn interactivity (#942): whether a person is watching
//! this turn while it runs.
//!
//! Some turns have a human in front of them and some do not. A turn from a
//! chat client does; a subagent, a parent-wake re-engagement, a scheduled
//! routine does not. Narration, cadence and token spend all want that answer,
//! so the turn loop needs to be able to ask for it.
//!
//! ## Why this is stated, not inferred at each use
//!
//! The answer is *derivable* today: [`crate::ports::session::SessionId`] is
//! unscoped when no client connection installed one, so "unscoped" and
//! "nobody is watching" happen to coincide. Re-deriving that at every use
//! would put a load-bearing decision on a sentinel that also means "this slot
//! was never installed", and the review of the 28 request-scoped slots found
//! several features silently off because one was missed at a spawn site.
//! A feature disabling itself because a sentinel fell through is invisible.
//!
//! So the derivation happens once, and the answer is a property of the turn:
//!
//! 1. **A stated value wins.** A caller that starts a turn deliberately
//!    headless says so with [`with_turn_interactivity`], and nothing about the
//!    ambient session can override it.
//! 2. **With nothing stated, the session decides.** A real session id means a
//!    client connection is attached, so a person may be watching.
//! 3. **With neither, the answer is [`TurnInteractivity::Headless`].** That is
//!    the safe direction: a missing scope suppresses reassurance rather than
//!    emitting it to nobody.
//!
//! Rule 3 is why this slot cannot repeat the silent-default bug. The failure
//! mode of a dropped scope is a quiet turn, not a turn narrating to an empty
//! room.
//!
//! ## Contract
//!
//! [`with_turn_interactivity`] installs the slot for the duration of one
//! future; [`current_turn_interactivity`] reads it, resolving rules 1-3 above.
//! Like every `tokio::task_local!`, the slot does not cross a `tokio::spawn`.
//! Turn bodies that run on a spawned task therefore take it from
//! [`crate::ports::request_scope::RequestScope`], which captures the resolved
//! answer before the spawn and re-installs it inside - or state it themselves,
//! as the subagent spawn path does.

use crate::ports::session::current_session_id;

/// Whether a person is watching this turn while it runs.
///
/// Modelled as an enum rather than a boolean so a later case - attached over a
/// transport that mints no session, or interactive-but-the-human-walked-away -
/// becomes a third variant that every `match` has to account for, instead of
/// silently collapsing into one side of a `bool`. For the same reason this
/// type deliberately exposes no `is_interactive()` predicate: callers match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurnInteractivity {
    /// A person may be watching the turn as it runs, so cadence matters: a
    /// sign of life now beats a complete account later.
    Interactive,
    /// Nobody is watching the turn as it runs. Whatever it says is read
    /// afterwards, if at all, so completeness matters and reassurance does
    /// not.
    ///
    /// The [`Default`], because emitting nothing to a watching human is a
    /// smaller fault than emitting reassurance to nobody.
    #[default]
    Headless,
}

tokio::task_local! {
    /// The interactivity a caller stated for this turn. Unset means nothing
    /// stated it, and [`current_turn_interactivity`] falls back to the
    /// session-derived answer.
    static TURN_INTERACTIVITY: TurnInteractivity;
}

/// Run `fut` with `interactivity` stated for the turn.
///
/// Use this at any site that starts a turn whose audience it knows: the
/// subagent and standalone-agent spawn path, the parent-wake coordinator, and
/// (per #413) a future scheduled routine. A stated value beats the
/// session-derived default, so this is how a caller keeps a turn headless even
/// when a client connection is open.
pub async fn with_turn_interactivity<F, T>(interactivity: TurnInteractivity, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TURN_INTERACTIVITY.scope(interactivity, fut).await
}

/// The interactivity of the current turn.
///
/// Returns the stated value if a caller installed one, otherwise derives it
/// from the login session: a real session id means a client connection is
/// attached, the unscoped sentinel means none is. Never panics, never blocks.
pub fn current_turn_interactivity() -> TurnInteractivity {
    TURN_INTERACTIVITY
        .try_with(|i| *i)
        .unwrap_or_else(|_| derive_from_session())
}

/// The session-derived default, used when nothing stated an interactivity.
fn derive_from_session() -> TurnInteractivity {
    if current_session_id().is_unscoped() {
        TurnInteractivity::Headless
    } else {
        TurnInteractivity::Interactive
    }
}
