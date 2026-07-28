//! Acceptance tests for explicit turn interactivity (#942).
//!
//! A turn either has a person watching it run or it does not. The daemon must
//! be able to say which, and a caller that starts a turn on purpose with no
//! one watching must be able to *state* that rather than leave it to a
//! sentinel falling through.
//!
//! Three rules, in precedence order, and one test per rule:
//!
//! 1. An explicit statement wins.
//! 2. With no statement, the login session decides: a real session means a
//!    client connection is attached, so a person may be watching.
//! 3. With neither, the answer is headless - the safe direction, because a
//!    missing scope must suppress reassurance rather than emit it to no one.

use desktop_assistant_core::ports::request_scope::RequestScope;
use desktop_assistant_core::ports::session::{SessionId, with_session_id};
use desktop_assistant_core::ports::turn_interactivity::{
    TurnInteractivity, current_turn_interactivity, with_turn_interactivity,
};

#[tokio::test]
async fn a_turn_from_a_client_session_is_interactive() {
    let observed = with_session_id(SessionId::new("conn-7"), async {
        current_turn_interactivity()
    })
    .await;

    assert_eq!(
        observed,
        TurnInteractivity::Interactive,
        "a turn dispatched on a real client connection has a person watching"
    );
}

#[tokio::test]
async fn a_turn_with_no_session_is_headless() {
    assert_eq!(
        current_turn_interactivity(),
        TurnInteractivity::Headless,
        "no session scope means no one is watching"
    );

    // The unscoped sentinel is the same answer as no scope at all: it is what
    // `current_session_id` returns when nothing installed the slot.
    let observed = with_session_id(SessionId::unscoped(), async {
        current_turn_interactivity()
    })
    .await;
    assert_eq!(observed, TurnInteractivity::Headless);
}

#[tokio::test]
async fn an_explicitly_headless_turn_stays_headless_under_an_interactive_session() {
    let observed = with_session_id(SessionId::new("conn-7"), async {
        with_turn_interactivity(TurnInteractivity::Headless, async {
            current_turn_interactivity()
        })
        .await
    })
    .await;

    assert_eq!(
        observed,
        TurnInteractivity::Headless,
        "an explicit statement beats the session-derived default"
    );
}

#[tokio::test]
async fn an_explicitly_interactive_turn_stays_interactive_with_no_session() {
    // The statement wins in both directions, so a future "attached over a
    // transport that mints no session" case is expressible without touching
    // the derivation.
    let observed = with_turn_interactivity(TurnInteractivity::Interactive, async {
        current_turn_interactivity()
    })
    .await;

    assert_eq!(observed, TurnInteractivity::Interactive);
}

#[tokio::test]
async fn nested_turn_interactivity_scopes_override_then_restore() {
    let (inner, after) = with_turn_interactivity(TurnInteractivity::Interactive, async {
        let inner = with_turn_interactivity(TurnInteractivity::Headless, async {
            current_turn_interactivity()
        })
        .await;
        (inner, current_turn_interactivity())
    })
    .await;

    assert_eq!(inner, TurnInteractivity::Headless);
    assert_eq!(after, TurnInteractivity::Interactive);
}

#[tokio::test]
async fn a_turn_that_loses_its_scope_across_a_spawn_is_headless() {
    // `task_local`s do not cross `tokio::spawn`. A spawn site that forgets to
    // re-install must land on the safe answer, never on reassurance emitted to
    // no one - the #261 bug class, pointed in the harmless direction.
    let observed = with_session_id(SessionId::new("conn-7"), async {
        with_turn_interactivity(TurnInteractivity::Interactive, async {
            tokio::spawn(async { current_turn_interactivity() })
                .await
                .expect("spawned probe must not panic")
        })
        .await
    })
    .await;

    assert_eq!(observed, TurnInteractivity::Headless);
}

#[tokio::test]
async fn a_request_scope_carries_the_derived_answer_across_a_spawn() {
    let captured =
        with_session_id(SessionId::new("conn-7"), async { RequestScope::capture() }).await;

    assert_eq!(
        captured.interactivity,
        TurnInteractivity::Interactive,
        "capture resolves the answer once, from the session installed at capture time"
    );

    let observed =
        tokio::spawn(async move { captured.scope(async { current_turn_interactivity() }).await })
            .await
            .expect("spawned turn body must not panic");

    assert_eq!(observed, TurnInteractivity::Interactive);
}

#[tokio::test]
async fn a_request_scope_pins_an_explicitly_headless_turn_across_a_spawn() {
    // The statement is made before the spawn; the bundle must carry it, not
    // re-derive from the session it also carries.
    let captured = with_session_id(SessionId::new("conn-7"), async {
        with_turn_interactivity(TurnInteractivity::Headless, async {
            RequestScope::capture()
        })
        .await
    })
    .await;

    assert_eq!(captured.session_id, SessionId::new("conn-7"));
    assert_eq!(captured.interactivity, TurnInteractivity::Headless);

    let observed =
        tokio::spawn(async move { captured.scope(async { current_turn_interactivity() }).await })
            .await
            .expect("spawned turn body must not panic");

    assert_eq!(
        observed,
        TurnInteractivity::Headless,
        "a stated headless turn stays headless inside the spawned turn body"
    );
}
