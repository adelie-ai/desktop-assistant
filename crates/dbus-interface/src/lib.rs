pub mod connections;
pub mod conversation;
pub mod knowledge;
pub mod settings;

use desktop_assistant_core::ports::auth::UserId;
use desktop_assistant_core::ports::inbound::AssistantService;

/// Resolve the user id for an inbound D-Bus call (#156).
///
/// The D-Bus session bus is local-only — there is no JWT to extract a
/// `sub` from like the WebSocket transport does, so the daemon takes
/// the OS-level user from the `USER` environment variable inherited
/// from the user's session. When `USER` is unset (containerised
/// deploys, scripted launches), the helper falls through to
/// [`UserId::default`] — the schema sentinel `"default"` — which
/// matches the single-tenant fallback every other code path uses.
///
/// ## Trust boundary
///
/// `$USER` is inherited from the local desktop session and is not
/// adversary-controlled in a single-tenant deploy. Multi-tenant /
/// shared-bus deployments need peer-credential lookup
/// (SO_PEERCRED / `sd_bus_get_credentials`) instead; that's an
/// explicit follow-up (see issue body) and out of scope for #156.
pub fn resolve_dbus_user_id() -> UserId {
    match std::env::var("USER") {
        Ok(value) if !value.is_empty() => UserId::new(value),
        _ => UserId::default(),
    }
}

/// D-Bus adapter that exposes an `AssistantService` over the session bus.
pub struct DbusAssistantAdapter<S: AssistantService> {
    service: S,
}

impl<S: AssistantService> DbusAssistantAdapter<S> {
    pub fn new(service: S) -> Self {
        Self { service }
    }

    pub fn service(&self) -> &S {
        &self.service
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::inbound::AssistantService;

    struct StubAssistant;

    impl AssistantService for StubAssistant {
        fn version(&self) -> &str {
            "0.1.0-test"
        }

        fn ping(&self) -> &str {
            "pong"
        }
    }

    #[test]
    fn adapter_wraps_service() {
        let adapter = DbusAssistantAdapter::new(StubAssistant);
        assert_eq!(adapter.service().version(), "0.1.0-test");
    }

    #[test]
    fn adapter_delegates_ping() {
        let adapter = DbusAssistantAdapter::new(StubAssistant);
        assert_eq!(adapter.service().ping(), "pong");
    }
}
