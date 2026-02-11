pub mod conversation;

use desktop_assistant_core::ports::inbound::AssistantService;

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
