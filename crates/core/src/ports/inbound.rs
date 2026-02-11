/// Inbound port for health/status queries.
///
/// Any adapter that wants to expose assistant status (D-Bus, HTTP, etc.)
/// implements a handler that calls through this trait.
pub trait AssistantService: Send + Sync {
    /// Returns a version string for the running assistant.
    fn version(&self) -> &str;

    /// Simple liveness check.
    fn ping(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAssistant;

    impl AssistantService for MockAssistant {
        fn version(&self) -> &str {
            env!("CARGO_PKG_VERSION")
        }

        fn ping(&self) -> &str {
            "pong"
        }
    }

    #[test]
    fn mock_assistant_returns_version() {
        let assistant = MockAssistant;
        assert!(!assistant.version().is_empty(), "version must not be empty");
    }

    #[test]
    fn mock_assistant_responds_to_ping() {
        let assistant = MockAssistant;
        assert_eq!(assistant.ping(), "pong");
    }
}
