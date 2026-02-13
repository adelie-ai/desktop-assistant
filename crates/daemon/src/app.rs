use desktop_assistant_core::ports::inbound::AssistantService;

/// Concrete implementation of the assistant service for the daemon.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Assistant;

impl AssistantService for Assistant {
    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn ping(&self) -> &str {
        "pong"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_version_matches_cargo_pkg() {
        let assistant = Assistant;
        assert_eq!(assistant.version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn assistant_ping_returns_pong() {
        let assistant = Assistant;
        assert_eq!(assistant.ping(), "pong");
    }
}
