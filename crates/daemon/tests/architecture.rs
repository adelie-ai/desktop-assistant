//! Integration tests validating that the workspace crates compose correctly.

use desktop_assistant_core::ports::inbound::AssistantService;

/// A test-only implementation to verify the trait is usable from outside the crate.
struct TestAssistant;

impl AssistantService for TestAssistant {
    fn version(&self) -> &str {
        "integration-test"
    }

    fn ping(&self) -> &str {
        "pong"
    }
}

#[test]
fn core_inbound_port_is_implementable_externally() {
    let assistant = TestAssistant;
    assert_eq!(assistant.version(), "integration-test");
    assert_eq!(assistant.ping(), "pong");
}

#[test]
fn dbus_adapter_accepts_external_service_impl() {
    let adapter = desktop_assistant_dbus::DbusAssistantAdapter::new(TestAssistant);
    assert_eq!(adapter.service().ping(), "pong");
}

#[test]
fn core_error_displays_correctly() {
    let err = desktop_assistant_core::CoreError::SystemService("connection refused".into());
    assert_eq!(err.to_string(), "system service error: connection refused");
}
