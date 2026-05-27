//! D-Bus adapters that translate session-bus calls into [`api::Command`]
//! dispatches over a [`BridgeTransport`].
//!
//! Each adapter mirrors the corresponding in-process adapter in
//! `crates/dbus-interface/src/*.rs` **method-for-method**: same
//! interface name, same method names, same signatures, same return
//! shapes. The wire-format compatibility is enforced by the
//! [`introspection`](crate::adapter::introspection) golden tests in
//! `tests/`.
//!
//! The big difference from the in-process adapters: there is no
//! `AssistantService`/`ConversationService`/`SettingsService` trait
//! object behind these. Calls go out as `api::Command`s over UDS to
//! the daemon, which already owns the business logic. The translation
//! is mechanical — we just re-shape arguments and pattern-match
//! results.
//!
//! Event-driven D-Bus signals (`ResponseChunk` / `ResponseComplete` /
//! `ResponseError` / `ConfigChanged`) are emitted by
//! [`event_forwarder::run`] which subscribes to the transport's event
//! channel and dispatches to the matching object path. Adapters
//! themselves don't emit those — they only emit the synchronous
//! reply.

pub mod background_tasks;
pub mod connections;
pub mod conversations;
pub mod event_forwarder;
pub mod knowledge;
pub mod settings;

pub use background_tasks::DbusBackgroundTasksAdapter;
pub use connections::DbusConnectionsAdapter;
pub use conversations::DbusConversationsAdapter;
pub use knowledge::DbusKnowledgeAdapter;
pub use settings::DbusSettingsAdapter;

/// Well-known D-Bus bus name. Same as the in-process daemon's name —
/// we are the **replacement** for that registration, not a parallel
/// one. The daemon must NOT also own the name in Option-A deployments
/// where the bridge is enabled (operators pick one or the other via
/// the daemon's `DESKTOP_ASSISTANT_DBUS_REQUIRED` env knob).
pub const DBUS_SERVICE_NAME: &str = "org.desktopAssistant";

/// Object paths the bridge exposes. Order matters for introspection
/// goldens.
pub mod paths {
    pub const CONVERSATIONS: &str = "/org/desktopAssistant/Conversations";
    pub const SETTINGS: &str = "/org/desktopAssistant/Settings";
    pub const CONNECTIONS: &str = "/org/desktopAssistant/Connections";
    pub const KNOWLEDGE: &str = "/org/desktopAssistant/Knowledge";
    pub const BACKGROUND_TASKS: &str = "/org/desktopAssistant/BackgroundTasks";
}
