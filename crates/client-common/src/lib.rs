pub mod auth;
pub mod config;
#[cfg(feature = "dbus")]
pub mod dbus_client;
pub mod signal;
pub mod transport;
pub mod types;
pub mod ws_client;

pub use config::{ConnectionConfig, TransportMode};
pub use signal::SignalEvent;
pub use transport::{AssistantClient, TransportClient, connect_transport};
pub use types::{ChatMessage, ConversationDetail, ConversationSummary};
