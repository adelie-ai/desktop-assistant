//! Shared client-side transport, config, and command types for the assistant frontends.

pub mod auth;
pub mod commands;
pub mod config;
#[cfg(feature = "dbus")]
pub mod dbus_client;
pub mod signal;
pub mod transport;
pub mod types;
pub mod uds_client;
pub mod ws_client;

pub use commands::AssistantCommands;
pub use config::{ConnectionConfig, TransportMode, default_desktop_socket_path};
pub use signal::SignalEvent;
pub use transport::{AssistantClient, TransportClient, connect_transport};
pub use types::{ChatMessage, ConversationDetail, ConversationSummary};
pub use uds_client::UdsClient;
