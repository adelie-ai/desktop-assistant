//! Shared client-side transport, config, and command types for the assistant frontends.

pub mod auth;
#[cfg(feature = "clap")]
pub mod cli;
pub mod commands;
pub mod config;
pub mod connector;
#[cfg(feature = "dbus")]
pub mod dbus_client;
#[cfg(feature = "mcp-host")]
pub mod mcp_host;
pub mod signal;
pub mod system_id;
pub mod timeouts;
pub mod transport;
pub mod types;
pub mod uds_client;
pub mod ws_client;

#[cfg(feature = "clap")]
pub use cli::TransportArgs;
pub use commands::AssistantCommands;
pub use config::{ConnectionConfig, TransportMode, default_desktop_socket_path};
pub use connector::Connector;
pub use signal::SignalEvent;
pub use transport::{AssistantClient, DropNotifier, TransportClient, connect_transport};
pub use types::{ChatMessage, ConversationDetail, ConversationSummary, MessageKind};
pub use uds_client::UdsClient;
