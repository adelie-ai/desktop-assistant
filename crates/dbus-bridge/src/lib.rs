//! adelie-dbus-bridge: standalone per-user D-Bus bridge.
//!
//! The bridge is the session-bus surface (well-known name
//! `org.desktopAssistant`, the object paths under
//! `/org/desktopAssistant/...`), and it does NOT link the daemon's
//! `application` crate. Instead, every D-Bus method call is translated
//! into a `WsRequest` and shipped over a local UDS connection to the
//! daemon, authenticated by kernel peer-cred (#407). `WsFrame::Event`s
//! coming back over the same UDS connection are translated into the
//! corresponding D-Bus signals.
//!
//! See `docs/architecture-evolution.md` Phase 1 for context. The cutover
//! (#318/#319) deleted the daemon's own in-process `dbus-interface`
//! adapters, so this crate is the only thing that serves the surface.
//!
//! The split between `lib.rs` and `main.rs` is deliberate: the binary wires a
//! client-common [`Connector`](desktop_assistant_client_common::Connector) to a
//! real D-Bus connection in a way that needs signals and the session bus;
//! everything testable is exposed as library API.
//!
//! ## Modules
//!
//! - [`transport`]: the [`BridgeTransport`] trait the
//!   adapters dispatch through, and [`ConnectorBridgeTransport`],
//!   a thin forwarder over the shared client-common `Connector` (which owns the
//!   authenticated UDS connection, reconnect, and JWT minting — #316).
//! - [`adapter`]: D-Bus adapter structs (one per object path) that
//!   speak only `api-model` types — no `core`/`application` deps.
//! - [`telemetry`]: what this binary tells the shared telemetry crate about
//!   itself. The binary installs the subscriber; this crate's library half
//!   never does.
//! - [`session`]: per-D-Bus-sender daemon sessions — each sender gets its own
//!   authenticated `Connector` + a unicast event forwarder, so turn responses
//!   (and, post-#367/#320, live sync and client tools) reach only that caller
//!   instead of broadcasting across the bus.

pub mod adapter;
pub mod session;
pub mod telemetry;
pub mod transport;

pub use session::{ConnectorSessionFactory, SessionRegistry, spawn_name_owner_watcher};
pub use transport::{BridgeTransport, BridgeTransportError, ConnectorBridgeTransport};
