//! adelie-dbus-bridge: standalone per-user D-Bus bridge.
//!
//! The bridge exposes the same session-bus surface the in-process
//! `dbus-interface` adapters expose today (well-known name
//! `org.desktopAssistant`, the four object paths under
//! `/org/desktopAssistant/...`) but does NOT link the daemon's
//! `application` crate. Instead, every D-Bus method call is translated
//! into a `WsRequest` and shipped over a JWT-authenticated UDS
//! connection to the daemon (issue #103). `WsFrame::Event`s coming
//! back over the same UDS connection are translated into the
//! corresponding D-Bus signals.
//!
//! See `docs/architecture-evolution.md` Phase 1 for context. This crate
//! is the binary half of the "D-Bus bridge" story; the daemon itself
//! still ships the in-process surface for Option A
//! (independently-shippable transition — see PR #106).
//!
//! The split between `lib.rs` and `main.rs` is deliberate: the binary wires a
//! client-common [`Connector`](desktop_assistant_client_common::Connector) to a
//! real D-Bus connection in a way that needs signals and the session bus;
//! everything testable is exposed as library API.
//!
//! ## Modules
//!
//! - [`transport`]: the [`BridgeTransport`](transport::BridgeTransport) trait the
//!   adapters dispatch through, and [`ConnectorBridgeTransport`](transport::ConnectorBridgeTransport),
//!   a thin forwarder over the shared client-common `Connector` (which owns the
//!   authenticated UDS connection, reconnect, and JWT minting — #316).
//! - [`adapter`]: D-Bus adapter structs (one per object path) that
//!   speak only `api-model` types — no `core`/`application` deps.

pub mod adapter;
pub mod transport;

pub use transport::{BridgeTransport, BridgeTransportError, ConnectorBridgeTransport};
