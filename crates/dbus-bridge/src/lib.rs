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
//! The split between `lib.rs` and `main.rs` is deliberate: the binary
//! wires together a [`minter::Minter`] and a [`transport::BridgeTransport`]
//! in a way that needs root, signals, and a real D-Bus connection;
//! everything testable is exposed as library API.
//!
//! ## Modules
//!
//! - [`minter`]: line-delimited JSON client for the local `adelie-mint`
//!   socket. Returns a freshly-minted JWT or an explicit error.
//! - [`transport`]: a [`BridgeTransport`](transport::BridgeTransport)
//!   trait + a concrete UDS impl that frames length-prefixed JSON to
//!   the daemon, performs the JWT handshake, and demuxes
//!   `WsFrame::Result`/`Error` (by request id) from `WsFrame::Event`
//!   (broadcast to subscribers).
//! - [`adapter`]: D-Bus adapter structs (one per object path) that
//!   speak only `api-model` types — no `core`/`application` deps.

pub mod adapter;
pub mod minter;
pub mod transport;

pub use transport::{BridgeTransport, BridgeTransportError, UdsBridgeTransport};
