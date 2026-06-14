//! The client-facing signal stream.
//!
//! `SignalEvent` (and the `api::Event` → `SignalEvent` projection) moved to
//! `api-model` (#377) so the shared, wasm-targeting client cores can consume the
//! signal stream without pulling this crate's native transport tail. It is
//! re-exported here so existing `client_common::SignalEvent` /
//! `client_common::signal::SignalEvent` paths are unchanged.

pub use desktop_assistant_api_model::signal::SignalEvent;
