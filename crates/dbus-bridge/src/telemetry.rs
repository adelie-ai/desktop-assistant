//! How this binary configures telemetry.
//!
//! The setup itself belongs to `adelie_telemetry`, which every Adelie binary
//! shares so an operator meets the same knobs everywhere. This module holds
//! only the answers that are the bridge's own: what it calls itself, how loud
//! it is when nobody asked, and whether a closing span writes a line.
//!
//! The bridge is a separate process from the daemon, so it installs its own
//! subscriber. No library does: a library hosted inside another binary
//! inherits that binary's subscriber.
//!
//! Where telemetry goes is not decided here. It comes from the standard
//! `OTEL_*` environment variables at run time, so an operator moves it without
//! a rebuild. See the Logging section of `docs/logging.md`.

use adelie_telemetry::Config;

/// The name this binary reports as `service.name`.
///
/// It separates the bridge's telemetry from the daemon's, so it has to be the
/// binary's own name and stay stable: a backend query written against it
/// outlives any one release.
pub const SERVICE_NAME: &str = "adele-dbus-bridge";

/// The filter that applies when `RUST_LOG` says nothing.
///
/// `info`, which is what the bridge has always done.
pub const DEFAULT_FILTER: &str = "info";

/// The bridge's telemetry configuration.
pub fn config() -> Config {
    Config::new(SERVICE_NAME)
        .with_default_filter(DEFAULT_FILTER)
        // A closing span writes how long it was open, which is what makes a
        // slow call readable in `journalctl` with no trace backend to open.
        .with_span_close_events(true)
}
