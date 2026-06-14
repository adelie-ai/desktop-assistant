//! Desktop-notification port.
//!
//! A capability-gated outbound closure for posting a desktop notification
//! (freedesktop `org.freedesktop.Notifications` on Linux). The closure is only
//! wired when a notification service is actually present on the session bus, so
//! the `builtin_notify` tool simply isn't offered on a headless host — "is the
//! capability present?" is decided at wiring time, distinct from "did a given
//! call succeed?".

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;

/// Notification urgency, mapping to the freedesktop `urgency` hint
/// (0 = low, 1 = normal, 2 = critical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NotifyUrgency {
    Low,
    #[default]
    Normal,
    Critical,
}

impl NotifyUrgency {
    /// Parse a tool-supplied urgency string; unknown/missing values map to
    /// `Normal`.
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("low") => Self::Low,
            Some("critical") => Self::Critical,
            _ => Self::Normal,
        }
    }

    /// The freedesktop `urgency` hint byte.
    pub fn hint(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::Critical => 2,
        }
    }
}

/// Boxed async closure that posts a desktop notification: `(summary, body,
/// urgency) → notification id`. Returns `Ok(None)` when the post was
/// suppressed by rate-limiting (e.g. an identical notification fired moments
/// ago), so callers can report "shown" vs "suppressed" without it being an
/// error.
pub type NotifyFn = Arc<
    dyn Fn(
            String,
            String,
            NotifyUrgency,
        ) -> Pin<Box<dyn Future<Output = Result<Option<u32>, CoreError>> + Send>>
        + Send
        + Sync,
>;
