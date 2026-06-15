//! Desktop-notification wiring for the `builtin_notify` tool.
//!
//! Capability-gated: [`build_notify_fn`] probes the session bus for a
//! freedesktop notification server and returns a [`NotifyFn`] only when one is
//! present. On a headless host (no session bus, or no notification daemon) it
//! returns `None`, so the daemon never wires the tool and the model is simply
//! not offered it — "is the capability present?" decided once, here, distinct
//! from "did a given call succeed?".

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::notify::{NotifyFn, NotifyUrgency};

/// App name shown by the notification server.
const APP_NAME: &str = "Adele";
/// Identical (summary + body + urgency) notifications fired within this window
/// are suppressed, so a misbehaving caller can't spam the user.
const DEDUP_WINDOW: Duration = Duration::from_secs(10);

#[zbus::proxy(
    interface = "org.freedesktop.Notifications",
    default_service = "org.freedesktop.Notifications",
    default_path = "/org/freedesktop/Notifications"
)]
trait Notifications {
    #[allow(clippy::too_many_arguments)]
    async fn notify(
        &self,
        app_name: &str,
        replaces_id: u32,
        app_icon: &str,
        summary: &str,
        body: &str,
        actions: &[&str],
        hints: HashMap<&str, zbus::zvariant::Value<'_>>,
        expire_timeout: i32,
    ) -> zbus::Result<u32>;

    /// `(name, vendor, version, spec_version)` — used purely as a liveness
    /// probe for the capability check.
    async fn get_server_information(&self) -> zbus::Result<(String, String, String, String)>;
}

/// Build a [`NotifyFn`] when a notification server is reachable, else `None`.
pub async fn build_notify_fn() -> Option<NotifyFn> {
    let connection = match zbus::Connection::session().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::info!("builtin_notify disabled: no session D-Bus bus ({e})");
            return None;
        }
    };

    let proxy = match NotificationsProxy::new(&connection).await {
        Ok(proxy) => proxy,
        Err(e) => {
            tracing::info!("builtin_notify disabled: notifications proxy unavailable ({e})");
            return None;
        }
    };

    // Capability probe: if no service owns the name this errors out, and we
    // degrade by leaving the tool unwired.
    match proxy.get_server_information().await {
        Ok((name, vendor, version, _spec)) => tracing::info!(
            "builtin_notify enabled: notification server {name} {version} ({vendor})"
        ),
        Err(e) => {
            tracing::info!("builtin_notify disabled: no notification server ({e})");
            return None;
        }
    }

    let proxy = Arc::new(proxy);
    // Last (content-hash, instant) for the dedup window.
    let last: Arc<Mutex<Option<(u64, Instant)>>> = Arc::new(Mutex::new(None));

    let notify_fn: NotifyFn = Arc::new(
        move |summary: String, body: String, urgency: NotifyUrgency| {
            let proxy = Arc::clone(&proxy);
            let last = Arc::clone(&last);
            Box::pin(async move {
                let key = {
                    let mut hasher = DefaultHasher::new();
                    summary.hash(&mut hasher);
                    body.hash(&mut hasher);
                    urgency.hint().hash(&mut hasher);
                    hasher.finish()
                };

                // Dedup. The guard is dropped before the await below — never held
                // across a suspension point.
                {
                    let mut guard = last.lock().unwrap();
                    if let Some((prev_key, when)) = *guard
                        && prev_key == key
                        && when.elapsed() < DEDUP_WINDOW
                    {
                        return Ok(None);
                    }
                    *guard = Some((key, Instant::now()));
                }

                let mut hints: HashMap<&str, zbus::zvariant::Value<'_>> = HashMap::new();
                hints.insert("urgency", zbus::zvariant::Value::U8(urgency.hint()));

                let id = proxy
                    .notify(
                        APP_NAME,
                        0,
                        "dialog-information",
                        &summary,
                        &body,
                        &[],
                        hints,
                        -1, // let the server choose the timeout
                    )
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("notification failed: {e}")))?;
                Ok(Some(id))
            })
        },
    );

    Some(notify_fn)
}
