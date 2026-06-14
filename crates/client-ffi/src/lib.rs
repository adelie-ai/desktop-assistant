//! `desktop-assistant-client-ffi` — a C ABI over `client-common`'s UDS
//! [`Connector`], so native (non-Rust) clients drive the assistant through the
//! same battle-tested path gtk/tui use instead of re-implementing it.
//!
//! First consumer: the adele-kde plasmoid, replacing its ~1900-line Python
//! helper (live multi-client sync #367, client tools #320). The data path is
//! **UDS-direct** — no D-Bus / `zbus` (`client-common` is depended on with
//! `default-features = false`) — so live sync + client tools come for free from
//! the shared `Connector`.
//!
//! ## Shape
//! - [`adele_client_connect`] returns an opaque handle owning a multi-thread
//!   tokio runtime (its workers keep the signal pump running) plus a connected
//!   `Connector`.
//! - [`adele_client_start_signals`] spawns a pump that turns each
//!   [`SignalEvent`] into a JSON string and hands it to a C callback. The Qt
//!   wrapper marshals that onto the GUI thread (a queued connection).
//! - Request/response calls ([`adele_client_send_prompt`],
//!   [`adele_client_subscribe_conversations`]) `block_on` the runtime.
//!
//! ## Safety contract
//! Pointer args must be NULL or valid NUL-terminated UTF-8 C strings. Strings
//! this library returns are owned by the caller and freed with
//! [`adele_string_free`]; the handle with [`adele_client_free`]. The signal
//! callback fires on a runtime worker thread, so it must be thread-safe (the Qt
//! wrapper posts onto the GUI thread).

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_client_common::minter::default_minter_socket_path;
use desktop_assistant_client_common::{
    ConnectionConfig, Connector, SignalEvent, TransportMode, default_desktop_socket_path,
};
use tokio::runtime::Runtime;

/// Opaque client handle (see module docs). Created by [`adele_client_connect`],
/// destroyed by [`adele_client_free`].
pub struct AdeleClient {
    rt: Runtime,
    connector: Arc<Connector>,
}

/// Borrow a C string as an owned `String`, or `None` if NULL / not UTF-8.
///
/// # Safety
/// `ptr` must be NULL or a valid NUL-terminated C string.
unsafe fn cstr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(str::to_owned)
}

/// Connect to the daemon over UDS. `socket_path` / `minter_socket` may be NULL
/// to use the platform defaults (`$XDG_RUNTIME_DIR/adelie/{sock,mint.sock}`).
/// Returns NULL on failure (bad runtime, daemon unreachable, mint failure).
///
/// # Safety
/// `socket_path` and `minter_socket` must each be NULL or a valid NUL-terminated
/// C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_client_connect(
    socket_path: *const c_char,
    minter_socket: *const c_char,
) -> *mut AdeleClient {
    let socket = unsafe { cstr(socket_path) }
        .map(PathBuf::from)
        .or_else(default_desktop_socket_path);
    let minter = unsafe { cstr(minter_socket) }
        .map(PathBuf::from)
        .or_else(default_minter_socket_path);

    let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    else {
        return ptr::null_mut();
    };

    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: socket,
        minter_socket: minter,
        ..ConnectionConfig::default()
    };

    match rt.block_on(Connector::connect(&config)) {
        Ok(connector) => Box::into_raw(Box::new(AdeleClient {
            rt,
            connector: Arc::new(connector),
        })),
        Err(_) => ptr::null_mut(),
    }
}

/// Start streaming `SignalEvent`s to `cb` as JSON strings. Call once per client.
/// The pump survives daemon restarts (it re-subscribes); after a `disconnected`
/// event the embedder should re-issue [`adele_client_subscribe_conversations`].
///
/// # Safety
/// `client` must be a live handle from [`adele_client_connect`]. `cb` is invoked
/// from a worker thread with a NUL-terminated JSON string (valid only for that
/// call) plus `user_data`; both must stay valid until [`adele_client_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_client_start_signals(
    client: *mut AdeleClient,
    cb: extern "C" fn(*const c_char, *mut c_void),
    user_data: *mut c_void,
) {
    let Some(client) = (unsafe { client.as_ref() }) else {
        return;
    };
    // `cb` is a fn pointer (Send + Copy). Carry `user_data` as a usize, not a raw
    // `*mut c_void`, so the spawned future is `Send` without an `unsafe impl`
    // (disjoint closure capture would otherwise grab the bare pointer field). The
    // embedder guarantees `user_data` stays valid and the callback is thread-safe
    // (the Qt wrapper posts onto its GUI thread).
    let user_data = user_data as usize;
    let connector = Arc::clone(&client.connector);
    client.rt.spawn(async move {
        let mut rx = connector.subscribe();
        loop {
            match rx.recv().await {
                Some(event) => {
                    let reconnect = matches!(event, SignalEvent::Disconnected { .. });
                    if let Some(json) = signal_to_json(&event)
                        && let Ok(c_json) = CString::new(json)
                    {
                        // The callback MUST copy out of `c_json`; it is freed when
                        // the call returns.
                        cb(c_json.as_ptr(), user_data as *mut c_void);
                    }
                    if reconnect {
                        // The Connector reconnects under us; re-subscribe for a
                        // fresh stream.
                        rx = connector.subscribe();
                    }
                }
                None => rx = connector.subscribe(),
            }
        }
    });
}

/// Set-replace the conversations this client is viewing (live sync). `*_json` is
/// a JSON array of conversation-id strings; `[]` unsubscribes from all. Returns
/// `false` on a bad handle / malformed JSON / dispatch failure.
///
/// # Safety
/// `client` must be a live handle from [`adele_client_connect`];
/// `conversation_ids_json` must be NULL or a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_client_subscribe_conversations(
    client: *mut AdeleClient,
    conversation_ids_json: *const c_char,
) -> bool {
    let Some(client) = (unsafe { client.as_ref() }) else {
        return false;
    };
    let Some(ids) = (unsafe { cstr(conversation_ids_json) })
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
    else {
        return false;
    };
    let connector = Arc::clone(&client.connector);
    client.rt.block_on(async move {
        match connector.client().as_commands() {
            Some(commands) => commands
                .send_command(api::Command::SubscribeConversations {
                    conversation_ids: ids,
                })
                .await
                .is_ok(),
            None => false,
        }
    })
}

/// Send a prompt; returns the turn `request_id` (the id the streamed events
/// carry) as an owned C string the caller frees with [`adele_string_free`], or
/// NULL on a bad handle / dispatch failure.
///
/// # Safety
/// `client` must be a live handle from [`adele_client_connect`]; `conversation_id`
/// and `prompt` must be valid NUL-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_client_send_prompt(
    client: *mut AdeleClient,
    conversation_id: *const c_char,
    prompt: *const c_char,
) -> *mut c_char {
    let Some(client) = (unsafe { client.as_ref() }) else {
        return ptr::null_mut();
    };
    let (Some(conversation_id), Some(prompt)) =
        (unsafe { cstr(conversation_id) }, unsafe { cstr(prompt) })
    else {
        return ptr::null_mut();
    };
    let connector = Arc::clone(&client.connector);
    let result = client
        .rt
        .block_on(async move { connector.send_prompt(&conversation_id, &prompt).await });
    match result {
        Ok(request_id) => CString::new(request_id)
            .map(CString::into_raw)
            .unwrap_or(ptr::null_mut()),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by this library.
///
/// # Safety
/// `s` must be NULL or a pointer returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: produced by `CString::into_raw` in this library.
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Disconnect and free the client. Drops the runtime (stopping the pump) and the
/// `Connector` (closing the daemon connection).
///
/// # Safety
/// `client` must be NULL or a handle from [`adele_client_connect`], not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adele_client_free(client: *mut AdeleClient) {
    if !client.is_null() {
        // SAFETY: came from `Box::into_raw` in `adele_client_connect`.
        unsafe { drop(Box::from_raw(client)) };
    }
}

/// Marshal a `SignalEvent` to a tagged JSON object for the C callback. Returns
/// `None` for variants the chat surface does not consume (v1): `ContextUsage`,
/// `ConversationWarning`, `ScratchpadChanged`, `Task*`.
fn signal_to_json(event: &SignalEvent) -> Option<String> {
    use SignalEvent as E;
    let value = match event {
        E::UserMessageAdded {
            conversation_id,
            request_id,
            content,
        } => serde_json::json!({
            "kind": "user_message_added", "conversation_id": conversation_id,
            "request_id": request_id, "content": content,
        }),
        E::Chunk {
            conversation_id,
            request_id,
            chunk,
        } => serde_json::json!({
            "kind": "chunk", "conversation_id": conversation_id,
            "request_id": request_id, "chunk": chunk,
        }),
        E::Complete {
            conversation_id,
            request_id,
            full_response,
        } => serde_json::json!({
            "kind": "complete", "conversation_id": conversation_id,
            "request_id": request_id, "full_response": full_response,
        }),
        E::Error {
            conversation_id,
            request_id,
            error,
        } => serde_json::json!({
            "kind": "error", "conversation_id": conversation_id,
            "request_id": request_id, "error": error,
        }),
        E::Status {
            conversation_id,
            request_id,
            message,
        } => serde_json::json!({
            "kind": "status", "conversation_id": conversation_id,
            "request_id": request_id, "message": message,
        }),
        E::TitleChanged {
            conversation_id,
            title,
        } => serde_json::json!({
            "kind": "title_changed", "conversation_id": conversation_id, "title": title,
        }),
        E::ConversationListChanged { conversation_id } => serde_json::json!({
            "kind": "conversation_list_changed", "conversation_id": conversation_id,
        }),
        E::ClientToolCall {
            task_id,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments,
        } => serde_json::json!({
            "kind": "client_tool_call", "task_id": task_id,
            "conversation_id": conversation_id, "tool_call_id": tool_call_id,
            "tool_name": tool_name, "arguments": arguments,
        }),
        E::Disconnected { reason } => serde_json::json!({
            "kind": "disconnected", "reason": reason,
        }),
        _ => return None,
    };
    serde_json::to_string(&value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_to_json_marshals_chat_events_with_a_kind_tag() {
        let json = signal_to_json(&SignalEvent::UserMessageAdded {
            conversation_id: "c".into(),
            request_id: "r".into(),
            content: "hi".into(),
        })
        .expect("user_message_added is forwarded");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "user_message_added");
        assert_eq!(v["conversation_id"], "c");
        assert_eq!(v["content"], "hi");
    }

    #[test]
    fn signal_to_json_preserves_client_tool_call_arguments_as_json() {
        let json = signal_to_json(&SignalEvent::ClientToolCall {
            task_id: "t".into(),
            conversation_id: "c".into(),
            tool_call_id: "tc".into(),
            tool_name: "echo".into(),
            arguments: serde_json::json!({"x": 1}),
        })
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "client_tool_call");
        assert_eq!(v["tool_name"], "echo");
        // The Value rides through structurally, not stringified.
        assert_eq!(v["arguments"]["x"], 1);
    }

    #[test]
    fn signal_to_json_drops_events_the_chat_surface_ignores() {
        assert!(
            signal_to_json(&SignalEvent::ScratchpadChanged {
                conversation_id: "c".into(),
            })
            .is_none()
        );
    }
}
