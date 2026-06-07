//! Generic D-Bus command channel (#213).
//!
//! The shared [`AssistantCommands`] command channel (config Settings, the
//! per-conversation model override / model selection, background tasks,
//! purposes, named-connection management) funnels every typed method through
//! one `send_command(api::Command) -> api::CommandResult` call. The WebSocket
//! and Unix-domain-socket transports implement that by sending a `WsRequest`
//! and awaiting the correlated `WsFrame`. This adapter gives the **D-Bus**
//! transport the same single entry point so `TransportClient::as_commands`
//! returns `Some` for every transport rather than dropping the management
//! surface on D-Bus.
//!
//! Like the [`crate::connections`] and [`crate::knowledge`] adapters, the
//! `api::CommandResult` is passed back as a JSON string (`busctl` shows
//! `s`); the client re-parses it with `serde_json`. The single method maps
//! every request/response `api::Command` 1:1, so there is no per-command
//! marshaling to teach zbus about — the command JSON in, the result JSON out.
//!
//! ## Hot-reload parity (`SetConfig`)
//!
//! `SetConfig` is **not** special-cased here. The daemon's hot-reload of a
//! config change happens *inside* `handle_command(SetConfig)` — the settings
//! service writes the new value to the config file and refreshes the daemon's
//! in-memory config (see `DaemonSettingsService::set_*` /
//! `RegistryHandle::set_personality`), so the next turn reads the new value
//! with no restart. The `Event::ConfigChanged` the socket dispatch loop emits
//! after `SetConfig` is a *client notification only* — there is no
//! server-side consumer of it (verified: no `ConfigChanged` handler exists in
//! the daemon/application/core crates). So routing `SetConfig` through
//! `handle_command` here hot-applies exactly as WS/UDS do; D-Bus clients that
//! want a change notification subscribe to the existing
//! `org.desktopAssistant.Settings.ConfigChanged` signal (emitted by the
//! granular `Settings.SetConfig` path the KCM already uses).
//!
//! ## Background-task subscription limitation
//!
//! `SubscribeBackgroundTasks` / `UnsubscribeBackgroundTasks` set up a
//! *streaming* subscription that pushes `Event::Task*` frames back over the
//! persistent socket for the lifetime of the connection. That cannot be
//! represented as a single request/response D-Bus method, so this adapter
//! rejects both with a clear error rather than pretending to succeed. The
//! in-process daemon's D-Bus interface does not expose background-task events
//! as signals (only the out-of-process `dbus-bridge` does, by holding a socket
//! `SubscribeBackgroundTasks` open itself); a D-Bus client that needs the
//! live stream should use a socket transport (WS/UDS).

use std::sync::Arc;

use desktop_assistant_api_model::{self as api};
use desktop_assistant_application::{AssistantApiHandler, RequestContext};
use zbus::{fdo, interface};

use crate::resolve_dbus_user_id;

fn to_fdo_error<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

/// D-Bus adapter exposing the shared command channel over the session bus.
///
/// Wraps the same `Arc<dyn AssistantApiHandler>` the socket transports'
/// dispatch loop holds, so the trust model and per-user scoping are identical
/// across transports (#156).
pub struct DbusCommandsAdapter {
    handler: Arc<dyn AssistantApiHandler>,
}

impl DbusCommandsAdapter {
    pub fn new(handler: Arc<dyn AssistantApiHandler>) -> Self {
        Self { handler }
    }
}

#[interface(name = "org.desktopAssistant.Commands")]
impl DbusCommandsAdapter {
    /// Run one request/response `api::Command` and return its
    /// `api::CommandResult` as a JSON string.
    ///
    /// `command_json` is a serialized [`api::Command`]; the reply is a
    /// serialized [`api::CommandResult`]. This is the D-Bus implementation of
    /// the shared [`AssistantCommands::send_command`] entry point, so it
    /// covers config Settings, the per-conversation model override, purposes,
    /// named-connection management, knowledge, scratchpad, and the rest of the
    /// management surface in one method.
    ///
    /// `SetConfig` hot-applies via the handler exactly as the socket
    /// transports do (see module docs). The two streaming background-task
    /// subscription commands are rejected — they need a persistent push
    /// channel a single D-Bus method can't provide.
    async fn send_command(&self, command_json: &str) -> fdo::Result<String> {
        let command: api::Command = serde_json::from_str(command_json)
            .map_err(|e| fdo::Error::InvalidArgs(format!("invalid command JSON: {e}")))?;

        // Reject the streaming-subscription commands up front: they cannot be
        // a single request/response (#213). The socket transports drive these
        // via the dispatch loop's per-connection forwarder over the persistent
        // socket; there is no equivalent over a one-shot D-Bus method, and the
        // in-process D-Bus interface does not surface Task* events as signals.
        if matches!(
            command,
            api::Command::SubscribeBackgroundTasks | api::Command::UnsubscribeBackgroundTasks
        ) {
            return Err(fdo::Error::NotSupported(
                "background-task streaming is not available over the generic D-Bus command \
                 channel (a single request/response method cannot push events); use a socket \
                 transport (WS/UDS) and Command::SubscribeBackgroundTasks for the live stream"
                    .to_string(),
            ));
        }

        // #156: install the per-user scope at the D-Bus dispatch boundary so
        // the handler (and the storage queries it composes) see the local OS
        // user instead of the `"default"` sentinel. `handle_command_for` is the
        // context-aware entry point the WS/UDS adapters already use; routing
        // through it keeps the auth wiring identical across transports. For
        // `SetConfig` this is the same call the socket dispatch loop makes, so
        // the in-daemon config hot-reload happens here too (see module docs).
        let ctx = RequestContext::for_user(resolve_dbus_user_id());
        let result = self
            .handler
            .handle_command_for(ctx, command)
            .await
            .map_err(|e| fdo::Error::Failed(format!("{e:?}")))?;

        serde_json::to_string(&result).map_err(to_fdo_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_application::{ApiError, ApiResult, EventSink};
    use desktop_assistant_core::ports::auth::current_user_id;
    use std::sync::Mutex;

    /// Records every `Command` it sees (with the resolved user id at the call
    /// site) and replies with a canned `CommandResult`. An optional error
    /// forces the handler to fail so we can assert error propagation.
    struct RecordingHandler {
        seen: Mutex<Vec<(api::Command, String)>>,
        reply: api::CommandResult,
        fail_with: Option<String>,
    }

    impl RecordingHandler {
        fn new(reply: api::CommandResult) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply,
                fail_with: None,
            })
        }

        fn failing(message: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: api::CommandResult::Ack,
                fail_with: Some(message.to_string()),
            })
        }

        fn last(&self) -> (api::Command, String) {
            self.seen
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("no command")
        }

        fn count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl AssistantApiHandler for RecordingHandler {
        async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
            self.seen
                .lock()
                .unwrap()
                .push((cmd, current_user_id().as_str().to_string()));
            if let Some(message) = &self.fail_with {
                return Err(ApiError::Core(message.clone()));
            }
            Ok(self.reply.clone())
        }

        async fn handle_send_message(
            &self,
            _conversation_id: String,
            _content: String,
            _request_id: String,
            _sink: Arc<dyn EventSink>,
        ) -> ApiResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn send_command_round_trips_request_and_response_json() {
        // A representative request/response command (ListConversations)
        // must reach the handler verbatim and its CommandResult must come
        // back as the JSON-string envelope the client deserializes.
        let reply = api::CommandResult::Conversations(vec![api::ConversationSummary {
            id: "c1".into(),
            title: "hello".into(),
            message_count: 3,
            updated_at: "2026-06-07T00:00:00Z".into(),
            archived: false,
        }]);
        let handler = RecordingHandler::new(reply.clone());
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let request = serde_json::to_string(&api::Command::ListConversations {
            max_age_days: None,
            include_archived: false,
        })
        .unwrap();

        let raw = adapter.send_command(&request).await.unwrap();
        let decoded: api::CommandResult = serde_json::from_str(&raw).unwrap();
        assert_eq!(decoded, reply);

        let (seen_cmd, _user) = handler.last();
        assert!(matches!(
            seen_cmd,
            api::Command::ListConversations {
                include_archived: false,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn send_command_routes_set_config_through_handler_for_hot_reload() {
        // SetConfig must reach the handler (where the in-daemon hot-reload
        // happens — the settings service writes the file and refreshes the
        // in-memory config) rather than being dropped or special-cased away.
        // The canned reply is `Config`, returned via the JSON envelope.
        let reply = api::CommandResult::Config(api::Config {
            embeddings: api::EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "text-embedding-3-small".into(),
                base_url: "https://api.openai.com/v1".into(),
                has_api_key: false,
                available: true,
                is_default: true,
            },
            persistence: api::PersistenceSettingsView {
                enabled: false,
                remote_url: String::new(),
                remote_name: "origin".into(),
                push_on_update: true,
            },
            personality: api::PersonalitySettingsView::default(),
        });
        let handler = RecordingHandler::new(reply.clone());
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let request = serde_json::to_string(&api::Command::SetConfig {
            changes: api::ConfigChanges::default(),
        })
        .unwrap();

        let raw = adapter.send_command(&request).await.unwrap();
        let decoded: api::CommandResult = serde_json::from_str(&raw).unwrap();
        assert_eq!(decoded, reply);

        let (seen_cmd, _user) = handler.last();
        assert!(
            matches!(seen_cmd, api::Command::SetConfig { .. }),
            "SetConfig must be dispatched through the handler so the config hot-reload runs"
        );
    }

    #[tokio::test]
    async fn send_command_installs_resolved_user_id_scope() {
        // #156: the handler (and downstream storage) must see the resolved
        // local OS user, not the "default" sentinel.
        let handler = RecordingHandler::new(api::CommandResult::Ack);
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let _guard = crate::testing::UserEnvGuard::set("alice-cmd");

        let request =
            serde_json::to_string(&api::Command::DeleteConversation { id: "c1".into() }).unwrap();
        adapter.send_command(&request).await.unwrap();

        let (_cmd, user) = handler.last();
        assert_eq!(
            user, "alice-cmd",
            "send_command must install the resolved user id before dispatching"
        );
    }

    #[tokio::test]
    async fn send_command_rejects_subscribe_background_tasks() {
        let handler = RecordingHandler::new(api::CommandResult::Ack);
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let request = serde_json::to_string(&api::Command::SubscribeBackgroundTasks).unwrap();
        let err = adapter
            .send_command(&request)
            .await
            .expect_err("subscribe must be rejected over the generic D-Bus channel");

        assert!(
            matches!(err, fdo::Error::NotSupported(_)),
            "expected NotSupported, got {err:?}"
        );
        assert!(
            format!("{err}").contains("background-task streaming"),
            "error must explain the streaming limitation: {err}"
        );
        assert_eq!(
            handler.count(),
            0,
            "a rejected subscription must never reach the handler"
        );
    }

    #[tokio::test]
    async fn send_command_rejects_unsubscribe_background_tasks() {
        let handler = RecordingHandler::new(api::CommandResult::Ack);
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let request = serde_json::to_string(&api::Command::UnsubscribeBackgroundTasks).unwrap();
        let err = adapter
            .send_command(&request)
            .await
            .expect_err("unsubscribe must be rejected over the generic D-Bus channel");

        assert!(matches!(err, fdo::Error::NotSupported(_)));
        assert_eq!(handler.count(), 0);
    }

    #[tokio::test]
    async fn send_command_propagates_handler_error_as_fdo_error() {
        let handler = RecordingHandler::failing("boom: connection refused");
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let request = serde_json::to_string(&api::Command::Ping).unwrap();
        let err = adapter
            .send_command(&request)
            .await
            .expect_err("a handler error must surface as an fdo error");

        assert!(matches!(err, fdo::Error::Failed(_)));
        assert!(
            format!("{err}").contains("boom: connection refused"),
            "the daemon's error message must be preserved: {err}"
        );
    }

    #[tokio::test]
    async fn send_command_rejects_malformed_json() {
        let handler = RecordingHandler::new(api::CommandResult::Ack);
        let adapter = DbusCommandsAdapter::new(handler.clone() as Arc<dyn AssistantApiHandler>);

        let err = adapter
            .send_command("{not valid json")
            .await
            .expect_err("malformed command JSON must be rejected");

        assert!(
            matches!(err, fdo::Error::InvalidArgs(_)),
            "expected InvalidArgs, got {err:?}"
        );
        assert_eq!(handler.count(), 0);
    }
}
