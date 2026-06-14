//! D-Bus adapter for `/org/desktopAssistant/Connections`.
//!
//! Mirrors `crates/dbus-interface/src/connections.rs` method-for-method:
//! same JSON-string envelope contracts for `list_connections`,
//! `get_purposes`, and `list_available_models` so existing KCM /
//! plasmoid code keeps parsing the same shape.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

fn to_fdo<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// D-Bus adapter for the named-connections + purposes API. Same
/// surface shape as the daemon's in-process adapter — methods return
/// JSON strings for complex payloads so KCM keeps using
/// `QJsonDocument`.
pub struct DbusConnectionsAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusConnectionsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }
}

#[interface(name = "org.desktopAssistant.Connections")]
impl<T: BridgeTransport + 'static> DbusConnectionsAdapter<T> {
    async fn list_connections(&self) -> fdo::Result<String> {
        let result = self.dispatch(api::Command::ListConnections).await?;
        match &result {
            api::CommandResult::Connections(_) => serde_json::to_string(&result).map_err(to_fdo),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListConnections result: {other:?}"
            ))),
        }
    }

    async fn create_connection(&self, id: &str, config_json: &str) -> fdo::Result<()> {
        let config: api::ConnectionConfigView =
            serde_json::from_str(config_json).map_err(to_fdo)?;
        let result = self
            .dispatch(api::Command::CreateConnection {
                id: id.to_string(),
                config,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected CreateConnection result: {other:?}"
            ))),
        }
    }

    async fn update_connection(&self, id: &str, config_json: &str) -> fdo::Result<()> {
        let config: api::ConnectionConfigView =
            serde_json::from_str(config_json).map_err(to_fdo)?;
        let result = self
            .dispatch(api::Command::UpdateConnection {
                id: id.to_string(),
                config,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected UpdateConnection result: {other:?}"
            ))),
        }
    }

    async fn delete_connection(&self, id: &str, force: bool) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::DeleteConnection {
                id: id.to_string(),
                force,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected DeleteConnection result: {other:?}"
            ))),
        }
    }

    async fn list_available_models(
        &self,
        connection_id: &str,
        refresh: bool,
    ) -> fdo::Result<String> {
        let connection_id = if connection_id.trim().is_empty() {
            None
        } else {
            Some(connection_id.to_string())
        };
        let result = self
            .dispatch(api::Command::ListAvailableModels {
                connection_id,
                refresh,
            })
            .await?;
        match &result {
            api::CommandResult::Models(_) => serde_json::to_string(&result).map_err(to_fdo),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListAvailableModels result: {other:?}"
            ))),
        }
    }

    async fn get_purposes(&self) -> fdo::Result<String> {
        let result = self.dispatch(api::Command::GetPurposes).await?;
        match &result {
            api::CommandResult::Purposes(_) => serde_json::to_string(&result).map_err(to_fdo),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetPurposes result: {other:?}"
            ))),
        }
    }

    async fn set_purpose(&self, purpose: &str, config_json: &str) -> fdo::Result<()> {
        let purpose: api::PurposeKindApi = serde_json::from_str(&format!("\"{purpose}\""))
            .map_err(|e| fdo::Error::Failed(format!("invalid purpose '{purpose}': {e}")))?;
        let config: api::PurposeConfigView = serde_json::from_str(config_json).map_err(to_fdo)?;
        let result = self
            .dispatch(api::Command::SetPurpose { purpose, config })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetPurpose result: {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Contract tests for the Connections adapter. The spec is the *canonical*
    //! `api::Command` each D-Bus method must build and the result it must map —
    //! the same command/result contract every transport (UDS/WS/D-Bus) honors,
    //! so the bridge behaves identically to any other client. Unhappy paths
    //! (malformed JSON, unknown purpose, result-variant mismatch, daemon error)
    //! are enumerated as named tests.
    use super::*;
    use std::sync::Mutex;

    /// Records each dispatched command and replies with a scripted result (or a
    /// daemon error). The recorded command is the assertion surface.
    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: Result<api::CommandResult, String>,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: Ok(reply),
            })
        }
        fn failing(daemon_msg: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: Err(daemon_msg.to_string()),
            })
        }
        fn count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
        fn last(&self) -> api::Command {
            self.seen
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("a command was dispatched")
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for FakeTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.seen.lock().unwrap().push(command);
            self.reply.clone().map_err(BridgeTransportError::Daemon)
        }
    }

    fn adapter(t: Arc<FakeTransport>) -> DbusConnectionsAdapter<FakeTransport> {
        DbusConnectionsAdapter::new(t)
    }

    // --- list_connections -----------------------------------------------------

    #[tokio::test]
    async fn list_connections_dispatches_list_and_returns_json_envelope() {
        let t = FakeTransport::replying(api::CommandResult::Connections(Vec::new()));
        let json = adapter(Arc::clone(&t)).list_connections().await.unwrap();
        assert!(matches!(t.last(), api::Command::ListConnections));
        // The wire contract is the JSON-serialized CommandResult envelope.
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::Connections(_)));
    }

    #[tokio::test]
    async fn list_connections_errors_on_unexpected_result_variant() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .list_connections()
            .await
            .expect_err("a non-Connections result must surface as an error, not a panic");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // --- create_connection ----------------------------------------------------

    #[tokio::test]
    async fn create_connection_builds_command_with_parsed_config() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        adapter(Arc::clone(&t))
            .create_connection(
                "anthropic-main",
                r#"{"type":"anthropic","base_url":"https://x"}"#,
            )
            .await
            .unwrap();
        match t.last() {
            api::Command::CreateConnection { id, config } => {
                assert_eq!(id, "anthropic-main");
                assert_eq!(
                    config,
                    api::ConnectionConfigView::Anthropic {
                        base_url: Some("https://x".to_string()),
                        api_key_env: None,
                        connect_timeout_secs: None,
                        stream_timeout_secs: None,
                        max_context_tokens: None,
                    }
                );
            }
            other => panic!("expected CreateConnection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_connection_rejects_malformed_config_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .create_connection("c", "{ not json")
            .await
            .expect_err("malformed config must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(
            t.count(),
            0,
            "a malformed command must never reach the daemon"
        );
    }

    #[tokio::test]
    async fn create_connection_errors_on_unexpected_result_variant() {
        let t = FakeTransport::replying(api::CommandResult::Connections(Vec::new()));
        let err = adapter(Arc::clone(&t))
            .create_connection("c", r#"{"type":"anthropic"}"#)
            .await
            .expect_err("a non-Ack result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // --- update_connection ----------------------------------------------------

    #[tokio::test]
    async fn update_connection_builds_command_with_parsed_config() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        adapter(Arc::clone(&t))
            .update_connection("c1", r#"{"type":"anthropic"}"#)
            .await
            .unwrap();
        assert!(matches!(
            t.last(),
            api::Command::UpdateConnection { id, config: api::ConnectionConfigView::Anthropic { .. } } if id == "c1"
        ));
    }

    #[tokio::test]
    async fn update_connection_rejects_malformed_config_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .update_connection("c", "nope")
            .await
            .expect_err("malformed config must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(t.count(), 0);
    }

    // --- delete_connection ----------------------------------------------------

    #[tokio::test]
    async fn delete_connection_carries_the_force_flag_verbatim() {
        for force in [false, true] {
            let t = FakeTransport::replying(api::CommandResult::Ack);
            adapter(Arc::clone(&t))
                .delete_connection("doomed", force)
                .await
                .unwrap();
            match t.last() {
                api::Command::DeleteConnection { id, force: f } => {
                    assert_eq!(id, "doomed");
                    assert_eq!(f, force);
                }
                other => panic!("expected DeleteConnection, got {other:?}"),
            }
        }
    }

    // --- list_available_models ------------------------------------------------

    #[tokio::test]
    async fn list_available_models_passes_connection_id_and_refresh() {
        let t = FakeTransport::replying(api::CommandResult::Models(Vec::new()));
        adapter(Arc::clone(&t))
            .list_available_models("bedrock", true)
            .await
            .unwrap();
        match t.last() {
            api::Command::ListAvailableModels {
                connection_id,
                refresh,
            } => {
                assert_eq!(connection_id, Some("bedrock".to_string()));
                assert!(refresh);
            }
            other => panic!("expected ListAvailableModels, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_available_models_blank_id_means_all_connections() {
        // Empty/whitespace connection id is the contract's "aggregate across all
        // healthy connections" signal — it must become `None`, not `Some("")`.
        for blank in ["", "   "] {
            let t = FakeTransport::replying(api::CommandResult::Models(Vec::new()));
            adapter(Arc::clone(&t))
                .list_available_models(blank, false)
                .await
                .unwrap();
            assert!(matches!(
                t.last(),
                api::Command::ListAvailableModels {
                    connection_id: None,
                    refresh: false
                }
            ));
        }
    }

    // --- get_purposes ---------------------------------------------------------

    #[tokio::test]
    async fn get_purposes_dispatches_and_returns_json_envelope() {
        let purposes = api::PurposesView {
            interactive: None,
            dreaming: None,
            consolidation: None,
            embedding: None,
            titling: None,
        };
        let t = FakeTransport::replying(api::CommandResult::Purposes(purposes));
        let json = adapter(Arc::clone(&t)).get_purposes().await.unwrap();
        assert!(matches!(t.last(), api::Command::GetPurposes));
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::Purposes(_)));
    }

    // --- set_purpose ----------------------------------------------------------

    #[tokio::test]
    async fn set_purpose_builds_command_with_parsed_purpose_and_config() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        adapter(Arc::clone(&t))
            .set_purpose(
                "interactive",
                r#"{"connection":"primary","model":"primary"}"#,
            )
            .await
            .unwrap();
        match t.last() {
            api::Command::SetPurpose { purpose, config } => {
                assert_eq!(purpose, api::PurposeKindApi::Interactive);
                assert_eq!(config.connection, "primary");
                assert_eq!(config.model, "primary");
            }
            other => panic!("expected SetPurpose, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_purpose_rejects_unknown_purpose_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .set_purpose("not-a-purpose", r#"{"connection":"p","model":"m"}"#)
            .await
            .expect_err("an unknown purpose kind must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(t.count(), 0);
    }

    #[tokio::test]
    async fn set_purpose_rejects_malformed_config_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .set_purpose("dreaming", "{ bad")
            .await
            .expect_err("malformed purpose config must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(t.count(), 0);
    }

    // --- transport / daemon errors --------------------------------------------

    #[tokio::test]
    async fn daemon_error_is_propagated_verbatim() {
        let t = FakeTransport::failing("connection pool exhausted");
        let err = adapter(Arc::clone(&t))
            .list_connections()
            .await
            .expect_err("a daemon error must surface");
        assert!(
            format!("{err}").contains("connection pool exhausted"),
            "daemon message must be preserved: {err}"
        );
    }
}
