use std::sync::Arc;

use desktop_assistant_api_model::{self as api};
use desktop_assistant_application::AssistantApiHandler;
use zbus::{fdo, interface};

fn to_fdo_error<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

/// D-Bus adapter for the multi-connection / purposes API.
///
/// Each method translates D-Bus arguments into an `api::Command`, dispatches
/// it through the shared `AssistantApiHandler` (the same path the WebSocket
/// adapter uses), and shapes the resulting `api::CommandResult` back onto the
/// wire. Complex payloads are passed as JSON strings to avoid teaching zbus
/// about every nested enum variant; this matches how the WS interface already
/// serializes them and keeps the KCM marshaling code minimal (it just hands
/// the same JSON to `QJsonDocument`).
pub struct DbusConnectionsAdapter {
    handler: Arc<dyn AssistantApiHandler>,
}

impl DbusConnectionsAdapter {
    pub fn new(handler: Arc<dyn AssistantApiHandler>) -> Self {
        Self { handler }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.handler
            .handle_command(cmd)
            .await
            .map_err(|e| fdo::Error::Failed(format!("{e:?}")))
    }
}

#[interface(name = "org.desktopAssistant.Connections")]
impl DbusConnectionsAdapter {
    /// Return the configured connections. Wire format is a JSON object
    /// `{"connections": [ConnectionView]}`, matching the WebSocket adapter.
    async fn list_connections(&self) -> fdo::Result<String> {
        let result = self.dispatch(api::Command::ListConnections).await?;
        match &result {
            api::CommandResult::Connections(_) => {
                serde_json::to_string(&result).map_err(to_fdo_error)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListConnections result: {other:?}"
            ))),
        }
    }

    /// Add a new connection. `config_json` must be the serialized
    /// `api::ConnectionConfigView`.
    async fn create_connection(&self, id: &str, config_json: &str) -> fdo::Result<()> {
        let config: api::ConnectionConfigView =
            serde_json::from_str(config_json).map_err(to_fdo_error)?;
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

    /// Update an existing connection.
    async fn update_connection(&self, id: &str, config_json: &str) -> fdo::Result<()> {
        let config: api::ConnectionConfigView =
            serde_json::from_str(config_json).map_err(to_fdo_error)?;
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

    /// Delete a connection. Set `force=true` to evict purposes that still
    /// reference it (the daemon falls them back to the interactive purpose).
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

    /// List models, optionally scoped to one connection. Empty
    /// `connection_id` means "all connections". Returns a JSON array of
    /// `ModelListing`.
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
            api::CommandResult::Models(_) => serde_json::to_string(&result).map_err(to_fdo_error),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListAvailableModels result: {other:?}"
            ))),
        }
    }

    /// Return the current purpose assignments. Wire format is a JSON object
    /// `{"purposes": PurposesView}`, matching the WebSocket adapter.
    async fn get_purposes(&self) -> fdo::Result<String> {
        let result = self.dispatch(api::Command::GetPurposes).await?;
        match &result {
            api::CommandResult::Purposes(_) => serde_json::to_string(&result).map_err(to_fdo_error),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetPurposes result: {other:?}"
            ))),
        }
    }

    /// Set a single purpose. `purpose` is one of `interactive`, `dreaming`,
    /// `embedding`, `titling`. `config_json` is the serialized
    /// `api::PurposeConfigView`.
    async fn set_purpose(&self, purpose: &str, config_json: &str) -> fdo::Result<()> {
        let purpose: api::PurposeKindApi = serde_json::from_str(&format!("\"{purpose}\""))
            .map_err(|e| fdo::Error::Failed(format!("invalid purpose '{purpose}': {e}")))?;
        let config: api::PurposeConfigView =
            serde_json::from_str(config_json).map_err(to_fdo_error)?;
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
