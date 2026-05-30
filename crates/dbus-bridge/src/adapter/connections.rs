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
