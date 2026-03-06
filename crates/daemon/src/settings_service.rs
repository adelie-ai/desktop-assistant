use std::path::PathBuf;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::inbound::{
    BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView, EmbeddingsSettingsView,
    LlmSettingsView, McpServerView, PersistenceSettingsView, SettingsService,
};
use desktop_assistant_mcp_client::executor::McpControlHandle;

use crate::config;

pub struct DaemonSettingsService {
    config_path: PathBuf,
    mcp_handle: Option<McpControlHandle>,
}

impl DaemonSettingsService {
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            config_path,
            mcp_handle: None,
        }
    }

    pub fn with_mcp_control(mut self, handle: McpControlHandle) -> Self {
        self.mcp_handle = Some(handle);
        self
    }

    fn mcp_handle(&self) -> Result<&McpControlHandle, CoreError> {
        self.mcp_handle
            .as_ref()
            .ok_or_else(|| CoreError::SystemService("MCP control not configured".to_string()))
    }
}

impl SettingsService for DaemonSettingsService {
    async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
        let view = config::get_llm_settings_view(&self.config_path)
            .map_err(|error| CoreError::SystemService(error.to_string()))?;

        Ok(LlmSettingsView {
            connector: view.connector,
            model: view.model,
            base_url: view.base_url,
            has_api_key: view.has_api_key,
            temperature: view.temperature,
            top_p: view.top_p,
            max_tokens: view.max_tokens,
        })
    }

    async fn set_llm_settings(
        &self,
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<(), CoreError> {
        config::set_llm_settings(
            &self.config_path,
            &connector,
            model.as_deref(),
            base_url.as_deref(),
            temperature,
            top_p,
            max_tokens,
        )
        .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn set_api_key(&self, api_key: String) -> Result<(), CoreError> {
        config::set_api_key(&self.config_path, &api_key)
            .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn generate_ws_jwt(&self, _subject: Option<String>) -> Result<String, CoreError> {
        config::generate_ws_jwt(Some(config::current_username()))
            .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
        config::validate_ws_jwt(&token).map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
        let view = config::get_embeddings_settings_view(&self.config_path)
            .map_err(|error| CoreError::SystemService(error.to_string()))?;

        Ok(EmbeddingsSettingsView {
            connector: view.connector,
            model: view.model,
            base_url: view.base_url,
            has_api_key: view.has_api_key,
            available: view.available,
            is_default: view.is_default,
        })
    }

    async fn set_embeddings_settings(
        &self,
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<(), CoreError> {
        config::set_embeddings_settings(
            &self.config_path,
            connector.as_deref(),
            model.as_deref(),
            base_url.as_deref(),
        )
        .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn get_connector_defaults(
        &self,
        connector: String,
    ) -> Result<ConnectorDefaultsView, CoreError> {
        let defaults = config::get_connector_defaults(&connector);
        Ok(ConnectorDefaultsView {
            llm_model: defaults.llm_model,
            llm_base_url: defaults.llm_base_url,
            embeddings_model: defaults.embeddings_model,
            embeddings_base_url: defaults.embeddings_base_url,
            embeddings_available: defaults.embeddings_available,
        })
    }

    async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
        let resolved = config::get_persistence_settings_view(&self.config_path)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        Ok(PersistenceSettingsView {
            enabled: resolved.enabled,
            remote_url: resolved.remote_url.unwrap_or_default(),
            remote_name: resolved.remote_name,
            push_on_update: resolved.push_on_update,
        })
    }

    async fn set_persistence_settings(
        &self,
        enabled: bool,
        remote_url: Option<String>,
        remote_name: Option<String>,
        push_on_update: bool,
    ) -> Result<(), CoreError> {
        config::set_persistence_settings(
            &self.config_path,
            enabled,
            remote_url.as_deref(),
            remote_name.as_deref(),
            push_on_update,
        )
        .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
        let (url, max_connections) = config::get_database_settings_view(&self.config_path)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        Ok(DatabaseSettingsView {
            url,
            max_connections,
        })
    }

    async fn set_database_settings(
        &self,
        url: Option<String>,
        max_connections: u32,
    ) -> Result<(), CoreError> {
        config::set_database_settings(&self.config_path, url.as_deref(), max_connections)
            .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
        let view = config::get_backend_tasks_settings_view(&self.config_path)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        Ok(BackendTasksSettingsView {
            has_separate_llm: view.has_separate_llm,
            llm_connector: view.llm_connector,
            llm_model: view.llm_model,
            llm_base_url: view.llm_base_url,
            dreaming_enabled: view.dreaming_enabled,
            dreaming_interval_secs: view.dreaming_interval_secs,
        })
    }

    async fn set_backend_tasks_settings(
        &self,
        llm_connector: Option<String>,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
    ) -> Result<(), CoreError> {
        config::set_backend_tasks_settings(
            &self.config_path,
            llm_connector.as_deref(),
            llm_model.as_deref(),
            llm_base_url.as_deref(),
            dreaming_enabled,
            dreaming_interval_secs,
        )
        .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn list_mcp_servers(&self) -> Result<Vec<McpServerView>, CoreError> {
        let handle = self.mcp_handle()?;
        let statuses = handle.status(None).await;
        Ok(statuses
            .into_iter()
            .map(|s| McpServerView {
                name: s.name,
                command: s.command,
                args: vec![],
                namespace: None,
                enabled: s.enabled,
                status: s.status,
                tool_count: s.tool_count,
            })
            .collect())
    }

    async fn add_mcp_server(
        &self,
        name: String,
        command: String,
        args: Vec<String>,
        namespace: Option<String>,
        enabled: bool,
    ) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        let config = desktop_assistant_mcp_client::executor::McpServerConfig {
            name,
            command,
            args,
            namespace,
            enabled,
        };
        handle
            .add_server(config)
            .await
            .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn remove_mcp_server(&self, name: String) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        handle
            .remove_server(&name)
            .await
            .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn set_mcp_server_enabled(&self, name: String, enabled: bool) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        if enabled {
            handle
                .enable_server(&name)
                .await
                .map_err(|e| CoreError::SystemService(e.to_string()))
        } else {
            handle
                .disable_server(&name)
                .await
                .map_err(|e| CoreError::SystemService(e.to_string()))
        }
    }

    async fn mcp_server_action(
        &self,
        action: String,
        server: Option<String>,
    ) -> Result<Vec<McpServerView>, CoreError> {
        let handle = self.mcp_handle()?;
        let server_ref = server.as_deref();

        match action.as_str() {
            "status" => {}
            "start" => {
                handle
                    .start_server(server_ref)
                    .await
                    .map_err(|e| CoreError::SystemService(e.to_string()))?;
            }
            "stop" => {
                handle
                    .stop_server(server_ref)
                    .await
                    .map_err(|e| CoreError::SystemService(e.to_string()))?;
            }
            "restart" => {
                handle
                    .restart_server(server_ref)
                    .await
                    .map_err(|e| CoreError::SystemService(e.to_string()))?;
            }
            _ => {
                return Err(CoreError::SystemService(format!(
                    "unknown MCP action: {action}"
                )));
            }
        }

        let statuses = handle.status(server_ref).await;
        Ok(statuses
            .into_iter()
            .map(|s| McpServerView {
                name: s.name,
                command: s.command,
                args: vec![],
                namespace: None,
                enabled: s.enabled,
                status: s.status,
                tool_count: s.tool_count,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_service_constructs() {
        let service = DaemonSettingsService::new(PathBuf::from("/tmp/desktop-assistant-test.toml"));
        assert!(service.config_path.ends_with("desktop-assistant-test.toml"));
    }
}
