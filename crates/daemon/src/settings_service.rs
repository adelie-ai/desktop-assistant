use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::inbound::{
    BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView, EmbeddingsSettingsView,
    LlmSettingsView, McpServerView, PersistenceSettingsView, PersonalitySettingsView,
    SettingsService, WsAuthSettingsView,
};
use desktop_assistant_mcp_client::executor::McpControlHandle;

use crate::api_surface::RegistryHandle;
use crate::config;

pub struct DaemonSettingsService {
    config_path: PathBuf,
    mcp_handle: Option<McpControlHandle>,
    /// Shared handle to the live registry + config. Personality (#226) is read
    /// and written through here rather than via disk-only `config::` helpers, so
    /// the settings GET, the dispatch wrapper's per-send read, and a `SetConfig`
    /// all share the registry's in-memory config — making personality changes
    /// take effect on the next turn without a separate reload.
    registry: Option<Arc<RegistryHandle>>,
}

impl DaemonSettingsService {
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            config_path,
            mcp_handle: None,
            registry: None,
        }
    }

    pub fn with_mcp_control(mut self, handle: McpControlHandle) -> Self {
        self.mcp_handle = Some(handle);
        self
    }

    pub fn with_registry(mut self, registry: Arc<RegistryHandle>) -> Self {
        self.registry = Some(registry);
        self
    }

    fn registry(&self) -> Result<&Arc<RegistryHandle>, CoreError> {
        self.registry
            .as_ref()
            .ok_or_else(|| CoreError::SystemService("registry handle not configured".to_string()))
    }

    fn mcp_handle(&self) -> Result<&McpControlHandle, CoreError> {
        self.mcp_handle
            .as_ref()
            .ok_or_else(|| CoreError::SystemService("MCP control not configured".to_string()))
    }

    /// Refresh the executor's in-memory secrets from `secrets.toml` so an OAuth
    /// server's `authorized` state reflects a refresh token minted out-of-band
    /// (the `--mcp-oauth-login` flow runs in a *separate* process, so the live
    /// daemon otherwise never sees the new token until restart). Best-effort: a
    /// missing/unreadable file leaves the current snapshot in place. Called
    /// before every MCP status read so the settings UI reports the truth.
    async fn reload_mcp_secrets(&self, handle: &McpControlHandle) {
        let path = desktop_assistant_mcp_client::config::default_secrets_path();
        // A missing file must not wipe the snapshot the daemon started with.
        if !path.exists() {
            return;
        }
        if let Ok(secrets) = desktop_assistant_mcp_client::config::load_secrets(&path) {
            handle.replace_secrets(secrets).await;
        }
    }

    /// Refresh the executor's in-memory service accounts from `mcp_servers.toml`
    /// so an account's `granted_scopes` (recorded by a separate `--mcp-oauth-login`
    /// process) are reflected in a server's coverage state without a restart.
    /// Best-effort, mirroring [`Self::reload_mcp_secrets`].
    async fn reload_mcp_service_accounts(&self, handle: &McpControlHandle) {
        let path = desktop_assistant_mcp_client::config::default_config_path();
        if !path.exists() {
            return;
        }
        if let Ok(accounts) = desktop_assistant_mcp_client::config::load_service_accounts(&path) {
            handle.replace_service_accounts(accounts).await;
        }
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
            hosted_tool_search: view.hosted_tool_search,
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
        hosted_tool_search: Option<bool>,
    ) -> Result<(), CoreError> {
        config::set_llm_settings(
            &self.config_path,
            &connector,
            model.as_deref(),
            base_url.as_deref(),
            temperature,
            top_p,
            max_tokens,
            hosted_tool_search,
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
            backend_llm_model: defaults.backend_llm_model,
            embeddings_model: defaults.embeddings_model,
            embeddings_base_url: defaults.embeddings_base_url,
            embeddings_available: defaults.embeddings_available,
            hosted_tool_search_available: defaults.hosted_tool_search_available,
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

    async fn get_personality_settings(&self) -> Result<PersonalitySettingsView, CoreError> {
        // Read from the registry's in-memory config — the same value the
        // dispatch wrapper installs as the per-turn task-local (#226).
        Ok(self.registry()?.personality())
    }

    async fn set_personality_settings(
        &self,
        personality: PersonalitySettingsView,
    ) -> Result<(), CoreError> {
        // Persists to the config file and refreshes the in-memory config, so
        // the next send's task-local reflects the change (hot reload).
        self.registry()?.set_personality(personality)
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
            archive_after_days: view.archive_after_days,
        })
    }

    async fn set_backend_tasks_settings(
        &self,
        llm_connector: Option<String>,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
        archive_after_days: u32,
    ) -> Result<(), CoreError> {
        config::set_backend_tasks_settings(
            &self.config_path,
            llm_connector.as_deref(),
            llm_model.as_deref(),
            llm_base_url.as_deref(),
            dreaming_enabled,
            dreaming_interval_secs,
            archive_after_days,
        )
        .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn list_mcp_servers(&self) -> Result<Vec<McpServerView>, CoreError> {
        let handle = self.mcp_handle()?;
        self.reload_mcp_secrets(handle).await;
        self.reload_mcp_service_accounts(handle).await;
        let statuses = handle.status(None).await;
        Ok(statuses
            .into_iter()
            .map(|s| McpServerView {
                name: s.name,
                command: s.command,
                args: s.args,
                namespace: s.namespace,
                enabled: s.enabled,
                status: s.status,
                tool_count: s.tool_count,
                transport: s.transport,
                target: s.target,
                detail: s.detail,
                configure_label: s.configure_label,
                configure_command: s.configure_command,
                auth_kind: s.auth_kind,
                oauth_authorized: s.oauth_authorized,
                oauth_account: s.oauth_account,
                oauth_scopes: s.oauth_scopes,
                oauth_client_id: s.oauth_client_id,
                oauth_token_url: s.oauth_token_url,
                oauth_authorize_url: s.oauth_authorize_url,
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
            env: std::collections::HashMap::new(),
            env_secrets: std::collections::HashMap::new(),
            http: None,
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
        self.reload_mcp_secrets(handle).await;
        self.reload_mcp_service_accounts(handle).await;
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
                args: s.args,
                namespace: s.namespace,
                enabled: s.enabled,
                status: s.status,
                tool_count: s.tool_count,
                transport: s.transport,
                target: s.target,
                detail: s.detail,
                configure_label: s.configure_label,
                configure_command: s.configure_command,
                auth_kind: s.auth_kind,
                oauth_authorized: s.oauth_authorized,
                oauth_account: s.oauth_account,
                oauth_scopes: s.oauth_scopes,
                oauth_client_id: s.oauth_client_id,
                oauth_token_url: s.oauth_token_url,
                oauth_authorize_url: s.oauth_authorize_url,
            })
            .collect())
    }

    async fn upsert_mcp_server(&self, config_json: String) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        let config: desktop_assistant_mcp_client::executor::McpServerConfig =
            serde_json::from_str(&config_json)
                .map_err(|e| CoreError::SystemService(format!("invalid MCP server config: {e}")))?;
        if config.name.trim().is_empty() {
            return Err(CoreError::SystemService(
                "MCP server name must not be empty".into(),
            ));
        }
        // A server is either stdio (a command) or remote (an http transport);
        // reject a config that is neither so we never persist an unusable entry.
        if config.command.trim().is_empty() && config.http.is_none() {
            return Err(CoreError::SystemService(
                "MCP server must set either a command (stdio) or an http transport".into(),
            ));
        }
        handle
            .upsert_server(config)
            .await
            .map_err(|e| CoreError::SystemService(e.to_string()))
    }

    async fn set_mcp_secret(&self, id: String, value: String) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        if id.trim().is_empty() {
            return Err(CoreError::SystemService(
                "secret id must not be empty".into(),
            ));
        }
        // Read-modify-write secrets.toml (preserving other entries), then push
        // the fresh snapshot into the live executor so a following upsert that
        // references this id resolves without a restart.
        let path = desktop_assistant_mcp_client::config::default_secrets_path();
        let secrets = desktop_assistant_mcp_client::config::upsert_secret(&path, &id, &value)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        handle.replace_secrets(secrets).await;
        Ok(())
    }

    async fn get_ws_auth_settings(&self) -> Result<WsAuthSettingsView, CoreError> {
        let ws_auth = config::get_ws_auth_settings(&self.config_path)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;

        let (oidc_issuer, oidc_auth_endpoint, oidc_token_endpoint, oidc_client_id, oidc_scopes) =
            match ws_auth.oidc {
                Some(oidc) => (
                    oidc.issuer_url,
                    oidc.authorization_endpoint,
                    oidc.token_endpoint,
                    oidc.client_id,
                    oidc.scopes,
                ),
                None => (
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                ),
            };

        Ok(WsAuthSettingsView {
            methods: ws_auth.methods,
            oidc_issuer,
            oidc_auth_endpoint,
            oidc_token_endpoint,
            oidc_client_id,
            oidc_scopes,
        })
    }

    async fn set_ws_auth_settings(
        &self,
        methods: Vec<String>,
        oidc_issuer: String,
        oidc_auth_endpoint: String,
        oidc_token_endpoint: String,
        oidc_client_id: String,
        oidc_scopes: String,
    ) -> Result<(), CoreError> {
        config::set_ws_auth_settings(
            &self.config_path,
            &methods,
            &oidc_issuer,
            &oidc_auth_endpoint,
            &oidc_token_endpoint,
            &oidc_client_id,
            &oidc_scopes,
        )
        .map_err(|e| CoreError::SystemService(e.to_string()))
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
