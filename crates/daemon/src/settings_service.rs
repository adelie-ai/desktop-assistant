use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::inbound::{
    BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView, EmbeddingHealth,
    EmbeddingsSettingsView, LlmSettingsView, McpServerView, PersistenceSettingsView,
    PersonalitySettingsView, ServiceAccountView, SettingsService, WsAuthSettingsView,
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
    /// Result of the daemon's startup embedding probe (#499). Set once at
    /// start-up; `get_embeddings_settings` reports it so `GetConfig` surfaces a
    /// real degraded/disabled state instead of a bare `available = true`. A
    /// config change after start-up does not re-probe, so this reflects the
    /// backend as it was at boot (re-probe on reload is a follow-up).
    embedding_health: Option<Arc<EmbeddingHealth>>,
}

impl DaemonSettingsService {
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            config_path,
            mcp_handle: None,
            registry: None,
            embedding_health: None,
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

    /// Inject the startup embedding-probe result (#499) so `GetConfig` reports
    /// the backend's real health.
    pub fn with_embedding_health(mut self, health: Arc<EmbeddingHealth>) -> Self {
        self.embedding_health = Some(health);
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

        // Prefer the startup probe result (#499) when it was injected; it knows
        // whether the configured backend can actually embed. Without a probe
        // handle (only in tests / degraded wiring) the honest answer is
        // `Unknown` — health was never determined. Deriving `Ok` from the shallow
        // `available` connector check would be exactly the false-green #499
        // exists to kill.
        let health = self
            .embedding_health
            .as_ref()
            .map(|health| (**health).clone())
            .unwrap_or(EmbeddingHealth::Unknown);

        Ok(EmbeddingsSettingsView {
            connector: view.connector,
            model: view.model,
            base_url: view.base_url,
            has_api_key: view.has_api_key,
            available: view.available,
            is_default: view.is_default,
            health,
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
                oauth_account_ref: s.oauth_account_ref,
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
            description: None,
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
                oauth_account_ref: s.oauth_account_ref,
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
        // Type-safe reference check (epic #477): a server that references a
        // service account must name an existing one, and must not also carry an
        // inline oauth block. resolve_server_oauth returns those exact errors, so
        // reuse it to reject a cross-wired / dangling config before persisting.
        if let Some(http) = &config.http {
            let path = desktop_assistant_mcp_client::config::default_config_path();
            let accounts = desktop_assistant_mcp_client::config::load_service_accounts(&path)
                .map_err(|e| CoreError::SystemService(e.to_string()))?;
            desktop_assistant_mcp_client::executor::resolve_server_oauth(
                http,
                &accounts,
                &config.name,
            )
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
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

    async fn list_service_accounts(&self) -> Result<Vec<ServiceAccountView>, CoreError> {
        let config_path = desktop_assistant_mcp_client::config::default_config_path();
        let accounts = desktop_assistant_mcp_client::config::load_service_accounts(&config_path)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        // `authorized` = a refresh token for the account is present in secrets.
        let secrets_path = desktop_assistant_mcp_client::config::default_secrets_path();
        let secrets =
            desktop_assistant_mcp_client::config::load_secrets(&secrets_path).unwrap_or_default();
        // The client spawns this argv (detached) to sign an account in; the
        // daemon reports it because only it knows its own binary path (mirrors
        // the MCP-server Sign-in action). Falls back to the bare name on PATH.
        let exe = std::env::current_exe()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| "desktop-assistant".to_string());
        Ok(accounts
            .into_iter()
            .map(|a| {
                let authorized = secrets.contains_key(&a.refresh_token_ref);
                let configure_command =
                    vec![exe.clone(), "--mcp-oauth-login".to_string(), a.id.clone()];
                ServiceAccountView {
                    id: a.id,
                    display_name: a.display_name,
                    client_id: a.client_id,
                    client_secret_ref: a.client_secret_ref,
                    authorize_url: a.authorize_url,
                    token_url: a.token_url,
                    account: a.account,
                    refresh_token_ref: a.refresh_token_ref,
                    granted_scopes: a.granted_scopes,
                    authorized,
                    configure_label: Some("Sign in".to_string()),
                    configure_command,
                }
            })
            .collect())
    }

    async fn upsert_service_account(&self, config_json: String) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        let account: desktop_assistant_mcp_client::executor::ServiceAccount =
            serde_json::from_str(&config_json).map_err(|e| {
                CoreError::SystemService(format!("invalid service account config: {e}"))
            })?;
        // Add-or-replace by id, then persist. save_service_accounts validates the
        // whole set (unique/non-empty id, non-empty client_id, https urls) and
        // fails closed, so a malformed account never lands on disk.
        let config_path = desktop_assistant_mcp_client::config::default_config_path();
        let mut accounts =
            desktop_assistant_mcp_client::config::load_service_accounts(&config_path)
                .map_err(|e| CoreError::SystemService(e.to_string()))?;
        match accounts.iter_mut().find(|a| a.id == account.id) {
            Some(existing) => *existing = account,
            None => accounts.push(account),
        }
        desktop_assistant_mcp_client::config::save_service_accounts(&config_path, &accounts)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        // Push the fresh set into the live executor so a following server upsert
        // that references it resolves without a restart.
        handle.replace_service_accounts(accounts).await;
        Ok(())
    }

    async fn remove_service_account(&self, id: String) -> Result<(), CoreError> {
        let handle = self.mcp_handle()?;
        let config_path = desktop_assistant_mcp_client::config::default_config_path();
        let mut accounts =
            desktop_assistant_mcp_client::config::load_service_accounts(&config_path)
                .map_err(|e| CoreError::SystemService(e.to_string()))?;
        let before = accounts.len();
        accounts.retain(|a| a.id != id);
        if accounts.len() == before {
            return Err(CoreError::SystemService(format!(
                "service account '{id}' not found"
            )));
        }
        desktop_assistant_mcp_client::config::save_service_accounts(&config_path, &accounts)
            .map_err(|e| CoreError::SystemService(e.to_string()))?;
        handle.replace_service_accounts(accounts).await;
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

    /// A path that does not exist resolves to the daemon's default config, whose
    /// default connector (`openai`) is not Anthropic, so `available` is `true`.
    /// That lets these tests exercise the health-vs-`available` mapping without
    /// touching the filesystem.
    fn available_config_path() -> PathBuf {
        PathBuf::from("/nonexistent/desktop-assistant-embed-health-499.toml")
    }

    #[tokio::test]
    async fn get_embeddings_reports_injected_unavailable_over_available_true() {
        // #499: `available` is a shallow connector check and is `true` here, but
        // the startup probe found the backend broken. The reported health MUST be
        // the probe's `Unavailable`, never a false-green derived from `available`.
        let service = DaemonSettingsService::new(available_config_path()).with_embedding_health(
            Arc::new(EmbeddingHealth::Unavailable {
                reason: "HTTP 501 Not Implemented".to_string(),
            }),
        );
        let view = service
            .get_embeddings_settings()
            .await
            .expect("resolving default embeddings settings should succeed");
        assert!(
            view.available,
            "default connector is available (not anthropic)"
        );
        match view.health {
            EmbeddingHealth::Unavailable { reason } => assert!(
                reason.contains("501"),
                "the probe's degraded reason must be surfaced, got: {reason}"
            ),
            other => panic!("expected Unavailable despite available=true, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_embeddings_without_probe_handle_reports_unknown_not_false_green() {
        // Without a probe handle (only in tests / degraded wiring), the honest
        // answer is `Unknown` — health was never determined — NOT the shallow
        // `available -> Ok` false-green that #499 exists to kill.
        let service = DaemonSettingsService::new(available_config_path());
        let view = service
            .get_embeddings_settings()
            .await
            .expect("resolving default embeddings settings should succeed");
        assert!(
            view.available,
            "default connector is available (not anthropic)"
        );
        assert_eq!(
            view.health,
            EmbeddingHealth::Unknown,
            "no probe handle must report Unknown, never a false-green Ok"
        );
    }
}
