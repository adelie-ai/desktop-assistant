use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::tools::ToolExecutor;
use tokio::sync::{Mutex, RwLock};

pub use crate::builtin::BuiltinToolService;
use crate::config::save_mcp_configs;
use crate::oauth::{InMemoryTokenStore, OAuthClient, TokenProvider, TokenStore};
use crate::{ListChangeFlags, McpClient, McpError};

fn default_enabled() -> bool {
    true
}

/// HTTP transport settings for reaching a remote MCP server (streamable-HTTP).
///
/// The presence of this table on an [`McpServerConfig`] selects the HTTP
/// transport instead of spawning `command` — so pointing Adele at Google's
/// hosted Gmail/Calendar/Drive/Chat MCP endpoints is one `[servers.http]`
/// table per service, and one server entry per account (with its own
/// `namespace` + `auth_bearer_secret`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HttpTransportConfig {
    /// Remote MCP endpoint URL, e.g. `https://gmailmcp.googleapis.com/mcp/v1`.
    pub url: String,
    /// Secret ID (looked up in secrets.toml) whose value is sent verbatim as an
    /// `Authorization: Bearer` token — a static token the daemon never
    /// refreshes. Prefer [`Self::oauth`] for tokens that expire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_bearer_secret: Option<String>,
    /// When set, authenticate with OAuth 2.0: the daemon exchanges a stored
    /// refresh token for short-lived access tokens and refreshes them on demand
    /// (and on `401`). Takes precedence over [`Self::auth_bearer_secret`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<OAuthServerConfig>,
}

/// OAuth 2.0 settings for a remote MCP server (issue #455 follow-up).
///
/// Secret **references** (`*_ref`) name entries in `secrets.toml`; the secret
/// values themselves never live in `mcp_servers.toml`. `client_id`, the URLs,
/// and scopes are non-secret and stored inline. The interactive login
/// (`desktop-assistant --mcp-oauth-login <server>`) uses `authorize_url` +
/// `scopes` to mint the initial refresh token.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OAuthServerConfig {
    /// OAuth client identifier (public; safe to store inline).
    pub client_id: String,
    /// Token endpoint, e.g. `https://oauth2.googleapis.com/token`.
    pub token_url: String,
    /// Secret ID (secrets.toml) holding the refresh token that bootstraps the
    /// daemon's access-token refresh. Obtain it once via the interactive login.
    pub refresh_token_ref: String,
    /// Secret ID (secrets.toml) for the OAuth client secret. Omit for public
    /// (PKCE) clients that have no client secret.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_ref: Option<String>,
    /// Authorization endpoint (used only by the interactive login flow).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorize_url: Option<String>,
    /// Scopes requested by the interactive login flow.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
    /// Token-store key (defaults to the server `name`). Use the account email
    /// so multiple servers for one account can share a token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    /// Seconds before hard expiry at which to refresh proactively (default 60).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_skew_seconds: Option<i64>,
}

/// Configuration for an MCP server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    /// Command to spawn for a stdio server. Ignored when [`Self::http`] is set.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional namespace prefix. When set, all tools from this server are
    /// exposed as `{namespace}__{tool_name}`. When absent, tool names are
    /// passed through unchanged.
    pub namespace: Option<String>,
    /// Whether this server is enabled. Disabled servers are not started.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Environment variables to set when spawning the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Environment variables whose values are resolved from secrets.toml.
    /// Keys are env var names, values are secret IDs.
    #[serde(default)]
    pub env_secrets: HashMap<String, String>,
    /// When set, reach this server over HTTP (streamable-HTTP) instead of
    /// spawning `command`. See [`HttpTransportConfig`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http: Option<HttpTransportConfig>,
}

/// Status information for an MCP server — the per-server *descriptor* the
/// settings/KCM surface renders. Richer than the old flat status string so the
/// UI can show an honest state and the one action that moves it forward.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpServerStatusInfo {
    pub name: String,
    pub command: String,
    /// Configured launch arguments. Surfaced so the settings layer can project
    /// an `McpServerView` that round-trips what `add_mcp_server` wrote (#314).
    pub args: Vec<String>,
    /// Optional tool-namespace prefix, mirrored from the config so it
    /// round-trips through the settings surface (#314).
    pub namespace: Option<String>,
    pub enabled: bool,
    /// Coarse state: `disabled` | `running` | `stopped` | `needs_auth` |
    /// `auth_expired` | `error`. `stopped` = enabled but not connected with no
    /// captured error; `error`/`auth_expired` = a connect attempt failed.
    pub status: String,
    pub tool_count: u32,
    /// Transport: `"stdio"` or `"http"`.
    pub transport: String,
    /// Human-facing connection target: the command (stdio) or url (http).
    pub target: String,
    /// Last connection error, when the server failed to connect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Label for a Configure/Sign-in button, if the server offers one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configure_label: Option<String>,
    /// argv the client spawns (detached) to configure/sign in. Empty = none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub configure_command: Vec<String>,
    /// For http servers: `"none"` | `"bearer"` | `"oauth"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_kind: Option<String>,
    /// For oauth servers: whether a refresh token is present in secrets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_authorized: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_account: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub oauth_scopes: Vec<String>,
}

/// A captured connection failure, kept per server name so `status()` can report
/// an honest `error`/`auth_expired` state with detail instead of a blank
/// `stopped`. `auth_expired` distinguishes a revoked/expired OAuth refresh token
/// (needs re-login) from a generic connect failure.
#[derive(Debug, Clone)]
struct ConnectError {
    message: String,
    auth_expired: bool,
}

/// A connected MCP server: the client behind its own lock plus a lock-free
/// handle to its list-change flags.
///
/// DS-1: each client has its OWN mutex so a slow or hung tool call on one
/// server cannot block tool calls, status checks, or cache refreshes on any
/// other server. The outer `clients` vector is only ever locked long enough
/// to clone a handle out of it.
struct ClientHandle {
    client: Arc<Mutex<McpClient>>,
    flags: Arc<ListChangeFlags>,
}

impl ClientHandle {
    fn new(client: McpClient) -> Self {
        let flags = client.list_change_flags();
        Self {
            client: Arc::new(Mutex::new(client)),
            flags,
        }
    }
}

/// Shared mutable state for MCP servers, accessible via `McpControlHandle`.
pub struct McpExecutorState {
    configs: RwLock<Vec<McpServerConfig>>,
    /// Connected MCP client instances, indexed by config position. RwLock
    /// because most accesses only need to clone an `Arc` out of a slot;
    /// only connect/disconnect/add/remove take the write lock.
    clients: RwLock<Vec<Option<ClientHandle>>>,
    /// Map from namespaced tool name (`server__original`) to (server index, original tool name).
    tool_routing: Mutex<HashMap<String, (usize, String)>>,
    /// Cached list of all available tools.
    cached_tools: Mutex<Vec<ToolDefinition>>,
    /// Cached metadata for MCP resources across all connected servers.
    cached_resources: Mutex<Vec<serde_json::Value>>,
    /// Cached metadata for MCP prompts across all connected servers.
    cached_prompts: Mutex<Vec<serde_json::Value>>,
    /// Path to the MCP config file (for persisting changes).
    config_path: PathBuf,
    /// Secrets loaded from secrets.toml, keyed by secret ID.
    secrets: HashMap<String, String>,
    /// Backend for persisting OAuth tokens across restarts (issue #455). The
    /// executor is deliberately agnostic about *where* tokens live: it holds a
    /// [`TokenStore`] trait object, defaulting to an in-memory store, so the
    /// daemon can inject a keyring-backed one (or, later, a server-side store)
    /// without any change here. Shared across all OAuth servers, keyed by
    /// account, so services for one account can share a cached token.
    token_store: Arc<dyn TokenStore>,
    /// Last connection failure per server *name* (stable across add/remove,
    /// unlike the index). Set when a connect attempt fails, cleared on success;
    /// read by `status()` to report `error`/`auth_expired` with detail.
    last_errors: Mutex<HashMap<String, ConnectError>>,
}

impl McpExecutorState {
    /// Resolve the final environment variables for a server config by merging
    /// `env` (plaintext) with `env_secrets` (looked up from secrets.toml).
    /// Secret references override plaintext if both set the same var.
    fn resolve_env(&self, config: &McpServerConfig) -> Result<HashMap<String, String>, McpError> {
        let mut env = config.env.clone();
        for (var_name, secret_id) in &config.env_secrets {
            let value = self.secrets.get(secret_id).ok_or_else(|| {
                McpError::UnexpectedResponse(format!(
                    "secret '{}' (referenced by env_secrets.{} in server '{}') not found in secrets.toml",
                    secret_id, var_name, config.name
                ))
            })?;
            env.insert(var_name.clone(), value.clone());
        }
        Ok(env)
    }

    /// Look up a required secret by ID, failing closed with a message that
    /// names the field and server so a misconfiguration is easy to fix.
    fn require_secret(
        &self,
        secret_id: &str,
        server_name: &str,
        field: &str,
    ) -> Result<String, McpError> {
        self.secrets.get(secret_id).cloned().ok_or_else(|| {
            McpError::UnexpectedResponse(format!(
                "secret '{secret_id}' (referenced by {field} in server '{server_name}') not found in secrets.toml"
            ))
        })
    }

    /// Resolve the bearer token for an HTTP transport from secrets.toml, if the
    /// config references one. A missing secret is an error (fail closed) rather
    /// than silently connecting unauthenticated.
    fn resolve_bearer(
        &self,
        http: &HttpTransportConfig,
        server_name: &str,
    ) -> Result<Option<String>, McpError> {
        match &http.auth_bearer_secret {
            None => Ok(None),
            Some(secret_id) => Ok(Some(self.require_secret(
                secret_id,
                server_name,
                "http.auth_bearer_secret",
            )?)),
        }
    }

    /// Build an OAuth [`TokenProvider`] for a server from its config + secrets.
    /// Fails closed if the client secret or bootstrap refresh token is missing.
    fn build_token_provider(
        &self,
        oauth: &OAuthServerConfig,
        server_name: &str,
    ) -> Result<TokenProvider, McpError> {
        let client_secret = match &oauth.client_secret_ref {
            Some(id) => {
                Some(self.require_secret(id, server_name, "http.oauth.client_secret_ref")?)
            }
            None => None,
        };
        let client = OAuthClient::new(&oauth.client_id, client_secret, &oauth.token_url)?;
        let refresh_token = self.require_secret(
            &oauth.refresh_token_ref,
            server_name,
            "http.oauth.refresh_token_ref",
        )?;
        let account_key = oauth
            .account
            .clone()
            .unwrap_or_else(|| server_name.to_string());
        let skew = chrono::Duration::seconds(oauth.refresh_skew_seconds.unwrap_or(60));
        Ok(TokenProvider::bootstrap_from_refresh_token(
            client,
            account_key,
            Arc::clone(&self.token_store),
            skew,
            refresh_token,
        ))
    }

    /// Connect to a server using its configured transport: HTTP when
    /// [`McpServerConfig::http`] is set, otherwise a stdio child process. An
    /// HTTP server authenticates via OAuth when `http.oauth` is present, else a
    /// static bearer.
    async fn connect_client(&self, config: &McpServerConfig) -> Result<McpClient, McpError> {
        match &config.http {
            Some(http) => match &http.oauth {
                Some(oauth) => {
                    let provider = self.build_token_provider(oauth, &config.name)?;
                    McpClient::connect_http_oauth(&http.url, Arc::new(provider)).await
                }
                None => {
                    let bearer = self.resolve_bearer(http, &config.name)?;
                    McpClient::connect_http(&http.url, bearer).await
                }
            },
            None => {
                let env = self.resolve_env(config)?;
                McpClient::connect(&config.command, &config.args, &env).await
            }
        }
    }

    async fn maybe_refresh_metadata(&self) -> Result<(), McpError> {
        let (tools_changed, resources_changed, prompts_changed) = {
            // Flags are read through the shared handles, NOT by locking the
            // clients — a client busy with a slow tool call must not block
            // this check (DS-1).
            let clients = self.clients.read().await;
            (
                clients
                    .iter()
                    .flatten()
                    .any(|handle| handle.flags.tools_changed()),
                clients
                    .iter()
                    .flatten()
                    .any(|handle| handle.flags.resources_changed()),
                clients
                    .iter()
                    .flatten()
                    .any(|handle| handle.flags.prompts_changed()),
            )
        };

        if tools_changed {
            tracing::info!("MCP reported tools/list_changed, refreshing tool cache");
            self.refresh_tool_cache().await?;
        }

        if resources_changed {
            tracing::info!("MCP reported resources/list_changed, refreshing resources cache");
            self.refresh_resources_cache().await?;
        }

        if prompts_changed {
            tracing::info!("MCP reported prompts/list_changed, refreshing prompts cache");
            self.refresh_prompts_cache().await?;
        }

        Ok(())
    }

    async fn refresh_all_metadata(&self) -> Result<(), McpError> {
        self.refresh_tool_cache().await?;
        self.refresh_resources_cache().await?;
        self.refresh_prompts_cache().await?;
        Ok(())
    }

    /// Snapshot the currently connected servers as `(config index, name,
    /// namespace, client)` tuples. Holds the vector/config locks only for
    /// the duration of the clone, so callers can talk to each server
    /// without blocking unrelated operations (DS-1).
    async fn connected_clients(
        &self,
    ) -> Vec<(usize, String, Option<String>, Arc<Mutex<McpClient>>)> {
        let configs = self.configs.read().await;
        let clients = self.clients.read().await;
        clients
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let handle = slot.as_ref()?;
                let config = configs.get(idx)?;
                Some((
                    idx,
                    config.name.clone(),
                    config.namespace.clone(),
                    Arc::clone(&handle.client),
                ))
            })
            .collect()
    }

    async fn refresh_tool_cache(&self) -> Result<(), McpError> {
        let mut all_tools = Vec::new();
        let mut new_routing = HashMap::new();

        for (idx, name, namespace, client) in self.connected_clients().await {
            let mut client = client.lock().await;
            match client.list_tools().await {
                Ok(tools) => {
                    tracing::info!("MCP server '{name}' provides {} tools", tools.len());
                    let ns = namespace.as_deref();
                    for tool in tools {
                        let exposed_name = match ns {
                            Some(prefix) => format!("{}__{}", prefix, tool.name),
                            None => tool.name.clone(),
                        };
                        if ns.is_some() {
                            tracing::debug!("  tool: {} (exposed as {})", tool.name, exposed_name);
                        } else {
                            tracing::debug!("  tool: {}", tool.name);
                        }
                        new_routing.insert(exposed_name.clone(), (idx, tool.name.clone()));
                        all_tools.push(ToolDefinition::new(
                            exposed_name,
                            tool.description,
                            tool.parameters,
                        ));
                    }
                }
                Err(e) => {
                    tracing::warn!("failed to refresh tools from MCP server '{name}': {e}");
                }
            }
        }

        *self.tool_routing.lock().await = new_routing;
        *self.cached_tools.lock().await = all_tools;

        Ok(())
    }

    async fn refresh_resources_cache(&self) -> Result<(), McpError> {
        let mut all_resources = Vec::new();

        for (_idx, name, _ns, client) in self.connected_clients().await {
            let mut client = client.lock().await;
            match client.list_resources().await {
                Ok(resources) => {
                    tracing::info!("MCP server '{name}' provides {} resources", resources.len());
                    all_resources.extend(resources);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!("MCP server '{name}' does not implement resources/list");
                }
                Err(e) => {
                    tracing::warn!("failed to refresh resources from MCP server '{name}': {e}");
                }
            }
        }

        *self.cached_resources.lock().await = all_resources;
        Ok(())
    }

    async fn refresh_prompts_cache(&self) -> Result<(), McpError> {
        let mut all_prompts = Vec::new();

        for (_idx, name, _ns, client) in self.connected_clients().await {
            let mut client = client.lock().await;
            match client.list_prompts().await {
                Ok(prompts) => {
                    tracing::info!("MCP server '{name}' provides {} prompts", prompts.len());
                    all_prompts.extend(prompts);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!("MCP server '{name}' does not implement prompts/list");
                }
                Err(e) => {
                    tracing::warn!("failed to refresh prompts from MCP server '{name}': {e}");
                }
            }
        }

        *self.cached_prompts.lock().await = all_prompts;
        Ok(())
    }

    /// Connect a single server by index.
    /// Record a connection failure for `name` so `status()` can report it.
    async fn record_connect_error(&self, name: &str, err: &McpError) {
        self.last_errors.lock().await.insert(
            name.to_string(),
            ConnectError {
                message: err.to_string(),
                auth_expired: is_auth_expired(err),
            },
        );
    }

    /// Clear any recorded connection failure for `name` (on a successful connect).
    async fn clear_connect_error(&self, name: &str) {
        self.last_errors.lock().await.remove(name);
    }

    async fn connect_server(&self, idx: usize) -> Result<(), McpError> {
        let configs = self.configs.read().await;
        let config = configs.get(idx).ok_or_else(|| {
            McpError::UnexpectedResponse(format!("server index {idx} out of range"))
        })?;

        tracing::info!(
            "connecting to MCP server '{}': {}",
            config.name,
            connect_target(config)
        );

        match self.connect_client(config).await {
            Ok(client) => {
                {
                    let mut clients = self.clients.write().await;
                    clients[idx] = Some(ClientHandle::new(client));
                }
                self.clear_connect_error(&config.name).await;
                Ok(())
            }
            Err(e) => {
                tracing::error!("failed to connect to MCP server '{}': {e}", config.name);
                self.record_connect_error(&config.name, &e).await;
                Err(e)
            }
        }
    }

    /// Disconnect a single server by index.
    async fn disconnect_server(&self, idx: usize) {
        let handle = {
            let mut clients = self.clients.write().await;
            clients.get_mut(idx).and_then(|slot| slot.take())
        };
        if let Some(handle) = handle {
            handle.client.lock().await.shutdown().await;
        }
    }

    /// Find the index of a server by name.
    async fn find_server_index(&self, name: &str) -> Option<usize> {
        let configs = self.configs.read().await;
        configs.iter().position(|c| c.name == name)
    }
}

/// Clonable handle for runtime control of MCP servers.
///
/// Created via `McpToolExecutor::control_handle()` before the executor is
/// moved into `ConversationHandler`.
#[derive(Clone)]
pub struct McpControlHandle {
    state: Arc<McpExecutorState>,
}

impl McpControlHandle {
    /// Get status for one or all servers.
    pub async fn status(&self, server: Option<&str>) -> Vec<McpServerStatusInfo> {
        let configs = self.state.configs.read().await;
        let clients = self.state.clients.read().await;
        let routing = self.state.tool_routing.lock().await;
        let last_errors = self.state.last_errors.lock().await;
        // Absolute path to this binary, for the OAuth "Sign in" launch command
        // the client spawns; fall back to the bare name (assumed on PATH).
        let exe = std::env::current_exe()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| "desktop-assistant".to_string());

        let indices: Vec<usize> = if let Some(name) = server {
            configs
                .iter()
                .position(|c| c.name == name)
                .into_iter()
                .collect()
        } else {
            (0..configs.len()).collect()
        };

        indices
            .into_iter()
            .filter_map(|idx| {
                let config = configs.get(idx)?;
                let connected = clients.get(idx).is_some_and(|c| c.is_some());
                let tool_count = routing
                    .values()
                    .filter(|(server_idx, _)| *server_idx == idx)
                    .count() as u32;

                // Transport + auth summary — never includes secret *values*.
                let (transport, target) = match &config.http {
                    Some(http) => ("http", http.url.clone()),
                    None => ("stdio", config.command.clone()),
                };
                let oauth = config.http.as_ref().and_then(|h| h.oauth.as_ref());
                let (auth_kind, oauth_authorized, oauth_account, oauth_scopes) = match &config.http
                {
                    None => (None, None, None, Vec::new()),
                    Some(http) => match &http.oauth {
                        Some(o) => {
                            let authorized = self.state.secrets.contains_key(&o.refresh_token_ref);
                            (
                                Some("oauth".to_string()),
                                Some(authorized),
                                o.account.clone(),
                                o.scopes.clone(),
                            )
                        }
                        None => {
                            let kind = if http.auth_bearer_secret.is_some() {
                                "bearer"
                            } else {
                                "none"
                            };
                            (Some(kind.to_string()), None, None, Vec::new())
                        }
                    },
                };

                let needs_auth = oauth.is_some() && oauth_authorized == Some(false);
                let recorded = last_errors.get(&config.name);

                let status = if !config.enabled {
                    "disabled"
                } else if connected {
                    "running"
                } else if needs_auth {
                    "needs_auth"
                } else if let Some(err) = recorded {
                    if err.auth_expired {
                        "auth_expired"
                    } else {
                        "error"
                    }
                } else {
                    "stopped"
                };

                // OAuth servers get a "Sign in" action; the client spawns the
                // daemon's own login command (detached). Phase 2 fills stdio
                // `--config-ui` here.
                let (configure_label, configure_command) = if oauth.is_some() {
                    (
                        Some("Sign in".to_string()),
                        vec![
                            exe.clone(),
                            "--mcp-oauth-login".to_string(),
                            config.name.clone(),
                        ],
                    )
                } else {
                    (None, Vec::new())
                };

                Some(McpServerStatusInfo {
                    name: config.name.clone(),
                    command: config.command.clone(),
                    args: config.args.clone(),
                    namespace: config.namespace.clone(),
                    enabled: config.enabled,
                    status: status.to_string(),
                    tool_count,
                    transport: transport.to_string(),
                    target,
                    detail: recorded.map(|e| e.message.clone()),
                    configure_label,
                    configure_command,
                    auth_kind,
                    oauth_authorized,
                    oauth_account,
                    oauth_scopes,
                })
            })
            .collect()
    }

    /// Start one or all servers.
    pub async fn start_server(&self, server: Option<&str>) -> Result<String, McpError> {
        let indices = self.resolve_indices(server).await?;
        let mut started = Vec::new();

        for idx in indices {
            let configs = self.state.configs.read().await;
            let config = &configs[idx];
            if !config.enabled {
                continue;
            }
            let name = config.name.clone();
            drop(configs);

            // Skip if already connected
            {
                let clients = self.state.clients.read().await;
                if clients[idx].is_some() {
                    continue;
                }
            }

            if self.state.connect_server(idx).await.is_ok() {
                started.push(name);
            }
        }

        self.state.refresh_all_metadata().await?;

        if started.is_empty() {
            Ok("no servers started".to_string())
        } else {
            Ok(format!("started: {}", started.join(", ")))
        }
    }

    /// Stop one or all servers.
    pub async fn stop_server(&self, server: Option<&str>) -> Result<String, McpError> {
        let indices = self.resolve_indices(server).await?;
        let mut stopped = Vec::new();

        for idx in indices {
            let was_connected = {
                let clients = self.state.clients.read().await;
                clients[idx].is_some()
            };

            if was_connected {
                let name = {
                    let configs = self.state.configs.read().await;
                    configs[idx].name.clone()
                };
                self.state.disconnect_server(idx).await;
                stopped.push(name);
            }
        }

        self.state.refresh_all_metadata().await?;

        if stopped.is_empty() {
            Ok("no servers stopped".to_string())
        } else {
            Ok(format!("stopped: {}", stopped.join(", ")))
        }
    }

    /// Restart one or all servers.
    pub async fn restart_server(&self, server: Option<&str>) -> Result<String, McpError> {
        self.stop_server(server).await?;
        self.start_server(server).await
    }

    /// Add a server config, persist to TOML, and auto-start if enabled.
    pub async fn add_server(&self, config: McpServerConfig) -> Result<(), McpError> {
        let auto_start = config.enabled;
        let idx = {
            let mut configs = self.state.configs.write().await;

            // Check for duplicate name
            if configs.iter().any(|c| c.name == config.name) {
                return Err(McpError::UnexpectedResponse(format!(
                    "server '{}' already exists",
                    config.name
                )));
            }

            configs.push(config);
            let idx = configs.len() - 1;

            // Extend clients vec to match
            let mut clients = self.state.clients.write().await;
            clients.push(None);

            idx
        };

        self.persist_configs().await?;

        if auto_start {
            let _ = self.state.connect_server(idx).await;
            let _ = self.state.refresh_all_metadata().await;
        }

        Ok(())
    }

    /// Remove a server by name: auto-stop, remove config, persist.
    pub async fn remove_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        // Stop if connected
        self.state.disconnect_server(idx).await;

        // Remove from configs and clients
        {
            let mut configs = self.state.configs.write().await;
            configs.remove(idx);

            let mut clients = self.state.clients.write().await;
            clients.remove(idx);
        }

        // Rebuild routing since indices shifted
        let _ = self.state.refresh_all_metadata().await;
        self.persist_configs().await?;

        Ok(())
    }

    /// Enable a server: set enabled=true, auto-start, persist.
    pub async fn enable_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        {
            let mut configs = self.state.configs.write().await;
            configs[idx].enabled = true;
        }

        self.persist_configs().await?;
        let _ = self.state.connect_server(idx).await;
        let _ = self.state.refresh_all_metadata().await;

        Ok(())
    }

    /// Disable a server: auto-stop, set enabled=false, persist.
    pub async fn disable_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        self.state.disconnect_server(idx).await;

        {
            let mut configs = self.state.configs.write().await;
            configs[idx].enabled = false;
        }

        let _ = self.state.refresh_all_metadata().await;
        self.persist_configs().await?;

        Ok(())
    }

    /// Persist current configs to the TOML file.
    pub async fn persist_configs(&self) -> Result<(), McpError> {
        let configs = self.state.configs.read().await;
        save_mcp_configs(&self.state.config_path, &configs)
    }

    async fn resolve_indices(&self, server: Option<&str>) -> Result<Vec<usize>, McpError> {
        let configs = self.state.configs.read().await;
        if let Some(name) = server {
            let idx = configs.iter().position(|c| c.name == name).ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;
            Ok(vec![idx])
        } else {
            Ok((0..configs.len()).collect())
        }
    }
}

/// Adapter implementing `ToolExecutor` by managing multiple MCP server connections.
/// Routes tool calls to the correct MCP server based on tool name.
pub struct McpToolExecutor {
    state: Arc<McpExecutorState>,
    /// Built-in in-process tools (knowledge base + tool search + sys props).
    builtin_tools: BuiltinToolService,
}

impl McpToolExecutor {
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        Self::with_builtin_tools(configs, BuiltinToolService::new())
    }

    pub fn with_builtin_tools(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
    ) -> Self {
        let clients: Vec<Option<ClientHandle>> = (0..configs.len()).map(|_| None).collect();
        Self {
            state: Arc::new(McpExecutorState {
                configs: RwLock::new(configs),
                clients: RwLock::new(clients),
                tool_routing: Mutex::new(HashMap::new()),
                cached_tools: Mutex::new(Vec::new()),
                cached_resources: Mutex::new(Vec::new()),
                cached_prompts: Mutex::new(Vec::new()),
                config_path: PathBuf::new(),
                secrets: HashMap::new(),
                token_store: Arc::new(InMemoryTokenStore::default()),
                last_errors: Mutex::new(HashMap::new()),
            }),
            builtin_tools,
        }
    }

    pub fn with_builtin_tools_and_config_path(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
        config_path: PathBuf,
        secrets: HashMap<String, String>,
    ) -> Self {
        Self::with_builtin_tools_config_and_token_store(
            configs,
            builtin_tools,
            config_path,
            secrets,
            Arc::new(InMemoryTokenStore::default()),
        )
    }

    /// Like [`Self::with_builtin_tools_and_config_path`] but with an injected
    /// [`TokenStore`] for persisting OAuth tokens (issue #455). The daemon
    /// passes a keyring-backed store here; tests and the default path get an
    /// in-memory one. Keeping this a trait object means the persistence backend
    /// (keyring today, possibly a server-side store later) is swappable without
    /// touching the transport or provider code.
    pub fn with_builtin_tools_config_and_token_store(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
        config_path: PathBuf,
        secrets: HashMap<String, String>,
        token_store: Arc<dyn TokenStore>,
    ) -> Self {
        let clients: Vec<Option<ClientHandle>> = (0..configs.len()).map(|_| None).collect();
        Self {
            state: Arc::new(McpExecutorState {
                configs: RwLock::new(configs),
                clients: RwLock::new(clients),
                tool_routing: Mutex::new(HashMap::new()),
                cached_tools: Mutex::new(Vec::new()),
                cached_resources: Mutex::new(Vec::new()),
                cached_prompts: Mutex::new(Vec::new()),
                config_path,
                secrets,
                token_store,
                last_errors: Mutex::new(HashMap::new()),
            }),
            builtin_tools,
        }
    }

    /// Get a clonable handle for runtime control of MCP servers.
    ///
    /// Call this before moving the executor into `ConversationHandler`.
    pub fn control_handle(&self) -> McpControlHandle {
        McpControlHandle {
            state: Arc::clone(&self.state),
        }
    }

    /// Get a mutable reference to the builtin tool service.
    pub fn builtin_tools_mut(&mut self) -> &mut BuiltinToolService {
        &mut self.builtin_tools
    }

    /// Connect to all configured MCP servers, discover their tools,
    /// and build the routing table.
    pub async fn start(&self) -> Result<(), McpError> {
        {
            let configs = self.state.configs.read().await;
            let mut clients = self.state.clients.write().await;

            for (idx, config) in configs.iter().enumerate() {
                if !config.enabled {
                    tracing::info!("skipping disabled MCP server '{}'", config.name);
                    continue;
                }

                tracing::info!(
                    "connecting to MCP server '{}': {}",
                    config.name,
                    connect_target(config)
                );

                match self.state.connect_client(config).await {
                    Ok(client) => {
                        clients[idx] = Some(ClientHandle::new(client));
                        self.state.clear_connect_error(&config.name).await;
                    }
                    Err(e) => {
                        tracing::error!("failed to connect to MCP server '{}': {e}", config.name);
                        self.state.record_connect_error(&config.name, &e).await;
                    }
                }
            }
        }

        self.state.refresh_all_metadata().await?;
        Ok(())
    }

    pub async fn available_resources(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP resources cache: {e}");
        }
        self.state.cached_resources.lock().await.clone()
    }

    pub async fn available_prompts(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP prompts cache: {e}");
        }
        self.state.cached_prompts.lock().await.clone()
    }

    /// Returns every registered tool as a `(service_name, tool_name)` pair.
    ///
    /// MCP server tools are labelled with their configured service name.
    /// Built-in tools are labelled `"builtin"`.
    /// Intended for startup diagnostics.
    pub async fn tools_by_service(&self) -> Vec<(String, String)> {
        let configs = self.state.configs.read().await;
        let routing = self.state.tool_routing.lock().await;
        let cached = self.state.cached_tools.lock().await;

        let mut entries: Vec<(String, String)> = cached
            .iter()
            .map(|tool| {
                let service = routing
                    .get(&tool.name)
                    .and_then(|(idx, _)| configs.get(*idx))
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                (service, tool.name.clone())
            })
            .collect();

        for tool in self.builtin_tools.tool_definitions() {
            entries.push(("builtin".to_string(), tool.name));
        }

        entries
    }

    /// Return all MCP (non-builtin) tool definitions.
    pub async fn all_mcp_tools(&self) -> Vec<ToolDefinition> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        self.state.cached_tools.lock().await.clone()
    }

    /// Shut down all connected MCP servers.
    pub async fn shutdown(&self) {
        let handles: Vec<ClientHandle> = {
            let mut clients = self.state.clients.write().await;
            clients.iter_mut().filter_map(|slot| slot.take()).collect()
        };
        for handle in handles {
            handle.client.lock().await.shutdown().await;
        }
    }
}

impl ToolExecutor for McpToolExecutor {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        // Only return builtin tools as core. MCP tools are discovered
        // dynamically via builtin_tool_search to avoid bloating every
        // request with dozens of tool definitions.
        self.builtin_tools.tool_definitions()
    }

    async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }

        let mut namespaces = Vec::new();

        // Builtins are always sent as core tools, so skip them here.
        // Only MCP server tools go into deferred namespaces.

        // MCP server tool namespaces — grouped by server
        let configs = self.state.configs.read().await;
        let cached = self.state.cached_tools.lock().await;
        let routing = self.state.tool_routing.lock().await;

        for (idx, config) in configs.iter().enumerate() {
            let server_tools: Vec<ToolDefinition> = cached
                .iter()
                .filter(|tool| {
                    routing
                        .get(&tool.name)
                        .is_some_and(|(server_idx, _)| *server_idx == idx)
                })
                .cloned()
                .collect();

            if !server_tools.is_empty() {
                let ns_name = config
                    .namespace
                    .as_deref()
                    .unwrap_or(&config.name)
                    .to_string();
                namespaces.push(ToolNamespace::new(
                    &ns_name,
                    format!("Tools from the {} MCP server", config.name),
                    server_tools,
                ));
            }
        }

        namespaces
    }

    async fn search_tools(&self, query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        let cached = self.state.cached_tools.lock().await;
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let results: Vec<ToolDefinition> = cached
            .iter()
            .filter(|tool| {
                let name = tool.name.to_lowercase();
                let desc = tool.description.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| name.contains(kw) || desc.contains(kw))
            })
            .cloned()
            .collect();

        Ok(results)
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        // Check builtins first
        if BuiltinToolService::supports_tool(name) {
            return Ok(self
                .builtin_tools
                .tool_definitions()
                .into_iter()
                .find(|t| t.name == name));
        }

        // Check cached MCP tools
        let cached = self.state.cached_tools.lock().await;
        Ok(cached.iter().find(|t| t.name == name).cloned())
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        if BuiltinToolService::supports_tool(name) {
            return self.builtin_tools.execute_tool(name, arguments).await;
        }

        self.state
            .maybe_refresh_metadata()
            .await
            .map_err(|e| CoreError::ToolExecution(format!("failed to refresh tools: {e}")))?;

        let routing = self.state.tool_routing.lock().await;
        let (idx, original_name) = routing
            .get(name)
            .ok_or_else(|| {
                // Find tools with a similar prefix to help the model self-correct.
                let prefix = name.find('_').map(|i| &name[..i]).unwrap_or(name);
                let similar: Vec<&str> = routing
                    .keys()
                    .filter(|k| k.starts_with(prefix))
                    .map(|k| k.as_str())
                    .collect();
                if similar.is_empty() {
                    CoreError::ToolExecution(format!("unknown tool: {name}"))
                } else {
                    CoreError::ToolExecution(format!(
                        "unknown tool: {name}. Similar tools available: {}",
                        similar.join(", ")
                    ))
                }
            })?
            .clone();
        drop(routing);

        // DS-1: clone the per-server handle out of the (briefly read-locked)
        // vector, then await the call holding only THAT server's lock — a
        // slow tool on one server no longer blocks every other server.
        let client = {
            let clients = self.state.clients.read().await;
            clients
                .get(idx)
                .and_then(|slot| slot.as_ref())
                .map(|handle| Arc::clone(&handle.client))
                .ok_or_else(|| {
                    CoreError::ToolExecution(format!(
                        "MCP server for tool '{name}' is not connected"
                    ))
                })?
        };

        client
            .lock()
            .await
            .call_tool(&original_name, arguments)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("tool '{name}' failed: {e}")))
    }
}

fn is_method_not_found(error: &McpError) -> bool {
    matches!(error, McpError::ServerError { code: -32601, .. })
}

/// True if a connect failure was an expired/revoked OAuth refresh token — the
/// one failure that maps to `auth_expired` ("re-run the login") rather than a
/// generic `error`. An OAuth server's initialize handshake makes a real token
/// call, so `invalid_grant` surfaces here as a connect error.
fn is_auth_expired(error: &McpError) -> bool {
    matches!(
        error,
        McpError::OAuth(crate::oauth::OAuthError::InvalidGrant(_))
    )
}

/// Human-facing connection target for logs: the HTTP url when configured,
/// otherwise the stdio command.
fn connect_target(config: &McpServerConfig) -> &str {
    match &config.http {
        Some(http) => http.url.as_str(),
        None => config.command.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_creation_with_empty_configs() {
        let executor = McpToolExecutor::new(vec![]);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let configs = rt.block_on(executor.state.configs.read());
        assert!(configs.is_empty());
    }

    #[test]
    fn server_config_construction() {
        let config = McpServerConfig {
            name: "fileio".into(),
            command: "fileio-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        };
        assert_eq!(config.name, "fileio");
        assert_eq!(config.command, "fileio-mcp");
        assert!(config.args.is_empty());
        assert!(config.namespace.is_none());
        assert!(config.enabled);
    }

    #[test]
    fn server_config_with_namespace() {
        let config = McpServerConfig {
            name: "tickets-jira".into(),
            command: "jira-mcp".into(),
            args: vec![],
            namespace: Some("jira".into()),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        };
        assert_eq!(config.namespace.as_deref(), Some("jira"));
    }

    #[test]
    fn server_config_with_args() {
        let config = McpServerConfig {
            name: "genmcp".into(),
            command: "genmcp".into(),
            args: vec!["--config".into(), "/path/to/config.toml".into()],
            namespace: None,
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        };
        assert_eq!(config.args.len(), 2);
    }

    #[tokio::test]
    async fn executor_no_configs_returns_builtin_tools() {
        let executor = McpToolExecutor::new(vec![]);
        let tools = executor.core_tools().await;
        assert!(!tools.is_empty());
        assert!(
            tools
                .iter()
                .any(|tool| tool.name == "builtin_knowledge_base_write")
        );
        assert!(tools.iter().any(|tool| tool.name == "builtin_tool_search"));
    }

    #[tokio::test]
    async fn executor_no_configs_returns_empty_resources_and_prompts() {
        let executor = McpToolExecutor::new(vec![]);
        let resources = executor.available_resources().await;
        let prompts = executor.available_prompts().await;
        assert!(resources.is_empty());
        assert!(prompts.is_empty());
    }

    #[tokio::test]
    async fn executor_unknown_tool_returns_error() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool("nonexistent", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn executor_includes_builtin_tools() {
        let executor = McpToolExecutor::new(vec![]);
        let tools = executor.core_tools().await;
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"builtin_knowledge_base_write"));
        assert!(names.contains(&"builtin_knowledge_base_search"));
        assert!(names.contains(&"builtin_knowledge_base_delete"));
        assert!(names.contains(&"builtin_tool_search"));
        assert!(names.contains(&"builtin_sys_props"));
    }

    #[tokio::test]
    async fn executor_executes_builtin_sys_props() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn control_handle_status_empty() {
        let executor = McpToolExecutor::new(vec![]);
        let handle = executor.control_handle();
        let status = handle.status(None).await;
        assert!(status.is_empty());
    }

    #[tokio::test]
    async fn control_handle_status_shows_configs() {
        let configs = vec![
            McpServerConfig {
                name: "fileio".into(),
                command: "fileio-mcp".into(),
                args: vec![],
                namespace: None,
                enabled: true,
                env: HashMap::new(),
                env_secrets: HashMap::new(),
                http: None,
            },
            McpServerConfig {
                name: "jira".into(),
                command: "jira-mcp".into(),
                args: vec![],
                namespace: Some("jira".into()),
                enabled: false,
                env: HashMap::new(),
                env_secrets: HashMap::new(),
                http: None,
            },
        ];
        let executor = McpToolExecutor::new(configs);
        let handle = executor.control_handle();
        let status = handle.status(None).await;
        assert_eq!(status.len(), 2);
        assert_eq!(status[0].name, "fileio");
        assert_eq!(status[0].status, "stopped");
        assert!(status[0].enabled);
        assert_eq!(status[1].name, "jira");
        assert_eq!(status[1].status, "disabled");
        assert!(!status[1].enabled);
    }

    fn oauth_server(name: &str, refresh_ref: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: String::new(),
            args: vec![],
            namespace: None,
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: Some(HttpTransportConfig {
                url: "https://calendarmcp.googleapis.com/mcp/v1".into(),
                auth_bearer_secret: None,
                oauth: Some(OAuthServerConfig {
                    client_id: "cid".into(),
                    token_url: "https://oauth2.googleapis.com/token".into(),
                    refresh_token_ref: refresh_ref.into(),
                    client_secret_ref: None,
                    authorize_url: Some("https://accounts.google.com/o/oauth2/v2/auth".into()),
                    scopes: vec!["https://www.googleapis.com/auth/calendar".into()],
                    account: Some("dave@spadea.tech".into()),
                    refresh_skew_seconds: None,
                }),
            }),
        }
    }

    #[tokio::test]
    async fn control_handle_status_reports_rich_states() {
        let stdio = |name: &str, enabled: bool| McpServerConfig {
            name: name.into(),
            command: "some-mcp".into(),
            args: vec![],
            namespace: None,
            enabled,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        };
        let configs = vec![
            oauth_server("authless", "missing_refresh"), // no secret -> needs_auth
            oauth_server("expired", "present_refresh"), // secret present + invalid_grant -> auth_expired
            stdio("broken", true),                      // recorded generic error -> error
            stdio("idle", true),                        // stopped
            stdio("off", false),                        // disabled
        ];
        let mut secrets = HashMap::new();
        secrets.insert("present_refresh".to_string(), "rt".to_string());
        let executor = McpToolExecutor::with_builtin_tools_config_and_token_store(
            configs,
            BuiltinToolService::new(),
            std::path::PathBuf::new(),
            secrets,
            std::sync::Arc::new(crate::oauth::InMemoryTokenStore::default()),
        );
        executor
            .state
            .record_connect_error(
                "expired",
                &McpError::OAuth(crate::oauth::OAuthError::InvalidGrant("revoked".into())),
            )
            .await;
        executor
            .state
            .record_connect_error("broken", &McpError::Http("connection refused".into()))
            .await;

        let status = executor.control_handle().status(None).await;
        let by_name: HashMap<_, _> = status.iter().map(|s| (s.name.as_str(), s)).collect();

        // OAuth server, no refresh token yet -> needs_auth + a Sign in action.
        let authless = by_name["authless"];
        assert_eq!(authless.status, "needs_auth");
        assert_eq!(authless.oauth_authorized, Some(false));
        assert_eq!(authless.auth_kind.as_deref(), Some("oauth"));
        assert_eq!(authless.transport, "http");
        assert_eq!(authless.configure_label.as_deref(), Some("Sign in"));
        assert!(
            authless
                .configure_command
                .iter()
                .any(|a| a == "--mcp-oauth-login")
        );
        assert!(authless.configure_command.iter().any(|a| a == "authless"));

        // OAuth server, token present but the refresh was rejected -> auth_expired.
        let expired = by_name["expired"];
        assert_eq!(expired.status, "auth_expired");
        assert_eq!(expired.oauth_authorized, Some(true));
        assert!(
            expired
                .detail
                .as_deref()
                .unwrap()
                .contains("refresh token is no longer valid"),
            "detail: {:?}",
            expired.detail
        );

        // stdio server with a recorded connect failure -> error, no action.
        let broken = by_name["broken"];
        assert_eq!(broken.status, "error");
        assert_eq!(broken.transport, "stdio");
        assert!(broken.configure_label.is_none());
        assert!(broken.auth_kind.is_none());
        assert!(
            broken
                .detail
                .as_deref()
                .unwrap()
                .contains("connection refused")
        );

        assert_eq!(by_name["idle"].status, "stopped");
        assert!(by_name["idle"].detail.is_none());
        assert_eq!(by_name["off"].status, "disabled");
    }

    #[tokio::test]
    async fn control_handle_status_carries_command_args_namespace() {
        // #314 MCP CRUD round-trip: `status()` must surface the full config
        // (command + args + namespace), not just the command, so the settings
        // layer can project an `McpServerView` that round-trips what was added.
        let configs = vec![McpServerConfig {
            name: "tasks".into(),
            command: "/usr/bin/tasks-mcp".into(),
            args: vec!["--mode".into(), "stdio".into()],
            namespace: Some("jira".into()),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        }];
        let executor = McpToolExecutor::new(configs);
        let handle = executor.control_handle();
        let status = handle.status(None).await;
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].command, "/usr/bin/tasks-mcp");
        assert_eq!(
            status[0].args,
            vec!["--mode".to_string(), "stdio".to_string()]
        );
        assert_eq!(status[0].namespace.as_deref(), Some("jira"));
    }

    #[tokio::test]
    async fn control_handle_status_by_name() {
        let configs = vec![McpServerConfig {
            name: "fileio".into(),
            command: "fileio-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
        }];
        let executor = McpToolExecutor::new(configs);
        let handle = executor.control_handle();
        let status = handle.status(Some("fileio")).await;
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].name, "fileio");

        let empty = handle.status(Some("nonexistent")).await;
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn tool_namespaces_excludes_builtins() {
        let executor = McpToolExecutor::new(vec![]);
        let namespaces = executor.tool_namespaces().await;

        // With no MCP servers, namespaces should be empty —
        // builtins are always core tools, not deferred.
        assert!(namespaces.is_empty());
    }

    #[tokio::test]
    async fn shutdown_non_consuming() {
        let executor = McpToolExecutor::new(vec![]);
        executor.shutdown().await;
        // Can still access after shutdown
        let tools = executor.core_tools().await;
        assert!(!tools.is_empty());
    }
}
