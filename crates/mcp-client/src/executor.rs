use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::tool_registry::{ReindexProvider, ToolReindexFn};
use desktop_assistant_core::ports::tools::ToolExecutor;
use tokio::sync::{Mutex, RwLock};

pub use crate::builtin::BuiltinToolService;
use crate::config::save_mcp_configs;
#[cfg(feature = "http")]
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
    /// Reference to a reusable [`ServiceAccount`] by `id` (epic #477).
    /// **Mutually exclusive** with the inline [`Self::oauth`] block — set one or
    /// the other. When set, the account supplies the OAuth client identity +
    /// refresh token, and this server contributes only `url`, [`Self::scopes`],
    /// and the server-level `namespace`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth_account: Option<String>,
    /// Required OAuth scopes for this server when it authenticates via
    /// [`Self::oauth_account`]. Checked against the account's *granted* scopes
    /// for coverage, and unioned across servers at sign-in. Ignored for inline
    /// [`Self::oauth`] (which carries its own scopes) or bearer auth.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
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

/// A named, reusable **outbound** OAuth credential — a *service account*.
///
/// This is the shared OAuth *client identity* that MCP servers reference by
/// [`id`](Self::id) instead of duplicating an inline `[servers.http.oauth]`
/// block per server (epic #477). Gmail/Calendar/Drive all sit behind one Google
/// Cloud OAuth client, so their servers point at one account and sign in once.
///
/// Direction matters: here Adele is the OAuth *client* — it **holds** a refresh
/// token, mints access tokens, and presents them **to** a remote service. That
/// is the opposite of the inbound WebSocket API-auth config, where Adele is the
/// relying party *validating* tokens clients present to it. The two are never
/// interchangeable (see #480 for the type-safe validation).
///
/// Secret **references** (`*_ref`) name entries in `secrets.toml`; secret values
/// never live in this struct nor in `mcp_servers.toml`. `client_id`, the URLs,
/// `account`, and `granted_scopes` are non-secret and stored inline.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceAccount {
    /// Stable, unique identifier a server references (e.g. `oauth_account = "<id>"`).
    pub id: String,
    /// Human-facing name shown in the settings UI (e.g. "Work Google Workspace").
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub display_name: String,
    /// OAuth client identifier (public; safe to store inline).
    pub client_id: String,
    /// Secret ID (secrets.toml) for the OAuth client secret. Omit for public
    /// (PKCE) clients that have no client secret.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_ref: Option<String>,
    /// Authorization endpoint used by the interactive sign-in flow.
    pub authorize_url: String,
    /// Token endpoint, e.g. `https://oauth2.googleapis.com/token`.
    pub token_url: String,
    /// Token-store key — typically the account email. Every server referencing
    /// this account shares the tokens minted under this key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    /// Secret ID (secrets.toml) holding the refresh token minted by sign-in.
    pub refresh_token_ref: String,
    /// Scopes actually granted by the last successful sign-in. A referencing
    /// server's *required* scopes are checked against these for coverage (#479).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub granted_scopes: Vec<String>,
}

/// A server's OAuth configuration after resolving any [`ServiceAccount`]
/// reference — the single shape the connect/status paths consume regardless of
/// whether the server carried an inline `oauth` block or pointed at an account.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOAuth {
    /// Effective config feeding the token provider / connect + login paths.
    pub effective: OAuthServerConfig,
    /// Scopes this server needs granted to function (for coverage checks).
    pub required_scopes: Vec<String>,
    /// Scopes currently granted. For an account, its `granted_scopes`; for an
    /// inline block, its own `scopes` (requested == granted by construction, so
    /// inline servers never regress into a coverage `needs_auth`).
    pub granted_scopes: Vec<String>,
    /// Id of the referenced service account, when resolved from one. Drives the
    /// per-account "Sign in" command; `None` for an inline oauth block.
    pub account_id: Option<String>,
}

/// Resolve a server's effective OAuth config from either its inline `oauth`
/// block (back-compat) or a referenced [`ServiceAccount`]. Returns `Ok(None)`
/// for a non-OAuth HTTP server (bearer or unauthenticated).
///
/// Fails closed with a clear [`McpError::InvalidConfig`] when the server both
/// sets an inline block **and** references an account (ambiguous), or points at
/// an account id that doesn't exist.
pub fn resolve_server_oauth(
    http: &HttpTransportConfig,
    accounts: &[ServiceAccount],
    server_name: &str,
) -> Result<Option<ResolvedOAuth>, McpError> {
    match (&http.oauth, &http.oauth_account) {
        (Some(_), Some(account_id)) => Err(McpError::InvalidConfig(format!(
            "server '{server_name}' sets both an inline [http.oauth] block and \
             oauth_account = '{account_id}'; use exactly one"
        ))),
        (Some(inline), None) => Ok(Some(ResolvedOAuth {
            effective: inline.clone(),
            required_scopes: inline.scopes.clone(),
            granted_scopes: inline.scopes.clone(),
            account_id: None,
        })),
        (None, Some(account_id)) => {
            let acct = accounts
                .iter()
                .find(|a| &a.id == account_id)
                .ok_or_else(|| {
                    McpError::InvalidConfig(format!(
                        "server '{server_name}' references unknown service account '{account_id}'"
                    ))
                })?;
            let effective = OAuthServerConfig {
                client_id: acct.client_id.clone(),
                token_url: acct.token_url.clone(),
                refresh_token_ref: acct.refresh_token_ref.clone(),
                client_secret_ref: acct.client_secret_ref.clone(),
                authorize_url: Some(acct.authorize_url.clone()),
                scopes: http.scopes.clone(),
                // Share minted tokens across every server for this account:
                // key by the account email when known, else the account id.
                account: acct.account.clone().or_else(|| Some(acct.id.clone())),
                refresh_skew_seconds: None,
            };
            Ok(Some(ResolvedOAuth {
                effective,
                required_scopes: http.scopes.clone(),
                granted_scopes: acct.granted_scopes.clone(),
                account_id: Some(acct.id.clone()),
            }))
        }
        (None, None) => Ok(None),
    }
}

/// True when every `required` scope is present in `granted`.
pub fn scopes_covered(required: &[String], granted: &[String]) -> bool {
    required.iter().all(|s| granted.contains(s))
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
    /// Human-facing description of what this server offers, used as the fallback
    /// seed for its provider description in tool-search surfacing when the server
    /// sends no `initialize` instructions. Phase-1 stopgap until every server
    /// emits its own instructions. Optional for TOML back-compat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Resolve a provider's description for tool-search surfacing:
/// server `initialize` instructions, else the configured `description`, else a
/// generic boilerplate naming the server. Trimmed inputs are the caller's job
/// (instructions arrive trimmed from [`crate::parse_server_instructions`]).
pub(crate) fn resolve_provider_description(
    instructions: Option<&str>,
    config_description: Option<&str>,
    server_name: &str,
) -> String {
    instructions
        .or(config_description)
        .map(String::from)
        .unwrap_or_else(|| format!("Tools from the {server_name} MCP server"))
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
    /// The id of the referenced [`ServiceAccount`], when this server resolved its
    /// OAuth from one (epic #477). `None` for an inline oauth block. Distinct
    /// from [`Self::oauth_account`] (the token-store key/email); this is the
    /// config *reference* the editor round-trips.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_account_ref: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub oauth_scopes: Vec<String>,
    /// Non-secret OAuth request fields, echoed so the editor can prefill them on
    /// edit (otherwise a re-save would blank a working server). Secret *values*
    /// are still never included — only these public config fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_token_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_authorize_url: Option<String>,
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
    /// The server's captured `initialize` instructions, cached here (read from
    /// the already-initialized client at construction) so provider grouping for
    /// reindexing can read it without taking the per-client lock.
    instructions: Option<String>,
}

impl ClientHandle {
    fn new(client: McpClient) -> Self {
        let flags = client.list_change_flags();
        let instructions = client.server_instructions().map(String::from);
        Self {
            client: Arc::new(Mutex::new(client)),
            flags,
            instructions,
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
    /// Secrets loaded from secrets.toml, keyed by secret ID. Behind a
    /// `std::sync::RwLock` (not the tokio one) so the sync resolve paths can
    /// read it without `.await`, and the settings surface can swap in a fresh
    /// snapshot via [`McpControlHandle::replace_secrets`] after a token is
    /// minted out-of-band (the `--mcp-oauth-login` flow) or a secret is set.
    secrets: std::sync::RwLock<HashMap<String, String>>,
    /// Reusable outbound OAuth **service accounts** (epic #477), the credentials
    /// an HTTP server may reference by id via `http.oauth_account`. Behind a sync
    /// `RwLock` like [`Self::secrets`] so the resolve paths read it without
    /// `.await`, and the settings surface can swap in a fresh snapshot (e.g. when
    /// a per-account sign-in records new `granted_scopes`) without a restart.
    service_accounts: std::sync::RwLock<Vec<ServiceAccount>>,
    /// Backend for persisting OAuth tokens across restarts (issue #455). The
    /// executor is deliberately agnostic about *where* tokens live: it holds a
    /// [`TokenStore`] trait object, defaulting to an in-memory store, so the
    /// daemon can inject a keyring-backed one (or, later, a server-side store)
    /// without any change here. Shared across all OAuth servers, keyed by
    /// account, so services for one account can share a cached token.
    #[cfg(feature = "http")]
    token_store: Arc<dyn TokenStore>,
    /// Last connection failure per server *name* (stable across add/remove,
    /// unlike the index). Set when a connect attempt fails, cleared on success;
    /// read by `status()` to report `error`/`auth_expired` with detail.
    last_errors: Mutex<HashMap<String, ConnectError>>,
    /// Serializes each enable/disable reindex (#498 review). Overlapping toggles
    /// otherwise race: two [`McpControlHandle::fire_tool_reindex`] calls could
    /// interleave their `cached_tools` snapshot and `reindex(..)` write so an
    /// earlier toggle's slower reindex lands *after* a later one and commits a
    /// stale snapshot as the final index (and it does not self-heal). This mutex
    /// is held across BOTH the snapshot and the closure await, making per-toggle
    /// snapshot+reindex mutually exclusive; the last toggle to serialize
    /// deterministically writes the current tool set. A `tokio::sync::Mutex`
    /// guard held across `.await` is correct and does not trip
    /// `clippy::await_holding_lock` (that lint targets std / parking_lot guards).
    reindex_lock: Mutex<()>,
    /// Injected reindex closure (#498): re-writes the persistent
    /// `tool_definitions` search index with the current connected-tool set
    /// after an enable/disable. `OnceLock` because it is wired once at startup
    /// (via [`McpControlHandle::set_tool_reindex`]) and never reset; unset means
    /// no persistent index to maintain (headless / no-Postgres), so
    /// [`McpControlHandle::fire_tool_reindex`] is a clean no-op. Held as a
    /// boxed closure rather than a store handle so this crate never depends on
    /// `storage`; the "mcp"-source delete-then-reinsert policy lives in the
    /// daemon's closure.
    tool_reindex: OnceLock<ToolReindexFn>,
}

impl McpExecutorState {
    /// Resolve the final environment variables for a server config by merging
    /// `env` (plaintext) with `env_secrets` (looked up from secrets.toml).
    /// Secret references override plaintext if both set the same var.
    fn resolve_env(&self, config: &McpServerConfig) -> Result<HashMap<String, String>, McpError> {
        let mut env = config.env.clone();
        let secrets = self.secrets.read().expect("secrets lock poisoned");
        for (var_name, secret_id) in &config.env_secrets {
            let value = secrets.get(secret_id).ok_or_else(|| {
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
    /// Used only by the HTTP transport's bearer/OAuth secret resolution.
    #[cfg(feature = "http")]
    fn require_secret(
        &self,
        secret_id: &str,
        server_name: &str,
        field: &str,
    ) -> Result<String, McpError> {
        self.secrets
            .read()
            .expect("secrets lock poisoned")
            .get(secret_id)
            .cloned()
            .ok_or_else(|| {
                McpError::UnexpectedResponse(format!(
                    "secret '{secret_id}' (referenced by {field} in server '{server_name}') not found in secrets.toml"
                ))
            })
    }

    /// Resolve the bearer token for an HTTP transport from secrets.toml, if the
    /// config references one. A missing secret is an error (fail closed) rather
    /// than silently connecting unauthenticated.
    #[cfg(feature = "http")]
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
    #[cfg(feature = "http")]
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
            #[cfg(feature = "http")]
            Some(http) => {
                // Resolve any service-account reference into an effective OAuth
                // config. Clone the account snapshot and finish resolving before
                // any `.await` so the std `RwLock` guard never crosses a suspend.
                let resolved = {
                    let accounts = self
                        .service_accounts
                        .read()
                        .expect("service_accounts lock poisoned")
                        .clone();
                    resolve_server_oauth(http, &accounts, &config.name)?
                };
                match resolved {
                    Some(resolved) => {
                        let provider =
                            self.build_token_provider(&resolved.effective, &config.name)?;
                        McpClient::connect_http_oauth(&http.url, Arc::new(provider)).await
                    }
                    None => {
                        let bearer = self.resolve_bearer(http, &config.name)?;
                        McpClient::connect_http(&http.url, bearer).await
                    }
                }
            }
            _ => {
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

    /// Group the current connected MCP tools by provider for reindexing. Modeled
    /// on `tool_namespaces` (keyed on `config.namespace.unwrap_or(config.name)`),
    /// but returns [`ReindexProvider`] with a resolved description
    /// (`initialize` instructions ?? config `description` ?? boilerplate) read
    /// from the per-server [`ClientHandle::instructions`] cache — no per-client
    /// lock needed. Servers contributing zero connected tools are skipped.
    async fn mcp_providers_with_tools(&self) -> Vec<ReindexProvider> {
        let configs = self.configs.read().await;
        let clients = self.clients.read().await;
        let cached = self.cached_tools.lock().await;
        let routing = self.tool_routing.lock().await;

        let mut providers = Vec::new();
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
            if server_tools.is_empty() {
                continue;
            }
            let name = config
                .namespace
                .as_deref()
                .unwrap_or(&config.name)
                .to_string();
            let instructions = clients
                .get(idx)
                .and_then(|slot| slot.as_ref())
                .and_then(|handle| handle.instructions.as_deref());
            let description = resolve_provider_description(
                instructions,
                config.description.as_deref(),
                &config.name,
            );
            providers.push(ReindexProvider {
                name,
                source: "mcp",
                description,
                tools: server_tools,
            });
        }
        providers
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
        // Snapshot the service accounts once so each server's OAuth config can be
        // resolved (inline block or account reference) inside the loop below.
        let accounts = self
            .state
            .service_accounts
            .read()
            .expect("service_accounts lock poisoned")
            .clone();

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
                // Resolve the effective OAuth config (inline block or referenced
                // service account). A *config* error (both set / missing account)
                // is surfaced honestly as an `error` state rather than hidden.
                let resolved = config
                    .http
                    .as_ref()
                    .map(|http| resolve_server_oauth(http, &accounts, &config.name));

                let mut auth_kind: Option<String> = None;
                let mut oauth_authorized: Option<bool> = None;
                let mut oauth_account: Option<String> = None;
                let mut oauth_account_ref: Option<String> = None;
                let mut oauth_scopes: Vec<String> = Vec::new();
                let mut oauth_client_id: Option<String> = None;
                let mut oauth_token_url: Option<String> = None;
                let mut oauth_authorize_url: Option<String> = None;
                let mut configure_label: Option<String> = None;
                let mut configure_command: Vec<String> = Vec::new();
                let mut needs_auth = false;
                let mut config_error: Option<String> = None;
                let mut coverage_detail: Option<String> = None;

                match resolved {
                    None => {} // stdio
                    Some(Err(e)) => {
                        // Misconfigured OAuth server (both set / missing account);
                        // no Sign-in button — signing in can't fix a bad config.
                        auth_kind = Some("oauth".to_string());
                        config_error = Some(e.to_string());
                    }
                    Some(Ok(None)) => {
                        // HTTP without OAuth: static bearer or unauthenticated.
                        let http = config.http.as_ref().expect("http present when resolve ran");
                        auth_kind = Some(
                            if http.auth_bearer_secret.is_some() {
                                "bearer"
                            } else {
                                "none"
                            }
                            .to_string(),
                        );
                    }
                    Some(Ok(Some(resolved))) => {
                        auth_kind = Some("oauth".to_string());
                        let authorized = self
                            .state
                            .secrets
                            .read()
                            .expect("secrets lock poisoned")
                            .contains_key(&resolved.effective.refresh_token_ref);
                        let covered =
                            scopes_covered(&resolved.required_scopes, &resolved.granted_scopes);
                        needs_auth = !authorized || !covered;
                        // Authorized but the account hasn't been granted every
                        // scope this server needs → prompt a re-authorize.
                        if authorized && !covered {
                            let missing: Vec<&str> = resolved
                                .required_scopes
                                .iter()
                                .filter(|s| !resolved.granted_scopes.contains(s))
                                .map(|s| s.as_str())
                                .collect();
                            coverage_detail = Some(format!(
                                "service account not authorized for required scope(s): {}",
                                missing.join(", ")
                            ));
                        }
                        oauth_authorized = Some(authorized);
                        oauth_account = resolved.effective.account.clone();
                        oauth_account_ref = resolved.account_id.clone();
                        oauth_scopes = resolved.required_scopes.clone();
                        oauth_client_id = Some(resolved.effective.client_id.clone());
                        oauth_token_url = Some(resolved.effective.token_url.clone());
                        oauth_authorize_url = resolved.effective.authorize_url.clone();
                        // Sign-in is per *account* when resolved from one (one
                        // login satisfies every server sharing it), else per
                        // server (inline oauth). Phase 2 fills stdio here.
                        let login_target = resolved
                            .account_id
                            .clone()
                            .unwrap_or_else(|| config.name.clone());
                        configure_label = Some("Sign in".to_string());
                        configure_command =
                            vec![exe.clone(), "--mcp-oauth-login".to_string(), login_target];
                    }
                }

                let recorded = last_errors.get(&config.name);

                let status = if !config.enabled {
                    "disabled"
                } else if connected {
                    "running"
                } else if config_error.is_some() {
                    "error"
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

                // Detail precedence: a config error, then a recorded connect
                // failure, then a scope-coverage gap.
                let detail = config_error
                    .or_else(|| recorded.map(|e| e.message.clone()))
                    .or(coverage_detail);

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
                    detail,
                    configure_label,
                    configure_command,
                    auth_kind,
                    oauth_authorized,
                    oauth_account,
                    oauth_account_ref,
                    oauth_scopes,
                    oauth_client_id,
                    oauth_token_url,
                    oauth_authorize_url,
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

    /// Add a server, or replace an existing one with the same name. This is the
    /// settings-UI write path (transport-aware: stdio or http+bearer/oauth).
    /// Unlike [`Self::add_server`] (which errors on a duplicate), it disconnects
    /// any live client for that name, swaps in the new config, persists, and
    /// reconnects when the new config is enabled.
    pub async fn upsert_server(&self, config: McpServerConfig) -> Result<(), McpError> {
        match self.state.find_server_index(&config.name).await {
            Some(idx) => {
                // Replace in place: stop the old client, swap config, persist,
                // then reconnect if enabled. The connect-error for this name is
                // cleared so a fixed config doesn't keep showing a stale failure.
                self.state.disconnect_server(idx).await;
                self.state.clear_connect_error(&config.name).await;
                let auto_start = config.enabled;
                {
                    let mut configs = self.state.configs.write().await;
                    configs[idx] = config;
                }
                self.persist_configs().await?;
                if auto_start {
                    let _ = self.state.connect_server(idx).await;
                }
                let _ = self.state.refresh_all_metadata().await;
                Ok(())
            }
            None => self.add_server(config).await,
        }
    }

    /// Swap in a fresh secrets snapshot (e.g. after `set_mcp_secret` writes
    /// `secrets.toml`, or a reload before a status read). Clears the stale
    /// connect error for any OAuth server whose bootstrap refresh token *just
    /// appeared*, so a freshly signed-in server reads as authorized/`stopped`
    /// rather than a stale `auth_expired`/`error`.
    pub async fn replace_secrets(&self, secrets: HashMap<String, String>) {
        let newly_authorized: Vec<String> = {
            let configs = self.state.configs.read().await;
            let old = self.state.secrets.read().expect("secrets lock poisoned");
            configs
                .iter()
                .filter_map(|c| {
                    let o = c.http.as_ref()?.oauth.as_ref()?;
                    (!old.contains_key(&o.refresh_token_ref)
                        && secrets.contains_key(&o.refresh_token_ref))
                    .then(|| c.name.clone())
                })
                .collect()
        };
        *self.state.secrets.write().expect("secrets lock poisoned") = secrets;
        for name in &newly_authorized {
            self.state.clear_connect_error(name).await;
        }
    }

    /// Swap in a fresh service-account snapshot (e.g. after a per-account sign-in
    /// records new `granted_scopes` in `mcp_servers.toml`, or a reload before a
    /// status read). Runs in a *separate* process, so the live daemon otherwise
    /// wouldn't see the new grants until restart.
    pub async fn replace_service_accounts(&self, accounts: Vec<ServiceAccount>) {
        *self
            .state
            .service_accounts
            .write()
            .expect("service_accounts lock poisoned") = accounts;
    }

    /// Wire the tool-registry reindex closure (#498), called once at startup by
    /// the daemon when a persistent tool index exists. `OnceLock::set` succeeds
    /// only the first time; a second call is silently ignored, which is fine -
    /// the closure is startup-immutable. Left unwired (headless / no-Postgres),
    /// [`Self::fire_tool_reindex`] is a no-op and toggling behaves exactly as
    /// before this change.
    pub fn set_tool_reindex(&self, reindex: ToolReindexFn) {
        let _ = self.state.tool_reindex.set(reindex);
    }

    /// Re-write the persistent tool-search index with the current connected-tool
    /// set (#498). Called at the end of [`Self::enable_server`] /
    /// [`Self::disable_server`] so a hot-toggled server's tools become (or stop
    /// being) discoverable without a daemon restart.
    ///
    /// No-op when no reindex closure is wired. Any error the closure returns is
    /// logged and swallowed: a persistence hiccup must never fail the toggle
    /// (the server has already connected/disconnected in memory), and the next
    /// toggle - or a restart - re-converges the index.
    ///
    /// Concurrency: `reindex_lock` is held across BOTH the `cached_tools`
    /// snapshot and the `reindex(..)` await, so overlapping enable/disable
    /// toggles serialize instead of interleaving. Without it, an earlier
    /// toggle's slower reindex could land *after* a later one and commit a stale
    /// snapshot as the final index; with it, the last toggle to acquire the lock
    /// snapshots the *current* tool set and writes it last (last-writer-wins).
    /// Holding a `tokio::sync::Mutex` guard across `.await` is correct and does
    /// not trip `clippy::await_holding_lock` (that lint targets std /
    /// parking_lot guards).
    pub(crate) async fn fire_tool_reindex(&self) {
        let Some(reindex) = self.state.tool_reindex.get() else {
            return;
        };
        // Snapshot + reindex must be atomic per toggle (see `reindex_lock`).
        let _serialize = self.state.reindex_lock.lock().await;
        let providers = self.state.mcp_providers_with_tools().await;
        if let Err(e) = reindex(providers).await {
            tracing::warn!("tool_definitions reindex after MCP enable/disable failed: {e}");
        }
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
        // Re-write the persistent tool index so the newly-enabled server's tools
        // are discoverable by tool-search without a daemon restart (#498).
        self.fire_tool_reindex().await;

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
        // Prune the disabled server's now-dead rows from the persistent tool
        // index so they stop being advertised to tool-search (#498).
        self.fire_tool_reindex().await;

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
                secrets: std::sync::RwLock::new(HashMap::new()),
                service_accounts: std::sync::RwLock::new(Vec::new()),
                #[cfg(feature = "http")]
                token_store: Arc::new(InMemoryTokenStore::default()),
                last_errors: Mutex::new(HashMap::new()),
                reindex_lock: Mutex::new(()),
                tool_reindex: OnceLock::new(),
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
                secrets: std::sync::RwLock::new(secrets),
                service_accounts: std::sync::RwLock::new(Vec::new()),
                #[cfg(feature = "http")]
                token_store: Arc::new(InMemoryTokenStore::default()),
                last_errors: Mutex::new(HashMap::new()),
                reindex_lock: Mutex::new(()),
                tool_reindex: OnceLock::new(),
            }),
            builtin_tools,
        }
    }

    /// Like [`Self::with_builtin_tools_and_config_path`] but with an injected
    /// [`TokenStore`] for persisting OAuth tokens (issue #455). The daemon
    /// passes a keyring-backed store here; tests and the default path get an
    /// in-memory one. Keeping this a trait object means the persistence backend
    /// (keyring today, possibly a server-side store later) is swappable without
    /// touching the transport or provider code.
    #[cfg(feature = "http")]
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
                secrets: std::sync::RwLock::new(secrets),
                service_accounts: std::sync::RwLock::new(Vec::new()),
                token_store,
                last_errors: Mutex::new(HashMap::new()),
                reindex_lock: Mutex::new(()),
                tool_reindex: OnceLock::new(),
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

    /// The connected MCP tools grouped by provider (server), each with its
    /// resolved description. The startup analogue of what the reindex closure
    /// receives, so a fresh boot registers provider rows immediately (not only
    /// after the first enable/disable toggle).
    pub async fn mcp_providers(&self) -> Vec<ReindexProvider> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        self.state.mcp_providers_with_tools().await
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
    #[cfg(feature = "http")]
    {
        matches!(
            error,
            McpError::OAuth(crate::oauth::OAuthError::InvalidGrant(_))
        )
    }
    // Without the HTTP transport there are no OAuth connects, so no failure can
    // be an expired-token failure.
    #[cfg(not(feature = "http"))]
    {
        let _ = error;
        false
    }
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
    fn mcp_server_config_description_deserializes_from_toml() {
        let toml = r#"
            name = "weather"
            command = "weather-mcp"
            description = "Live weather and forecasts."
        "#;
        let config: McpServerConfig = toml::from_str(toml).expect("parse config");
        assert_eq!(
            config.description.as_deref(),
            Some("Live weather and forecasts."),
            "the description field must round-trip from TOML"
        );
    }

    #[test]
    fn mcp_server_config_description_absent_defaults_none() {
        // TOML back-compat: an existing config with no `description` still parses.
        let toml = r#"
            name = "weather"
            command = "weather-mcp"
        "#;
        let config: McpServerConfig = toml::from_str(toml).expect("parse config");
        assert_eq!(
            config.description, None,
            "an absent description defaults to None"
        );
    }

    #[test]
    fn resolved_description_prefers_instructions() {
        // instructions ?? config.description ?? boilerplate.
        assert_eq!(
            resolve_provider_description(Some("live instructions"), Some("cfg desc"), "weather"),
            "live instructions"
        );
    }

    #[test]
    fn resolved_description_falls_back_to_config() {
        assert_eq!(
            resolve_provider_description(None, Some("cfg desc"), "weather"),
            "cfg desc"
        );
    }

    #[test]
    fn resolved_description_falls_back_to_boilerplate() {
        assert_eq!(
            resolve_provider_description(None, None, "weather"),
            "Tools from the weather MCP server"
        );
    }

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
            description: None,
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
            description: None,
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
            description: None,
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
                description: None,
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
                description: None,
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
                    account: Some("dave@example.com".into()),
                    refresh_skew_seconds: None,
                }),
                oauth_account: None,
                scopes: vec![],
            }),
            description: None,
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
            description: None,
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
        // Non-secret OAuth request fields are echoed so an edit can prefill them.
        assert_eq!(authless.oauth_client_id.as_deref(), Some("cid"));
        assert_eq!(
            authless.oauth_token_url.as_deref(),
            Some("https://oauth2.googleapis.com/token")
        );
        assert_eq!(
            authless.oauth_authorize_url.as_deref(),
            Some("https://accounts.google.com/o/oauth2/v2/auth")
        );

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
            description: None,
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
            description: None,
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
    async fn upsert_server_adds_then_replaces_by_name() {
        let dir = std::env::temp_dir().join("mcp_upsert_server_test");
        let path = dir.join("mcp_servers.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let executor = McpToolExecutor::with_builtin_tools_and_config_path(
            vec![],
            BuiltinToolService::new(),
            path.clone(),
            HashMap::new(),
        );
        let handle = executor.control_handle();

        let disabled_stdio = |command: &str| McpServerConfig {
            name: "weather".into(),
            command: command.into(),
            args: vec!["--v1".into()],
            namespace: None,
            enabled: false, // disabled: no connect attempt, no child spawn
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        };

        // Absent -> added.
        handle
            .upsert_server(disabled_stdio("weather-mcp-v1"))
            .await
            .unwrap();
        let status = handle.status(None).await;
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].command, "weather-mcp-v1");

        // Same name -> replaced in place (not a second entry).
        handle
            .upsert_server(disabled_stdio("weather-mcp-v2"))
            .await
            .unwrap();
        let status = handle.status(None).await;
        assert_eq!(status.len(), 1, "upsert must replace, not duplicate");
        assert_eq!(status[0].command, "weather-mcp-v2");

        // Persisted to TOML so it survives a restart.
        let on_disk = crate::config::load_mcp_configs(&path).unwrap();
        assert_eq!(on_disk.len(), 1);
        assert_eq!(on_disk[0].command, "weather-mcp-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn replace_secrets_flips_authorized_and_clears_stale_error() {
        let executor = McpToolExecutor::with_builtin_tools_config_and_token_store(
            vec![oauth_server("cal", "cal_refresh")],
            BuiltinToolService::new(),
            std::path::PathBuf::new(),
            HashMap::new(),
            std::sync::Arc::new(crate::oauth::InMemoryTokenStore::default()),
        );
        // Simulate a prior failed connect (no token yet).
        executor
            .state
            .record_connect_error("cal", &McpError::Http("connection refused".into()))
            .await;
        let handle = executor.control_handle();

        // No token yet: needs_auth, unauthorized.
        let before = handle.status(Some("cal")).await;
        assert_eq!(before[0].status, "needs_auth");
        assert_eq!(before[0].oauth_authorized, Some(false));

        // Token appears out-of-band (the sign-in flow) -> reload the snapshot.
        let mut secrets = HashMap::new();
        secrets.insert("cal_refresh".to_string(), "rt-value".to_string());
        handle.replace_secrets(secrets).await;

        let after = handle.status(Some("cal")).await;
        assert_eq!(after[0].oauth_authorized, Some(true));
        // Stale error was cleared, so it reads as stopped (authorized, not yet
        // connected) rather than a leftover `error`.
        assert_eq!(after[0].status, "stopped");
        assert!(after[0].detail.is_none());
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

    // --- Service-account resolution (issue #479) ------------------------------

    fn service_account(id: &str, granted: &[&str]) -> ServiceAccount {
        ServiceAccount {
            id: id.into(),
            display_name: "Work Google".into(),
            client_id: "acct-client".into(),
            client_secret_ref: Some("acct_secret".into()),
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            account: Some("user@example.com".into()),
            refresh_token_ref: "acct_refresh".into(),
            granted_scopes: granted.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn resolve_inline_oauth_is_passthrough() {
        // Back-compat: an inline oauth block resolves to itself, required ==
        // granted (so an inline server never regresses into coverage needs_auth).
        let http = HttpTransportConfig {
            url: "https://x/mcp".into(),
            auth_bearer_secret: None,
            oauth: Some(OAuthServerConfig {
                client_id: "cid".into(),
                token_url: "https://oauth2.googleapis.com/token".into(),
                refresh_token_ref: "rt".into(),
                client_secret_ref: None,
                authorize_url: Some("https://auth".into()),
                scopes: vec!["scope.a".into()],
                account: Some("acc".into()),
                refresh_skew_seconds: None,
            }),
            oauth_account: None,
            scopes: vec![],
        };
        let resolved = resolve_server_oauth(&http, &[], "srv").unwrap().unwrap();
        assert_eq!(resolved.account_id, None);
        assert_eq!(resolved.effective.client_id, "cid");
        assert_eq!(resolved.required_scopes, vec!["scope.a".to_string()]);
        assert_eq!(resolved.granted_scopes, vec!["scope.a".to_string()]);
    }

    #[test]
    fn resolve_account_reference_builds_effective_config() {
        let http = HttpTransportConfig {
            url: "https://gmail/mcp".into(),
            auth_bearer_secret: None,
            oauth: None,
            oauth_account: Some("work".into()),
            scopes: vec!["gmail.modify".into()],
        };
        let accounts = vec![service_account("work", &["gmail.modify", "calendar"])];
        let resolved = resolve_server_oauth(&http, &accounts, "gmail")
            .unwrap()
            .unwrap();
        assert_eq!(resolved.account_id.as_deref(), Some("work"));
        assert_eq!(resolved.effective.client_id, "acct-client");
        assert_eq!(resolved.effective.refresh_token_ref, "acct_refresh");
        assert_eq!(
            resolved.effective.client_secret_ref.as_deref(),
            Some("acct_secret")
        );
        assert_eq!(
            resolved.effective.authorize_url.as_deref(),
            Some("https://accounts.google.com/o/oauth2/v2/auth")
        );
        // Token-store key = the account email, so servers sharing the account
        // share the minted token.
        assert_eq!(
            resolved.effective.account.as_deref(),
            Some("user@example.com")
        );
        assert_eq!(resolved.required_scopes, vec!["gmail.modify".to_string()]);
        assert_eq!(
            resolved.granted_scopes,
            vec!["gmail.modify".to_string(), "calendar".to_string()]
        );
    }

    #[test]
    fn resolve_account_keys_by_id_when_no_email() {
        let mut acct = service_account("work", &[]);
        acct.account = None;
        let http = HttpTransportConfig {
            url: "https://gmail/mcp".into(),
            auth_bearer_secret: None,
            oauth: None,
            oauth_account: Some("work".into()),
            scopes: vec![],
        };
        let resolved = resolve_server_oauth(&http, &[acct], "gmail")
            .unwrap()
            .unwrap();
        assert_eq!(resolved.effective.account.as_deref(), Some("work"));
    }

    #[test]
    fn resolve_rejects_both_inline_and_account() {
        let http = HttpTransportConfig {
            url: "https://x/mcp".into(),
            auth_bearer_secret: None,
            oauth: Some(OAuthServerConfig {
                client_id: "cid".into(),
                token_url: "https://t".into(),
                refresh_token_ref: "rt".into(),
                client_secret_ref: None,
                authorize_url: Some("https://a".into()),
                scopes: vec![],
                account: None,
                refresh_skew_seconds: None,
            }),
            oauth_account: Some("work".into()),
            scopes: vec![],
        };
        let err = resolve_server_oauth(&http, &[service_account("work", &[])], "srv").unwrap_err();
        assert!(
            matches!(err, McpError::InvalidConfig(ref m) if m.contains("both")),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_rejects_missing_account() {
        let http = HttpTransportConfig {
            url: "https://x/mcp".into(),
            auth_bearer_secret: None,
            oauth: None,
            oauth_account: Some("nope".into()),
            scopes: vec![],
        };
        let err = resolve_server_oauth(&http, &[service_account("work", &[])], "srv").unwrap_err();
        assert!(
            matches!(err, McpError::InvalidConfig(ref m) if m.contains("nope")),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_non_oauth_http_is_none() {
        let http = HttpTransportConfig {
            url: "https://x/mcp".into(),
            auth_bearer_secret: Some("bearer".into()),
            oauth: None,
            oauth_account: None,
            scopes: vec![],
        };
        assert!(resolve_server_oauth(&http, &[], "srv").unwrap().is_none());
    }

    #[test]
    fn scopes_covered_is_subset_check() {
        let granted = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(scopes_covered(&["a".into(), "c".into()], &granted));
        assert!(scopes_covered(&[], &granted));
        assert!(!scopes_covered(&["a".into(), "z".into()], &granted));
    }

    /// An HTTP server that references a service account by id, requiring
    /// `required` scopes.
    fn account_server(name: &str, account_id: &str, required: &[&str]) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: String::new(),
            args: vec![],
            namespace: Some(name.into()),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: Some(HttpTransportConfig {
                url: format!("https://{name}/mcp"),
                auth_bearer_secret: None,
                oauth: None,
                oauth_account: Some(account_id.into()),
                scopes: required.iter().map(|s| s.to_string()).collect(),
            }),
            description: None,
        }
    }

    /// Build an executor seeded with configs, secrets, and service accounts.
    async fn executor_with_accounts(
        configs: Vec<McpServerConfig>,
        secrets: HashMap<String, String>,
        accounts: Vec<ServiceAccount>,
    ) -> McpToolExecutor {
        let executor = McpToolExecutor::with_builtin_tools_config_and_token_store(
            configs,
            BuiltinToolService::new(),
            std::path::PathBuf::new(),
            secrets,
            std::sync::Arc::new(crate::oauth::InMemoryTokenStore::default()),
        );
        executor
            .control_handle()
            .replace_service_accounts(accounts)
            .await;
        executor
    }

    #[tokio::test]
    async fn account_server_needs_auth_when_no_refresh_token() {
        // Account exists, scopes covered, but no refresh token in secrets yet.
        let configs = vec![account_server("gmail", "work", &["gmail.modify"])];
        let accounts = vec![service_account("work", &["gmail.modify"])];
        let executor = executor_with_accounts(configs, HashMap::new(), accounts).await;

        let status = executor.control_handle().status(Some("gmail")).await;
        let s = &status[0];
        assert_eq!(s.status, "needs_auth");
        assert_eq!(s.auth_kind.as_deref(), Some("oauth"));
        assert_eq!(s.oauth_authorized, Some(false));
        // The referenced account id is surfaced so the editor round-trips it into
        // the account picker (distinct from the token-store `oauth_account`).
        assert_eq!(s.oauth_account_ref.as_deref(), Some("work"));
        // Sign-in targets the *account id*, not the server name — one login
        // satisfies every server sharing it.
        assert!(s.configure_command.iter().any(|a| a == "--mcp-oauth-login"));
        assert!(
            s.configure_command.iter().any(|a| a == "work"),
            "configure_command: {:?}",
            s.configure_command
        );
        assert!(!s.configure_command.iter().any(|a| a == "gmail"));
    }

    #[tokio::test]
    async fn account_server_covered_and_authorized_is_stopped() {
        // Refresh token present + required scopes ⊆ granted → ready (stopped,
        // since we don't actually connect in this unit test).
        let mut secrets = HashMap::new();
        secrets.insert("acct_refresh".to_string(), "rt".to_string());
        let configs = vec![account_server("gmail", "work", &["gmail.modify"])];
        let accounts = vec![service_account("work", &["gmail.modify", "calendar"])];
        let executor = executor_with_accounts(configs, secrets, accounts).await;

        let s = &executor.control_handle().status(Some("gmail")).await[0];
        assert_eq!(s.status, "stopped");
        assert_eq!(s.oauth_authorized, Some(true));
        assert!(s.detail.is_none());
    }

    #[tokio::test]
    async fn account_server_needs_auth_on_scope_gap() {
        // Authorized, but a required scope isn't granted → needs_auth + a
        // coverage detail naming the missing scope.
        let mut secrets = HashMap::new();
        secrets.insert("acct_refresh".to_string(), "rt".to_string());
        let configs = vec![account_server(
            "gmail",
            "work",
            &["gmail.modify", "gmail.send"],
        )];
        let accounts = vec![service_account("work", &["gmail.modify"])];
        let executor = executor_with_accounts(configs, secrets, accounts).await;

        let s = &executor.control_handle().status(Some("gmail")).await[0];
        assert_eq!(s.status, "needs_auth");
        assert_eq!(s.oauth_authorized, Some(true));
        assert!(
            s.detail.as_deref().unwrap().contains("gmail.send"),
            "detail should name the missing scope: {:?}",
            s.detail
        );
    }

    #[tokio::test]
    async fn misconfigured_account_reference_reports_error() {
        // References an account that doesn't exist → honest `error` with detail,
        // no Sign-in button (signing in can't fix a bad config).
        let configs = vec![account_server("gmail", "ghost", &["gmail.modify"])];
        let executor = executor_with_accounts(configs, HashMap::new(), vec![]).await;

        let s = &executor.control_handle().status(Some("gmail")).await[0];
        assert_eq!(s.status, "error");
        assert!(s.detail.as_deref().unwrap().contains("ghost"));
        assert!(s.configure_label.is_none());
    }

    #[tokio::test]
    async fn two_servers_share_one_account_signin_target() {
        // Gmail + Calendar both reference the same account → both point their
        // Sign-in at that account id (shared login).
        let configs = vec![
            account_server("gmail", "work", &["gmail.modify"]),
            account_server("calendar", "work", &["calendar"]),
        ];
        let accounts = vec![service_account("work", &[])];
        let executor = executor_with_accounts(configs, HashMap::new(), accounts).await;

        let status = executor.control_handle().status(None).await;
        for s in &status {
            assert_eq!(s.status, "needs_auth", "{}", s.name);
            assert!(
                s.configure_command.iter().any(|a| a == "work"),
                "{} signs in via the account: {:?}",
                s.name,
                s.configure_command
            );
        }
    }

    // --- runtime enable/disable tool-search reindex (#498) -----------------
    //
    // The injected `ToolReindexFn` is the seam that lets `enable_server` /
    // `disable_server` re-write the persistent `tool_definitions` index without
    // `mcp-client` depending on `storage`. These pin the executor half of the
    // contract: it fires the closure with the current connected-tool set, is a
    // clean no-op when no closure is wired (headless / no-Postgres), and never
    // lets a reindex error fail the toggle.

    use desktop_assistant_core::ports::tool_registry::ToolReindexFn;

    /// A minimal enabled stdio config (never actually connected in these tests).
    fn stdio_cfg(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: "x".into(),
            args: vec![],
            namespace: None,
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        }
    }

    /// Seed the connected state so `mcp_providers_with_tools` produces providers:
    /// each `(server_idx, exposed_name)` adds a cached tool routed to the config
    /// at `server_idx`. The executor must already hold matching configs.
    async fn seed_routed_tools(executor: &McpToolExecutor, tools: &[(usize, &str)]) {
        let mut cached = executor.state.cached_tools.lock().await;
        let mut routing = executor.state.tool_routing.lock().await;
        for (idx, name) in tools {
            cached.push(ToolDefinition::new(
                *name,
                format!("{name} description"),
                serde_json::json!({"type": "object"}),
            ));
            routing.insert((*name).to_string(), (*idx, (*name).to_string()));
        }
    }

    /// Recording reindex closure: captures each call's provider groups so a test
    /// can assert exactly what the executor handed over.
    fn recording_reindex() -> (ToolReindexFn, Arc<Mutex<Vec<Vec<ReindexProvider>>>>) {
        let calls: Arc<Mutex<Vec<Vec<ReindexProvider>>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&calls);
        let f: ToolReindexFn = Arc::new(move |providers| {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                sink.lock().await.push(providers);
                Ok(())
            })
        });
        (f, calls)
    }

    #[tokio::test]
    async fn fire_tool_reindex_hands_over_provider_groups() {
        // Two servers each contributing one tool -> the reindex receives one
        // provider group per server, keyed on its name, carrying its member tool.
        let executor = McpToolExecutor::new(vec![stdio_cfg("servera"), stdio_cfg("serverb")]);
        seed_routed_tools(&executor, &[(0, "servera__alpha"), (1, "serverb__beta")]).await;

        let handle = executor.control_handle();
        let (reindex, calls) = recording_reindex();
        handle.set_tool_reindex(reindex);

        handle.fire_tool_reindex().await;

        let calls = calls.lock().await;
        assert_eq!(calls.len(), 1, "reindex closure must fire exactly once");
        let providers = &calls[0];
        let mut names: Vec<&str> = providers.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["servera", "serverb"],
            "reindex must receive one provider group per connected server"
        );
        let a = providers
            .iter()
            .find(|p| p.name == "servera")
            .expect("servera group present");
        assert_eq!(a.source, "mcp", "MCP providers carry the mcp source");
        assert_eq!(
            a.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["servera__alpha"],
            "each provider group carries only its own member tools"
        );
    }

    #[tokio::test]
    async fn fire_tool_reindex_is_noop_when_reindex_fn_unset() {
        // Headless / no-Postgres path: no reindex closure is ever wired. Both a
        // bare `fire_tool_reindex` and a real enable/disable toggle must stay
        // fine - the no-op fire can never fail or panic the toggle.
        let dir = std::env::temp_dir().join("mcp_noop_reindex_test");
        let path = dir.join("mcp_servers.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let config = McpServerConfig {
            name: "noopserver".into(),
            command: "definitely-not-a-real-mcp-binary".into(),
            args: vec![],
            namespace: None,
            enabled: false,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        };
        let executor = McpToolExecutor::with_builtin_tools_and_config_path(
            vec![config],
            BuiltinToolService::new(),
            path.clone(),
            HashMap::new(),
        );
        let handle = executor.control_handle();

        // A bare fire with no closure wired is a clean no-op (nothing to invoke).
        handle.fire_tool_reindex().await;

        // ...and driving a real toggle stays Ok despite the unwired reindex.
        handle
            .enable_server("noopserver")
            .await
            .expect("enable stays Ok with no reindex wired");
        handle
            .disable_server("noopserver")
            .await
            .expect("disable stays Ok with no reindex wired");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fire_tool_reindex_error_does_not_fail_the_toggle() {
        let executor = McpToolExecutor::new(vec![stdio_cfg("servera")]);
        seed_routed_tools(&executor, &[(0, "servera__alpha")]).await;

        let invoked: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let flag = Arc::clone(&invoked);
        let failing: ToolReindexFn = Arc::new(move |_providers| {
            let flag = Arc::clone(&flag);
            Box::pin(async move {
                *flag.lock().await = true;
                Err(CoreError::Storage("simulated reindex failure".to_string()))
            })
        });

        let handle = executor.control_handle();
        handle.set_tool_reindex(failing);

        // `fire_tool_reindex` returns `()` — the error is logged and swallowed,
        // so `enable_server` / `disable_server` stay `Ok`. Reaching the line
        // after the await is the assertion that the error did not propagate.
        handle.fire_tool_reindex().await;
        assert!(
            *invoked.lock().await,
            "the failing closure must actually have run"
        );
    }

    #[tokio::test]
    async fn enable_and_disable_server_fire_reindex_with_current_tools() {
        // Pins the ACTUAL fix: the `fire_tool_reindex().await` calls at the end
        // of `enable_server` and `disable_server`. The other #498 unit tests
        // drive `fire_tool_reindex` directly, so deleting those two call sites
        // would leave them green; this test drives the toggles themselves and so
        // goes red if either call is removed. (A temp config path is required
        // because both toggles `persist_configs` to TOML.)
        let dir = std::env::temp_dir().join("mcp_toggle_reindex_test");
        let path = dir.join("mcp_servers.toml");
        let _ = std::fs::remove_dir_all(&dir);

        // A disabled stdio server whose command does not exist: enabling it
        // attempts a connect that fails and is swallowed, leaving an empty
        // connected-tool set — exactly the current set the reindex must receive.
        let config = McpServerConfig {
            name: "toggleme".into(),
            command: "definitely-not-a-real-mcp-binary".into(),
            args: vec![],
            namespace: None,
            enabled: false,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        };
        let executor = McpToolExecutor::with_builtin_tools_and_config_path(
            vec![config],
            BuiltinToolService::new(),
            path.clone(),
            HashMap::new(),
        );
        let handle = executor.control_handle();
        let (reindex, calls) = recording_reindex();
        handle.set_tool_reindex(reindex);

        // Enabling drives connect + refresh + the reindex fire at the end. The
        // fake server never connects, so it contributes zero tools and thus no
        // provider group — but the reindex still fires (the pinned behavior).
        handle
            .enable_server("toggleme")
            .await
            .expect("enable_server");
        {
            let calls = calls.lock().await;
            assert_eq!(
                calls.len(),
                1,
                "enable_server must fire the reindex exactly once"
            );
            assert!(
                calls[0].is_empty(),
                "a server that connected zero tools yields no provider group"
            );
        }

        // Disabling drives disconnect + refresh + the reindex fire at the end.
        handle
            .disable_server("toggleme")
            .await
            .expect("disable_server");
        {
            let calls = calls.lock().await;
            assert_eq!(
                calls.len(),
                2,
                "disable_server must fire the reindex exactly once more"
            );
            assert!(
                calls[1].is_empty(),
                "still no provider group after the disable reindex"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn overlapping_toggles_leave_index_consistent() {
        // Two reindexes fired concurrently through the handle must serialize:
        // the final committed index equals the final connected-tool set, never a
        // stale snapshot torn across the interleave (#498 review, FIX 2).
        let executor = McpToolExecutor::new(vec![stdio_cfg("servera")]);
        let handle = executor.control_handle();
        // Both tools route to the single server so the provider grouping picks up
        // whatever cached set is current when a toggle serializes.
        {
            let mut routing = executor.state.tool_routing.lock().await;
            routing.insert("servera__alpha".into(), (0, "alpha".into()));
            routing.insert("serverb__beta".into(), (0, "beta".into()));
        }

        // Models the persistent index: snapshot the handed-over member names,
        // yield (so a concurrent toggle can interleave), then OVERWRITE a shared
        // committed cell — mirroring the daemon's delete-then-reinsert of "mcp".
        let committed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&committed);
        let reindex: ToolReindexFn = Arc::new(move |providers| {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                let names: Vec<String> = providers
                    .iter()
                    .flat_map(|p| p.tools.iter().map(|t| t.name.clone()))
                    .collect();
                tokio::task::yield_now().await;
                *sink.lock().await = names;
                Ok(())
            })
        });
        handle.set_tool_reindex(reindex);

        let tool = |n: &str| ToolDefinition::new(n, "", serde_json::json!({"type": "object"}));

        // Toggle 1 may snapshot the pre-change set...
        *executor.state.cached_tools.lock().await = vec![tool("servera__alpha")];
        let h1 = handle.clone();
        let t1 = tokio::spawn(async move { h1.fire_tool_reindex().await });

        // ...then the connected-tool set changes to its final value before
        // toggle 2. Because snapshot+reindex is serialized, whichever toggle
        // acquires the lock last snapshots this current set and writes it last.
        *executor.state.cached_tools.lock().await =
            vec![tool("servera__alpha"), tool("serverb__beta")];
        let h2 = handle.clone();
        let t2 = tokio::spawn(async move { h2.fire_tool_reindex().await });

        t1.await.expect("toggle 1 task");
        t2.await.expect("toggle 2 task");

        let final_names: Vec<String> = executor
            .state
            .cached_tools
            .lock()
            .await
            .iter()
            .map(|t| t.name.clone())
            .collect();
        assert_eq!(
            *committed.lock().await,
            final_names,
            "the final committed index must equal the current connected-tool set \
             (serialized last-writer-wins, not a torn stale snapshot)"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_grouping_and_diagnostics_never_deadlock() {
        // Regression for the `cached_tools` <-> `tool_routing` lock-order
        // inversion (#520). `tools_by_service` acquired the two mutexes in the
        // opposite order from `tool_namespaces` / `mcp_providers_with_tools`, so
        // two callers racing on a multi-thread runtime could wedge AB-BA (one
        // holds `cached_tools` waiting on `tool_routing`, the other the reverse).
        // With a single canonical order they always make progress; this test
        // times out (fails) if the inversion is ever reintroduced.
        let configs: Vec<McpServerConfig> = (0..8).map(|i| stdio_cfg(&format!("srv{i}"))).collect();
        let executor = Arc::new(McpToolExecutor::new(configs));
        {
            let mut cached = executor.state.cached_tools.lock().await;
            let mut routing = executor.state.tool_routing.lock().await;
            for s in 0..8usize {
                for t in 0..8usize {
                    let name = format!("srv{s}__tool{t}");
                    cached.push(ToolDefinition::new(
                        name.clone(),
                        "d",
                        serde_json::json!({"type": "object"}),
                    ));
                    routing.insert(name, (s, format!("tool{t}")));
                }
            }
        }

        let mut handles = Vec::new();
        for _ in 0..32 {
            let e = Arc::clone(&executor);
            handles.push(tokio::spawn(async move {
                for _ in 0..200 {
                    let _ = e.tools_by_service().await;
                }
            }));
            let e = Arc::clone(&executor);
            handles.push(tokio::spawn(async move {
                for _ in 0..200 {
                    let _ = e.tool_namespaces().await;
                }
            }));
        }

        let join_all = async {
            for h in handles {
                h.await.expect("stress task panicked");
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(30), join_all)
            .await
            .expect(
                "tools_by_service and tool_namespaces deadlocked — \
                 cached_tools/tool_routing lock-order inversion (#520)",
            );
    }
}
