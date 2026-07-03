//! D-Bus adapter for `/org/desktopAssistant/Settings`.
//!
//! Mirrors the in-process settings adapter
//! (`crates/dbus-interface/src/settings.rs`) method-for-method, dispatching
//! each method as an [`api::Command`] over the [`BridgeTransport`] instead of
//! calling `SettingsService` directly. The introspection parity gate
//! (`tests/introspection.rs`) enforces that the surface matches.
//!
//! ## Intentional surface differences (the introspection gate's Q2 carve-out)
//!
//! Three methods on the in-process surface are **deliberately not** mirrored
//! here (#314 Q1/Q2); they have no caller in adele-kde and no `api::Command`
//! wire equivalent:
//!
//! - `GetLlmSettings` / `SetLlmSettings` — the legacy single-connection LLM
//!   surface, removed from `api-model` and supplanted by named connections
//!   (the `Connections` adapter). No KDE caller.
//! - `GenerateWsJwt` — JWT minting is **off D-Bus entirely** (#281). Local
//!   transports no longer use a JWT at all: the bridge reaches the daemon over a
//!   peer-cred-authenticated UDS (#407). A JWT is only a network-door (WebSocket)
//!   concern, where clients obtain one via the daemon's WS `/login`; generation
//!   and validation stay factored in `auth-jwt`.
//!
//! These are the only entries the parity gate is allowed to find missing; see
//! the `Q2_DROPS` list in `tests/introspection.rs`.
//!
//! ## Database settings carry a secret
//!
//! `get_database_settings` returns the connection `url` verbatim, which for a
//! password-auth deployment embeds the DB password. This faithfully mirrors
//! the in-process method (no regression); removing the password from the
//! returned URL is tracked in the secrets-hardening epic (#365), a prerequisite
//! before this surface is exposed to any less-trusted (e.g. remote WS) client.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use serde::{Deserialize, Serialize};
use zbus::object_server::SignalEmitter;
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

/// Wire-format aggregate config returned to D-Bus clients. Strict
/// subset of the in-process adapter's `ConfigData` — only fields the
/// wire `Config` carries are populated; the rest default to "unset"
/// sentinels (empty strings, `false`, `0`) to keep the D-Bus signature
/// stable across daemon upgrades.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, zbus::zvariant::Type)]
pub struct ConfigData {
    pub llm_connector: String,
    pub llm_model: String,
    pub llm_base_url: String,
    pub llm_has_api_key: bool,
    pub embeddings_connector: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_has_api_key: bool,
    pub embeddings_available: bool,
    pub embeddings_is_default: bool,
    pub persistence_enabled: bool,
    pub persistence_remote_url: String,
    pub persistence_remote_name: String,
    pub persistence_push_on_update: bool,
    pub llm_temperature: f64,
    pub llm_top_p: f64,
    pub llm_max_tokens: u32,
    pub llm_hosted_tool_search: i32,
    // Personality (#226): each trait as a 0..=4 ordinal (Never=0..=Always=4),
    // matching the in-process adapter's contract so the KCM sees the same shape
    // regardless of which D-Bus surface it talks to.
    pub personality_professionalism: u32,
    pub personality_warmth: u32,
    pub personality_directness: u32,
    pub personality_enthusiasm: u32,
    pub personality_humor: u32,
    pub personality_sarcasm: u32,
    pub personality_pretentiousness: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, zbus::zvariant::Type)]
pub struct ConfigPatchArgs {
    pub set_llm_connector: bool,
    pub llm_connector: String,
    pub set_llm_model: bool,
    pub llm_model: String,
    pub set_llm_base_url: bool,
    pub llm_base_url: String,
    pub set_llm_api_key: bool,
    pub llm_api_key: String,
    pub set_embeddings_connector: bool,
    pub embeddings_connector: String,
    pub set_embeddings_model: bool,
    pub embeddings_model: String,
    pub set_embeddings_base_url: bool,
    pub embeddings_base_url: String,
    pub set_persistence_enabled: bool,
    pub persistence_enabled: bool,
    pub set_persistence_remote_url: bool,
    pub persistence_remote_url: String,
    pub set_persistence_remote_name: bool,
    pub persistence_remote_name: String,
    pub set_persistence_push_on_update: bool,
    pub persistence_push_on_update: bool,
    pub set_llm_temperature: bool,
    pub llm_temperature: f64,
    pub set_llm_top_p: bool,
    pub llm_top_p: f64,
    pub set_llm_max_tokens: bool,
    pub llm_max_tokens: u32,
    pub set_llm_hosted_tool_search: bool,
    pub llm_hosted_tool_search: i32,
    // Personality (#226): set `set_personality_* = true` to apply the paired
    // 0..=4 ordinal. Validation/clamping happens daemon-side.
    pub set_personality_professionalism: bool,
    pub personality_professionalism: u32,
    pub set_personality_warmth: bool,
    pub personality_warmth: u32,
    pub set_personality_directness: bool,
    pub personality_directness: u32,
    pub set_personality_enthusiasm: bool,
    pub personality_enthusiasm: u32,
    pub set_personality_humor: bool,
    pub personality_humor: u32,
    pub set_personality_sarcasm: bool,
    pub personality_sarcasm: u32,
    pub set_personality_pretentiousness: bool,
    pub personality_pretentiousness: u32,
}

/// Build a `ConfigData` from the wire-level `api::Config`. LLM fields
/// are derived from the named-connections surface in a follow-up;
/// today the bridge surfaces empty strings so the D-Bus signature
/// stays stable.
fn config_from_wire(c: &api::Config) -> ConfigData {
    ConfigData {
        // LLM fields are no longer on the legacy `Config` surface;
        // clients that want them should call the Connections adapter.
        llm_connector: String::new(),
        llm_model: String::new(),
        llm_base_url: String::new(),
        llm_has_api_key: false,
        embeddings_connector: c.embeddings.connector.clone(),
        embeddings_model: c.embeddings.model.clone(),
        embeddings_base_url: c.embeddings.base_url.clone(),
        embeddings_has_api_key: c.embeddings.has_api_key,
        embeddings_available: c.embeddings.available,
        embeddings_is_default: c.embeddings.is_default,
        persistence_enabled: c.persistence.enabled,
        persistence_remote_url: c.persistence.remote_url.clone(),
        persistence_remote_name: c.persistence.remote_name.clone(),
        persistence_push_on_update: c.persistence.push_on_update,
        llm_temperature: -1.0,
        llm_top_p: -1.0,
        llm_max_tokens: 0,
        llm_hosted_tool_search: -1,
        personality_professionalism: c.personality.professionalism.as_ordinal() as u32,
        personality_warmth: c.personality.warmth.as_ordinal() as u32,
        personality_directness: c.personality.directness.as_ordinal() as u32,
        personality_enthusiasm: c.personality.enthusiasm.as_ordinal() as u32,
        personality_humor: c.personality.humor.as_ordinal() as u32,
        personality_sarcasm: c.personality.sarcasm.as_ordinal() as u32,
        personality_pretentiousness: c.personality.pretentiousness.as_ordinal() as u32,
    }
}

pub struct DbusSettingsAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusSettingsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }
}

#[interface(name = "org.desktopAssistant.Settings")]
impl<T: BridgeTransport + 'static> DbusSettingsAdapter<T> {
    /// Write the API key for the daemon's primary LLM.
    async fn set_api_key(&self, api_key: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetApiKey {
                api_key: api_key.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetApiKey result: {other:?}"
            ))),
        }
    }

    /// Return resolved embeddings settings:
    /// `(connector, model, base_url, has_api_key, available, is_default)`.
    async fn get_embeddings_settings(
        &self,
    ) -> fdo::Result<(String, String, String, bool, bool, bool)> {
        let result = self.dispatch(api::Command::GetEmbeddingsSettings).await?;
        match result {
            api::CommandResult::EmbeddingsSettings(v) => Ok((
                v.connector,
                v.model,
                v.base_url,
                v.has_api_key,
                v.available,
                v.is_default,
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetEmbeddingsSettings result: {other:?}"
            ))),
        }
    }

    /// Update embeddings settings; empty strings clear optional fields.
    async fn set_embeddings_settings(
        &self,
        connector: &str,
        model: &str,
        base_url: &str,
    ) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetEmbeddingsSettings {
                connector: normalize(connector),
                model: normalize(model),
                base_url: normalize(base_url),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetEmbeddingsSettings result: {other:?}"
            ))),
        }
    }

    /// Return connector defaults:
    /// `(llm_model, llm_base_url, embeddings_model, embeddings_base_url, embeddings_available, hosted_tool_search_available, backend_llm_model)`.
    async fn get_connector_defaults(
        &self,
        connector: &str,
    ) -> fdo::Result<(String, String, String, String, bool, bool, String)> {
        let result = self
            .dispatch(api::Command::GetConnectorDefaults {
                connector: connector.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::ConnectorDefaults(d) => Ok((
                d.llm_model,
                d.llm_base_url,
                d.embeddings_model,
                d.embeddings_base_url,
                d.embeddings_available,
                d.hosted_tool_search_available,
                d.backend_llm_model,
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetConnectorDefaults result: {other:?}"
            ))),
        }
    }

    /// Return persistence settings:
    /// `(enabled, remote_url, remote_name, push_on_update)`.
    async fn get_persistence_settings(&self) -> fdo::Result<(bool, String, String, bool)> {
        let result = self.dispatch(api::Command::GetPersistenceSettings).await?;
        match result {
            api::CommandResult::PersistenceSettings(p) => {
                Ok((p.enabled, p.remote_url, p.remote_name, p.push_on_update))
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetPersistenceSettings result: {other:?}"
            ))),
        }
    }

    async fn set_persistence_settings(
        &self,
        enabled: bool,
        remote_url: &str,
        remote_name: &str,
        push_on_update: bool,
    ) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetPersistenceSettings {
                enabled,
                remote_url: normalize(remote_url),
                remote_name: normalize(remote_name),
                push_on_update,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetPersistenceSettings result: {other:?}"
            ))),
        }
    }

    /// Return aggregate config. LLM-section fields are populated from
    /// the named-connections surface in a follow-up; today they are
    /// empty / sentinel values so the D-Bus signature is stable.
    async fn get_config(&self) -> fdo::Result<ConfigData> {
        let result = self.dispatch(api::Command::GetConfig).await?;
        match result {
            api::CommandResult::Config(c) => Ok(config_from_wire(&c)),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetConfig result: {other:?}"
            ))),
        }
    }

    /// Apply a partial aggregate config update. Only the embeddings +
    /// persistence sections round-trip through the wire today; the
    /// LLM section is a no-op pending the follow-up that widens the
    /// api-model. Returns the updated config snapshot.
    async fn set_config(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        changes: ConfigPatchArgs,
    ) -> fdo::Result<ConfigData> {
        if changes.set_llm_api_key {
            self.dispatch(api::Command::SetApiKey {
                api_key: changes.llm_api_key.clone(),
            })
            .await?;
        }

        let mut wire_changes = api::ConfigChanges::default();
        if changes.set_embeddings_connector {
            wire_changes.embeddings_connector = Some(changes.embeddings_connector.clone());
        }
        if changes.set_embeddings_model {
            wire_changes.embeddings_model = Some(changes.embeddings_model.clone());
        }
        if changes.set_embeddings_base_url {
            wire_changes.embeddings_base_url = Some(changes.embeddings_base_url.clone());
        }
        if changes.set_persistence_enabled {
            wire_changes.persistence_enabled = Some(changes.persistence_enabled);
        }
        if changes.set_persistence_remote_url {
            wire_changes.persistence_remote_url = Some(changes.persistence_remote_url.clone());
        }
        if changes.set_persistence_remote_name {
            wire_changes.persistence_remote_name = Some(changes.persistence_remote_name.clone());
        }
        if changes.set_persistence_push_on_update {
            wire_changes.persistence_push_on_update = Some(changes.persistence_push_on_update);
        }

        // Personality (#226): translate each set ordinal into the wire level.
        // An out-of-range ordinal is a clean InvalidArgs error rather than a
        // silent clamp, matching the in-process adapter.
        ordinal_to_level(
            changes.set_personality_professionalism,
            changes.personality_professionalism,
            &mut wire_changes.personality_professionalism,
        )?;
        ordinal_to_level(
            changes.set_personality_warmth,
            changes.personality_warmth,
            &mut wire_changes.personality_warmth,
        )?;
        ordinal_to_level(
            changes.set_personality_directness,
            changes.personality_directness,
            &mut wire_changes.personality_directness,
        )?;
        ordinal_to_level(
            changes.set_personality_enthusiasm,
            changes.personality_enthusiasm,
            &mut wire_changes.personality_enthusiasm,
        )?;
        ordinal_to_level(
            changes.set_personality_humor,
            changes.personality_humor,
            &mut wire_changes.personality_humor,
        )?;
        ordinal_to_level(
            changes.set_personality_sarcasm,
            changes.personality_sarcasm,
            &mut wire_changes.personality_sarcasm,
        )?;
        ordinal_to_level(
            changes.set_personality_pretentiousness,
            changes.personality_pretentiousness,
            &mut wire_changes.personality_pretentiousness,
        )?;

        let result = self
            .dispatch(api::Command::SetConfig {
                changes: wire_changes,
            })
            .await?;
        let updated = match result {
            api::CommandResult::Config(c) => config_from_wire(&c),
            other => {
                return Err(fdo::Error::Failed(format!(
                    "unexpected SetConfig result: {other:?}"
                )));
            }
        };

        let emitter = emitter.to_owned();
        // Best-effort: a failed signal isn't a method-call failure.
        if let Err(e) = Self::config_changed(&emitter, &updated).await {
            tracing::warn!("config_changed signal emit failed: {e}");
        }
        Ok(updated)
    }

    /// List configured MCP servers.
    /// Returns: `Vec<(name, command, enabled, status, tool_count)>`.
    ///
    /// Legacy five-tuple form, kept for D-Bus parity. New clients (the KCM MCP
    /// Servers tab) use [`Self::list_mcp_servers_json`], which carries the full
    /// per-server descriptor.
    async fn list_mcp_servers(&self) -> fdo::Result<Vec<(String, String, bool, String, u32)>> {
        let result = self.dispatch(api::Command::ListMcpServers).await?;
        match result {
            api::CommandResult::McpServers(servers) => Ok(servers
                .into_iter()
                .map(|s| (s.name, s.command, s.enabled, s.status, s.tool_count))
                .collect()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListMcpServers result: {other:?}"
            ))),
        }
    }

    /// List configured MCP servers as a JSON array of full descriptors
    /// (MCP-servers-UI epic). Each element is a serialized [`api::McpServerView`]
    /// carrying transport/target/state/detail/configure/oauth fields (never any
    /// secret *value*). Chosen over a widening D-Bus tuple so the config surface
    /// can grow without re-churning the D-Bus signature + introspection gate; the
    /// KCM `JSON.parse`s the reply. Reuses the `ListMcpServers` command.
    async fn list_mcp_servers_json(&self) -> fdo::Result<String> {
        let result = self.dispatch(api::Command::ListMcpServers).await?;
        match result {
            api::CommandResult::McpServers(servers) => serde_json::to_string(&servers)
                .map_err(|e| fdo::Error::Failed(format!("failed to serialize MCP servers: {e}"))),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListMcpServers result: {other:?}"
            ))),
        }
    }

    /// Return database settings: `(url, max_connections)`.
    ///
    /// SECURITY: `url` is the raw connection string and, for password auth,
    /// embeds the password inline — returned verbatim, exactly as the
    /// in-process `get_database_settings` does (#314 / #365 walks this back).
    async fn get_database_settings(&self) -> fdo::Result<(String, u32)> {
        let result = self.dispatch(api::Command::GetDatabaseSettings).await?;
        match result {
            api::CommandResult::DatabaseSettings(v) => Ok((v.url, v.max_connections)),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetDatabaseSettings result: {other:?}"
            ))),
        }
    }

    /// Update database settings. An empty `url` clears it — the daemon's
    /// `SetDatabaseSettings` handler normalizes the empty string, so the
    /// bridge forwards it verbatim (no client-side normalization).
    async fn set_database_settings(&self, url: &str, max_connections: u32) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetDatabaseSettings {
                url: url.to_string(),
                max_connections,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetDatabaseSettings result: {other:?}"
            ))),
        }
    }

    /// Return backend-tasks settings: `(has_separate_llm, llm_connector,
    /// llm_model, llm_base_url, dreaming_enabled, dreaming_interval_secs,
    /// archive_after_days)`.
    async fn get_backend_tasks_settings(
        &self,
    ) -> fdo::Result<(bool, String, String, String, bool, u64, u32)> {
        let result = self.dispatch(api::Command::GetBackendTasksSettings).await?;
        match result {
            api::CommandResult::BackendTasksSettings(v) => Ok((
                v.has_separate_llm,
                v.llm_connector,
                v.llm_model,
                v.llm_base_url,
                v.dreaming_enabled,
                v.dreaming_interval_secs,
                v.archive_after_days,
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetBackendTasksSettings result: {other:?}"
            ))),
        }
    }

    /// Update backend-tasks settings. Empty `llm_connector`/`llm_model`/
    /// `llm_base_url` clear the override; the daemon normalizes the empties, so
    /// the bridge forwards them verbatim.
    async fn set_backend_tasks_settings(
        &self,
        llm_connector: &str,
        llm_model: &str,
        llm_base_url: &str,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
        archive_after_days: u32,
    ) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetBackendTasksSettings {
                llm_connector: llm_connector.to_string(),
                llm_model: llm_model.to_string(),
                llm_base_url: llm_base_url.to_string(),
                dreaming_enabled,
                dreaming_interval_secs,
                archive_after_days,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetBackendTasksSettings result: {other:?}"
            ))),
        }
    }

    /// Add a new MCP server. `args` is whitespace-split into argv (matching the
    /// in-process adapter), and an empty `namespace` clears it.
    async fn add_mcp_server(
        &self,
        name: &str,
        command: &str,
        args: &str,
        namespace: &str,
        enabled: bool,
    ) -> fdo::Result<()> {
        let args: Vec<String> = if args.trim().is_empty() {
            Vec::new()
        } else {
            args.split_whitespace().map(|s| s.to_string()).collect()
        };
        let result = self
            .dispatch(api::Command::AddMcpServer {
                name: name.to_string(),
                command: command.to_string(),
                args,
                namespace: normalize(namespace),
                enabled,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected AddMcpServer result: {other:?}"
            ))),
        }
    }

    /// Remove an MCP server by name.
    async fn remove_mcp_server(&self, name: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::RemoveMcpServer {
                name: name.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected RemoveMcpServer result: {other:?}"
            ))),
        }
    }

    /// Enable or disable an MCP server.
    async fn set_mcp_server_enabled(&self, name: &str, enabled: bool) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetMcpServerEnabled {
                name: name.to_string(),
                enabled,
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetMcpServerEnabled result: {other:?}"
            ))),
        }
    }

    /// Perform an action (status/start/stop/restart) on MCP server(s). An empty
    /// `server` targets all of them. Returns the resulting server list as
    /// `Vec<(name, command, enabled, status, tool_count)>`.
    async fn mcp_server_action(
        &self,
        action: &str,
        server: &str,
    ) -> fdo::Result<Vec<(String, String, bool, String, u32)>> {
        let result = self
            .dispatch(api::Command::McpServerAction {
                action: action.to_string(),
                server: normalize(server),
            })
            .await?;
        match result {
            api::CommandResult::McpServers(servers) => Ok(servers
                .into_iter()
                .map(|s| (s.name, s.command, s.enabled, s.status, s.tool_count))
                .collect()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected McpServerAction result: {other:?}"
            ))),
        }
    }

    /// Return WebSocket auth settings: `(methods, oidc_issuer,
    /// oidc_auth_endpoint, oidc_token_endpoint, oidc_client_id, oidc_scopes)`.
    /// No secret is returned (the HS256 signing key never leaves the daemon).
    async fn get_ws_auth_settings(
        &self,
    ) -> fdo::Result<(Vec<String>, String, String, String, String, String)> {
        let result = self.dispatch(api::Command::GetWsAuthSettings).await?;
        match result {
            api::CommandResult::WsAuthSettings(v) => Ok((
                v.methods,
                v.oidc_issuer,
                v.oidc_auth_endpoint,
                v.oidc_token_endpoint,
                v.oidc_client_id,
                v.oidc_scopes,
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetWsAuthSettings result: {other:?}"
            ))),
        }
    }

    /// Update WebSocket auth settings. Strings are forwarded verbatim (the
    /// in-process adapter does no normalization here either).
    async fn set_ws_auth_settings(
        &self,
        methods: Vec<String>,
        oidc_issuer: &str,
        oidc_auth_endpoint: &str,
        oidc_token_endpoint: &str,
        oidc_client_id: &str,
        oidc_scopes: &str,
    ) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::SetWsAuthSettings {
                methods,
                oidc_issuer: oidc_issuer.to_string(),
                oidc_auth_endpoint: oidc_auth_endpoint.to_string(),
                oidc_token_endpoint: oidc_token_endpoint.to_string(),
                oidc_client_id: oidc_client_id.to_string(),
                oidc_scopes: oidc_scopes.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetWsAuthSettings result: {other:?}"
            ))),
        }
    }

    /// Signal emitted after a successful aggregate config update.
    #[zbus(signal)]
    async fn config_changed(emitter: &SignalEmitter<'_>, config: &ConfigData) -> zbus::Result<()>;
}

fn normalize(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// When `set` is true, validate the 0..=4 `ordinal` into a [`api::PersonalityLevel`]
/// and store it in `slot`; otherwise leave `slot` as `None`. Out-of-range input
/// is a clean `InvalidArgs` error rather than a silent clamp (#226).
fn ordinal_to_level(
    set: bool,
    ordinal: u32,
    slot: &mut Option<api::PersonalityLevel>,
) -> fdo::Result<()> {
    if set {
        let level = u8::try_from(ordinal)
            .ok()
            .and_then(api::PersonalityLevel::from_ordinal)
            .ok_or_else(|| {
                fdo::Error::InvalidArgs(format!(
                    "personality level {ordinal} out of range; expected 0..=4 (Never..=Always)"
                ))
            })?;
        *slot = Some(level);
    }
    Ok(())
}

/// Translate an `api::Config` from a `ConfigChanged` wire event into
/// the D-Bus signal payload. Used by the event forwarder so the same
/// translation lives in one place.
pub fn config_data_from_event(c: &api::Config) -> ConfigData {
    config_from_wire(c)
}

// `to_fdo` is in the file scope so the helper can be re-used by other
// adapter modules without crossing privacy. Quieten the compiler if
// it's unused inside this module's body.
const _: fn(&str) -> fdo::Error = |s| to_fdo(s);

#[cfg(test)]
mod tests {
    //! Behaviour tests for the #315 G2 settings methods: command construction
    //! and result mapping. Signature parity is covered separately by
    //! `tests/introspection.rs`.
    use super::*;
    use crate::transport::BridgeTransport;
    use std::sync::Mutex;

    /// Records each dispatched command and replies with a canned result.
    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: api::CommandResult,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply,
            })
        }

        fn last(&self) -> api::Command {
            self.seen
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("no command dispatched")
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for FakeTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.seen.lock().unwrap().push(command);
            Ok(self.reply.clone())
        }
    }

    fn settings(transport: Arc<FakeTransport>) -> DbusSettingsAdapter<FakeTransport> {
        DbusSettingsAdapter::new(transport)
    }

    #[tokio::test]
    async fn get_database_settings_maps_view_to_tuple() {
        let t = FakeTransport::replying(api::CommandResult::DatabaseSettings(
            api::DatabaseSettingsView {
                url: "postgres://u:p@h/db".into(),
                max_connections: 7,
            },
        ));
        let (url, max) = settings(Arc::clone(&t))
            .get_database_settings()
            .await
            .unwrap();
        assert_eq!(url, "postgres://u:p@h/db");
        assert_eq!(max, 7);
        assert!(matches!(t.last(), api::Command::GetDatabaseSettings));
    }

    #[tokio::test]
    async fn set_database_settings_forwards_url_verbatim() {
        // The bridge does NOT normalize; the daemon collapses the empty string.
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .set_database_settings("   ", 3)
            .await
            .unwrap();
        match t.last() {
            api::Command::SetDatabaseSettings {
                url,
                max_connections,
            } => {
                assert_eq!(url, "   ");
                assert_eq!(max_connections, 3);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test]
    async fn add_mcp_server_splits_args_and_clears_empty_namespace() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .add_mcp_server("srv", "/usr/bin/mcp", "--port 8080  --verbose", "", true)
            .await
            .unwrap();
        match t.last() {
            api::Command::AddMcpServer {
                name,
                command,
                args,
                namespace,
                enabled,
            } => {
                assert_eq!(name, "srv");
                assert_eq!(command, "/usr/bin/mcp");
                assert_eq!(args, vec!["--port", "8080", "--verbose"]);
                assert_eq!(namespace, None);
                assert!(enabled);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test]
    async fn add_mcp_server_keeps_namespace_and_handles_blank_args() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .add_mcp_server("srv", "cmd", "   ", "tools", false)
            .await
            .unwrap();
        match t.last() {
            api::Command::AddMcpServer {
                args,
                namespace,
                enabled,
                ..
            } => {
                assert!(args.is_empty());
                assert_eq!(namespace, Some("tools".to_string()));
                assert!(!enabled);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test]
    async fn mcp_server_action_maps_to_five_tuples_and_clears_blank_server() {
        let t = FakeTransport::replying(api::CommandResult::McpServers(vec![api::McpServerView {
            name: "a".into(),
            command: "c".into(),
            args: vec!["x".into()],
            namespace: Some("ns".into()),
            enabled: true,
            status: "running".into(),
            tool_count: 4,
            ..Default::default()
        }]));
        let rows = settings(Arc::clone(&t))
            .mcp_server_action("restart", "")
            .await
            .unwrap();
        // The D-Bus tuple intentionally drops args/namespace (5-wide for parity).
        assert_eq!(
            rows,
            vec![(
                "a".to_string(),
                "c".to_string(),
                true,
                "running".to_string(),
                4
            )]
        );
        match t.last() {
            api::Command::McpServerAction { action, server } => {
                assert_eq!(action, "restart");
                assert_eq!(server, None);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_ws_auth_settings_maps_view_without_secret() {
        let t = FakeTransport::replying(api::CommandResult::WsAuthSettings(
            api::WsAuthSettingsView {
                methods: vec!["jwt".into(), "oidc".into()],
                oidc_issuer: "iss".into(),
                oidc_auth_endpoint: "auth".into(),
                oidc_token_endpoint: "tok".into(),
                oidc_client_id: "cid".into(),
                oidc_scopes: "openid".into(),
            },
        ));
        let (methods, issuer, _auth, _tok, cid, scopes) = settings(Arc::clone(&t))
            .get_ws_auth_settings()
            .await
            .unwrap();
        assert_eq!(methods, vec!["jwt", "oidc"]);
        assert_eq!(issuer, "iss");
        assert_eq!(cid, "cid");
        assert_eq!(scopes, "openid");
    }

    // --- pure helpers ---------------------------------------------------------

    #[test]
    fn normalize_trims_and_maps_blank_to_none() {
        assert_eq!(normalize(""), None);
        assert_eq!(normalize("   "), None);
        assert_eq!(normalize("  ollama  "), Some("ollama".to_string()));
    }

    #[test]
    fn ordinal_to_level_pins_in_range_unsets_absent_and_rejects_out_of_range() {
        // Not set → the field is left untouched (it stays at the ConfigChanges
        // default of None for that trait); the ordinal is ignored.
        let mut slot = Some(api::PersonalityLevel::Always);
        ordinal_to_level(false, 0, &mut slot).unwrap();
        assert_eq!(
            slot,
            Some(api::PersonalityLevel::Always),
            "an unset trait must not be mutated"
        );

        // In-range bounds pin the level.
        let mut slot = None;
        ordinal_to_level(true, 0, &mut slot).unwrap();
        assert_eq!(slot, Some(api::PersonalityLevel::Never));
        let mut slot = None;
        ordinal_to_level(true, 4, &mut slot).unwrap();
        assert_eq!(slot, Some(api::PersonalityLevel::Always));

        // Out of range (>4) is a clean error, not a silent clamp.
        let mut slot = None;
        assert!(ordinal_to_level(true, 5, &mut slot).is_err());
    }

    #[test]
    fn config_from_wire_carries_embeddings_persistence_and_sentinels_llm() {
        let wire = api::Config {
            embeddings: api::EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "text-embedding-3-small".into(),
                base_url: "https://api.openai.com/v1".into(),
                has_api_key: true,
                available: true,
                is_default: false,
            },
            persistence: api::PersistenceSettingsView {
                enabled: true,
                remote_url: "git@h:r.git".into(),
                remote_name: "origin".into(),
                push_on_update: false,
            },
            personality: api::PersonalitySettingsView::default(),
        };
        let data = config_from_wire(&wire);
        assert_eq!(data.embeddings_connector, "openai");
        assert!(data.embeddings_has_api_key);
        assert!(data.persistence_enabled);
        assert_eq!(data.persistence_remote_name, "origin");
        // The legacy LLM block is no longer on `Config`; the bridge surfaces
        // stable sentinels so the D-Bus signature doesn't drift.
        assert_eq!(data.llm_connector, "");
        assert_eq!(data.llm_max_tokens, 0);
    }

    // --- untested methods: command construction + result mapping --------------

    #[tokio::test]
    async fn set_api_key_builds_command_and_acks() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .set_api_key("sk-123")
            .await
            .unwrap();
        assert!(matches!(t.last(), api::Command::SetApiKey { api_key } if api_key == "sk-123"));
    }

    #[tokio::test]
    async fn get_embeddings_settings_maps_view_to_tuple() {
        let t = FakeTransport::replying(api::CommandResult::EmbeddingsSettings(
            api::EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "m".into(),
                base_url: "u".into(),
                has_api_key: true,
                available: true,
                is_default: false,
            },
        ));
        let (connector, model, _url, has_key, available, is_default) = settings(Arc::clone(&t))
            .get_embeddings_settings()
            .await
            .unwrap();
        assert_eq!(connector, "openai");
        assert_eq!(model, "m");
        assert!(has_key && available && !is_default);
    }

    #[tokio::test]
    async fn set_embeddings_settings_normalizes_blank_fields_to_none() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .set_embeddings_settings("ollama", "  ", "")
            .await
            .unwrap();
        match t.last() {
            api::Command::SetEmbeddingsSettings {
                connector,
                model,
                base_url,
            } => {
                assert_eq!(connector, Some("ollama".to_string()));
                assert_eq!(model, None, "blank model clears the field");
                assert_eq!(base_url, None, "blank base_url clears the field");
            }
            other => panic!("expected SetEmbeddingsSettings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_connector_defaults_passes_connector_and_maps_result() {
        let t = FakeTransport::replying(api::CommandResult::ConnectorDefaults(
            api::ConnectorDefaultsView {
                llm_model: "claude".into(),
                llm_base_url: "lu".into(),
                embeddings_model: "emb".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: true,
                hosted_tool_search_available: false,
                backend_llm_model: "haiku".into(),
            },
        ));
        let (llm_model, ..) = settings(Arc::clone(&t))
            .get_connector_defaults("anthropic")
            .await
            .unwrap();
        assert_eq!(llm_model, "claude");
        assert!(matches!(
            t.last(),
            api::Command::GetConnectorDefaults { connector } if connector == "anthropic"
        ));
    }

    #[tokio::test]
    async fn set_persistence_settings_normalizes_blank_remote_fields() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .set_persistence_settings(true, "", "  ", false)
            .await
            .unwrap();
        match t.last() {
            api::Command::SetPersistenceSettings {
                enabled,
                remote_url,
                remote_name,
                push_on_update,
            } => {
                assert!(enabled);
                assert_eq!(remote_url, None);
                assert_eq!(remote_name, None);
                assert!(!push_on_update);
            }
            other => panic!("expected SetPersistenceSettings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_config_maps_wire_config_via_config_from_wire() {
        let t = FakeTransport::replying(api::CommandResult::Config(api::Config {
            embeddings: api::EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "m".into(),
                base_url: "u".into(),
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
        }));
        let data = settings(Arc::clone(&t)).get_config().await.unwrap();
        assert_eq!(data.embeddings_connector, "openai");
        assert!(matches!(t.last(), api::Command::GetConfig));
    }

    #[tokio::test]
    async fn list_mcp_servers_maps_to_five_tuples() {
        let t = FakeTransport::replying(api::CommandResult::McpServers(vec![api::McpServerView {
            name: "weather".into(),
            command: "weather-mcp".into(),
            args: vec!["serve".into()],
            namespace: None,
            enabled: true,
            status: "running".into(),
            tool_count: 2,
            ..Default::default()
        }]));
        let rows = settings(Arc::clone(&t)).list_mcp_servers().await.unwrap();
        assert_eq!(
            rows,
            vec![(
                "weather".to_string(),
                "weather-mcp".to_string(),
                true,
                "running".to_string(),
                2
            )]
        );
    }

    #[tokio::test]
    async fn list_mcp_servers_json_carries_the_full_descriptor() {
        // MCP-servers-UI epic: the JSON read path must preserve the rich fields
        // the five-tuple form drops (transport/target/state/detail/configure/
        // oauth), so the KCM can render an honest state + Sign-in action.
        let t = FakeTransport::replying(api::CommandResult::McpServers(vec![api::McpServerView {
            name: "gmail-work".into(),
            command: String::new(),
            args: vec![],
            namespace: Some("gmail".into()),
            enabled: true,
            status: "needs_auth".into(),
            tool_count: 0,
            transport: "http".into(),
            target: "https://example.test/mcp".into(),
            detail: None,
            configure_label: Some("Sign in".into()),
            configure_command: vec![
                "/usr/bin/desktop-assistant".into(),
                "--mcp-oauth-login".into(),
                "gmail-work".into(),
            ],
            auth_kind: Some("oauth".into()),
            oauth_authorized: Some(false),
            oauth_account: Some("dave@spadea.tech".into()),
            oauth_scopes: vec!["https://www.googleapis.com/auth/gmail.modify".into()],
        }]));

        let json = settings(Arc::clone(&t))
            .list_mcp_servers_json()
            .await
            .unwrap();
        assert!(matches!(t.last(), api::Command::ListMcpServers));

        let arr: serde_json::Value = serde_json::from_str(&json).unwrap();
        let s = &arr[0];
        assert_eq!(s["name"], "gmail-work");
        assert_eq!(s["transport"], "http");
        assert_eq!(s["target"], "https://example.test/mcp");
        assert_eq!(s["status"], "needs_auth");
        assert_eq!(s["auth_kind"], "oauth");
        assert_eq!(s["oauth_authorized"], false);
        assert_eq!(s["oauth_account"], "dave@spadea.tech");
        assert_eq!(s["configure_label"], "Sign in");
        assert_eq!(s["configure_command"][1], "--mcp-oauth-login");
        // A secret value must never appear anywhere in the descriptor.
        assert!(!json.contains("refresh_token"));
    }

    #[tokio::test]
    async fn get_backend_tasks_settings_maps_view_to_tuple() {
        let t = FakeTransport::replying(api::CommandResult::BackendTasksSettings(
            api::BackendTasksSettingsView {
                has_separate_llm: true,
                llm_connector: "ollama".into(),
                llm_model: "qwen".into(),
                llm_base_url: "u".into(),
                dreaming_enabled: true,
                dreaming_interval_secs: 3600,
                archive_after_days: 30,
            },
        ));
        let (has_separate, connector, model, _u, dreaming, interval, archive) =
            settings(Arc::clone(&t))
                .get_backend_tasks_settings()
                .await
                .unwrap();
        assert!(has_separate && dreaming);
        assert_eq!(
            (connector, model, interval, archive),
            ("ollama".to_string(), "qwen".to_string(), 3600, 30)
        );
    }

    #[tokio::test]
    async fn remove_and_toggle_mcp_server_build_their_commands() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .remove_mcp_server("weather")
            .await
            .unwrap();
        assert!(matches!(t.last(), api::Command::RemoveMcpServer { name } if name == "weather"));

        let t = FakeTransport::replying(api::CommandResult::Ack);
        settings(Arc::clone(&t))
            .set_mcp_server_enabled("weather", false)
            .await
            .unwrap();
        assert!(matches!(
            t.last(),
            api::Command::SetMcpServerEnabled { name, enabled } if name == "weather" && !enabled
        ));
    }

    #[tokio::test]
    async fn unexpected_result_variant_is_a_clean_error() {
        // Representative: a getter that receives the wrong CommandResult must
        // surface an error rather than panic. The same match-arm guards every
        // method.
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = settings(Arc::clone(&t))
            .get_embeddings_settings()
            .await
            .expect_err("a non-EmbeddingsSettings result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }
}
