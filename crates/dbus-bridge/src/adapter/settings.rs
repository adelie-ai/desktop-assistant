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
//! - `GenerateWsJwt` — JWT minting is **off D-Bus entirely** (#281): the
//!   bridge already holds a JWT from the local minter, and any client that
//!   needs one calls the minter directly. JWT generation/validation stays
//!   factored in `auth-jwt` / `jwt-minter`, a WS/web concern.
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
    use tokio::sync::broadcast;

    /// Records each dispatched command and replies with a canned result.
    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: api::CommandResult,
        events_tx: broadcast::Sender<api::Event>,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            let (events_tx, _rx) = broadcast::channel(1);
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply,
                events_tx,
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

        fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
            self.events_tx.subscribe()
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
}
