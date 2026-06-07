//! D-Bus adapter for `/org/desktopAssistant/Settings`.
//!
//! ## Coverage gap
//!
//! The in-process settings adapter
//! (`crates/dbus-interface/src/settings.rs`) calls methods on
//! `SettingsService` that have **no `api::Command` equivalent on the
//! wire today**:
//!
//! - `GetLlmSettings` / `SetLlmSettings` (legacy single-connection
//!   surface, removed from `api-model`; supplanted by named
//!   connections via the `Connections` adapter).
//! - `GenerateWsJwt`, `ValidateWsJwt` (the bridge already has a JWT
//!   from the local minter; clients that need their own should call
//!   the minter directly).
//! - `GetDatabaseSettings` / `SetDatabaseSettings`.
//! - `GetBackendTasksSettings` / `SetBackendTasksSettings`.
//! - `GetWsAuthSettings` / `SetWsAuthSettings`.
//! - The MCP server CRUD methods
//!   (`AddMcpServer`/`RemoveMcpServer`/...) are wire-modeled but the
//!   bridge currently only proxies `ListMcpServers`. MCP CRUD is
//!   wired up in a follow-up because the wire surface for
//!   `ServerView` does not yet round-trip the `command` arg the
//!   in-process adapter requires.
//!
//! Per Option A in PR #106, this is acceptable: the daemon still
//! ships the in-process surface, and existing TUI/KCM/plasmoid clients
//! talk to that. The bridge exposes the wire-proxyable subset under a
//! configurable name (default `org.desktopAssistant.Bridge`), and the
//! follow-up issue widens the api-model so the bridge can subsume the
//! full surface.

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
