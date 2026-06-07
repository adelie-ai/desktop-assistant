use std::sync::Arc;

use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::inbound::SettingsService;
use desktop_assistant_core::prompts::PersonalityLevel;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

use crate::resolve_dbus_user_id;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, zbus::zvariant::Type)]
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
    // Personality (#226). Each trait is an integer 0..=4 (Never=0, Rarely=1,
    // Sometimes=2, Often=3, Always=4) so the KCM can bind a 0..=4 slider
    // directly. See `PersonalityLevel::as_ordinal` / `from_ordinal` for the
    // canonical mapping.
    pub personality_professionalism: u32,
    pub personality_warmth: u32,
    pub personality_directness: u32,
    pub personality_enthusiasm: u32,
    pub personality_humor: u32,
    pub personality_sarcasm: u32,
    pub personality_pretentiousness: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, zbus::zvariant::Type)]
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
    // Personality (#226). For each trait, set `set_personality_* = true` to
    // apply the paired ordinal value (0..=4, Never..=Always). An out-of-range
    // ordinal is rejected with an error rather than clamped.
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

#[derive(Debug, Default)]
struct ConfigPatch {
    llm_connector: Option<String>,
    llm_model: Option<String>,
    llm_base_url: Option<String>,
    llm_api_key: Option<String>,
    embeddings_connector: Option<String>,
    embeddings_model: Option<String>,
    embeddings_base_url: Option<String>,
    persistence_enabled: Option<bool>,
    persistence_remote_url: Option<String>,
    persistence_remote_name: Option<String>,
    persistence_push_on_update: Option<bool>,
    llm_temperature: Option<f64>,
    llm_top_p: Option<f64>,
    llm_max_tokens: Option<u32>,
    llm_hosted_tool_search: Option<bool>,
    // Personality (#226): each `Some(ordinal)` overrides that trait. The
    // ordinal is validated against `PersonalityLevel::from_ordinal` when applied.
    personality_professionalism: Option<u32>,
    personality_warmth: Option<u32>,
    personality_directness: Option<u32>,
    personality_enthusiasm: Option<u32>,
    personality_humor: Option<u32>,
    personality_sarcasm: Option<u32>,
    personality_pretentiousness: Option<u32>,
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn to_fdo_error<E: ToString>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

/// D-Bus adapter for assistant settings.
///
/// Exposes both granular settings methods and transport-level aggregate
/// config methods (`GetConfig`/`SetConfig`) for API parity.
pub struct DbusSettingsAdapter<S: SettingsService + 'static> {
    service: Arc<S>,
}

impl<S: SettingsService + 'static> DbusSettingsAdapter<S> {
    pub fn new(service: Arc<S>) -> Self {
        Self { service }
    }

    async fn get_config_tuple(&self) -> fdo::Result<ConfigData> {
        let llm = self
            .service
            .get_llm_settings()
            .await
            .map_err(to_fdo_error)?;
        let embeddings = self
            .service
            .get_embeddings_settings()
            .await
            .map_err(to_fdo_error)?;
        let persistence = self
            .service
            .get_persistence_settings()
            .await
            .map_err(to_fdo_error)?;
        let personality = self
            .service
            .get_personality_settings()
            .await
            .map_err(to_fdo_error)?;

        Ok(ConfigData {
            llm_connector: llm.connector,
            llm_model: llm.model,
            llm_base_url: llm.base_url,
            llm_has_api_key: llm.has_api_key,
            embeddings_connector: embeddings.connector,
            embeddings_model: embeddings.model,
            embeddings_base_url: embeddings.base_url,
            embeddings_has_api_key: embeddings.has_api_key,
            embeddings_available: embeddings.available,
            embeddings_is_default: embeddings.is_default,
            persistence_enabled: persistence.enabled,
            persistence_remote_url: persistence.remote_url,
            persistence_remote_name: persistence.remote_name,
            persistence_push_on_update: persistence.push_on_update,
            llm_temperature: llm.temperature.unwrap_or(-1.0),
            llm_top_p: llm.top_p.unwrap_or(-1.0),
            llm_max_tokens: llm.max_tokens.unwrap_or(0),
            llm_hosted_tool_search: llm.hosted_tool_search.map(|v| v as i32).unwrap_or(-1),
            // Expose each trait as its 0..=4 ordinal for the KCM (#226).
            personality_professionalism: personality.professionalism.as_ordinal() as u32,
            personality_warmth: personality.warmth.as_ordinal() as u32,
            personality_directness: personality.directness.as_ordinal() as u32,
            personality_enthusiasm: personality.enthusiasm.as_ordinal() as u32,
            personality_humor: personality.humor.as_ordinal() as u32,
            personality_sarcasm: personality.sarcasm.as_ordinal() as u32,
            personality_pretentiousness: personality.pretentiousness.as_ordinal() as u32,
        })
    }

    async fn apply_config_patch(&self, patch: ConfigPatch) -> fdo::Result<ConfigData> {
        let ConfigPatch {
            llm_connector,
            llm_model,
            llm_base_url,
            llm_api_key,
            embeddings_connector,
            embeddings_model,
            embeddings_base_url,
            persistence_enabled,
            persistence_remote_url,
            persistence_remote_name,
            persistence_push_on_update,
            llm_temperature,
            llm_top_p,
            llm_max_tokens,
            llm_hosted_tool_search,
            personality_professionalism,
            personality_warmth,
            personality_directness,
            personality_enthusiasm,
            personality_humor,
            personality_sarcasm,
            personality_pretentiousness,
        } = patch;

        let llm_changed = llm_connector.is_some()
            || llm_model.is_some()
            || llm_base_url.is_some()
            || llm_temperature.is_some()
            || llm_top_p.is_some()
            || llm_max_tokens.is_some()
            || llm_hosted_tool_search.is_some();
        if llm_changed {
            let current = self
                .service
                .get_llm_settings()
                .await
                .map_err(to_fdo_error)?;
            let llm_model_set = llm_model.is_some();
            let llm_base_url_set = llm_base_url.is_some();

            let connector = normalize_optional_string(llm_connector).unwrap_or(current.connector);
            let model = if llm_model_set {
                normalize_optional_string(llm_model)
            } else {
                Some(current.model)
            };
            let base_url = if llm_base_url_set {
                normalize_optional_string(llm_base_url)
            } else {
                Some(current.base_url)
            };

            let temperature = if llm_temperature.is_some() {
                llm_temperature
            } else {
                current.temperature
            };
            let top_p = if llm_top_p.is_some() {
                llm_top_p
            } else {
                current.top_p
            };
            let max_tokens = if llm_max_tokens.is_some() {
                llm_max_tokens
            } else {
                current.max_tokens
            };

            let hosted_tool_search = if llm_hosted_tool_search.is_some() {
                llm_hosted_tool_search
            } else {
                current.hosted_tool_search
            };

            self.service
                .set_llm_settings(
                    connector,
                    model,
                    base_url,
                    temperature,
                    top_p,
                    max_tokens,
                    hosted_tool_search,
                )
                .await
                .map_err(to_fdo_error)?;
        }

        if let Some(api_key) = normalize_optional_string(llm_api_key) {
            self.service
                .set_api_key(api_key)
                .await
                .map_err(to_fdo_error)?;
        }

        let embeddings_changed = embeddings_connector.is_some()
            || embeddings_model.is_some()
            || embeddings_base_url.is_some();
        if embeddings_changed {
            let current = self
                .service
                .get_embeddings_settings()
                .await
                .map_err(to_fdo_error)?;
            let embeddings_connector_set = embeddings_connector.is_some();
            let embeddings_model_set = embeddings_model.is_some();
            let embeddings_base_url_set = embeddings_base_url.is_some();

            let connector = if embeddings_connector_set {
                normalize_optional_string(embeddings_connector)
            } else if current.is_default {
                None
            } else {
                Some(current.connector)
            };
            let model = if embeddings_model_set {
                normalize_optional_string(embeddings_model)
            } else {
                Some(current.model)
            };
            let base_url = if embeddings_base_url_set {
                normalize_optional_string(embeddings_base_url)
            } else {
                Some(current.base_url)
            };

            self.service
                .set_embeddings_settings(connector, model, base_url)
                .await
                .map_err(to_fdo_error)?;
        }

        let persistence_changed = persistence_enabled.is_some()
            || persistence_remote_url.is_some()
            || persistence_remote_name.is_some()
            || persistence_push_on_update.is_some();
        if persistence_changed {
            let current = self
                .service
                .get_persistence_settings()
                .await
                .map_err(to_fdo_error)?;
            let persistence_remote_url_set = persistence_remote_url.is_some();
            let persistence_remote_name_set = persistence_remote_name.is_some();

            let enabled = persistence_enabled.unwrap_or(current.enabled);
            let remote_url = if persistence_remote_url_set {
                normalize_optional_string(persistence_remote_url)
            } else {
                Some(current.remote_url)
            };
            let remote_name = if persistence_remote_name_set {
                normalize_optional_string(persistence_remote_name)
            } else {
                Some(current.remote_name)
            };
            let push_on_update = persistence_push_on_update.unwrap_or(current.push_on_update);

            self.service
                .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                .await
                .map_err(to_fdo_error)?;
        }

        // Personality (#226): apply per-trait ordinal overrides. Each
        // `Some(ordinal)` is validated and overlaid onto the current value;
        // an out-of-range ordinal is a clean error rather than a silent clamp.
        let personality_changed = personality_professionalism.is_some()
            || personality_warmth.is_some()
            || personality_directness.is_some()
            || personality_enthusiasm.is_some()
            || personality_humor.is_some()
            || personality_sarcasm.is_some()
            || personality_pretentiousness.is_some();
        if personality_changed {
            let mut p = self
                .service
                .get_personality_settings()
                .await
                .map_err(to_fdo_error)?;
            apply_level(&mut p.professionalism, personality_professionalism)?;
            apply_level(&mut p.warmth, personality_warmth)?;
            apply_level(&mut p.directness, personality_directness)?;
            apply_level(&mut p.enthusiasm, personality_enthusiasm)?;
            apply_level(&mut p.humor, personality_humor)?;
            apply_level(&mut p.sarcasm, personality_sarcasm)?;
            apply_level(&mut p.pretentiousness, personality_pretentiousness)?;
            self.service
                .set_personality_settings(p)
                .await
                .map_err(to_fdo_error)?;
        }

        self.get_config_tuple().await
    }
}

/// Overlay an optional 0..=4 ordinal onto a personality level, validating it.
/// `None` leaves the level unchanged; out-of-range input is a clean error.
fn apply_level(slot: &mut PersonalityLevel, ordinal: Option<u32>) -> fdo::Result<()> {
    if let Some(n) = ordinal {
        let level = u8::try_from(n)
            .ok()
            .and_then(PersonalityLevel::from_ordinal)
            .ok_or_else(|| {
                fdo::Error::InvalidArgs(format!(
                    "personality level {n} out of range; expected 0..=4 (Never..=Always)"
                ))
            })?;
        *slot = level;
    }
    Ok(())
}

#[interface(name = "org.desktopAssistant.Settings")]
impl<S: SettingsService + 'static> DbusSettingsAdapter<S> {
    /// Return non-sensitive LLM settings and whether an API key is available.
    async fn get_llm_settings(
        &self,
    ) -> fdo::Result<(String, String, String, bool, f64, f64, u32, i32)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_llm_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((
                settings.connector,
                settings.model,
                settings.base_url,
                settings.has_api_key,
                settings.temperature.unwrap_or(-1.0),
                settings.top_p.unwrap_or(-1.0),
                settings.max_tokens.unwrap_or(0),
                settings.hosted_tool_search.map(|v| v as i32).unwrap_or(-1),
            ))
        })
        .await
    }

    /// Update non-sensitive LLM settings.
    async fn set_llm_settings(
        &self,
        connector: &str,
        model: &str,
        base_url: &str,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let model = if model.trim().is_empty() {
                None
            } else {
                Some(model.to_string())
            };

            let base_url = if base_url.trim().is_empty() {
                None
            } else {
                Some(base_url.to_string())
            };

            self.service
                .set_llm_settings(
                    connector.to_string(),
                    model,
                    base_url,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Write API key to configured secret backend.
    ///
    /// This is intentionally write-only; there is no D-Bus method to read back secrets.
    async fn set_api_key(&self, api_key: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .set_api_key(api_key.to_string())
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Generate a signed WS JWT for connection authentication.
    ///
    /// Returns the token string. Subject defaults to `desktop-client` when blank.
    async fn generate_ws_jwt(&self, subject: &str) -> fdo::Result<String> {
        with_user_id(resolve_dbus_user_id(), async {
            let subject = if subject.trim().is_empty() {
                None
            } else {
                Some(subject.to_string())
            };

            self.service
                .generate_ws_jwt(subject)
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Return resolved embeddings settings.
    ///
    /// Returns: (connector, model, base_url, has_api_key, available, is_default)
    async fn get_embeddings_settings(
        &self,
    ) -> fdo::Result<(String, String, String, bool, bool, bool)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_embeddings_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((
                settings.connector,
                settings.model,
                settings.base_url,
                settings.has_api_key,
                settings.available,
                settings.is_default,
            ))
        })
        .await
    }

    /// Update embeddings settings. Empty connector clears override (reverts to LLM default).
    async fn set_embeddings_settings(
        &self,
        connector: &str,
        model: &str,
        base_url: &str,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let connector = if connector.trim().is_empty() {
                None
            } else {
                Some(connector.to_string())
            };

            let model = if model.trim().is_empty() {
                None
            } else {
                Some(model.to_string())
            };

            let base_url = if base_url.trim().is_empty() {
                None
            } else {
                Some(base_url.to_string())
            };

            self.service
                .set_embeddings_settings(connector, model, base_url)
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Return connector defaults.
    ///
    /// Returns: (llm_model, llm_base_url, embeddings_model, embeddings_base_url, embeddings_available)
    async fn get_connector_defaults(
        &self,
        connector: &str,
    ) -> fdo::Result<(String, String, String, String, bool, bool, String)> {
        with_user_id(resolve_dbus_user_id(), async {
            let defaults = self
                .service
                .get_connector_defaults(connector.to_string())
                .await
                .map_err(to_fdo_error)?;

            Ok((
                defaults.llm_model,
                defaults.llm_base_url,
                defaults.embeddings_model,
                defaults.embeddings_base_url,
                defaults.embeddings_available,
                defaults.hosted_tool_search_available,
                defaults.backend_llm_model,
            ))
        })
        .await
    }

    /// Return git persistence settings.
    ///
    /// Returns: (enabled, remote_url, remote_name, push_on_update)
    async fn get_persistence_settings(&self) -> fdo::Result<(bool, String, String, bool)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_persistence_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((
                settings.enabled,
                settings.remote_url,
                settings.remote_name,
                settings.push_on_update,
            ))
        })
        .await
    }

    /// Update git persistence settings.
    async fn set_persistence_settings(
        &self,
        enabled: bool,
        remote_url: &str,
        remote_name: &str,
        push_on_update: bool,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let remote_url = if remote_url.trim().is_empty() {
                None
            } else {
                Some(remote_url.to_string())
            };

            let remote_name = if remote_name.trim().is_empty() {
                None
            } else {
                Some(remote_name.to_string())
            };

            self.service
                .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Return database settings.
    ///
    /// Returns: (url, max_connections)
    async fn get_database_settings(&self) -> fdo::Result<(String, u32)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_database_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((settings.url, settings.max_connections))
        })
        .await
    }

    /// Update database settings. Empty url clears it.
    async fn set_database_settings(&self, url: &str, max_connections: u32) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let url = if url.trim().is_empty() {
                None
            } else {
                Some(url.to_string())
            };

            self.service
                .set_database_settings(url, max_connections)
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Return aggregate config tuple:
    /// (llm_connector, llm_model, llm_base_url, llm_has_api_key,
    ///  embeddings_connector, embeddings_model, embeddings_base_url, embeddings_has_api_key, embeddings_available, embeddings_is_default,
    ///  persistence_enabled, persistence_remote_url, persistence_remote_name, persistence_push_on_update)
    async fn get_config(&self) -> fdo::Result<ConfigData> {
        with_user_id(resolve_dbus_user_id(), self.get_config_tuple()).await
    }

    /// Apply a partial aggregate config update and emit `ConfigChanged`.
    ///
    /// For each string field, set `changes.set_* = true` to apply the provided value.
    /// Passing an empty string with `set_* = true` clears optional fields where supported.
    async fn set_config(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        changes: ConfigPatchArgs,
    ) -> fdo::Result<ConfigData> {
        let user_id = resolve_dbus_user_id();
        let emitter = emitter.to_owned();
        let updated = with_user_id(user_id, async move {
            let ConfigPatchArgs {
                set_llm_connector,
                llm_connector,
                set_llm_model,
                llm_model,
                set_llm_base_url,
                llm_base_url,
                set_llm_api_key,
                llm_api_key,
                set_embeddings_connector,
                embeddings_connector,
                set_embeddings_model,
                embeddings_model,
                set_embeddings_base_url,
                embeddings_base_url,
                set_persistence_enabled,
                persistence_enabled,
                set_persistence_remote_url,
                persistence_remote_url,
                set_persistence_remote_name,
                persistence_remote_name,
                set_persistence_push_on_update,
                persistence_push_on_update,
                set_llm_temperature,
                llm_temperature,
                set_llm_top_p,
                llm_top_p,
                set_llm_max_tokens,
                llm_max_tokens,
                set_llm_hosted_tool_search,
                llm_hosted_tool_search,
                set_personality_professionalism,
                personality_professionalism,
                set_personality_warmth,
                personality_warmth,
                set_personality_directness,
                personality_directness,
                set_personality_enthusiasm,
                personality_enthusiasm,
                set_personality_humor,
                personality_humor,
                set_personality_sarcasm,
                personality_sarcasm,
                set_personality_pretentiousness,
                personality_pretentiousness,
            } = changes;

            self.apply_config_patch(ConfigPatch {
                llm_connector: set_llm_connector.then_some(llm_connector),
                llm_model: set_llm_model.then_some(llm_model),
                llm_base_url: set_llm_base_url.then_some(llm_base_url),
                llm_api_key: set_llm_api_key.then_some(llm_api_key),
                embeddings_connector: set_embeddings_connector.then_some(embeddings_connector),
                embeddings_model: set_embeddings_model.then_some(embeddings_model),
                embeddings_base_url: set_embeddings_base_url.then_some(embeddings_base_url),
                persistence_enabled: set_persistence_enabled.then_some(persistence_enabled),
                persistence_remote_url: set_persistence_remote_url
                    .then_some(persistence_remote_url),
                persistence_remote_name: set_persistence_remote_name
                    .then_some(persistence_remote_name),
                persistence_push_on_update: set_persistence_push_on_update
                    .then_some(persistence_push_on_update),
                llm_temperature: set_llm_temperature.then_some(llm_temperature),
                llm_top_p: set_llm_top_p.then_some(llm_top_p),
                llm_max_tokens: set_llm_max_tokens.then_some(llm_max_tokens),
                llm_hosted_tool_search: set_llm_hosted_tool_search
                    .then_some(llm_hosted_tool_search == 1),
                personality_professionalism: set_personality_professionalism
                    .then_some(personality_professionalism),
                personality_warmth: set_personality_warmth.then_some(personality_warmth),
                personality_directness: set_personality_directness
                    .then_some(personality_directness),
                personality_enthusiasm: set_personality_enthusiasm
                    .then_some(personality_enthusiasm),
                personality_humor: set_personality_humor.then_some(personality_humor),
                personality_sarcasm: set_personality_sarcasm.then_some(personality_sarcasm),
                personality_pretentiousness: set_personality_pretentiousness
                    .then_some(personality_pretentiousness),
            })
            .await
        })
        .await?;

        Self::config_changed(&emitter, &updated)
            .await
            .map_err(to_fdo_error)?;

        Ok(updated)
    }

    /// Return backend-tasks settings (LLM override + dreaming config).
    ///
    /// Returns: (has_separate_llm, llm_connector, llm_model, llm_base_url, dreaming_enabled, dreaming_interval_secs, archive_after_days)
    async fn get_backend_tasks_settings(
        &self,
    ) -> fdo::Result<(bool, String, String, String, bool, u64, u32)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_backend_tasks_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((
                settings.has_separate_llm,
                settings.llm_connector,
                settings.llm_model,
                settings.llm_base_url,
                settings.dreaming_enabled,
                settings.dreaming_interval_secs,
                settings.archive_after_days,
            ))
        })
        .await
    }

    /// Update backend-tasks settings. Empty llm_connector clears the LLM override.
    async fn set_backend_tasks_settings(
        &self,
        llm_connector: &str,
        llm_model: &str,
        llm_base_url: &str,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
        archive_after_days: u32,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let llm_connector = if llm_connector.trim().is_empty() {
                None
            } else {
                Some(llm_connector.to_string())
            };

            let llm_model = if llm_model.trim().is_empty() {
                None
            } else {
                Some(llm_model.to_string())
            };

            let llm_base_url = if llm_base_url.trim().is_empty() {
                None
            } else {
                Some(llm_base_url.to_string())
            };

            self.service
                .set_backend_tasks_settings(
                    llm_connector,
                    llm_model,
                    llm_base_url,
                    dreaming_enabled,
                    dreaming_interval_secs,
                    archive_after_days,
                )
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// List configured MCP servers with status.
    ///
    /// Returns: Vec<(name, command, enabled, status, tool_count)>
    async fn list_mcp_servers(&self) -> fdo::Result<Vec<(String, String, bool, String, u32)>> {
        with_user_id(resolve_dbus_user_id(), async {
            let servers = self
                .service
                .list_mcp_servers()
                .await
                .map_err(to_fdo_error)?;

            Ok(servers
                .into_iter()
                .map(|s| (s.name, s.command, s.enabled, s.status, s.tool_count))
                .collect())
        })
        .await
    }

    /// Add a new MCP server.
    async fn add_mcp_server(
        &self,
        name: &str,
        command: &str,
        args: &str,
        namespace: &str,
        enabled: bool,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            let args: Vec<String> = if args.trim().is_empty() {
                vec![]
            } else {
                args.split_whitespace().map(|s| s.to_string()).collect()
            };

            let namespace = if namespace.trim().is_empty() {
                None
            } else {
                Some(namespace.to_string())
            };

            self.service
                .add_mcp_server(
                    name.to_string(),
                    command.to_string(),
                    args,
                    namespace,
                    enabled,
                )
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Remove an MCP server by name.
    async fn remove_mcp_server(&self, name: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .remove_mcp_server(name.to_string())
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Enable or disable an MCP server.
    async fn set_mcp_server_enabled(&self, name: &str, enabled: bool) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .set_mcp_server_enabled(name.to_string(), enabled)
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Perform an action (status/start/stop/restart) on MCP server(s).
    ///
    /// Returns: Vec<(name, command, enabled, status, tool_count)>
    async fn mcp_server_action(
        &self,
        action: &str,
        server: &str,
    ) -> fdo::Result<Vec<(String, String, bool, String, u32)>> {
        with_user_id(resolve_dbus_user_id(), async {
            let server = if server.trim().is_empty() {
                None
            } else {
                Some(server.to_string())
            };

            let servers = self
                .service
                .mcp_server_action(action.to_string(), server)
                .await
                .map_err(to_fdo_error)?;

            Ok(servers
                .into_iter()
                .map(|s| (s.name, s.command, s.enabled, s.status, s.tool_count))
                .collect())
        })
        .await
    }

    /// Return WebSocket auth settings.
    ///
    /// Returns: (methods, oidc_issuer, oidc_auth_endpoint, oidc_token_endpoint, oidc_client_id, oidc_scopes)
    async fn get_ws_auth_settings(
        &self,
    ) -> fdo::Result<(Vec<String>, String, String, String, String, String)> {
        with_user_id(resolve_dbus_user_id(), async {
            let settings = self
                .service
                .get_ws_auth_settings()
                .await
                .map_err(to_fdo_error)?;

            Ok((
                settings.methods,
                settings.oidc_issuer,
                settings.oidc_auth_endpoint,
                settings.oidc_token_endpoint,
                settings.oidc_client_id,
                settings.oidc_scopes,
            ))
        })
        .await
    }

    /// Update WebSocket auth settings.
    async fn set_ws_auth_settings(
        &self,
        methods: Vec<String>,
        oidc_issuer: &str,
        oidc_auth_endpoint: &str,
        oidc_token_endpoint: &str,
        oidc_client_id: &str,
        oidc_scopes: &str,
    ) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .set_ws_auth_settings(
                    methods,
                    oidc_issuer.to_string(),
                    oidc_auth_endpoint.to_string(),
                    oidc_token_endpoint.to_string(),
                    oidc_client_id.to_string(),
                    oidc_scopes.to_string(),
                )
                .await
                .map_err(to_fdo_error)
        })
        .await
    }

    /// Signal emitted after a successful aggregate config update.
    #[zbus(signal)]
    async fn config_changed(emitter: &SignalEmitter<'_>, config: &ConfigData) -> zbus::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::ports::inbound::{
        BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView,
        EmbeddingsSettingsView, LlmSettingsView, PersistenceSettingsView, PersonalitySettingsView,
        SettingsService, WsAuthSettingsView,
    };
    use std::sync::Mutex;

    #[derive(Clone)]
    struct SettingsState {
        llm: LlmSettingsView,
        embeddings: EmbeddingsSettingsView,
        persistence: PersistenceSettingsView,
        personality: PersonalitySettingsView,
        database: DatabaseSettingsView,
        backend_tasks: BackendTasksSettingsView,
        api_key_set: bool,
    }

    struct StatefulSettingsService {
        state: Mutex<SettingsState>,
    }

    impl StatefulSettingsService {
        fn new() -> Self {
            Self {
                state: Mutex::new(SettingsState {
                    llm: LlmSettingsView {
                        connector: "openai".to_string(),
                        model: "gpt-5.4".to_string(),
                        base_url: "https://api.openai.com/v1".to_string(),
                        has_api_key: false,
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        hosted_tool_search: None,
                    },
                    embeddings: EmbeddingsSettingsView {
                        connector: "openai".to_string(),
                        model: "text-embedding-3-small".to_string(),
                        base_url: "https://api.openai.com/v1".to_string(),
                        has_api_key: false,
                        available: true,
                        is_default: true,
                    },
                    persistence: PersistenceSettingsView {
                        enabled: false,
                        remote_url: String::new(),
                        remote_name: "origin".to_string(),
                        push_on_update: true,
                    },
                    personality: PersonalitySettingsView::default(),
                    database: DatabaseSettingsView {
                        url: String::new(),
                        max_connections: 5,
                    },
                    backend_tasks: BackendTasksSettingsView {
                        has_separate_llm: false,
                        llm_connector: "openai".to_string(),
                        llm_model: "gpt-5.4".to_string(),
                        llm_base_url: "https://api.openai.com/v1".to_string(),
                        dreaming_enabled: false,
                        dreaming_interval_secs: 3600,
                        archive_after_days: 0,
                    },
                    api_key_set: false,
                }),
            }
        }
    }

    impl SettingsService for StatefulSettingsService {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().llm.clone())
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
            let mut state = self.state.lock().unwrap();
            state.llm.connector = connector;
            if let Some(model) = model {
                state.llm.model = model;
            }
            if let Some(base_url) = base_url {
                state.llm.base_url = base_url;
            }
            state.llm.temperature = temperature;
            state.llm.top_p = top_p;
            state.llm.max_tokens = max_tokens;
            state.llm.hosted_tool_search = hosted_tool_search;
            Ok(())
        }

        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.api_key_set = true;
            state.llm.has_api_key = true;
            Ok(())
        }

        async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
            Ok(format!(
                "jwt-for-{}",
                subject.unwrap_or_else(|| "desktop-client".to_string())
            ))
        }

        async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
            Ok(token.starts_with("jwt-for-"))
        }

        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().embeddings.clone())
        }

        async fn set_embeddings_settings(
            &self,
            connector: Option<String>,
            model: Option<String>,
            base_url: Option<String>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            if let Some(connector) = connector {
                state.embeddings.connector = connector;
                state.embeddings.is_default = false;
            } else {
                state.embeddings.is_default = true;
            }
            if let Some(model) = model {
                state.embeddings.model = model;
            }
            if let Some(base_url) = base_url {
                state.embeddings.base_url = base_url;
            }
            Ok(())
        }

        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
            Ok(ConnectorDefaultsView {
                llm_model: "gpt-5.4".to_string(),
                llm_base_url: "https://api.openai.com/v1".to_string(),
                backend_llm_model: "gpt-4o-mini".to_string(),
                embeddings_model: "text-embedding-3-small".to_string(),
                embeddings_base_url: "https://api.openai.com/v1".to_string(),
                embeddings_available: true,
                hosted_tool_search_available: true,
            })
        }

        async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().persistence.clone())
        }

        async fn set_persistence_settings(
            &self,
            enabled: bool,
            remote_url: Option<String>,
            remote_name: Option<String>,
            push_on_update: bool,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.persistence.enabled = enabled;
            if let Some(remote_url) = remote_url {
                state.persistence.remote_url = remote_url;
            }
            if let Some(remote_name) = remote_name {
                state.persistence.remote_name = remote_name;
            }
            state.persistence.push_on_update = push_on_update;
            Ok(())
        }

        async fn get_personality_settings(&self) -> Result<PersonalitySettingsView, CoreError> {
            Ok(self.state.lock().unwrap().personality)
        }

        async fn set_personality_settings(
            &self,
            personality: PersonalitySettingsView,
        ) -> Result<(), CoreError> {
            self.state.lock().unwrap().personality = personality;
            Ok(())
        }

        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().database.clone())
        }

        async fn set_database_settings(
            &self,
            url: Option<String>,
            max_connections: u32,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.database.url = url.unwrap_or_default();
            state.database.max_connections = max_connections;
            Ok(())
        }

        async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().backend_tasks.clone())
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
            let mut state = self.state.lock().unwrap();
            state.backend_tasks.has_separate_llm = llm_connector.is_some();
            if let Some(connector) = llm_connector {
                state.backend_tasks.llm_connector = connector;
            }
            if let Some(model) = llm_model {
                state.backend_tasks.llm_model = model;
            }
            if let Some(base_url) = llm_base_url {
                state.backend_tasks.llm_base_url = base_url;
            }
            state.backend_tasks.dreaming_enabled = dreaming_enabled;
            state.backend_tasks.dreaming_interval_secs = dreaming_interval_secs;
            state.backend_tasks.archive_after_days = archive_after_days;
            Ok(())
        }

        async fn list_mcp_servers(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn add_mcp_server(
            &self,
            _name: String,
            _command: String,
            _args: Vec<String>,
            _namespace: Option<String>,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            _name: String,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            _action: String,
            _server: Option<String>,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }

        async fn get_ws_auth_settings(&self) -> Result<WsAuthSettingsView, CoreError> {
            Ok(WsAuthSettingsView {
                methods: vec!["password".to_string()],
                oidc_issuer: String::new(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            })
        }

        async fn set_ws_auth_settings(
            &self,
            _methods: Vec<String>,
            _oidc_issuer: String,
            _oidc_auth_endpoint: String,
            _oidc_token_endpoint: String,
            _oidc_client_id: String,
            _oidc_scopes: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[test]
    fn adapter_construction() {
        let service = Arc::new(StatefulSettingsService::new());
        let _adapter = DbusSettingsAdapter::new(service);
    }

    #[tokio::test]
    async fn get_config_tuple_aggregates_settings() {
        let service = Arc::new(StatefulSettingsService::new());
        let adapter = DbusSettingsAdapter::new(service);
        let config = adapter.get_config_tuple().await.unwrap();

        assert_eq!(config.llm_connector, "openai");
        assert_eq!(config.embeddings_model, "text-embedding-3-small");
        assert_eq!(config.persistence_remote_name, "origin");
    }

    #[tokio::test]
    async fn apply_config_patch_updates_config_and_api_key() {
        let service = Arc::new(StatefulSettingsService::new());
        let adapter = DbusSettingsAdapter::new(Arc::clone(&service));

        let updated = adapter
            .apply_config_patch(ConfigPatch {
                llm_connector: Some("ollama".into()),
                llm_model: Some("llama3.1:8b".into()),
                llm_base_url: Some("http://localhost:11434".into()),
                llm_api_key: Some("secret".into()),
                persistence_enabled: Some(true),
                persistence_remote_url: Some("git@example.com/repo.git".into()),
                persistence_remote_name: Some("upstream".into()),
                persistence_push_on_update: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(updated.llm_connector, "ollama");
        assert_eq!(updated.llm_model, "llama3.1:8b");
        assert!(updated.llm_has_api_key);
        assert!(updated.persistence_enabled);
        assert_eq!(updated.persistence_remote_name, "upstream");
        assert!(!updated.persistence_push_on_update);

        assert!(service.state.lock().unwrap().api_key_set);
    }

    #[tokio::test]
    async fn generate_ws_jwt_delegates_to_settings_service() {
        let service = Arc::new(StatefulSettingsService::new());
        let adapter = DbusSettingsAdapter::new(service);

        let token = adapter.generate_ws_jwt("tui").await.unwrap();
        assert_eq!(token, "jwt-for-tui");
    }

    // --- Personality int<->level contract (#226) ---------------------------

    #[tokio::test]
    async fn get_config_exposes_personality_as_ordinals() {
        // The KCM binds sliders to integers 0..=4 (Never=0 .. Always=4).
        // `GetConfig` must surface the default Expressive-7 levels as those
        // ordinals.
        let service = Arc::new(StatefulSettingsService::new());
        let adapter = DbusSettingsAdapter::new(service);
        let config = adapter.get_config_tuple().await.unwrap();

        assert_eq!(config.personality_professionalism, 4); // Always
        assert_eq!(config.personality_warmth, 3); // Often
        assert_eq!(config.personality_directness, 3); // Often
        assert_eq!(config.personality_enthusiasm, 2); // Sometimes
        assert_eq!(config.personality_humor, 2); // Sometimes
        assert_eq!(config.personality_sarcasm, 1); // Rarely
        assert_eq!(config.personality_pretentiousness, 1); // Rarely
    }

    #[tokio::test]
    async fn set_config_personality_ordinal_round_trips() {
        // Setting Humor=Never (0) via the patch must be reflected back as an
        // ordinal in the returned `ConfigData`.
        let service = Arc::new(StatefulSettingsService::new());
        let adapter = DbusSettingsAdapter::new(Arc::clone(&service));

        let updated = adapter
            .apply_config_patch(ConfigPatch {
                personality_humor: Some(0),   // Never
                personality_sarcasm: Some(4), // Always
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(updated.personality_humor, 0);
        assert_eq!(updated.personality_sarcasm, 4);
        // Untouched traits keep their defaults.
        assert_eq!(updated.personality_professionalism, 4);
    }

    /// Issue #156: settings methods that touch per-user storage must
    /// scope to the local OS user, not the `"default"` sentinel. The
    /// recording fake captures `current_user_id()` at the inbound call
    /// site so we can assert the D-Bus method entry installed the
    /// scope before calling into the service.
    #[tokio::test]
    async fn dbus_settings_methods_install_user_id_scope_at_method_entry() {
        use desktop_assistant_core::ports::auth::{UserId, current_user_id};
        use std::sync::Mutex;

        struct RecordingSettings {
            seen: Mutex<Vec<String>>,
        }

        impl RecordingSettings {
            fn new() -> Self {
                Self {
                    seen: Mutex::new(Vec::new()),
                }
            }
            fn record(&self) {
                self.seen
                    .lock()
                    .unwrap()
                    .push(current_user_id().as_str().to_string());
            }
            fn observed(&self) -> Vec<String> {
                self.seen.lock().unwrap().clone()
            }
        }

        impl SettingsService for RecordingSettings {
            async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
                self.record();
                Ok(LlmSettingsView {
                    connector: "x".into(),
                    model: "y".into(),
                    base_url: "z".into(),
                    has_api_key: false,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    hosted_tool_search: None,
                })
            }
            async fn set_llm_settings(
                &self,
                _: String,
                _: Option<String>,
                _: Option<String>,
                _: Option<f64>,
                _: Option<f64>,
                _: Option<u32>,
                _: Option<bool>,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn set_api_key(&self, _: String) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn generate_ws_jwt(&self, _: Option<String>) -> Result<String, CoreError> {
                self.record();
                Ok("t".into())
            }
            async fn validate_ws_jwt(&self, _: String) -> Result<bool, CoreError> {
                self.record();
                Ok(true)
            }
            async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
                self.record();
                Ok(EmbeddingsSettingsView {
                    connector: "x".into(),
                    model: "y".into(),
                    base_url: "z".into(),
                    has_api_key: false,
                    available: true,
                    is_default: true,
                })
            }
            async fn set_embeddings_settings(
                &self,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn get_connector_defaults(
                &self,
                _: String,
            ) -> Result<ConnectorDefaultsView, CoreError> {
                self.record();
                Ok(ConnectorDefaultsView {
                    llm_model: "m".into(),
                    llm_base_url: "u".into(),
                    backend_llm_model: "bm".into(),
                    embeddings_model: "em".into(),
                    embeddings_base_url: "eu".into(),
                    embeddings_available: false,
                    hosted_tool_search_available: false,
                })
            }
            async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
                self.record();
                Ok(PersistenceSettingsView {
                    enabled: false,
                    remote_url: String::new(),
                    remote_name: "origin".into(),
                    push_on_update: false,
                })
            }
            async fn set_persistence_settings(
                &self,
                _: bool,
                _: Option<String>,
                _: Option<String>,
                _: bool,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
                self.record();
                Ok(DatabaseSettingsView {
                    url: String::new(),
                    max_connections: 5,
                })
            }
            async fn set_database_settings(
                &self,
                _: Option<String>,
                _: u32,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn get_backend_tasks_settings(
                &self,
            ) -> Result<BackendTasksSettingsView, CoreError> {
                self.record();
                Ok(BackendTasksSettingsView {
                    has_separate_llm: false,
                    llm_connector: "x".into(),
                    llm_model: "y".into(),
                    llm_base_url: "z".into(),
                    dreaming_enabled: false,
                    dreaming_interval_secs: 0,
                    archive_after_days: 0,
                })
            }
            async fn set_backend_tasks_settings(
                &self,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
                _: bool,
                _: u64,
                _: u32,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn list_mcp_servers(
                &self,
            ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError>
            {
                self.record();
                Ok(vec![])
            }
            async fn add_mcp_server(
                &self,
                _: String,
                _: String,
                _: Vec<String>,
                _: Option<String>,
                _: bool,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn remove_mcp_server(&self, _: String) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn set_mcp_server_enabled(&self, _: String, _: bool) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
            async fn mcp_server_action(
                &self,
                _: String,
                _: Option<String>,
            ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError>
            {
                self.record();
                Ok(vec![])
            }
            async fn get_ws_auth_settings(&self) -> Result<WsAuthSettingsView, CoreError> {
                self.record();
                Ok(WsAuthSettingsView {
                    methods: vec![],
                    oidc_issuer: String::new(),
                    oidc_auth_endpoint: String::new(),
                    oidc_token_endpoint: String::new(),
                    oidc_client_id: String::new(),
                    oidc_scopes: String::new(),
                })
            }
            async fn set_ws_auth_settings(
                &self,
                _: Vec<String>,
                _: String,
                _: String,
                _: String,
                _: String,
                _: String,
            ) -> Result<(), CoreError> {
                self.record();
                Ok(())
            }
        }

        let service = Arc::new(RecordingSettings::new());
        let adapter = DbusSettingsAdapter::new(Arc::clone(&service));

        let _guard = crate::testing::UserEnvGuard::set("alice-settings");

        // Exercise representative methods: read-paths, write-paths, JWT.
        // We don't need to call every method — one per dispatcher style
        // is enough to pin the contract.
        let _llm = adapter.get_llm_settings().await.unwrap();
        adapter
            .set_llm_settings("openai", "gpt-5.4", "https://api")
            .await
            .unwrap();
        adapter.set_api_key("k").await.unwrap();
        let _jwt = adapter.generate_ws_jwt("tui").await.unwrap();
        let _emb = adapter.get_embeddings_settings().await.unwrap();
        let _persist = adapter.get_persistence_settings().await.unwrap();
        let _db = adapter.get_database_settings().await.unwrap();
        let _bt = adapter.get_backend_tasks_settings().await.unwrap();
        let _mcp = adapter.list_mcp_servers().await.unwrap();
        let _ws = adapter.get_ws_auth_settings().await.unwrap();

        let observed = service.observed();
        assert!(!observed.is_empty());
        for seen in observed {
            assert_eq!(
                seen,
                UserId::new("alice-settings").as_str(),
                "every D-Bus settings method must scope storage to the resolved local user"
            );
        }
    }
}
