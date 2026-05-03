//! Settings views — the `get_*` / `set_*` pairs that the inbound API
//! handler dispatches into when clients edit settings (the [llm] block,
//! embeddings, persistence, database, backend tasks, WS auth).
//!
//! Extracted from `config.rs` (#41). Each pair is a thin wrapper over
//! `load_daemon_config` → resolve / mutate → `save_daemon_config`. The
//! resolved view types (`LlmSettingsView`, `EmbeddingsSettingsView`,
//! etc.) and the resolution helpers stay in `mod.rs` since they're
//! shared with non-view code paths.

use std::path::Path;

use anyhow::anyhow;

use crate::connections::Connector;

use super::{
    ConnectorDefaultsView, EmbeddingsSettingsView, LlmSettingsView, OidcConfig, OidcDiscoveryInfo,
    ResolvedPersistenceConfig, SecretConfig, WsAuthConfig, WsAuthDiscoveryInfo, bucket_secret_len,
    default_archive_after_days, default_backend_llm_model, default_base_url, default_connector,
    default_dreaming_interval_secs, default_git_remote_name, default_llm_model,
    default_oidc_scopes, is_placeholder_secret_value, load_daemon_config, normalize_optional_value,
    parse_connector_or_openai, redacted_secret_audit, resolve_backend_tasks_llm_config,
    resolve_database_config, resolve_embeddings_config, resolve_llm_config,
    resolve_persistence_config, save_daemon_config, write_secret_to_backend,
};

pub fn get_llm_settings_view(path: &Path) -> anyhow::Result<LlmSettingsView> {
    let config = load_daemon_config(path)?;
    let resolved = resolve_llm_config(config.as_ref());

    Ok(LlmSettingsView {
        connector: resolved.connector,
        model: resolved.model,
        base_url: resolved.base_url,
        has_api_key: !resolved.api_key.is_empty(),
        temperature: resolved.temperature,
        top_p: resolved.top_p,
        max_tokens: resolved.max_tokens,
        hosted_tool_search: resolved.hosted_tool_search,
    })
}

pub fn set_llm_settings(
    path: &Path,
    connector: &str,
    model: Option<&str>,
    base_url: Option<&str>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    hosted_tool_search: Option<bool>,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    let connector = connector.trim().to_lowercase();
    if connector.is_empty() {
        return Err(anyhow!("connector must not be empty"));
    }

    if let Some(t) = temperature
        && !(0.0..=2.0).contains(&t)
    {
        return Err(anyhow!("temperature must be between 0.0 and 2.0"));
    }
    if let Some(p) = top_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(anyhow!("top_p must be between 0.0 and 1.0"));
    }
    if let Some(m) = max_tokens
        && m == 0
    {
        return Err(anyhow!("max_tokens must be greater than 0"));
    }

    config.llm.connector = connector;
    config.llm.model = normalize_optional_value(model);
    config.llm.base_url = normalize_optional_value(base_url);
    config.llm.temperature = temperature;
    config.llm.top_p = top_p;
    config.llm.max_tokens = max_tokens;
    config.llm.hosted_tool_search = hosted_tool_search;

    save_daemon_config(path, &config)
}

pub fn set_api_key(path: &Path, api_key: &str) -> anyhow::Result<()> {
    let api_key = api_key.trim();
    let (key_len, key_fingerprint) = redacted_secret_audit(api_key);
    // Logging the precise length narrows the search space for guessing the
    // connector type from logs (e.g. 51 chars ≈ Anthropic, 32 ≈ OpenAI).
    let key_len_bucket = bucket_secret_len(key_len);

    tracing::info!(
        secret_len_bucket = key_len_bucket,
        secret_fingerprint = %key_fingerprint,
        "received SetApiKey request"
    );

    if api_key.is_empty() {
        return Err(anyhow!("api key must not be empty"));
    }

    if is_placeholder_secret_value(api_key) {
        tracing::warn!(
            secret_len_bucket = key_len_bucket,
            secret_fingerprint = %key_fingerprint,
            "rejecting placeholder-like SetApiKey value"
        );
        return Err(anyhow!(
            "api key looks like a placeholder or masked value; provide the real key"
        ));
    }

    let mut config = load_daemon_config(path)?.unwrap_or_default();
    if config.llm.secret.is_none() {
        config.llm.secret = Some(SecretConfig::default());
    }

    let secret = config
        .llm
        .secret
        .clone()
        .unwrap_or_else(SecretConfig::default);

    let connector = config.llm.connector.trim().to_lowercase();
    let connector = if connector.is_empty() {
        default_connector()
    } else {
        connector
    };

    write_secret_to_backend(&secret, api_key, &connector)?;
    save_daemon_config(path, &config)
}

pub fn get_embeddings_settings_view(path: &Path) -> anyhow::Result<EmbeddingsSettingsView> {
    let config = load_daemon_config(path)?;
    let resolved = resolve_embeddings_config(config.as_ref());
    Ok(resolved)
}

pub fn set_embeddings_settings(
    path: &Path,
    connector: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.embeddings.connector = connector
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_lowercase());
    config.embeddings.model = normalize_optional_value(model);
    config.embeddings.base_url = normalize_optional_value(base_url);

    save_daemon_config(path, &config)
}

pub fn get_persistence_settings_view(path: &Path) -> anyhow::Result<ResolvedPersistenceConfig> {
    let config = load_daemon_config(path)?;
    Ok(resolve_persistence_config(config.as_ref()))
}

pub fn set_persistence_settings(
    path: &Path,
    enabled: bool,
    remote_url: Option<&str>,
    remote_name: Option<&str>,
    push_on_update: bool,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.persistence.git.enabled = enabled;
    config.persistence.git.remote_url = normalize_optional_value(remote_url);
    config.persistence.git.remote_name = remote_name
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(default_git_remote_name);
    config.persistence.git.push_on_update = push_on_update;

    save_daemon_config(path, &config)
}

pub fn get_database_settings_view(path: &Path) -> anyhow::Result<(String, u32)> {
    let config = load_daemon_config(path)?;
    let (url, max_connections) = resolve_database_config(config.as_ref());
    Ok((url.unwrap_or_default(), max_connections))
}

pub fn set_database_settings(
    path: &Path,
    url: Option<&str>,
    max_connections: u32,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.database.url = normalize_optional_value(url);
    config.database.max_connections = max_connections;

    save_daemon_config(path, &config)
}

pub struct BackendTasksSettingsViewConfig {
    pub has_separate_llm: bool,
    pub llm_connector: String,
    pub llm_model: String,
    pub llm_base_url: String,
    pub dreaming_enabled: bool,
    pub dreaming_interval_secs: u64,
    pub archive_after_days: u32,
}

pub fn get_backend_tasks_settings_view(
    path: &Path,
) -> anyhow::Result<BackendTasksSettingsViewConfig> {
    let config = load_daemon_config(path)?;
    let bt = config.as_ref().map(|c| &c.backend_tasks);
    let has_separate_llm = bt.is_some_and(|b| b.llm.is_some());
    let dreaming_enabled = bt.map(|b| b.dreaming_enabled).unwrap_or(false);
    let dreaming_interval_secs = bt
        .map(|b| b.dreaming_interval_secs)
        .unwrap_or_else(default_dreaming_interval_secs);
    let archive_after_days = bt
        .map(|b| b.archive_after_days)
        .unwrap_or_else(default_archive_after_days);

    let resolved = resolve_backend_tasks_llm_config(config.as_ref());

    Ok(BackendTasksSettingsViewConfig {
        has_separate_llm,
        llm_connector: resolved.connector,
        llm_model: resolved.model,
        llm_base_url: resolved.base_url,
        dreaming_enabled,
        dreaming_interval_secs,
        archive_after_days,
    })
}

pub fn set_backend_tasks_settings(
    path: &Path,
    llm_connector: Option<&str>,
    llm_model: Option<&str>,
    llm_base_url: Option<&str>,
    dreaming_enabled: bool,
    dreaming_interval_secs: u64,
    archive_after_days: u32,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.backend_tasks.dreaming_enabled = dreaming_enabled;
    config.backend_tasks.dreaming_interval_secs = dreaming_interval_secs;
    config.backend_tasks.archive_after_days = archive_after_days;

    // If connector is provided, configure a separate backend-tasks LLM.
    // If connector is None/empty, clear the override (fall back to primary).
    let connector = llm_connector.map(str::trim).filter(|v| !v.is_empty());

    if let Some(connector) = connector {
        let mut llm = config.backend_tasks.llm.unwrap_or_default();
        llm.connector = connector.to_lowercase();
        llm.model = normalize_optional_value(llm_model);
        llm.base_url = normalize_optional_value(llm_base_url);
        config.backend_tasks.llm = Some(llm);
    } else {
        config.backend_tasks.llm = None;
    }

    save_daemon_config(path, &config)
}

pub fn get_connector_defaults(connector: &str) -> ConnectorDefaultsView {
    let connector = connector.trim().to_lowercase();
    let connector = if connector.is_empty() {
        default_connector()
    } else {
        connector
    };

    let typed = parse_connector_or_openai(&connector);
    let llm_model = default_llm_model(&connector);
    let llm_base_url = default_base_url(&connector);

    let embeddings_available = typed.supports_embeddings();
    // Substitute OpenAI for the embedding lookup when this connector
    // doesn't ship one (Anthropic) — preserves legacy behaviour where
    // `embeddings_model` always resolves to a real value.
    let embeddings_connector = if embeddings_available {
        typed
    } else {
        Connector::OpenAi
    };

    ConnectorDefaultsView {
        llm_model,
        llm_base_url,
        backend_llm_model: default_backend_llm_model(&connector),
        embeddings_model: embeddings_connector.default_embedding_model().to_string(),
        embeddings_base_url: embeddings_connector.default_base_url().to_string(),
        embeddings_available,
        hosted_tool_search_available: typed.supports_hosted_tool_search(),
    }
}

pub fn get_ws_auth_discovery(config_path: &Path) -> anyhow::Result<WsAuthDiscoveryInfo> {
    let config = load_daemon_config(config_path)?.unwrap_or_default();
    let ws_auth = config.ws_auth;

    let oidc_info = if ws_auth.methods.contains(&"oidc".to_string()) {
        ws_auth.oidc.as_ref().map(|oidc| OidcDiscoveryInfo {
            authorization_endpoint: oidc.authorization_endpoint.clone(),
            token_endpoint: oidc.token_endpoint.clone(),
            client_id: oidc.client_id.clone(),
            scopes: oidc.scopes.clone(),
        })
    } else {
        None
    };

    Ok(WsAuthDiscoveryInfo {
        methods: ws_auth.methods,
        oidc: oidc_info,
    })
}

pub fn get_ws_auth_settings(config_path: &Path) -> anyhow::Result<WsAuthConfig> {
    let config = load_daemon_config(config_path)?.unwrap_or_default();
    Ok(config.ws_auth)
}

pub fn set_ws_auth_settings(
    config_path: &Path,
    methods: &[String],
    oidc_issuer: &str,
    oidc_auth_endpoint: &str,
    oidc_token_endpoint: &str,
    oidc_client_id: &str,
    oidc_scopes: &str,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(config_path)?.unwrap_or_default();

    config.ws_auth.methods = methods.to_vec();

    if methods.contains(&"oidc".to_string()) && !oidc_issuer.is_empty() {
        let existing_oidc = config.ws_auth.oidc.unwrap_or_else(|| OidcConfig {
            issuer_url: String::new(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
            client_id: String::new(),
            scopes: default_oidc_scopes(),
            jwks_uri: String::new(),
            audience: String::new(),
        });
        config.ws_auth.oidc = Some(OidcConfig {
            issuer_url: oidc_issuer.to_string(),
            authorization_endpoint: oidc_auth_endpoint.to_string(),
            token_endpoint: oidc_token_endpoint.to_string(),
            client_id: oidc_client_id.to_string(),
            scopes: if oidc_scopes.is_empty() {
                existing_oidc.scopes
            } else {
                oidc_scopes.to_string()
            },
            jwks_uri: existing_oidc.jwks_uri,
            audience: existing_oidc.audience,
        });
    } else {
        config.ws_auth.oidc = None;
    }

    save_daemon_config(config_path, &config)
}
