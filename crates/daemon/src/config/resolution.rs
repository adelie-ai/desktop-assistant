//! Config resolution helpers — turn the raw [`DaemonConfig`] schema
//! into the resolved views consumed by the rest of the daemon.
//!
//! Extracted from `config.rs` (#41). All of the `resolve_*` functions
//! plus the small connector-default helpers live here. The schema
//! itself, the secret-store env helpers (`default_api_key_env`,
//! `default_model_env`, `default_base_url_env`), and `default_connector`
//! stay in [`super`] because they are shared with serde defaults and
//! the secret backends.

use desktop_assistant_core::ports::llm::{BudgetSource, ContextBudget};

use crate::connections::{
    AnthropicConnection, BedrockConnection, ConnectionConfig, Connector, OllamaConnection,
    OpenAiConnection,
};
use crate::purposes::PurposeKind;

use super::secrets::read_secret_from_backend;
use super::{
    DaemonConfig, EmbeddingsSettingsView, LlmConfig, ResolvedLlmConfig, ResolvedPersistenceConfig,
    SecretConfig, default_api_key_env, default_base_url_env, default_connector,
    default_database_max_connections, default_git_remote_name, default_model_env,
    default_push_on_update,
};

pub fn resolve_embeddings_config(config: Option<&DaemonConfig>) -> EmbeddingsSettingsView {
    // Purpose-driven path: when `[purposes.embedding]` is configured, it wins
    // over the legacy `[embeddings]` block. The daemon API surface
    // (`set_purpose("embedding", ...)`) writes into `[purposes]`, so without
    // this short-circuit user-set purposes silently get ignored at startup.
    if let Some(view) = resolve_purpose_embeddings_view(config) {
        return view;
    }

    let llm_connector = config
        .map(|c| c.llm.connector.trim().to_lowercase())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(default_connector);

    let emb_config = config.map(|c| &c.embeddings);

    let explicit_connector = emb_config
        .and_then(|c| c.connector.as_deref())
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty());

    let is_default = explicit_connector.is_none();
    let connector = explicit_connector.unwrap_or_else(|| llm_connector.clone());
    let available = connector != "anthropic";

    let model = emb_config
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_embedding_model(&connector));

    let base_url = emb_config
        .and_then(|c| c.base_url.clone())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_base_url(&connector));

    // Resolve API key: reuse LLM secret config if connectors match, otherwise use env fallback
    let api_key = if is_default || connector == llm_connector {
        resolve_llm_config(config).api_key
    } else {
        let env_key = default_api_key_env(&connector);
        std::env::var(env_key).unwrap_or_default()
    };
    let has_api_key = !api_key.trim().is_empty();

    EmbeddingsSettingsView {
        connector,
        model,
        base_url,
        api_key,
        has_api_key,
        available,
        is_default,
    }
}

/// Build an `EmbeddingsSettingsView` from `purposes.embedding` if it is
/// configured, otherwise return `None`. Centralises the purpose-aware
/// short-circuit so the legacy resolver can skip the rest of its work.
fn resolve_purpose_embeddings_view(
    config: Option<&DaemonConfig>,
) -> Option<EmbeddingsSettingsView> {
    let resolved = resolve_purpose_llm_config(config, PurposeKind::Embedding)?;
    let available = resolved.connector != "anthropic";
    let has_api_key = !resolved.api_key.trim().is_empty();
    Some(EmbeddingsSettingsView {
        connector: resolved.connector,
        model: resolved.model,
        base_url: resolved.base_url,
        api_key: resolved.api_key,
        has_api_key,
        available,
        // Always `false` for purpose-driven config: the user explicitly chose
        // a connection/model, so this is no longer "the inferred default".
        is_default: false,
    })
}

pub fn resolve_persistence_config(config: Option<&DaemonConfig>) -> ResolvedPersistenceConfig {
    let persistence = config.map(|c| &c.persistence.git);

    let enabled = persistence.map(|p| p.enabled).unwrap_or(false);
    let remote_url = persistence
        .and_then(|p| p.remote_url.as_deref())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);

    let remote_name = persistence
        .map(|p| p.remote_name.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_git_remote_name);

    let push_on_update = persistence
        .map(|p| p.push_on_update)
        .unwrap_or_else(default_push_on_update);

    ResolvedPersistenceConfig {
        enabled,
        remote_url,
        remote_name,
        push_on_update,
    }
}

/// Resolve the database URL from config, then env var fallback.
/// Returns `None` if no database URL is configured anywhere.
pub fn resolve_database_config(config: Option<&DaemonConfig>) -> (Option<String>, u32) {
    let db = config.map(|c| &c.database);
    let url = db
        .and_then(|d| d.url.clone())
        .or_else(|| std::env::var("DESKTOP_ASSISTANT_DATABASE_URL").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let max_conns = db
        .map(|d| d.max_connections)
        .unwrap_or_else(default_database_max_connections);
    (url, max_conns)
}

/// Resolve `connector` to a typed [`Connector`], falling back to
/// `Connector::OpenAi` for unrecognised values — the historical
/// "default to OpenAI for unknown connector strings" behaviour, now
/// concentrated in one helper instead of repeated as a `_` arm in
/// every match (#47).
pub(crate) fn parse_connector_or_openai(connector: &str) -> Connector {
    Connector::parse(connector).unwrap_or(Connector::OpenAi)
}

fn default_embedding_model(connector: &str) -> String {
    let c = parse_connector_or_openai(connector);
    let model = c.default_embedding_model();
    // Anthropic has no embeddings; the legacy default for that case
    // was `text-embedding-3-small` (the OpenAI default).
    if model.is_empty() {
        Connector::OpenAi.default_embedding_model().to_string()
    } else {
        model.to_string()
    }
}

pub(crate) fn default_base_url(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_base_url()
        .to_string()
}

pub(crate) fn default_llm_model(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_chat_model()
        .to_string()
}

pub(crate) fn default_backend_llm_model(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_backend_chat_model()
        .to_string()
}

pub(crate) fn normalize_optional_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub fn resolve_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    resolve_llm_config_from(config.map(|c| &c.llm))
}

/// Resolve backend-tasks LLM config: uses `[backend_tasks.llm]` if set,
/// otherwise falls back to the top-level `[llm]`.
pub fn resolve_backend_tasks_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    let bt_llm = config.and_then(|c| c.backend_tasks.llm.as_ref());
    if bt_llm.is_some() {
        resolve_llm_config_from(bt_llm)
    } else {
        resolve_llm_config(config)
    }
}

/// Resolve the LLM config for a given [`PurposeKind`] when the user has
/// configured `[purposes.<kind>]`. Returns `None` when no purpose is set
/// (callers fall back to the legacy resolvers — `resolve_embeddings_config`
/// for embedding, `resolve_backend_tasks_llm_config` for dreaming/titling).
///
/// Resolution flow:
/// 1. Look up `cfg.purposes.<kind>`. If absent, return `None`.
/// 2. Validate `[connections]`. If the map fails to validate, log + `None` —
///    the legacy resolver still produces something usable from `[llm]`.
/// 3. Run [`crate::purposes::resolve_purpose`] which handles `"primary"`
///    inheritance (connection and model both fall through to interactive)
///    and dangling-connection warnings.
/// 4. Build a [`ResolvedLlmConfig`] from the purpose's connection via
///    [`resolve_connection_llm_config`], then override the model with the
///    purpose's `model_id`. The connection's resolved `api_key` /
///    `base_url` / connector type are preserved as-is — the purpose layer
///    only chooses *which* connection + model, not credentials.
///
/// Effort threading is handled at the call site (see
/// `api_surface::RoutingConversationHandler::apply_effort_mapping` for the
/// interactive path; backend tasks call the same mapper directly). The
/// effort hint lives on `cfg.purposes.<kind>.effort` and can be read back
/// via `cfg.purposes.get(kind).effort`.
pub fn resolve_purpose_llm_config(
    config: Option<&DaemonConfig>,
    kind: PurposeKind,
) -> Option<ResolvedLlmConfig> {
    let cfg = config?;
    cfg.purposes.get(kind)?;

    let connections = match cfg.validated_connections() {
        Ok(map) => map,
        Err(err) => {
            tracing::warn!(
                purpose = kind.as_key(),
                error = %err,
                "cannot resolve purpose: [connections] failed validation; falling back to legacy resolver"
            );
            return None;
        }
    };

    let resolved = match crate::purposes::resolve_purpose(kind, &cfg.purposes, &connections) {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!(
                purpose = kind.as_key(),
                error = %err,
                "purpose resolution failed; falling back to legacy resolver"
            );
            return None;
        }
    };

    // The connection must exist after `resolve_purpose` — it returns the
    // interactive fallback id for dangling refs, and interactive itself is
    // checked by `expect_interactive_connection`. Map miss here would be a
    // logic bug in `resolve_purpose`, not a config issue.
    let conn = connections.get(&resolved.connection_id)?;
    let mut llm = resolve_connection_llm_config(conn, Some(&cfg.llm));
    llm.model = resolved.model_id;
    Some(llm)
}

/// Universal fallback for purpose-aware context-window resolution.
/// Used when no purpose override is set and the connector's curated
/// table reports nothing for the model. Most modern frontier models
/// meet or exceed this; under-stating is safe (we compact slightly
/// earlier than necessary), over-stating is not (the LLM rejects).
pub const DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS: u64 = 200_000;

/// Three-tier resolution for "what's the context window for this purpose?"
///
/// Resolution order:
///   1. The purpose's `max_context_tokens` override, if explicitly set —
///      the user always wins. Tagged [`BudgetSource::PurposeOverride`].
///   2. The connector's curated table for the configured model, surfaced
///      via `LlmClient::max_context_tokens()` (or any equivalent the
///      caller passes through `connector_max`). Tagged
///      [`BudgetSource::ConnectorTable`].
///   3. [`DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS`] — a conservative universal
///      fallback so token-based compaction stays on for non-curated
///      models instead of silently disabling. Tagged
///      [`BudgetSource::UniversalFallback`].
///
/// `purpose_override` carries tier 1; `connector_max` carries tier 2.
/// Both are optional so callers without a live value can pass `None` and
/// still get the fallback.
///
/// Why a typed [`ContextBudget`]: the previous `u64`-only signature lost
/// the tier provenance, so callers couldn't tell whether the value came
/// from user config, the connector, or the silent fallback. Surfacing
/// the source as a tag lets the dispatch wrapper log which tier won and
/// gives operators a clean signal for "this model's window is unknown,
/// we're guessing 200K".
pub fn resolve_context_budget(
    purpose_override: Option<u64>,
    connector_max: Option<u64>,
) -> ContextBudget {
    if let Some(value) = purpose_override {
        return ContextBudget {
            max_input_tokens: value,
            source: BudgetSource::PurposeOverride,
        };
    }
    if let Some(value) = connector_max {
        return ContextBudget {
            max_input_tokens: value,
            source: BudgetSource::ConnectorTable,
        };
    }
    ContextBudget {
        max_input_tokens: DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS,
        source: BudgetSource::UniversalFallback,
    }
}

/// Convenience: pull `purposes.<kind>.max_context_tokens` from a
/// `DaemonConfig`. Returns `None` when no purpose is configured for `kind`
/// or the override is unset; in that case the caller should drop into
/// tier 2 / tier 3 of [`resolve_context_budget`].
pub fn purpose_max_context_override(
    config: Option<&DaemonConfig>,
    kind: PurposeKind,
) -> Option<u64> {
    config
        .and_then(|cfg| cfg.purposes.get(kind))
        .and_then(|p| p.max_context_tokens)
}

/// Shared resolution logic: takes an optional `LlmConfig` reference and
/// resolves connector, model, base_url, api_key with env-var fallbacks.
fn resolve_llm_config_from(llm_config: Option<&LlmConfig>) -> ResolvedLlmConfig {
    let connector = llm_config
        .map(|c| c.connector.trim().to_lowercase())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(default_connector);

    let default_api_key_env = default_api_key_env(&connector);
    let default_model_env = default_model_env(&connector);
    let default_base_url_env = default_base_url_env(&connector);

    let api_key_env = llm_config
        .and_then(|c| c.api_key_env.as_deref())
        .unwrap_or(default_api_key_env.as_str());

    let mut api_key = llm_config
        .and_then(|c| c.secret.as_ref())
        .and_then(|secret| read_secret_from_backend(secret, &connector))
        .unwrap_or_default();

    if api_key.is_empty() {
        api_key = std::env::var(api_key_env).unwrap_or_default();
    }

    let model = llm_config
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(default_model_env).ok())
        .unwrap_or_else(|| default_llm_model(&connector));

    let base_url = llm_config
        .and_then(|c| c.base_url.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(default_base_url_env).ok())
        .unwrap_or_else(|| {
            parse_connector_or_openai(&connector)
                .default_http_base_url()
                .to_string()
        });

    let temperature = llm_config.and_then(|c| c.temperature);
    let top_p = llm_config.and_then(|c| c.top_p);
    let max_tokens = llm_config.and_then(|c| c.max_tokens);
    let hosted_tool_search = llm_config.and_then(|c| c.hosted_tool_search);
    let aws_profile = llm_config.and_then(|c| c.aws_profile.clone());

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_tokens,
        hosted_tool_search,
        aws_profile,
    }
}

/// Resolve a per-connection [`ResolvedLlmConfig`] from a [`ConnectionConfig`].
///
/// Used by the connection registry (#9) to build one client per declared
/// connection. A [`ConnectionConfig`] holds only connector-identity fields
/// (endpoint, credentials, aws profile); it does not carry model, temperature,
/// hosted-tool-search, or `max_tokens` — those belong to purpose configs
/// (#10), which will supply overrides at dispatch time.
///
/// For now, this resolver fills the missing per-purpose fields from
/// `fallback_llm` (the top-level `[llm]` block) when present, then from
/// connector defaults / env vars. That keeps existing single-config installs
/// working until #10 lands.
pub fn resolve_connection_llm_config(
    connection: &ConnectionConfig,
    fallback_llm: Option<&LlmConfig>,
) -> ResolvedLlmConfig {
    let connector = connection.connector_type().to_string();
    let default_api_key_env = default_api_key_env(&connector);
    let default_model_env = default_model_env(&connector);
    let default_base_url_env = default_base_url_env(&connector);

    // Per-connector fields.
    let (conn_base_url, conn_api_key_env, conn_secret, conn_aws_profile): (
        Option<String>,
        Option<String>,
        Option<SecretConfig>,
        Option<String>,
    ) = match connection {
        ConnectionConfig::OpenAi(OpenAiConnection {
            base_url,
            api_key_env,
            secret,
        })
        | ConnectionConfig::Anthropic(AnthropicConnection {
            base_url,
            api_key_env,
            secret,
        }) => (base_url.clone(), api_key_env.clone(), secret.clone(), None),
        ConnectionConfig::Ollama(OllamaConnection { base_url }) => {
            (base_url.clone(), None, None, None)
        }
        ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile,
            region,
            base_url,
        }) => {
            // Bedrock historically used `base_url` to encode the region when
            // no explicit URL was set. Preserve that shape: prefer `base_url`,
            // fall back to `region`.
            let effective_base = base_url
                .clone()
                .or_else(|| region.clone())
                .filter(|v| !v.trim().is_empty());
            (effective_base, None, None, aws_profile.clone())
        }
    };

    // API key: connection secret → connection env var → fallback env var.
    let api_key_env_name = conn_api_key_env
        .as_deref()
        .unwrap_or(default_api_key_env.as_str());
    let mut api_key = conn_secret
        .as_ref()
        .and_then(|secret| read_secret_from_backend(secret, &connector))
        .unwrap_or_default();
    if api_key.is_empty() {
        api_key = std::env::var(api_key_env_name).unwrap_or_default();
    }

    // Base URL resolution.
    let base_url = conn_base_url
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(&default_base_url_env).ok())
        .unwrap_or_else(|| {
            parse_connector_or_openai(&connector)
                .default_http_base_url()
                .to_string()
        });

    // Model / tuning: not on the connection. Use the legacy `[llm]` block as
    // a placeholder until purpose configs (#10) provide per-request overrides.
    // If the fallback's connector differs from this connection's, its `model`
    // value is wrong for this connector, so we skip it.
    let fallback_model = fallback_llm
        .filter(|c| c.connector.trim().to_lowercase() == connector)
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty());
    let model = fallback_model
        .or_else(|| std::env::var(&default_model_env).ok())
        .unwrap_or_else(|| default_llm_model(&connector));

    let (temperature, top_p, max_tokens, hosted_tool_search) = fallback_llm
        .filter(|c| c.connector.trim().to_lowercase() == connector)
        .map(|c| (c.temperature, c.top_p, c.max_tokens, c.hosted_tool_search))
        .unwrap_or((None, None, None, None));

    let aws_profile = conn_aws_profile.or_else(|| {
        fallback_llm
            .filter(|c| c.connector.trim().to_lowercase() == connector)
            .and_then(|c| c.aws_profile.clone())
    });

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_tokens,
        hosted_tool_search,
        aws_profile,
    }
}
