//! Config resolution helpers — turn the raw [`DaemonConfig`] schema
//! into the resolved views consumed by the rest of the daemon.
//!
//! Extracted from `config.rs` (#41). All of the `resolve_*` functions
//! plus the small connector-default helpers live here. The schema
//! itself, the secret-store env helpers (`default_api_key_env`,
//! `default_model_env`, `default_base_url_env`), and `default_connector`
//! stay in [`super`] because they are shared with serde defaults and
//! the secret backends.

use desktop_assistant_core::context_window::snap_down_to_common;
use desktop_assistant_core::ports::llm::{BudgetSource, ContextBudget};
use desktop_assistant_core::ports::store::LearnedWindow;

use crate::connections::{
    AnthropicConnection, AzureConnection, BedrockConnection, ConnectionConfig, Connector,
    GoogleConnection, OllamaConnection, OpenAiConnection, OpenRouterConnection,
};
use crate::purposes::PurposeKind;

use super::secrets::read_secret_from_backend;
use super::{
    ConnectorExtras, DaemonConfig, EmbeddingsSettingsView, LlmConfig, ResolvedLlmConfig,
    ResolvedPersistenceConfig, SecretConfig, default_api_key_env, default_base_url_env,
    default_connector, default_database_max_connections, default_git_remote_name,
    default_model_env, default_push_on_update,
};

/// Azure's conventional key env var, used when a connection doesn't set an
/// explicit `api_key_env`. The connector key `azure` derives `AZURE_API_KEY`
/// (see [`default_api_key_env`]), but Azure OpenAI's documented variable is
/// `AZURE_OPENAI_API_KEY`, so the resolver defaults to that instead.
const AZURE_DEFAULT_API_KEY_ENV: &str = "AZURE_OPENAI_API_KEY";

/// Resolve a value from the connection field first, then a prioritized list of
/// environment variables, returning the first non-empty match. Used to fill
/// Google's `project` / `location` from the connection or the GCP-conventional
/// env vars.
fn field_or_env(field: Option<&str>, env_vars: &[&str]) -> Option<String> {
    if let Some(v) = field.map(str::trim).filter(|v| !v.is_empty()) {
        return Some(v.to_string());
    }
    env_vars.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

/// Whether `model` is, by name, a clearly text-*generation* model family that
/// cannot produce embeddings.
///
/// Deliberately conservative: it matches only on the *start* of the model name
/// (the family), and it first excludes anything that self-identifies as an
/// embedding model. The startup embed probe (see [`crate::embedding_probe`]) is
/// the general, model-agnostic safety net; this name check just gives a faster,
/// clearer signal for the common misconfiguration, so a false positive (wrongly
/// rejecting a valid embedder) is worse than a false negative (the probe still
/// catches it).
fn is_known_generation_model(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }

    // Never reject a model that self-identifies as an embedding model — this
    // protects embedding variants that share a family name with a generation
    // model (e.g. `qwen3-embedding`, `mistral-embed`, `granite-embedding`).
    const EMBED_MARKERS: &[&str] = &["embed", "bge", "gte", "e5-", "minilm", "arctic-embed"];
    if EMBED_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        return false;
    }

    // Families that are unambiguously chat/generation models. Matched as a
    // prefix so a substring elsewhere in a longer, unrelated name cannot
    // trigger a false positive.
    const GEN_FAMILIES: &[&str] = &[
        "gpt-oss",
        "gpt-4",
        "gpt-3.5",
        "gpt2",
        "llama",
        "codellama",
        "tinyllama",
        "mistral",
        "mixtral",
        "gemma",
        "phi",
        "qwen",
        "deepseek",
        "command-r",
        "vicuna",
        "orca",
        "solar",
        "dolphin",
        "wizardlm",
        "starling",
        "zephyr",
        "falcon",
    ];
    GEN_FAMILIES
        .iter()
        .any(|family| normalized.starts_with(family))
}

/// Secondary early-UX guard (#499): reject a model that is clearly a
/// text-generation model configured as the embedder.
///
/// Returns `Err(reason)` to reject. This is deliberately conservative and
/// name-based — the startup embed probe (see [`crate::embedding_probe`]) is the
/// general, model-agnostic mechanism that catches *any* non-embedding backend
/// regardless of name. This guard only gives a faster, clearer signal for the
/// common misconfiguration, so it must never false-reject an unusual-but-valid
/// embedding model.
pub(crate) fn reject_generation_model_embedder(model: &str) -> Result<(), String> {
    if is_known_generation_model(model) {
        Err(format!(
            "'{}' is a known text-generation model and cannot produce embeddings; \
             configure a dedicated embedding model (for example nomic-embed-text, \
             mxbai-embed-large, or text-embedding-3-small)",
            model.trim()
        ))
    } else {
        Ok(())
    }
}

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
    let available = connector_supports_embeddings(&connector);

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
    let available = connector_supports_embeddings(&resolved.connector);
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

/// Whether a connector string exposes embeddings, via the typed
/// [`Connector::supports_embeddings`] allowlist. An unrecognised connector
/// defaults to `true` (offer embeddings) so a bespoke/future connector isn't
/// silently denied; the startup embed probe is the general safety net.
///
/// Replaces the historical literal `connector != "anthropic"`: OpenRouter has
/// no embeddings (`false`), while Azure and Google do (`true`).
fn connector_supports_embeddings(connector: &str) -> bool {
    Connector::parse(connector)
        .map(|c| c.supports_embeddings())
        .unwrap_or(true)
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

/// Resolve the holistic-consolidation LLM: `[backend_tasks.consolidation_llm]`
/// if set, otherwise the same resolution as the rest of the backend tasks
/// (`[backend_tasks.llm]` → top-level `[llm]`). This lets extraction run on a
/// cheap model while consolidation uses a stronger one.
pub fn resolve_consolidation_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    let consolidation_llm = config.and_then(|c| c.backend_tasks.consolidation_llm.as_ref());
    if consolidation_llm.is_some() {
        resolve_llm_config_from(consolidation_llm)
    } else {
        resolve_backend_tasks_llm_config(config)
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
///      the user always wins. Tagged [`BudgetSource::PurposeOverride`]. For
///      Ollama this same value is read back via the per-turn budget and
///      provisioned as `num_ctx` (clamped to the model ceiling and the
///      per-connection hard cap), so budget and runtime window agree.
///   2. The connector's effective window for the configured model, surfaced
///      via `LlmClient::max_context_tokens()` (or any equivalent the caller
///      passes through `connector_max`). For Ollama this already folds the
///      per-connection `max_context_tokens` hard cap. Tagged
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
/// Note: a purpose override is deliberately *not* clamped to `connector_max`
/// here — `connector_max` is read before the per-turn budget is installed, so
/// for Ollama it reflects the default window, not the override. Clamping would
/// wrongly cap an override down to that default. The model's real ceiling and
/// the per-connection hard cap bind at the connector (`effective_num_ctx`)
/// instead, and the learned-overflow safety-net catches a genuinely over-long
/// prompt.
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

/// Apply a learned context-window observation to an already-resolved
/// [`ContextBudget`] (issues #343, #425).
///
/// This composes *after* [`resolve_context_budget`] so the documented
/// precedence holds: `purpose-override > learned (DOWN-ONLY cap, floored by a
/// proven success) > connector / Ollama-effective-window > 200K fallback`.
///
/// Two forces combine, in order:
///
/// **1. Overflow cap (down-only, snapped).** An observed overflow ceiling caps
/// the budget DOWN, but never to the scraped integer directly — it is snapped to
/// a stable rung via [`snap_down_to_common`]. Guards:
///   - **Invalidation.** The observation carries the `configured_window` in
///     force when it was seen; if that differs from the current budget the user
///     changed the window, so the stale observation is ignored and the fresh
///     ceiling stands.
///   - **Snap sanity.** [`snap_down_to_common`] returns `None` for a value below
///     its smallest rung, so a pathological parse (the 534-token poison of #425)
///     can never pin the budget — it's simply not applied.
///   - **Down-only.** The snapped cap applies only when strictly below the
///     resolved budget.
///
/// **2. Success floor (recovery).** The largest provider-measured input we've
/// seen this model ACCEPT floors the result (bounded by the configured budget so
/// we never exceed it). This is the #425 safety net: even if an overflow cap or
/// a bad parse tries to drop the budget below a size the model has demonstrably
/// handled, the floor holds it up — and as larger prompts succeed the budget
/// climbs back, which the old pure down-only ratchet could never do.
///
/// Returns the (possibly unchanged) budget; when either force moves it, the
/// source is re-tagged [`BudgetSource::LearnedCap`].
pub fn apply_learned_cap(budget: ContextBudget, learned: Option<LearnedWindow>) -> ContextBudget {
    let Some(learned) = learned else {
        return budget;
    };
    let resolved = budget.max_input_tokens;
    let mut effective = resolved;

    // Force 1: down-only overflow cap, snapped. Invalidation (configured_window
    // must match), snap-sanity (None for pathologically small), and down-only
    // are all expressed in this single chained guard.
    if let (Some(observed), Some(configured)) = (learned.observed_limit, learned.configured_window)
        && configured == resolved
        && let Some(snapped) = snap_down_to_common(observed)
        && snapped < effective
    {
        effective = snapped;
    }

    // Force 2: success floor — never cap below a size the model has proven it
    // accepts (bounded by the configured budget). Independent of the overflow
    // observation's invalidation: proven-good is proven-good.
    if let Some(high_water) = learned.max_success_input {
        effective = effective.max(high_water.min(resolved));
    }

    if effective == resolved {
        return budget; // neither force moved it — keep the resolved tier/source
    }
    ContextBudget {
        max_input_tokens: effective,
        source: BudgetSource::LearnedCap,
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
        // Legacy `[llm]` block has no per-connection timeout / context-cap
        // fields; connectors fall back to their shared stall-budget defaults
        // and "max available" context.
        connect_timeout_secs: None,
        stream_timeout_secs: None,
        keep_warm: false,
        max_context_tokens: None,
        // The legacy `[llm]` block has no Azure/Google surface fields; the
        // multi-field configs are `[connections]`-only.
        extras: ConnectorExtras::None,
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
            ..
        })
        | ConnectionConfig::Anthropic(AnthropicConnection {
            base_url,
            api_key_env,
            secret,
            ..
        })
        | ConnectionConfig::OpenRouter(OpenRouterConnection {
            base_url,
            api_key_env,
            secret,
            ..
        }) => (base_url.clone(), api_key_env.clone(), secret.clone(), None),
        ConnectionConfig::Azure(AzureConnection {
            base_url,
            api_key_env,
            secret,
            ..
        }) => (
            base_url.clone(),
            // Azure's key env defaults to AZURE_OPENAI_API_KEY (not the derived
            // AZURE_API_KEY) when the connection doesn't set one explicitly.
            Some(
                api_key_env
                    .clone()
                    .unwrap_or_else(|| AZURE_DEFAULT_API_KEY_ENV.to_string()),
            ),
            secret.clone(),
            None,
        ),
        ConnectionConfig::Google(GoogleConnection {
            base_url,
            api_key_env,
            secret,
            ..
        }) => (base_url.clone(), api_key_env.clone(), secret.clone(), None),
        ConnectionConfig::Ollama(OllamaConnection { base_url, .. }) => {
            (base_url.clone(), None, None, None)
        }
        ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile,
            region,
            base_url,
            secret,
            ..
        }) => {
            // Bedrock historically used `base_url` to encode the region when
            // no explicit URL was set. Preserve that shape: prefer `base_url`,
            // fall back to `region`.
            let effective_base = base_url
                .clone()
                .or_else(|| region.clone())
                .filter(|v| !v.trim().is_empty());
            // A stored secret (`ACCESS:SECRET[:SESSION]`) resolves into `api_key`,
            // which the Bedrock client parses into static AWS credentials.
            (effective_base, None, secret.clone(), aws_profile.clone())
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

    // Per-connection knobs present on every variant: streaming stall budgets
    // (`None` keeps the connector's shared default) and the context-window hard
    // cap (`None` = "max available"). `keep_warm` is Ollama-only and off
    // everywhere else.
    let (connect_timeout_secs, stream_timeout_secs, keep_warm, max_context_tokens) =
        match connection {
            ConnectionConfig::Anthropic(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::OpenAi(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::OpenRouter(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::Azure(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::Google(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::Bedrock(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                false,
                c.max_context_tokens,
            ),
            ConnectionConfig::Ollama(c) => (
                c.connect_timeout_secs,
                c.stream_timeout_secs,
                c.keep_warm.unwrap_or(false),
                c.max_context_tokens,
            ),
        };

    // Provider-specific surface/auth/endpoint knobs the flat struct can't hold.
    // Google resolves project/location from the connection then GCP-conventional
    // env vars; Azure passes its surface/auth/version strings through for the
    // factory to parse.
    let extras = match connection {
        ConnectionConfig::Azure(c) => ConnectorExtras::Azure {
            api_surface: c.api_surface.clone(),
            auth_mode: c.auth_mode.clone(),
            api_version: c.api_version.clone(),
        },
        ConnectionConfig::Google(c) => ConnectorExtras::Google {
            project: field_or_env(
                c.project.as_deref(),
                &["GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT"],
            ),
            location: field_or_env(
                c.location.as_deref(),
                &["GOOGLE_CLOUD_LOCATION", "GOOGLE_LOCATION"],
            ),
            auth_mode: c.auth_mode.clone(),
            credentials_path: c.credentials_path.clone(),
        },
        _ => ConnectorExtras::None,
    };

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
        connect_timeout_secs,
        stream_timeout_secs,
        keep_warm,
        max_context_tokens,
        extras,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_model_as_embedder_is_rejected_at_config_resolve() {
        // Clearly text-generation models configured as the embedder are
        // rejected loudly at resolve time (secondary early-UX guard; the
        // startup probe is the general mechanism).
        for gen_model in [
            "gpt-oss:120b",
            "llama3.1:8b",
            "qwen2.5:7b",
            "mistral:7b",
            "gemma2:9b",
            "phi3:mini",
            "deepseek-r1:14b",
        ] {
            assert!(
                reject_generation_model_embedder(gen_model).is_err(),
                "{gen_model} is a generation model and must be rejected as an embedder"
            );
        }

        // Real embedding models (including ones that share a family name with a
        // generation model, e.g. qwen/mistral) must NOT be false-rejected — the
        // probe is the general safety net, so this guard stays conservative.
        for emb in [
            "nomic-embed-text",
            "mxbai-embed-large",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "snowflake-arctic-embed",
            "all-minilm",
            "bge-large-en-v1.5",
            "qwen3-embedding",
            "mistral-embed",
        ] {
            assert!(
                reject_generation_model_embedder(emb).is_ok(),
                "{emb} is a valid embedding model and must not be rejected"
            );
        }
    }
}
