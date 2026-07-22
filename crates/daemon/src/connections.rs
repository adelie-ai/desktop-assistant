//! Named-connection config schema.
//!
//! A `connections` map keyed by a user-chosen slug ([`ConnectionId`]).
//! Each connection owns its own credentials/endpoint and declares its connector
//! type via a `#[serde(tag = "type")]` payload, replacing the legacy single
//! `[llm]` block which hard-coded one global connector.
//!
//! Schema-only: migration lives in [`super::config`] so it can share I/O
//! helpers with the wider config layer.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::{LlmConfig, SecretConfig};

/// Maximum length of a [`ConnectionId`] slug.
pub const CONNECTION_ID_MAX_LEN: usize = 64;

/// Validated slug identifying a connection entry.
///
/// Matches `^[a-z0-9][a-z0-9_-]*$` and is at most [`CONNECTION_ID_MAX_LEN`]
/// characters long. The first character must be alphanumeric so ids remain
/// usable as TOML bare keys and path fragments without quoting.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnectionId(String);

/// Error raised when a raw string is not a valid [`ConnectionId`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConnectionIdError {
    #[error("connection id must not be empty")]
    Empty,
    #[error(
        "connection id {id:?} is too long ({len} chars); \
         maximum is {max} characters"
    )]
    TooLong { id: String, len: usize, max: usize },
    #[error(
        "connection id {id:?} is invalid; must match [a-z0-9][a-z0-9_-]* \
         (lowercase ASCII alphanumerics, underscores, and hyphens; \
         first char must be alphanumeric)"
    )]
    InvalidChars { id: String },
}

impl ConnectionId {
    /// Parse and validate a connection id slug.
    pub fn new(raw: impl Into<String>) -> Result<Self, ConnectionIdError> {
        let raw = raw.into();
        if raw.is_empty() {
            return Err(ConnectionIdError::Empty);
        }
        if raw.len() > CONNECTION_ID_MAX_LEN {
            return Err(ConnectionIdError::TooLong {
                id: raw.clone(),
                len: raw.len(),
                max: CONNECTION_ID_MAX_LEN,
            });
        }

        // First char: lowercase ASCII letter or digit.
        // Remaining chars: same set plus `_` and `-`.
        let is_valid_first = |c: char| c.is_ascii_digit() || c.is_ascii_lowercase();
        let is_valid_rest = |c: char| is_valid_first(c) || c == '_' || c == '-';

        let mut chars = raw.chars();
        let first = chars.next().expect("non-empty checked above");
        if !is_valid_first(first) || chars.any(|c| !is_valid_rest(c)) {
            return Err(ConnectionIdError::InvalidChars { id: raw });
        }

        Ok(Self(raw))
    }

    /// Borrow the underlying slug as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying slug.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ConnectionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for ConnectionId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ConnectionId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        ConnectionId::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Connector-specific configuration for a single named connection.
///
/// Serialized as a tagged enum under `type = "..."`. Each variant owns only the
/// fields relevant to its connector type. The connection id (map key) and any
/// purpose-level model/parameter overrides are intentionally *not* stored here
/// — those live on the [`RootConnectionsConfig`] wrapper and on purpose configs
/// (#10) respectively.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum ConnectionConfig {
    Anthropic(AnthropicConnection),
    #[serde(rename = "openai")]
    OpenAi(OpenAiConnection),
    #[serde(rename = "openrouter")]
    OpenRouter(OpenRouterConnection),
    #[serde(rename = "azure")]
    Azure(AzureConnection),
    #[serde(rename = "google")]
    Google(GoogleConnection),
    Bedrock(BedrockConnection),
    Ollama(OllamaConnection),
}

impl ConnectionConfig {
    /// Short connector-type identifier (matches the `type =` tag).
    pub fn connector_type(&self) -> &'static str {
        self.connector().as_str()
    }

    /// Typed [`Connector`] discriminant — same shape as
    /// [`Self::connector_type`] but lifted into an enum so per-connector
    /// defaults (base URL, default model, etc.) can hang off the type
    /// instead of leaking string-match tables across the daemon (#47).
    pub fn connector(&self) -> Connector {
        match self {
            Self::Anthropic(_) => Connector::Anthropic,
            Self::OpenAi(_) => Connector::OpenAi,
            Self::OpenRouter(_) => Connector::OpenRouter,
            Self::Azure(_) => Connector::Azure,
            Self::Google(_) => Connector::Google,
            Self::Bedrock(_) => Connector::Bedrock,
            Self::Ollama(_) => Connector::Ollama,
        }
    }

    /// Set (or clear, with `None`) this connection's secret-store coordinate.
    ///
    /// Only credential-bearing connectors carry a `secret` field. Ollama talks
    /// to a local/self-hosted endpoint with no API key, so setting a credential
    /// on it is a caller error rather than a silent no-op.
    pub fn set_secret(&mut self, secret: Option<SecretConfig>) -> Result<(), &'static str> {
        match self {
            Self::Anthropic(c) => c.secret = secret,
            Self::OpenAi(c) => c.secret = secret,
            Self::OpenRouter(c) => c.secret = secret,
            Self::Azure(c) => c.secret = secret,
            // Google's secret is consumed only in api-key mode, but allow
            // setting it so operators can pre-provision the credential
            // regardless of the connection's current `auth_mode`.
            Self::Google(c) => c.secret = secret,
            Self::Bedrock(c) => c.secret = secret,
            Self::Ollama(_) => return Err("ollama connections do not use a stored credential"),
        }
        Ok(())
    }
}

/// Typed connector identity. The wire/config layer continues to round-trip
/// through `&str` (TOML, env vars, the legacy `[llm].connector` field) but
/// internally every per-connector default — base URL, default chat model,
/// embedding model, hosted-tool-search availability, etc. — is a method on
/// this enum so adding a new connector or fixing an alias is a single
/// match-arm change instead of a 5-table edit (#47).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Connector {
    Ollama,
    Anthropic,
    Bedrock,
    OpenAi,
    OpenRouter,
    Azure,
    Google,
}

impl Connector {
    /// Canonical short name. Matches the `type =` tag in
    /// `[connections.<id>]` and the legacy `[llm].connector` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Anthropic => "anthropic",
            Self::Bedrock => "bedrock",
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
            Self::Azure => "azure",
            Self::Google => "google",
        }
    }

    /// Parse a connector identifier with alias support.
    ///
    /// Accepts:
    /// - canonical names (`ollama`, `anthropic`, `bedrock`, `openai`)
    /// - the legacy `aws-bedrock` alias for [`Self::Bedrock`]
    /// - leading/trailing whitespace and any case
    ///
    /// Returns `None` for unrecognised values; callers that need a
    /// default for unknown input should chain `.unwrap_or(Connector::OpenAi)`
    /// (or whichever default is right for their context).
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "anthropic" => Some(Self::Anthropic),
            "bedrock" | "aws-bedrock" => Some(Self::Bedrock),
            "openai" => Some(Self::OpenAi),
            "openrouter" => Some(Self::OpenRouter),
            "azure" => Some(Self::Azure),
            // Deliberately no "gemini" alias: an unknown `type = "gemini"` must
            // stay unrecognised so the negative config test keeps rejecting it.
            "google" => Some(Self::Google),
            _ => None,
        }
    }

    /// Default base URL for this connector. Empty string for connectors
    /// that don't ship a default (so `.to_string()` and `format!` callers
    /// don't have to special-case `Option`).
    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Ollama => {
                desktop_assistant_llm_ollama::OllamaClient::get_default_base_url().unwrap_or("")
            }
            Self::Anthropic => {
                desktop_assistant_llm_anthropic::AnthropicClient::get_default_base_url()
                    .unwrap_or("")
            }
            Self::Bedrock => {
                desktop_assistant_llm_bedrock::BedrockClient::get_default_base_url().unwrap_or("")
            }
            Self::OpenAi => {
                desktop_assistant_llm_openai::OpenAiClient::get_default_base_url().unwrap_or("")
            }
            Self::OpenRouter => {
                desktop_assistant_llm_openrouter::OpenRouterClient::get_default_base_url()
                    .unwrap_or("")
            }
            // Azure's host is resource-specific (`https://<name>.openai.azure.com`),
            // so there is no shippable default — the operator must set it.
            Self::Azure => "",
            Self::Google => {
                desktop_assistant_llm_google::GoogleClient::get_default_base_url().unwrap_or("")
            }
        }
    }

    /// Default chat-completion model for this connector. Empty string if
    /// the connector doesn't ship a default.
    pub fn default_chat_model(self) -> &'static str {
        match self {
            Self::Ollama => {
                desktop_assistant_llm_ollama::OllamaClient::get_default_model().unwrap_or("")
            }
            Self::Anthropic => {
                desktop_assistant_llm_anthropic::AnthropicClient::get_default_model().unwrap_or("")
            }
            Self::Bedrock => {
                desktop_assistant_llm_bedrock::BedrockClient::get_default_model().unwrap_or("")
            }
            Self::OpenAi => {
                desktop_assistant_llm_openai::OpenAiClient::get_default_model().unwrap_or("")
            }
            Self::OpenRouter => {
                desktop_assistant_llm_openrouter::OpenRouterClient::get_default_model()
                    .unwrap_or("")
            }
            // Azure deployments are operator-named; no default deployment exists.
            Self::Azure => "",
            Self::Google => {
                desktop_assistant_llm_google::GoogleClient::get_default_model().unwrap_or("")
            }
        }
    }

    /// Default model for backend tasks (titling, dreaming, summary).
    /// Diverges from [`Self::default_chat_model`] for non-Ollama
    /// connectors — picks a smaller/cheaper model when the connector has
    /// one.
    pub fn default_backend_chat_model(self) -> &'static str {
        match self {
            Self::Ollama => {
                desktop_assistant_llm_ollama::OllamaClient::get_default_model().unwrap_or("")
            }
            Self::Anthropic => "claude-haiku-4-5-20251001",
            Self::Bedrock => "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            Self::OpenAi => "gpt-4o-mini",
            // OpenRouter has no dedicated cheaper-backend static; reuse its
            // vetted default model rather than ship a possibly-stale cheap slug
            // (operators override per purpose when they want a cheaper backend).
            Self::OpenRouter => {
                desktop_assistant_llm_openrouter::OpenRouterClient::get_default_model()
                    .unwrap_or("")
            }
            // Azure deployments are operator-named; no default backend deployment.
            Self::Azure => "",
            Self::Google => desktop_assistant_llm_google::GoogleClient::get_default_backend_model()
                .unwrap_or(""),
        }
    }

    /// Default embedding model for this connector. Anthropic doesn't
    /// ship embeddings, so `Self::Anthropic` returns an empty string —
    /// callers should check [`Self::supports_embeddings`] first or
    /// substitute `Connector::OpenAi`.
    pub fn default_embedding_model(self) -> &'static str {
        match self {
            Self::Ollama => "nomic-embed-text",
            Self::Bedrock => "amazon.titan-embed-text-v2:0",
            Self::OpenAi => "text-embedding-3-small",
            // Anthropic and OpenRouter ship no embeddings — callers must check
            // `supports_embeddings` first (returns `false` for both).
            Self::Anthropic => "",
            Self::OpenRouter => "",
            // Azure serves OpenAI's embedding models under an operator-named
            // deployment; the base-model default is the small text-embedding-3.
            Self::Azure => "text-embedding-3-small",
            Self::Google => {
                desktop_assistant_llm_google::GoogleClient::get_default_embedding_model()
                    .unwrap_or("")
            }
        }
    }

    /// Default base URL for connectors that target an HTTP endpoint
    /// directly (i.e. not Bedrock, which uses a region instead). Used
    /// as the fallback when a [`crate::config::ResolvedLlmConfig`]
    /// resolver runs out of more specific sources.
    pub fn default_http_base_url(self) -> &'static str {
        match self {
            Self::Ollama => "http://localhost:11434",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Bedrock => "us-east-1",
            Self::OpenAi => "https://api.openai.com/v1",
            Self::OpenRouter => {
                desktop_assistant_llm_openrouter::OpenRouterClient::get_default_base_url()
                    .unwrap_or("")
            }
            // Azure's endpoint is resource-specific; empty so the resolver
            // leaves `base_url` empty and preflight names the missing endpoint.
            Self::Azure => "",
            // Empty on purpose: Google composes the Vertex host from `location`
            // when no explicit base_url is set, so the resolver must NOT fill a
            // fixed regional URL here (that would pin the host to one region and
            // mismatch a differently-located deployment). The display default is
            // still available via `default_base_url`.
            Self::Google => "",
        }
    }

    /// Whether this connector exposes an embeddings endpoint. Explicit
    /// per-variant allowlist: Anthropic and OpenRouter don't; every other
    /// connector (including Azure and Google) does.
    pub fn supports_embeddings(self) -> bool {
        match self {
            Self::Anthropic | Self::OpenRouter => false,
            Self::Ollama | Self::Bedrock | Self::OpenAi | Self::Azure | Self::Google => true,
        }
    }

    /// Whether this connector supports server-side hosted tool search
    /// (used by the model-defaults view to gate the toggle in the KCM).
    pub fn supports_hosted_tool_search(self) -> bool {
        matches!(self, Self::OpenAi | Self::Anthropic)
    }
}

impl fmt::Display for Connector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Anthropic-specific connection fields.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// Seconds to wait for the first streaming response (headers/first event)
    /// before treating the request as stalled. Overrides the connector-shared
    /// [`STREAM_CONNECT_TIMEOUT`](desktop_assistant_llm_http::STREAM_CONNECT_TIMEOUT)
    /// default (30s). Useful for slow local models (e.g. a large GGUF doing a
    /// long prompt-eval on CPU).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks before treating the stream as
    /// stalled. Overrides the connector-shared
    /// [`STREAM_EVENT_TIMEOUT`](desktop_assistant_llm_http::STREAM_EVENT_TIMEOUT)
    /// default (60s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available" (use the model's curated/reported maximum). `Some(n)` clamps
    /// the daemon's input budget to `min(n, reported)` — e.g. to bound prompt
    /// size for billing. Cloud connectors have no `num_ctx` to pin, so this
    /// only constrains how much input the daemon packs per turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// OpenAI-compatible connection fields.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// Seconds to wait for the first streaming response before treating the
    /// request as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available". See [`AnthropicConnection::max_context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// OpenRouter connection fields.
///
/// OpenRouter is an OpenAI-compatible aggregator, so its config surface is
/// identical to [`OpenAiConnection`]; a distinct struct keeps the connector
/// identity typed and lets the two diverge later without a wire break.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// Seconds to wait for the first streaming response before treating the
    /// request as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available". See [`AnthropicConnection::max_context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Azure OpenAI (Microsoft Foundry) connection fields.
///
/// Extends the OpenAI-compatible base with the resource-specific knobs Azure
/// needs: which REST surface to speak, how to authenticate, and (classic only)
/// the `api-version`. The `model` is an operator-provisioned *deployment* name,
/// carried through the shared resolver/purpose layer like every other model.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureConnection {
    /// The resource endpoint, e.g. `https://<name>.openai.azure.com`. Required
    /// (there is no shippable default); preflight names it when missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Env var holding the resource key. Defaults to `AZURE_OPENAI_API_KEY`
    /// (resolved in [`crate::config::resolve_connection_llm_config`]) rather
    /// than the derived `AZURE_API_KEY`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// Which REST surface to target: `v1` (GA, default) or `classic` (legacy
    /// deployments path). Parsed by the factory into the connector's
    /// `ApiSurface` enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_surface: Option<String>,
    /// How to authenticate: `api_key` (default) or `entra` (Entra ID / managed
    /// identity). Parsed by the factory into the connector's `AuthMode` enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
    /// `api-version` for the classic surface; ignored on `v1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    /// Seconds to wait for the first streaming response before treating the
    /// request as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available". See [`AnthropicConnection::max_context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Google Vertex AI / Gemini connection fields.
///
/// Vertex is project/region-scoped and cloud-credential authenticated; the
/// simpler Gemini API (AI Studio) is folded in as `auth_mode = api_key`. The
/// `model` is the Gemini model id, carried through the shared resolver/purpose
/// layer.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleConnection {
    /// Explicit host override; when unset the connector composes the Vertex
    /// host from `location` (or uses the fixed Gemini API host in api-key mode),
    /// so this stays `None` for a normal region-scoped setup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Env var holding the API key (api-key / AI-Studio mode only). Defaults to
    /// the derived `GOOGLE_API_KEY`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Used only in `auth_mode = api_key`; allowed on any connection so the
    /// credential can be pre-provisioned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// GCP project id (Vertex). Falls back to `GOOGLE_CLOUD_PROJECT` /
    /// `GOOGLE_PROJECT` at resolve time when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Vertex region, e.g. `us-central1`. Falls back to `GOOGLE_CLOUD_LOCATION`
    /// / `GOOGLE_LOCATION` at resolve time when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// How to authenticate: `vertex` (default, OAuth2 bearer) or `api_key`
    /// (AI Studio). Parsed by the factory into the connector's `AuthMode` enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
    /// Path to a service-account JSON key file (Vertex). When unset the
    /// connector falls back to Application Default Credentials at request time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_path: Option<String>,
    /// Seconds to wait for the first streaming response before treating the
    /// request as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available". See [`AnthropicConnection::max_context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// AWS Bedrock connection fields.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BedrockConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aws_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Present only for environments that proxy Bedrock through a private
    /// endpoint. The AWS SDK default is usually correct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Secret-store coordinate for a raw static-credential string
    /// (`ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]`). Set via the
    /// `SetConnectionSecret` command; the raw value lives only in the secret
    /// backend, never in daemon.toml. When absent, the daemon falls back to the
    /// AWS credential chain / `aws_profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    /// Seconds to wait for the first streaming response before treating the
    /// request as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// Hard ceiling on the effective context window, in tokens. `None` = "max
    /// available". See [`AnthropicConnection::max_context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Ollama (local or self-hosted) connection fields.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Seconds to wait for the first streaming response (Ollama's prompt-eval
    /// can be very slow for large models on CPU) before treating the request
    /// as stalled. Overrides the shared 30s default. See
    /// [`AnthropicConnection::connect_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Seconds to wait between streaming chunks. Overrides the shared 60s
    /// default. See [`AnthropicConnection::stream_timeout_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout_secs: Option<u64>,
    /// When `true`, the daemon keeps this connection's **interactive-purpose**
    /// model resident in Ollama's memory by periodically re-loading it, so a
    /// chat reply isn't preceded by a cold model load. Only the interactive
    /// model is kept warm — background purposes (dreaming/titling) are allowed
    /// to be unloaded when idle. Defaults to `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_warm: Option<bool>,
    /// User-imposed hard ceiling on the context window, in tokens. `None`
    /// means **"max available"** — float to whatever the model reports via
    /// `/api/show` (e.g. 32768 for qwen2.5). `Some(n)` clamps the effective
    /// window (and the `num_ctx` sent to Ollama, and the daemon's input
    /// budget) to `min(n, reported)`, e.g. to fit a machine that can't afford
    /// the model's full KV cache on CPU. Defaults to "max available".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Errors raised while validating the `[connections]` map.
#[derive(Debug, Error, PartialEq)]
pub enum ConnectionsError {
    #[error("`connections` must contain at least one entry")]
    Empty,
    #[error("duplicate connection id {0:?}")]
    DuplicateId(String),
    #[error("invalid connection id: {0}")]
    InvalidId(#[from] ConnectionIdError),
}

/// Validated collection of named connections.
///
/// Wrapping the raw `IndexMap` gives us one place to enforce "at least one
/// entry" and "no duplicate ids" without every call site re-validating.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConnectionsMap(IndexMap<ConnectionId, ConnectionConfig>);

impl ConnectionsMap {
    /// Build a validated map from an ordered list of `(id, config)` pairs.
    ///
    /// Fails on duplicate ids. Accepts any iterable so callers can build from
    /// literal arrays, vec builders, or streaming sources.
    pub fn from_pairs<I>(pairs: I) -> Result<Self, ConnectionsError>
    where
        I: IntoIterator<Item = (ConnectionId, ConnectionConfig)>,
    {
        let mut map: IndexMap<ConnectionId, ConnectionConfig> = IndexMap::new();
        for (id, conn) in pairs {
            if map.contains_key(&id) {
                return Err(ConnectionsError::DuplicateId(id.into_string()));
            }
            map.insert(id, conn);
        }
        if map.is_empty() {
            return Err(ConnectionsError::Empty);
        }
        Ok(Self(map))
    }

    /// Iterate connection id / config pairs in declaration order.
    pub fn iter(&self) -> indexmap::map::Iter<'_, ConnectionId, ConnectionConfig> {
        self.0.iter()
    }

    /// Look up a connection by id.
    pub fn get(&self, id: &ConnectionId) -> Option<&ConnectionConfig> {
        self.0.get(id)
    }
}

/// Build a [`ConnectionConfig`] from a legacy `[llm]` block.
///
/// Used by the auto-migration path. Unknown/invalid connector strings fall
/// through to OpenAI (matches legacy default behaviour).
pub(crate) fn connection_from_legacy_llm(llm: &LlmConfig) -> ConnectionConfig {
    let connector = llm.connector.trim().to_ascii_lowercase();
    match connector.as_str() {
        "anthropic" => ConnectionConfig::Anthropic(AnthropicConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
            ..Default::default()
        }),
        "ollama" => ConnectionConfig::Ollama(OllamaConnection {
            base_url: llm.base_url.clone(),
            ..Default::default()
        }),
        "openrouter" => ConnectionConfig::OpenRouter(OpenRouterConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
            ..Default::default()
        }),
        "azure" => ConnectionConfig::Azure(AzureConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
            ..Default::default()
        }),
        "google" => ConnectionConfig::Google(GoogleConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
            ..Default::default()
        }),
        "bedrock" | "aws-bedrock" => ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile: llm.aws_profile.clone(),
            // Legacy `base_url` for bedrock was actually the AWS region string
            // (see resolve_llm_config_from). Prefer it as `region` unless it
            // looks like a URL.
            region: llm
                .base_url
                .as_ref()
                .filter(|v| !v.trim().is_empty() && !v.contains("://"))
                .cloned(),
            base_url: llm.base_url.as_ref().filter(|v| v.contains("://")).cloned(),
            ..Default::default()
        }),
        // Anything else (including the legacy default "openai") maps to OpenAI.
        _ => ConnectionConfig::OpenAi(OpenAiConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
            ..Default::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_accepts_lowercase_digits_underscore_hyphen() {
        for good in [
            "a",
            "1",
            "work",
            "work_openai",
            "work-openai",
            "home_bedrock_2",
            "a0",
        ] {
            ConnectionId::new(good).unwrap_or_else(|e| panic!("{good:?} should parse: {e}"));
        }
    }

    #[test]
    fn id_rejects_empty() {
        assert_eq!(ConnectionId::new(""), Err(ConnectionIdError::Empty));
    }

    #[test]
    fn id_rejects_invalid_first_char() {
        for bad in ["_work", "-work", ".openai"] {
            let err = ConnectionId::new(bad).unwrap_err();
            assert!(
                matches!(err, ConnectionIdError::InvalidChars { .. }),
                "expected InvalidChars for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn id_rejects_invalid_chars() {
        for bad in ["Work", "work openai", "work/openai", "café", "WORK"] {
            let err = ConnectionId::new(bad).unwrap_err();
            assert!(
                matches!(err, ConnectionIdError::InvalidChars { .. }),
                "expected InvalidChars for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn id_rejects_too_long() {
        let s = "a".repeat(CONNECTION_ID_MAX_LEN + 1);
        let err = ConnectionId::new(&s).unwrap_err();
        assert!(matches!(err, ConnectionIdError::TooLong { .. }));
    }

    #[test]
    fn id_accepts_max_length() {
        let s = "a".repeat(CONNECTION_ID_MAX_LEN);
        ConnectionId::new(&s).unwrap();
    }

    #[test]
    fn id_invalid_error_cites_the_id() {
        let err = ConnectionId::new("Bad Id!").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Bad Id!"), "expected bad id in message: {msg}");
    }

    #[test]
    fn connections_map_rejects_empty() {
        let pairs: Vec<(ConnectionId, ConnectionConfig)> = vec![];
        assert_eq!(
            ConnectionsMap::from_pairs(pairs).unwrap_err(),
            ConnectionsError::Empty
        );
    }

    #[test]
    fn connections_map_rejects_duplicate_ids() {
        let id = ConnectionId::new("default").unwrap();
        let pairs = vec![
            (
                id.clone(),
                ConnectionConfig::OpenAi(OpenAiConnection::default()),
            ),
            (
                id.clone(),
                ConnectionConfig::OpenAi(OpenAiConnection::default()),
            ),
        ];
        let err = ConnectionsMap::from_pairs(pairs).unwrap_err();
        assert_eq!(err, ConnectionsError::DuplicateId("default".to_string()));
    }

    #[test]
    fn connections_map_preserves_declaration_order() {
        let pairs = vec![
            (
                ConnectionId::new("b").unwrap(),
                ConnectionConfig::OpenAi(OpenAiConnection::default()),
            ),
            (
                ConnectionId::new("a").unwrap(),
                ConnectionConfig::Anthropic(AnthropicConnection::default()),
            ),
            (
                ConnectionId::new("c").unwrap(),
                ConnectionConfig::Ollama(OllamaConnection::default()),
            ),
        ];
        let map = ConnectionsMap::from_pairs(pairs).unwrap();
        let ids: Vec<_> = map.iter().map(|(id, _)| id.as_str().to_string()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn legacy_to_connection_openai() {
        let llm = LlmConfig {
            connector: "openai".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            ..LlmConfig::default()
        };
        match connection_from_legacy_llm(&llm) {
            ConnectionConfig::OpenAi(c) => {
                assert_eq!(c.base_url.as_deref(), Some("https://api.openai.com/v1"));
                assert_eq!(c.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
            }
            other => panic!("expected OpenAi, got {other:?}"),
        }
    }

    #[test]
    fn legacy_to_connection_anthropic() {
        let llm = LlmConfig {
            connector: "anthropic".to_string(),
            base_url: Some("https://api.anthropic.com".to_string()),
            api_key_env: None,
            secret: Some(SecretConfig::default()),
            ..LlmConfig::default()
        };
        match connection_from_legacy_llm(&llm) {
            ConnectionConfig::Anthropic(c) => {
                assert_eq!(c.base_url.as_deref(), Some("https://api.anthropic.com"));
                assert!(c.api_key_env.is_none());
                assert!(c.secret.is_some());
            }
            other => panic!("expected Anthropic, got {other:?}"),
        }
    }

    #[test]
    fn legacy_to_connection_bedrock_region() {
        let llm = LlmConfig {
            connector: "bedrock".to_string(),
            base_url: Some("us-west-2".to_string()),
            aws_profile: Some("work".to_string()),
            ..LlmConfig::default()
        };
        match connection_from_legacy_llm(&llm) {
            ConnectionConfig::Bedrock(c) => {
                assert_eq!(c.region.as_deref(), Some("us-west-2"));
                assert_eq!(c.aws_profile.as_deref(), Some("work"));
                assert!(c.base_url.is_none());
            }
            other => panic!("expected Bedrock, got {other:?}"),
        }
    }

    #[test]
    fn legacy_to_connection_bedrock_with_url_base() {
        let llm = LlmConfig {
            connector: "aws-bedrock".to_string(),
            base_url: Some("https://bedrock.internal.example.com".to_string()),
            ..LlmConfig::default()
        };
        match connection_from_legacy_llm(&llm) {
            ConnectionConfig::Bedrock(c) => {
                assert!(c.region.is_none());
                assert_eq!(
                    c.base_url.as_deref(),
                    Some("https://bedrock.internal.example.com")
                );
            }
            other => panic!("expected Bedrock, got {other:?}"),
        }
    }

    #[test]
    fn legacy_to_connection_ollama() {
        let llm = LlmConfig {
            connector: "ollama".to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            ..LlmConfig::default()
        };
        match connection_from_legacy_llm(&llm) {
            ConnectionConfig::Ollama(c) => {
                assert_eq!(c.base_url.as_deref(), Some("http://localhost:11434"));
            }
            other => panic!("expected Ollama, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_openai_toml() {
        let toml_src = r#"
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_WORK_KEY"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "openai");
    }

    #[test]
    fn roundtrip_bedrock_toml() {
        let toml_src = r#"
type = "bedrock"
aws_profile = "home"
region = "us-west-2"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "bedrock");
    }

    #[test]
    fn roundtrip_ollama_toml() {
        let toml_src = r#"
type = "ollama"
base_url = "http://localhost:11434"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "ollama");
    }

    #[test]
    fn roundtrip_anthropic_toml() {
        let toml_src = r#"
type = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "anthropic");
    }

    #[test]
    fn roundtrip_openrouter_toml() {
        let toml_src = r#"
type = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "openrouter");
    }

    #[test]
    fn roundtrip_azure_toml() {
        let toml_src = r#"
type = "azure"
base_url = "https://my-resource.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
api_surface = "classic"
auth_mode = "entra"
api_version = "2024-10-21"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "azure");
        match parsed {
            ConnectionConfig::Azure(c) => {
                assert_eq!(c.api_surface.as_deref(), Some("classic"));
                assert_eq!(c.auth_mode.as_deref(), Some("entra"));
                assert_eq!(c.api_version.as_deref(), Some("2024-10-21"));
            }
            other => panic!("expected Azure, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_google_toml() {
        let toml_src = r#"
type = "google"
project = "my-gcp-project"
location = "us-central1"
auth_mode = "vertex"
credentials_path = "/etc/adele/sa.json"
"#;
        let parsed: ConnectionConfig = toml::from_str(toml_src).unwrap();
        let serialized = toml::to_string(&parsed).unwrap();
        let reparsed: ConnectionConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
        assert_eq!(parsed.connector_type(), "google");
        match parsed {
            ConnectionConfig::Google(c) => {
                assert_eq!(c.project.as_deref(), Some("my-gcp-project"));
                assert_eq!(c.location.as_deref(), Some("us-central1"));
                assert_eq!(c.auth_mode.as_deref(), Some("vertex"));
                assert_eq!(c.credentials_path.as_deref(), Some("/etc/adele/sa.json"));
            }
            other => panic!("expected Google, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_type() {
        let toml_src = r#"
type = "gemini"
base_url = "https://example.com"
"#;
        let err = toml::from_str::<ConnectionConfig>(toml_src).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown variant") || msg.contains("gemini"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn rejects_unknown_field() {
        let toml_src = r#"
type = "openai"
mystery_key = "x"
"#;
        let err = toml::from_str::<ConnectionConfig>(toml_src).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown field"), "unexpected error: {msg}");
    }

    // --- Connector enum (#47) -----------------------------------------------

    #[test]
    fn connector_parse_canonical_names() {
        assert_eq!(Connector::parse("ollama"), Some(Connector::Ollama));
        assert_eq!(Connector::parse("anthropic"), Some(Connector::Anthropic));
        assert_eq!(Connector::parse("bedrock"), Some(Connector::Bedrock));
        assert_eq!(Connector::parse("openai"), Some(Connector::OpenAi));
        assert_eq!(Connector::parse("openrouter"), Some(Connector::OpenRouter));
        assert_eq!(Connector::parse("azure"), Some(Connector::Azure));
        assert_eq!(Connector::parse("google"), Some(Connector::Google));
    }

    #[test]
    fn connector_parse_rejects_gemini_alias() {
        // `google` is the canonical id; `gemini` must stay unrecognised so the
        // negative config fixture (`rejects_unknown_type`) keeps rejecting it.
        assert_eq!(Connector::parse("gemini"), None);
    }

    #[test]
    fn connector_parse_accepts_aws_bedrock_alias() {
        assert_eq!(Connector::parse("aws-bedrock"), Some(Connector::Bedrock));
        assert_eq!(Connector::parse("AWS-BEDROCK"), Some(Connector::Bedrock));
        assert_eq!(
            Connector::parse("  aws-bedrock  "),
            Some(Connector::Bedrock)
        );
    }

    #[test]
    fn connector_parse_is_case_insensitive() {
        assert_eq!(Connector::parse("OpenAI"), Some(Connector::OpenAi));
        assert_eq!(Connector::parse("BEDROCK"), Some(Connector::Bedrock));
    }

    #[test]
    fn connector_parse_rejects_unknown() {
        assert_eq!(Connector::parse(""), None);
        assert_eq!(Connector::parse("gemini"), None);
        assert_eq!(Connector::parse("anthrop"), None);
    }

    #[test]
    fn connector_as_str_round_trips_through_parse() {
        for &c in &[
            Connector::Ollama,
            Connector::Anthropic,
            Connector::Bedrock,
            Connector::OpenAi,
            Connector::OpenRouter,
            Connector::Azure,
            Connector::Google,
        ] {
            assert_eq!(Connector::parse(c.as_str()), Some(c));
        }
    }

    #[test]
    fn connector_capability_flags_match_legacy_string_checks() {
        // Pre-#47 the legacy `embeddings_available = connector != "anthropic"`
        // and `hosted_tool_search_available = connector == "openai" || ==
        // "anthropic"` lived inline in `get_connector_defaults`. Pin the
        // mapping here so the typed methods can't drift.
        assert!(Connector::Ollama.supports_embeddings());
        assert!(Connector::Bedrock.supports_embeddings());
        assert!(Connector::OpenAi.supports_embeddings());
        assert!(!Connector::Anthropic.supports_embeddings());
        // New connectors: OpenRouter has no embeddings; Azure and Google do.
        assert!(!Connector::OpenRouter.supports_embeddings());
        assert!(Connector::Azure.supports_embeddings());
        assert!(Connector::Google.supports_embeddings());

        assert!(!Connector::Ollama.supports_hosted_tool_search());
        assert!(!Connector::Bedrock.supports_hosted_tool_search());
        assert!(Connector::OpenAi.supports_hosted_tool_search());
        assert!(Connector::Anthropic.supports_hosted_tool_search());
        // None of the new connectors expose hosted tool search in v1.
        assert!(!Connector::OpenRouter.supports_hosted_tool_search());
        assert!(!Connector::Azure.supports_hosted_tool_search());
        assert!(!Connector::Google.supports_hosted_tool_search());
    }

    #[test]
    fn connection_config_connector_method_matches_type_tag() {
        let cases: [(ConnectionConfig, Connector); 7] = [
            (
                ConnectionConfig::Ollama(OllamaConnection::default()),
                Connector::Ollama,
            ),
            (
                ConnectionConfig::Anthropic(AnthropicConnection::default()),
                Connector::Anthropic,
            ),
            (
                ConnectionConfig::Bedrock(BedrockConnection::default()),
                Connector::Bedrock,
            ),
            (
                ConnectionConfig::OpenAi(OpenAiConnection::default()),
                Connector::OpenAi,
            ),
            (
                ConnectionConfig::OpenRouter(OpenRouterConnection::default()),
                Connector::OpenRouter,
            ),
            (
                ConnectionConfig::Azure(AzureConnection::default()),
                Connector::Azure,
            ),
            (
                ConnectionConfig::Google(GoogleConnection::default()),
                Connector::Google,
            ),
        ];
        for (cfg, expected) in cases {
            assert_eq!(cfg.connector(), expected);
            assert_eq!(cfg.connector_type(), expected.as_str());
        }
    }
}
