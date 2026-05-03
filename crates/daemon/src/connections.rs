//! Named-connection config schema.
//!
//! A `connections` map keyed by a user-chosen slug ([`ConnectionId`]).
//! Each connection owns its own credentials/endpoint and declares its connector
//! type via a `#[serde(tag = "type")]` payload, replacing the legacy single
//! `[llm]` block which hard-coded one global connector.
//!
//! Schema-only: migration lives in [`super::config`] so it can share I/O
//! helpers with the wider config layer. The blanket `#[allow(dead_code)]`
//! covers a handful of `ConnectionsMap` accessors that are exposed for
//! symmetry but have no current call site.
#![allow(dead_code)]

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
            Self::Bedrock(_) => Connector::Bedrock,
            Self::Ollama(_) => Connector::Ollama,
        }
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
            Self::Anthropic => "",
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
        }
    }

    /// Whether this connector exposes an embeddings endpoint. Anthropic
    /// doesn't.
    pub fn supports_embeddings(self) -> bool {
        !matches!(self, Self::Anthropic)
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
}

/// Ollama (local or self-hosted) connection fields.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConnection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
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

    /// Borrow the underlying `IndexMap`.
    pub fn as_map(&self) -> &IndexMap<ConnectionId, ConnectionConfig> {
        &self.0
    }

    /// Consume and return the underlying `IndexMap`.
    pub fn into_map(self) -> IndexMap<ConnectionId, ConnectionConfig> {
        self.0
    }

    /// Iterate connection id / config pairs in declaration order.
    pub fn iter(&self) -> indexmap::map::Iter<'_, ConnectionId, ConnectionConfig> {
        self.0.iter()
    }

    /// Number of configured connections.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are any connections.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
        }),
        "ollama" => ConnectionConfig::Ollama(OllamaConnection {
            base_url: llm.base_url.clone(),
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
        }),
        // Anything else (including the legacy default "openai") maps to OpenAI.
        _ => ConnectionConfig::OpenAi(OpenAiConnection {
            base_url: llm.base_url.clone(),
            api_key_env: llm.api_key_env.clone(),
            secret: llm.secret.clone(),
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

        assert!(!Connector::Ollama.supports_hosted_tool_search());
        assert!(!Connector::Bedrock.supports_hosted_tool_search());
        assert!(Connector::OpenAi.supports_hosted_tool_search());
        assert!(Connector::Anthropic.supports_hosted_tool_search());
    }

    #[test]
    fn connection_config_connector_method_matches_type_tag() {
        let cases: [(ConnectionConfig, Connector); 4] = [
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
        ];
        for (cfg, expected) in cases {
            assert_eq!(cfg.connector(), expected);
            assert_eq!(cfg.connector_type(), expected.as_str());
        }
    }
}
