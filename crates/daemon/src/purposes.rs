//! Purpose configs.
//!
//! Each LLM *purpose* (interactive chat, dreaming, embedding, titling)
//! references a named connection by id and picks a model from that connection,
//! optionally with an effort level. Replaces the legacy `[backend_tasks.llm]`
//! block, which duplicated credentials for every extra purpose and didn't
//! scale past two call sites.
//!
//! The wire format is narrow on purpose: a `connection` reference, a `model`
//! reference, and an optional `effort` hint. Both refs support a literal
//! `"primary"` sentinel that inherits from the `interactive` purpose at load
//! time. Resolution is one level deep — `primary -> primary` chains are
//! explicitly forbidden — so cycles are structurally impossible, not just
//! rejected.
//!
//! Schema + resolution only: registry instantiation lives in `registry.rs`,
//! and `Effort → ReasoningConfig` mapping lives in `api_surface.rs`.
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// `PurposeKind` is the canonical enum in `core::ports::inbound` (#43).
// Re-exported here so the daemon can reach it through the same path it
// always has (`crate::purposes::PurposeKind`).
pub use desktop_assistant_core::ports::inbound::PurposeKind;

use crate::connections::{ConnectionId, ConnectionIdError, ConnectionsMap};

/// A reference to a [`ConnectionId`] as it appears in a purpose config.
///
/// The literal string `"primary"` deserializes as [`ConnectionRef::Primary`]
/// and is resolved against the `interactive` purpose at load time. Any other
/// string is validated as a [`ConnectionId`] eagerly so broken slugs fail at
/// parse time rather than at dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionRef {
    Named(ConnectionId),
    Primary,
}

/// A reference to a model name as it appears in a purpose config.
///
/// Model names are connector-specific and not validated up front (we do not
/// have the connection's listing available at config-parse time), so this is
/// a plain `String`. The literal `"primary"` means "inherit from interactive".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    Named(String),
    Primary,
}

/// Reserved sentinel that represents the `primary` inherit token.
const PRIMARY_SENTINEL: &str = "primary";

// `Effort` is the canonical enum in `core::ports::inbound` (#43).
// Re-exported so the daemon's purpose configs can keep referring to
// `crate::purposes::Effort`. Maps to per-connector knobs at dispatch
// time (Anthropic `thinking.budget_tokens`, OpenAI `reasoning_effort`,
// Bedrock per-model; Ollama ignores).
pub use desktop_assistant_core::ports::inbound::Effort;

/// Raw config for a single purpose, as parsed from TOML.
///
/// Deserializes from a table like:
/// ```toml
/// [purposes.dreaming]
/// connection = "home_bedrock"
/// model = "claude-haiku-4-5"
/// effort = "low"
/// max_context_tokens = 1_000_000
/// ```
///
/// `max_context_tokens` is a user-supplied override for the model's context
/// window in tokens. When set, it takes priority over the connector's
/// curated table. Leaving it unset (the default) lets the daemon-side
/// resolver consult the connector's per-model table and fall back to a
/// conservative universal default.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PurposeConfig {
    #[serde(
        serialize_with = "serialize_connection_ref",
        deserialize_with = "deserialize_connection_ref"
    )]
    pub connection: ConnectionRef,
    #[serde(
        serialize_with = "serialize_model_ref",
        deserialize_with = "deserialize_model_ref"
    )]
    pub model: ModelRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Optional override for the model's max context window, in tokens.
    /// When `Some`, it wins over the connector's curated default and the
    /// universal fallback (see `crate::config::resolve_context_budget`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// All purpose configs, keyed by [`PurposeKind`].
///
/// TOML shape:
/// ```toml
/// [purposes.interactive]
/// connection = "work_openai"
/// model = "gpt-5.4"
/// effort = "medium"
///
/// [purposes.dreaming]
/// connection = "primary"    # inherit from interactive
/// model = "claude-haiku-4-5"
/// ```
///
/// The `interactive` purpose is required when the `[purposes]` table is
/// present; without it there is nothing for `"primary"` to inherit from.
/// Empty / absent `[purposes]` is represented by `Purposes::default()` and is
/// a valid state (first-run, no migration) — [`load_daemon_config`] decides
/// whether to synthesize a set.
///
/// Internally a `BTreeMap<PurposeKind, PurposeConfig>` so adding a new
/// variant doesn't require editing four named-field arms. Custom
/// serde implementations preserve the existing TOML named-table shape
/// (`[purposes.interactive]`, `[purposes.dreaming]`, ...) so on-disk
/// configs stay byte-compatible.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Purposes(BTreeMap<PurposeKind, PurposeConfig>);

impl Purposes {
    /// Whether any purpose is configured.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the raw [`PurposeConfig`] for a given kind, if present.
    pub fn get(&self, kind: PurposeKind) -> Option<&PurposeConfig> {
        self.0.get(&kind)
    }

    /// Mutable accessor used by config-mutation paths (delete-cascade,
    /// purpose editing) that need to tweak fields on an existing entry
    /// without going through the full `set` round-trip.
    pub fn get_mut(&mut self, kind: PurposeKind) -> Option<&mut PurposeConfig> {
        self.0.get_mut(&kind)
    }

    /// Mutably set the raw [`PurposeConfig`] for a given kind. Passing
    /// `None` clears the slot.
    pub fn set(&mut self, kind: PurposeKind, cfg: Option<PurposeConfig>) {
        match cfg {
            Some(c) => {
                self.0.insert(kind, c);
            }
            None => {
                self.0.remove(&kind);
            }
        }
    }

    /// Iterate (kind, config) pairs in [`PurposeKind::all`] order, skipping
    /// absent slots. Used by [`resolve_all`] and tests; kept on the
    /// public surface because the daemon's purpose-config introspection
    /// path will need this once we expose a "show me everything that's
    /// configured" command.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = (PurposeKind, &PurposeConfig)> {
        // Drive iteration order from `PurposeKind::all` rather than
        // `BTreeMap`'s natural ordering so the public iter contract is
        // independent of how `PurposeKind` happens to derive `Ord`.
        PurposeKind::all()
            .into_iter()
            .filter_map(|k| self.get(k).map(|c| (k, c)))
    }

    /// Build a `Purposes` from a list of `(kind, config)` pairs. Used
    /// by tests; the daemon paths build incrementally via `set`.
    /// Duplicate keys overwrite — last write wins, matching `set`.
    #[cfg(test)]
    pub fn from_pairs<I: IntoIterator<Item = (PurposeKind, PurposeConfig)>>(pairs: I) -> Self {
        let mut out = Self::default();
        for (kind, cfg) in pairs {
            out.set(kind, Some(cfg));
        }
        out
    }

    /// Validate the set at load time. Currently enforces:
    ///
    /// - When any purpose is set, `interactive` must be set (required anchor
    ///   for `"primary"` inheritance).
    /// - Interactive's `connection` and `model` must not be `Primary`
    ///   (nothing to inherit from).
    pub fn validate(&self) -> Result<(), PurposeError> {
        if self.is_empty() {
            return Ok(());
        }
        let Some(interactive) = self.get(PurposeKind::Interactive) else {
            return Err(PurposeError::MissingInteractive);
        };
        if matches!(interactive.connection, ConnectionRef::Primary) {
            return Err(PurposeError::InteractivePrimaryConnection);
        }
        if matches!(interactive.model, ModelRef::Primary) {
            return Err(PurposeError::InteractivePrimaryModel);
        }
        Ok(())
    }
}

/// Custom serde: preserve the existing TOML named-table shape on the
/// wire (so user configs and tests don't need to change) while letting
/// the in-memory representation be a single map. Serializes only the
/// purposes that are present, in [`PurposeKind::all`] order; deserializes
/// any subset of `interactive` / `dreaming` / `embedding` / `titling`
/// keys. Unknown keys are rejected so typos surface at parse time.
mod purposes_serde {
    use super::{PurposeConfig, PurposeKind, Purposes};
    use serde::de::{Error as DeError, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    impl Serialize for Purposes {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(self.0.len()))?;
            for kind in PurposeKind::all() {
                if let Some(cfg) = self.get(kind) {
                    map.serialize_entry(kind.as_key(), cfg)?;
                }
            }
            map.end()
        }
    }

    impl<'de> Deserialize<'de> for Purposes {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct PurposesVisitor;

            impl<'de> Visitor<'de> for PurposesVisitor {
                type Value = Purposes;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a `[purposes]` table with known purpose keys")
                }

                fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Purposes, M::Error> {
                    let mut out = Purposes::default();
                    while let Some(key) = access.next_key::<String>()? {
                        let Some(kind) = PurposeKind::from_key(&key) else {
                            return Err(M::Error::custom(format!(
                                "unknown purpose `{key}`; expected one of \
                                 `interactive`, `dreaming`, `embedding`, `titling`"
                            )));
                        };
                        if out.get(kind).is_some() {
                            return Err(M::Error::custom(format!(
                                "duplicate purpose `{key}` in `[purposes]`"
                            )));
                        }
                        let cfg: PurposeConfig = access.next_value()?;
                        out.set(kind, Some(cfg));
                    }
                    Ok(out)
                }
            }

            deserializer.deserialize_map(PurposesVisitor)
        }
    }
}

/// A purpose with every reference resolved to a concrete value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPurpose {
    pub kind: PurposeKind,
    pub connection_id: ConnectionId,
    pub model_id: String,
    pub effort: Option<Effort>,
}

/// Errors raised while parsing or validating purpose configs.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PurposeError {
    #[error("`[purposes]` is configured but `purposes.interactive` is missing")]
    MissingInteractive,
    #[error(
        "`purposes.interactive.connection` must name a concrete connection \
         (cannot be `\"primary\"` — nothing to inherit from)"
    )]
    InteractivePrimaryConnection,
    #[error(
        "`purposes.interactive.model` must be a concrete model name \
         (cannot be `\"primary\"` — nothing to inherit from)"
    )]
    InteractivePrimaryModel,
    #[error("purpose {purpose:?} is not configured")]
    Missing { purpose: &'static str },
    #[error(
        "purpose {purpose:?}: connection \"primary\" resolves to interactive, \
         which also uses \"primary\" (depth exceeded 1)"
    )]
    PrimaryChainTooDeep { purpose: &'static str },
    #[error(
        "purpose \"interactive\": connection {connection:?} is not configured \
         in `[connections]`"
    )]
    DanglingInteractiveConnection { connection: String },
}

/// Resolve a single purpose to a [`ResolvedPurpose`] against a validated
/// [`ConnectionsMap`].
///
/// Resolution rules:
/// - `ConnectionRef::Primary` and `ModelRef::Primary` inherit from the
///   `interactive` purpose. Since interactive itself must be concrete
///   (enforced at load time), the chain is capped at depth 1.
/// - If a purpose references a named connection that is not in the map, we
///   log a `tracing::warn!` and fall back to interactive's connection. The
///   model is left as-authored (there is no sensible auto-fallback for
///   models, and an incorrect model surfaces clearly at dispatch time).
/// - If *interactive* itself references a missing connection, we return
///   [`PurposeError::DanglingInteractiveConnection`] — the daemon refuses to
///   start with a broken primary.
pub fn resolve_purpose(
    kind: PurposeKind,
    purposes: &Purposes,
    connections: &ConnectionsMap,
) -> Result<ResolvedPurpose, PurposeError> {
    let Some(cfg) = purposes.get(kind) else {
        return Err(PurposeError::Missing {
            purpose: kind.as_key(),
        });
    };
    let interactive = purposes
        .get(PurposeKind::Interactive)
        .ok_or(PurposeError::MissingInteractive)?;

    // Resolve connection (depth 1 max).
    let connection_id = match &cfg.connection {
        ConnectionRef::Named(id) => {
            if connections.get(id).is_some() {
                id.clone()
            } else if kind == PurposeKind::Interactive {
                return Err(PurposeError::DanglingInteractiveConnection {
                    connection: id.as_str().to_string(),
                });
            } else {
                // Dangling non-interactive ref: warn and fall back to the
                // interactive connection. Interactive's connection is already
                // known-concrete by validation; re-check membership so a
                // config where interactive *also* became dangling returns a
                // clear error rather than silently succeeding.
                let fallback = expect_interactive_connection(interactive, connections)?;
                tracing::warn!(
                    purpose = kind.as_key(),
                    missing_connection = %id,
                    fallback = %fallback,
                    "purpose references a connection id that is not configured \
                     in `[connections]`; falling back to interactive's connection"
                );
                fallback
            }
        }
        ConnectionRef::Primary => {
            if kind == PurposeKind::Interactive {
                // Structurally impossible after validate(), but guard anyway.
                return Err(PurposeError::InteractivePrimaryConnection);
            }
            // Depth check: interactive must already be concrete.
            match &interactive.connection {
                ConnectionRef::Named(_) => {}
                ConnectionRef::Primary => {
                    return Err(PurposeError::PrimaryChainTooDeep {
                        purpose: kind.as_key(),
                    });
                }
            }
            expect_interactive_connection(interactive, connections)?
        }
    };

    // Resolve model.
    let model_id = match &cfg.model {
        ModelRef::Named(m) => m.clone(),
        ModelRef::Primary => {
            if kind == PurposeKind::Interactive {
                return Err(PurposeError::InteractivePrimaryModel);
            }
            match &interactive.model {
                ModelRef::Named(m) => m.clone(),
                ModelRef::Primary => {
                    return Err(PurposeError::PrimaryChainTooDeep {
                        purpose: kind.as_key(),
                    });
                }
            }
        }
    };

    Ok(ResolvedPurpose {
        kind,
        connection_id,
        model_id,
        effort: cfg.effort,
    })
}

/// Resolve every configured purpose. Returns a map keyed by [`PurposeKind`].
/// Missing purposes are simply absent from the output; it is up to call sites
/// to decide whether a given absence is a hard error.
///
/// Currently only exercised by tests — the daemon resolves purposes
/// per-call via `resolve_purpose` rather than batching at startup —
/// but kept as part of the public surface for diagnostic and
/// configuration-introspection use cases.
#[allow(dead_code)]
pub fn resolve_all(
    purposes: &Purposes,
    connections: &ConnectionsMap,
) -> Result<BTreeMap<PurposeKind, ResolvedPurpose>, PurposeError> {
    let mut out = BTreeMap::new();
    for (kind, _) in purposes.iter() {
        let resolved = resolve_purpose(kind, purposes, connections)?;
        out.insert(kind, resolved);
    }
    Ok(out)
}

/// Pull the (already-validated) interactive connection id and confirm it is
/// present in the connections map.
fn expect_interactive_connection(
    interactive: &PurposeConfig,
    connections: &ConnectionsMap,
) -> Result<ConnectionId, PurposeError> {
    let id = match &interactive.connection {
        ConnectionRef::Named(id) => id.clone(),
        ConnectionRef::Primary => {
            return Err(PurposeError::InteractivePrimaryConnection);
        }
    };
    if connections.get(&id).is_none() {
        return Err(PurposeError::DanglingInteractiveConnection {
            connection: id.as_str().to_string(),
        });
    }
    Ok(id)
}

// --- FromStr / serde glue ---------------------------------------------------

impl FromStr for ConnectionRef {
    type Err = ConnectionIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == PRIMARY_SENTINEL {
            Ok(ConnectionRef::Primary)
        } else {
            ConnectionId::new(s).map(ConnectionRef::Named)
        }
    }
}

impl fmt::Display for ConnectionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(id) => f.write_str(id.as_str()),
            Self::Primary => f.write_str(PRIMARY_SENTINEL),
        }
    }
}

impl FromStr for ModelRef {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == PRIMARY_SENTINEL {
            Ok(ModelRef::Primary)
        } else {
            Ok(ModelRef::Named(s.to_string()))
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(m) => f.write_str(m),
            Self::Primary => f.write_str(PRIMARY_SENTINEL),
        }
    }
}

fn serialize_connection_ref<S: serde::Serializer>(
    value: &ConnectionRef,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&value.to_string())
}

fn deserialize_connection_ref<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ConnectionRef, D::Error> {
    let raw = String::deserialize(deserializer)?;
    ConnectionRef::from_str(&raw).map_err(serde::de::Error::custom)
}

fn serialize_model_ref<S: serde::Serializer>(
    value: &ModelRef,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&value.to_string())
}

fn deserialize_model_ref<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ModelRef, D::Error> {
    let raw = String::deserialize(deserializer)?;
    // `ModelRef::from_str` is infallible.
    Ok(ModelRef::from_str(&raw).expect("ModelRef::from_str is infallible"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{ConnectionConfig, OpenAiConnection};

    fn conn_id(s: &str) -> ConnectionId {
        ConnectionId::new(s).unwrap()
    }

    fn connections_with(ids: &[&str]) -> ConnectionsMap {
        let pairs: Vec<_> = ids
            .iter()
            .map(|id| {
                (
                    conn_id(id),
                    ConnectionConfig::OpenAi(OpenAiConnection::default()),
                )
            })
            .collect();
        ConnectionsMap::from_pairs(pairs).unwrap()
    }

    fn interactive_for(conn: &str, model: &str) -> PurposeConfig {
        PurposeConfig {
            connection: ConnectionRef::Named(conn_id(conn)),
            model: ModelRef::Named(model.to_string()),
            effort: Some(Effort::Medium),
            max_context_tokens: None,
        }
    }

    // --- FromStr / parsing ------------------------------------------------

    #[test]
    fn connection_ref_parses_primary_sentinel() {
        assert_eq!(
            ConnectionRef::from_str("primary").unwrap(),
            ConnectionRef::Primary
        );
    }

    #[test]
    fn connection_ref_parses_concrete_id() {
        let r = ConnectionRef::from_str("work_openai").unwrap();
        assert_eq!(r, ConnectionRef::Named(conn_id("work_openai")));
    }

    #[test]
    fn connection_ref_rejects_invalid_slug() {
        ConnectionRef::from_str("Bad Id!").unwrap_err();
    }

    #[test]
    fn model_ref_parses_primary_and_named() {
        assert_eq!(ModelRef::from_str("primary").unwrap(), ModelRef::Primary);
        assert_eq!(
            ModelRef::from_str("gpt-5.4").unwrap(),
            ModelRef::Named("gpt-5.4".to_string())
        );
    }

    #[test]
    fn purpose_config_roundtrip_toml() {
        let toml_src = r#"
connection = "work_openai"
model = "gpt-5.4"
effort = "medium"
"#;
        let parsed: PurposeConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            parsed.connection,
            ConnectionRef::Named(conn_id("work_openai"))
        );
        assert_eq!(parsed.model, ModelRef::Named("gpt-5.4".to_string()));
        assert_eq!(parsed.effort, Some(Effort::Medium));

        let reserialized = toml::to_string(&parsed).unwrap();
        let reparsed: PurposeConfig = toml::from_str(&reserialized).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn purpose_config_parses_primary_sentinels() {
        let toml_src = r#"
connection = "primary"
model = "primary"
"#;
        let parsed: PurposeConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(parsed.connection, ConnectionRef::Primary);
        assert_eq!(parsed.model, ModelRef::Primary);
        assert_eq!(parsed.effort, None);
    }

    #[test]
    fn purpose_config_rejects_unknown_field() {
        let toml_src = r#"
connection = "work_openai"
model = "gpt-5.4"
effort = "medium"
mystery = "x"
"#;
        let err = toml::from_str::<PurposeConfig>(toml_src).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn effort_serde_lowercase() {
        #[derive(Deserialize, Serialize)]
        struct Holder {
            v: Effort,
        }
        for (lit, variant) in [
            ("low", Effort::Low),
            ("medium", Effort::Medium),
            ("high", Effort::High),
        ] {
            let src = format!("v = \"{lit}\"");
            let h: Holder = toml::from_str(&src).unwrap();
            assert_eq!(h.v, variant);
            // Round-trip
            let reserialized = toml::to_string(&h).unwrap();
            assert!(reserialized.contains(&format!("v = \"{lit}\"")));
        }
    }

    // --- Validation -------------------------------------------------------

    #[test]
    fn validate_empty_is_ok() {
        assert!(Purposes::default().validate().is_ok());
    }

    #[test]
    fn validate_requires_interactive_when_any_purpose_set() {
        let p = Purposes::from_pairs([(
            PurposeKind::Dreaming,
            PurposeConfig {
                connection: ConnectionRef::Primary,
                model: ModelRef::Primary,
                effort: None,
                max_context_tokens: None,
            },
        )]);
        assert_eq!(p.validate().unwrap_err(), PurposeError::MissingInteractive);
    }

    #[test]
    fn validate_rejects_primary_in_interactive_connection() {
        let p = Purposes::from_pairs([(
            PurposeKind::Interactive,
            PurposeConfig {
                connection: ConnectionRef::Primary,
                model: ModelRef::Named("gpt-5.4".to_string()),
                effort: None,
                max_context_tokens: None,
            },
        )]);
        assert_eq!(
            p.validate().unwrap_err(),
            PurposeError::InteractivePrimaryConnection
        );
    }

    #[test]
    fn validate_rejects_primary_in_interactive_model() {
        let p = Purposes::from_pairs([(
            PurposeKind::Interactive,
            PurposeConfig {
                connection: ConnectionRef::Named(conn_id("work")),
                model: ModelRef::Primary,
                effort: None,
                max_context_tokens: None,
            },
        )]);
        assert_eq!(
            p.validate().unwrap_err(),
            PurposeError::InteractivePrimaryModel
        );
    }

    // --- Resolution -------------------------------------------------------

    #[test]
    fn resolve_concrete_interactive() {
        let p =
            Purposes::from_pairs([(PurposeKind::Interactive, interactive_for("work", "gpt-5.4"))]);
        let conns = connections_with(&["work"]);

        let r = resolve_purpose(PurposeKind::Interactive, &p, &conns).unwrap();
        assert_eq!(r.connection_id, conn_id("work"));
        assert_eq!(r.model_id, "gpt-5.4");
        assert_eq!(r.effort, Some(Effort::Medium));
    }

    #[test]
    fn resolve_primary_inherits_from_interactive() {
        let p = Purposes::from_pairs([
            (PurposeKind::Interactive, interactive_for("work", "gpt-5.4")),
            (
                PurposeKind::Dreaming,
                PurposeConfig {
                    connection: ConnectionRef::Primary,
                    model: ModelRef::Primary,
                    effort: Some(Effort::Low),
                    max_context_tokens: None,
                },
            ),
        ]);
        let conns = connections_with(&["work"]);

        let r = resolve_purpose(PurposeKind::Dreaming, &p, &conns).unwrap();
        assert_eq!(r.connection_id, conn_id("work"));
        assert_eq!(r.model_id, "gpt-5.4");
        // Effort stays from dreaming's own config, not interactive's.
        assert_eq!(r.effort, Some(Effort::Low));
    }

    #[test]
    fn resolve_partial_primary_keeps_named_model() {
        let p = Purposes::from_pairs([
            (PurposeKind::Interactive, interactive_for("work", "gpt-5.4")),
            (
                PurposeKind::Dreaming,
                PurposeConfig {
                    connection: ConnectionRef::Primary,
                    model: ModelRef::Named("claude-haiku-4-5".to_string()),
                    effort: None,
                    max_context_tokens: None,
                },
            ),
        ]);
        let conns = connections_with(&["work"]);

        let r = resolve_purpose(PurposeKind::Dreaming, &p, &conns).unwrap();
        assert_eq!(r.connection_id, conn_id("work"));
        assert_eq!(r.model_id, "claude-haiku-4-5");
    }

    #[test]
    fn resolve_reports_missing_interactive() {
        let p = Purposes::default();
        let conns = connections_with(&["work"]);
        let err = resolve_purpose(PurposeKind::Interactive, &p, &conns).unwrap_err();
        assert!(matches!(
            err,
            PurposeError::Missing {
                purpose: "interactive"
            }
        ));
    }

    #[test]
    fn resolve_dangling_nonprimary_falls_back_to_interactive() {
        let p = Purposes::from_pairs([
            (PurposeKind::Interactive, interactive_for("work", "gpt-5.4")),
            (
                PurposeKind::Dreaming,
                PurposeConfig {
                    connection: ConnectionRef::Named(conn_id("ghost")),
                    model: ModelRef::Named("claude-haiku-4-5".to_string()),
                    effort: None,
                    max_context_tokens: None,
                },
            ),
        ]);
        let conns = connections_with(&["work"]);

        let r = resolve_purpose(PurposeKind::Dreaming, &p, &conns).unwrap();
        // Fell back to interactive's connection.
        assert_eq!(r.connection_id, conn_id("work"));
        // Model left as authored.
        assert_eq!(r.model_id, "claude-haiku-4-5");
    }

    #[test]
    fn resolve_dangling_interactive_connection_errors() {
        let p = Purposes::from_pairs([(
            PurposeKind::Interactive,
            interactive_for("ghost", "gpt-5.4"),
        )]);
        let conns = connections_with(&["work"]);
        let err = resolve_purpose(PurposeKind::Interactive, &p, &conns).unwrap_err();
        match err {
            PurposeError::DanglingInteractiveConnection { connection } => {
                assert_eq!(connection, "ghost");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn resolve_all_skips_absent_purposes() {
        let p = Purposes::from_pairs([
            (PurposeKind::Interactive, interactive_for("work", "gpt-5.4")),
            (
                PurposeKind::Titling,
                PurposeConfig {
                    connection: ConnectionRef::Primary,
                    model: ModelRef::Primary,
                    effort: None,
                    max_context_tokens: None,
                },
            ),
        ]);
        let conns = connections_with(&["work"]);
        let resolved = resolve_all(&p, &conns).unwrap();
        assert_eq!(resolved.len(), 2);
        assert!(resolved.contains_key(&PurposeKind::Interactive));
        assert!(resolved.contains_key(&PurposeKind::Titling));
        assert!(!resolved.contains_key(&PurposeKind::Dreaming));
    }

    #[test]
    fn purposes_toml_roundtrip_full() {
        let toml_src = r#"
[interactive]
connection = "work_openai"
model = "gpt-5.4"
effort = "medium"

[dreaming]
connection = "primary"
model = "claude-haiku-4-5"

[titling]
connection = "primary"
model = "primary"
"#;
        let parsed: Purposes = toml::from_str(toml_src).unwrap();
        assert!(parsed.get(PurposeKind::Interactive).is_some());
        assert!(parsed.get(PurposeKind::Dreaming).is_some());
        assert!(parsed.get(PurposeKind::Titling).is_some());
        assert!(parsed.get(PurposeKind::Embedding).is_none());
        parsed.validate().expect("valid");

        let reserialized = toml::to_string(&parsed).unwrap();
        let reparsed: Purposes = toml::from_str(&reserialized).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn kind_keys_roundtrip() {
        for k in PurposeKind::all() {
            assert_eq!(PurposeKind::from_key(k.as_key()), Some(k));
        }
        assert_eq!(PurposeKind::from_key("nope"), None);
    }

    // --- max_context_tokens (#51) ----------------------------------------

    #[test]
    fn purpose_config_parses_max_context_tokens() {
        let toml_src = r#"
connection = "work_bedrock"
model = "us.amazon.nova-premier-v1:0"
effort = "medium"
max_context_tokens = 1000000
"#;
        let parsed: PurposeConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(parsed.max_context_tokens, Some(1_000_000));
    }

    #[test]
    fn purpose_config_omits_max_context_tokens_when_none() {
        // Round-trip a config with no override; the serialized form must not
        // mention `max_context_tokens` at all so legacy configs stay clean.
        let cfg = PurposeConfig {
            connection: ConnectionRef::Named(conn_id("work")),
            model: ModelRef::Named("gpt-5.4".to_string()),
            effort: Some(Effort::Medium),
            max_context_tokens: None,
        };
        let serialized = toml::to_string(&cfg).unwrap();
        assert!(
            !serialized.contains("max_context_tokens"),
            "None should be skipped: {serialized}"
        );
        let reparsed: PurposeConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn purpose_config_legacy_toml_without_field_deserializes() {
        // Migration: configs predating #51 must still parse — `max_context_tokens`
        // has `#[serde(default)]` so the absence is fine even with
        // `deny_unknown_fields` on the struct.
        let legacy = r#"
connection = "work_openai"
model = "gpt-5.4"
effort = "high"
"#;
        let parsed: PurposeConfig = toml::from_str(legacy).unwrap();
        assert_eq!(parsed.max_context_tokens, None);
        assert_eq!(parsed.effort, Some(Effort::High));
    }
}
