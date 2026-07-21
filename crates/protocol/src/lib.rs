//! Shared, dependency-light protocol/domain enums.
//!
//! These types were extracted from `desktop-assistant-core` (#377) so that
//! `api-model` and the shared client cores can depend on them **without**
//! pulling `core`'s native dependency tail (tokio, aws-lc-sys crypto, …) — which
//! is what blocked compiling the wire types to `wasm32-unknown-unknown` for the
//! web client. `core` re-exports these at their original module paths so existing
//! `core::ports::inbound::*` / `core::prompts::*` call sites are unchanged.
//!
//! Serde representations here are wire-/storage-visible (JSON columns, config
//! files, the D-Bus int contract). Do not change variant names or `rename_all`
//! attributes without a migration.

// ---------------------------------------------------------------------------
// Effort
// ---------------------------------------------------------------------------

/// Effort hint passed to connectors and mapped to per-connector request
/// parameters at dispatch time.
///
/// Serializes as the lowercase variant name (`"low"`, `"medium"`,
/// `"high"`) for JSON columns and wire payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------------------
// PurposeKind
// ---------------------------------------------------------------------------

/// The LLM purposes the daemon resolves independently. Used as a stable
/// keyed map via [`Self::as_key`] / [`Self::from_key`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PurposeKind {
    /// The user-facing chat LLM. Cannot inherit (nothing to inherit from).
    Interactive,
    /// Periodic fact extraction (the frequent, cheap "dreaming" pass).
    Dreaming,
    /// Holistic knowledge-base consolidation (the slower, heavier daily pass
    /// that recomputes the whole KB — typically a stronger model).
    Consolidation,
    /// Vector embeddings for memory and retrieval.
    Embedding,
    /// Short-title generation for conversations.
    Titling,
    /// The voice assistant's interactive turns (routed by the `"voice"`
    /// conversation tag the voice daemon sets). Inherits interactive by default;
    /// point it at a stronger tool-calling model when voice's on-demand speech
    /// mode needs one (voice#126).
    Voice,
}

impl PurposeKind {
    /// Canonical lowercase key used in TOML and error messages.
    pub fn as_key(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Dreaming => "dreaming",
            Self::Consolidation => "consolidation",
            Self::Embedding => "embedding",
            Self::Titling => "titling",
            Self::Voice => "voice",
        }
    }

    /// Parse a canonical key back into a [`PurposeKind`]. Inverse of
    /// [`Self::as_key`]; used by adapters that round-trip key strings.
    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "interactive" => Some(Self::Interactive),
            "dreaming" => Some(Self::Dreaming),
            "consolidation" => Some(Self::Consolidation),
            "embedding" => Some(Self::Embedding),
            "titling" => Some(Self::Titling),
            "voice" => Some(Self::Voice),
            _ => None,
        }
    }

    /// Every purpose kind, in a stable order. Useful for iteration in
    /// tests and serialization round-trips. Order matches the schema
    /// migration order (interactive first because every other purpose
    /// can inherit from it).
    pub fn all() -> [Self; 6] {
        [
            Self::Interactive,
            Self::Dreaming,
            Self::Consolidation,
            Self::Embedding,
            Self::Titling,
            Self::Voice,
        ]
    }
}

impl std::fmt::Display for PurposeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_key())
    }
}

// ---------------------------------------------------------------------------
// Personality
// ---------------------------------------------------------------------------

/// One trait's strength. Levels are an *initial disposition*, not a rulebook —
/// the rendered blurb (built in `core::prompts`) always appends an adaptation
/// clause telling the model to take cues from the conversation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum PersonalityLevel {
    Never,
    Rarely,
    Sometimes,
    Often,
    Always,
}

impl PersonalityLevel {
    /// Stable ordinal 0..=4 used by the D-Bus int contract (Never=0 … Always=4).
    pub fn as_ordinal(self) -> u8 {
        match self {
            Self::Never => 0,
            Self::Rarely => 1,
            Self::Sometimes => 2,
            Self::Often => 3,
            Self::Always => 4,
        }
    }

    /// Inverse of [`Self::as_ordinal`]. Returns `None` for out-of-range input
    /// rather than clamping, so a malformed wire value surfaces as an error at
    /// the boundary instead of silently snapping to a level.
    pub fn from_ordinal(n: u8) -> Option<Self> {
        match n {
            0 => Some(Self::Never),
            1 => Some(Self::Rarely),
            2 => Some(Self::Sometimes),
            3 => Some(Self::Often),
            4 => Some(Self::Always),
            _ => None,
        }
    }
}

/// The assistant's configurable disposition — the "Expressive 7" traits, each
/// at a [`PersonalityLevel`] (issue #226, Phase 1: global).
///
/// Why a typed struct rather than a free `HashMap<String, Level>`: the trait
/// set is fixed and small, so naming each field gives compile-time safety (no
/// typo'd keys, exhaustive blurb matching) and a **stable wire schema** — config
/// files, the api-model `Config` view, and the D-Bus `ConfigData` tuple all
/// derive from these named fields, so adding/removing a trait is a deliberate,
/// type-checked change rather than a silent string drift.
///
/// The levels are an *initial disposition*, not a rulebook: the rendered blurb
/// (`core::prompts::render_blurb`) always appends an adaptation clause telling
/// the model to take cues from the conversation and adapt both ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Personality {
    #[serde(default = "default_professionalism")]
    pub professionalism: PersonalityLevel,
    #[serde(default = "default_warmth")]
    pub warmth: PersonalityLevel,
    #[serde(default = "default_directness")]
    pub directness: PersonalityLevel,
    #[serde(default = "default_enthusiasm")]
    pub enthusiasm: PersonalityLevel,
    #[serde(default = "default_humor")]
    pub humor: PersonalityLevel,
    #[serde(default = "default_sarcasm")]
    pub sarcasm: PersonalityLevel,
    #[serde(default = "default_pretentiousness")]
    pub pretentiousness: PersonalityLevel,
}

// Per-field default fns so a partial `[personality]` TOML block (only some
// traits specified) fills the rest from the Expressive-7 table rather than
// from `PersonalityLevel`'s arbitrary first variant.
fn default_professionalism() -> PersonalityLevel {
    PersonalityLevel::Always
}
fn default_warmth() -> PersonalityLevel {
    PersonalityLevel::Often
}
fn default_directness() -> PersonalityLevel {
    PersonalityLevel::Often
}
fn default_enthusiasm() -> PersonalityLevel {
    PersonalityLevel::Sometimes
}
fn default_humor() -> PersonalityLevel {
    PersonalityLevel::Sometimes
}
fn default_sarcasm() -> PersonalityLevel {
    PersonalityLevel::Rarely
}
fn default_pretentiousness() -> PersonalityLevel {
    PersonalityLevel::Rarely
}

impl Default for Personality {
    /// The "Expressive 7" defaults from the issue table.
    fn default() -> Self {
        Self {
            professionalism: default_professionalism(),
            warmth: default_warmth(),
            directness: default_directness(),
            enthusiasm: default_enthusiasm(),
            humor: default_humor(),
            sarcasm: default_sarcasm(),
            pretentiousness: default_pretentiousness(),
        }
    }
}

/// A partial, per-conversation override of the global [`Personality`] (issue
/// #227, Phase 2). Each trait is an `Option<PersonalityLevel>`: `Some(level)`
/// pins that trait for the conversation, `None` falls back to the global value.
///
/// Why a separate type rather than reusing `Personality` directly: the global
/// config is always a *complete* disposition (every trait has a level), but a
/// conversation override is *partial by design* — a "no-nonsense" client may
/// only want to force `humor = Never` and `directness = Always` and inherit the
/// rest of the user's global tuning. Modeling each trait as `Option` makes that
/// partial intent explicit and type-checked, and keeps [`Self::resolve`] a
/// trait-by-trait merge rather than an all-or-nothing replacement. The override
/// only sets the *initial disposition*; it still flows through the rendered
/// blurb, so the adaptation clause continues to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct PersonalityOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub professionalism: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmth: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directness: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enthusiasm: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub humor: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sarcasm: Option<PersonalityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pretentiousness: Option<PersonalityLevel>,
}

impl PersonalityOverride {
    /// Resolve this partial override against the `global` disposition into a
    /// concrete [`Personality`]: each `Some` trait wins, each `None` falls back
    /// to `global`. An all-`None` override resolves to `global` unchanged.
    pub fn resolve(&self, global: &Personality) -> Personality {
        Personality {
            professionalism: self.professionalism.unwrap_or(global.professionalism),
            warmth: self.warmth.unwrap_or(global.warmth),
            directness: self.directness.unwrap_or(global.directness),
            enthusiasm: self.enthusiasm.unwrap_or(global.enthusiasm),
            humor: self.humor.unwrap_or(global.humor),
            sarcasm: self.sarcasm.unwrap_or(global.sarcasm),
            pretentiousness: self.pretentiousness.unwrap_or(global.pretentiousness),
        }
    }

    /// `true` when every trait is `None` — i.e. the override pins nothing and
    /// [`Self::resolve`] returns the global value verbatim. Used by the
    /// persistence layer to store `NULL` rather than an empty object.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

// ---------------------------------------------------------------------------
// ClientContext (#549)
// ---------------------------------------------------------------------------

/// Best-effort, self-reported context about the person using a client and the
/// device they are on (issue #549). A client fills in whatever it can discover
/// and omits the rest; the daemon renders only the fields that are present.
///
/// # Trust posture
///
/// Like the per-machine `system_id` handshake hint, this is **untrusted display
/// data, not a trust boundary**: it is self-reported by the client, no privilege
/// is gated on it, it is sanitized before it is templated into the system
/// prompt, and it is kept out of logs.
///
/// # Fail-closed
///
/// Every field is optional and an absent field is simply omitted. The daemon
/// **never** substitutes its own host's `HOME` / `USER` / hostname for a missing
/// value — doing so would leak the daemon host into a multi-tenant prompt.
///
/// Each field is `#[serde(default, skip_serializing_if = "Option::is_none")]`
/// so an all-absent value serializes to `{}` and never widens the wire shape,
/// matching the `system_id` / `host_label` handshake convention.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClientContext {
    /// The user's real / display name (e.g. `"Ada Lovelace"`), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_name: Option<String>,
    /// The user's account / login name on their device (e.g. `"ada"`), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    /// The user's home directory on their device (e.g. `"/home/ada"`), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home_dir: Option<String>,
    /// The client device's hostname (e.g. `"analytical-engine"`), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// The user's IANA timezone (e.g. `"Europe/London"`), if known. The
    /// highest-value field: it lets the assistant resolve relative local times
    /// ("now", "tonight", "this morning") in the user's own zone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
    /// The client device's operating system description (e.g. `"Ubuntu 24.04"`
    /// or `"macOS 15.1"`), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
}

impl ClientContext {
    /// Whether every field is absent. The daemon emits no prompt section for an
    /// empty context (fail-closed).
    pub fn is_empty(&self) -> bool {
        self.real_name.is_none()
            && self.username.is_none()
            && self.home_dir.is_none()
            && self.hostname.is_none()
            && self.timezone.is_none()
            && self.os.is_none()
    }
}

#[cfg(test)]
mod client_context_tests {
    use super::ClientContext;

    fn full() -> ClientContext {
        ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada".into()),
            home_dir: Some("/home/ada".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("Ubuntu 24.04".into()),
        }
    }

    #[test]
    fn default_is_empty_and_absent_fields_are_skipped_on_the_wire() {
        // A fully-absent context is `is_empty()` and serializes to `{}` — the
        // `skip_serializing_if` on every field keeps an all-`None` value from
        // widening the wire shape (mirrors the `system_id`/`host_label` pattern).
        let ctx = ClientContext::default();
        assert!(ctx.is_empty());
        assert_eq!(serde_json::to_string(&ctx).unwrap(), "{}");
    }

    #[test]
    fn full_context_round_trips_losslessly() {
        let ctx = full();
        assert!(!ctx.is_empty());
        let json = serde_json::to_string(&ctx).unwrap();
        let back: ClientContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ctx);
    }

    #[test]
    fn partial_context_omits_absent_fields_but_round_trips() {
        // Only timezone present: the wire form carries just that key, and a
        // decode preserves exactly the present field (the rest stay `None`).
        let ctx = ClientContext {
            timezone: Some("America/New_York".into()),
            ..ClientContext::default()
        };
        assert!(!ctx.is_empty());
        let json = serde_json::to_string(&ctx).unwrap();
        assert_eq!(json, r#"{"timezone":"America/New_York"}"#);
        let back: ClientContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ctx);
    }

    #[test]
    fn unknown_and_missing_keys_decode_leniently() {
        // Forward/backward compatibility: an unknown key is ignored and any
        // missing key defaults to `None`, so an older/newer client's payload
        // never fails to parse.
        let back: ClientContext =
            serde_json::from_str(r#"{"username":"ada","future_field":"x"}"#).unwrap();
        assert_eq!(back.username.as_deref(), Some("ada"));
        assert!(back.real_name.is_none());
    }
}
