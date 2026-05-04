//! Per-connection LLM client registry.
//!
//! Issue #9. Builds one `Arc<dyn LlmClient>` per entry in
//! [`ConnectionsMap`] (from #8) and tracks availability so a single
//! misconfigured connection does not prevent daemon startup.
//!
//! Downstream:
//! - #10 layers purpose configs (model / temperature / hosted-tool-search
//!   overrides) on top. Purposes reference a connection by id and the registry
//!   supplies the underlying client.
//! - #11 exposes the registry's [`ConnectionStatus`] list over the API.
//!
//! For now, the "active" connection (used as the single dispatch target until
//! purposes land) is the first entry in declaration order — see
//! [`ConnectionRegistry::active_connection`]. `IndexMap` preserves insertion
//! order so this is deterministic across startups.
//!
//! Reload: [`ConnectionRegistry::rebuild_from`] fully rebuilds the registry
//! from a fresh [`DaemonConfig`]. This is deliberately naive for #9; a future
//! ticket can diff and reuse live clients.
use std::fmt;
use std::sync::Arc;

use desktop_assistant_core::ports::llm::LlmClient;
use indexmap::IndexMap;

use crate::config::{
    DaemonConfig, ResolvedLlmConfig, resolve_connection_llm_config, resolve_llm_config,
};
use crate::connections::{ConnectionConfig, ConnectionId};

/// Availability of a single connection in the registry.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionHealth {
    /// Client was built successfully and is ready to dispatch requests.
    Ok,
    /// Client build failed for this connection. The daemon continued starting;
    /// requests routed to this id will be rejected until the config is fixed.
    Unavailable { reason: String },
}

impl fmt::Display for ConnectionHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => f.write_str("ok"),
            Self::Unavailable { reason } => write!(f, "unavailable: {reason}"),
        }
    }
}

/// Per-connection status surface for diagnostics and the future API (#11).
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionStatus {
    pub id: ConnectionId,
    /// Connector-type tag (`"openai"`, `"anthropic"`, etc.) from the config.
    pub connector_type: String,
    pub health: ConnectionHealth,
}

/// Registry of per-connection LLM clients plus their status.
///
/// Built at daemon startup via [`build_registry`]. Clients are stored as
/// `Arc<dyn LlmClient>` (#44) so dispatch can wrap them in retry/profiling
/// layers without committing to a concrete connector type. `IndexMap`
/// preserves declaration order so [`ConnectionRegistry::active_connection`]
/// is stable.
pub struct ConnectionRegistry {
    clients: IndexMap<ConnectionId, Arc<dyn LlmClient>>,
    status: IndexMap<ConnectionId, ConnectionStatus>,
    active: Option<ConnectionId>,
}

// Several accessors on the registry aren't consumed by the daemon binary at
// the #9 boundary — they exist for #10 (purpose-based dispatch via `get`) and
// #11 (status API via `status` / `status_of`). `#[allow(dead_code)]` silences
// the warnings until those tickets land.
#[allow(dead_code)]
impl ConnectionRegistry {
    /// Empty registry (used for tests and as a placeholder before
    /// [`build_registry`] runs).
    pub fn empty() -> Self {
        Self {
            clients: IndexMap::new(),
            status: IndexMap::new(),
            active: None,
        }
    }

    /// Look up a live client by connection id. Returns `None` for unknown ids
    /// and for ids whose client failed to build.
    ///
    /// Returns a cloned `Arc` handle so callers can await `stream_completion`
    /// without holding the registry lock — required by the #11 routing
    /// handler, which resolves connections under a read lock and then
    /// dispatches async.
    pub fn get(&self, id: &ConnectionId) -> Option<Arc<dyn LlmClient>> {
        self.clients.get(id).cloned()
    }

    /// Status of every declared connection in declaration order (includes
    /// both ok and unavailable entries).
    pub fn status(&self) -> Vec<ConnectionStatus> {
        self.status.values().cloned().collect()
    }

    /// Status for a specific connection id, if declared.
    pub fn status_of(&self, id: &ConnectionId) -> Option<&ConnectionStatus> {
        self.status.get(id)
    }

    /// The "active" connection id used for request dispatch until #10 wires
    /// purpose configs.
    ///
    /// Resolution order:
    /// 1. The first connection in declaration order whose client built
    ///    successfully.
    /// 2. Otherwise `None` (all connections failed to build — the daemon will
    ///    start but requests will be rejected with a clear error).
    ///
    /// This is recorded once at build time so the choice is stable after
    /// construction; if config reloads change the order, [`rebuild_from`]
    /// recomputes it.
    pub fn active_connection(&self) -> Option<&ConnectionId> {
        self.active.as_ref()
    }

    /// Count of live (healthy) clients.
    pub fn live_count(&self) -> usize {
        self.clients.len()
    }

    /// Count of declared connections regardless of health.
    pub fn declared_count(&self) -> usize {
        self.status.len()
    }

    /// Move the active client out of the registry.
    ///
    /// Legacy accessor from before purpose-based dispatch landed —
    /// production callers use [`ConnectionRegistry::get`] now. Retained
    /// for diagnostics and legacy tests.
    pub fn take_active(&mut self) -> Option<(ConnectionId, Arc<dyn LlmClient>)> {
        let id = self.active.clone()?;
        let client = self.clients.shift_remove(&id)?;
        Some((id, client))
    }

    /// Full rebuild from a fresh [`DaemonConfig`]. Scaffolds the reload path;
    /// #9 does not try to reuse live clients across rebuilds.
    pub fn rebuild_from(&mut self, config: &DaemonConfig) {
        *self = build_registry(config);
    }

    /// Fire-and-forget [`LlmClient::warmup`] on every live client.
    /// Spawns one detached task per connection. Today only Ollama
    /// overrides the default no-op — it populates a GGUF context-length
    /// cache so [`LlmClient::max_context_tokens`] returns the declared
    /// window instead of `None`. Failures (server down, model not
    /// pulled) are silently swallowed by the implementation — the
    /// daemon's universal fallback applies if the cache stays empty.
    ///
    /// Called once at daemon startup after [`build_registry`] returns.
    /// Must be invoked from inside a Tokio runtime.
    pub fn spawn_warmups(&self) {
        for (id, client) in &self.clients {
            let id = id.clone();
            let client = Arc::clone(client);
            tokio::spawn(async move {
                client.warmup().await;
                tracing::debug!(connection = %id, "warmup completed");
            });
        }
    }
}

impl Default for ConnectionRegistry {
    fn default() -> Self {
        Self::empty()
    }
}

/// Build an `Arc<dyn LlmClient>` from a resolved LLM configuration.
///
/// Infallible by design: the underlying client constructors never fail
/// synchronously. Errors (bad credentials, unreachable endpoint) surface on
/// the first request. [`build_registry`] does synchronous sanity checks
/// *before* calling this so misconfigured connections can be marked
/// unavailable up front.
pub fn build_llm_client(resolved: ResolvedLlmConfig) -> Arc<dyn LlmClient> {
    match resolved.connector.as_str() {
        "ollama" => Arc::new(
            desktop_assistant_llm_ollama::OllamaClient::new(resolved.base_url, resolved.model)
                .with_temperature(resolved.temperature)
                .with_top_p(resolved.top_p)
                .with_max_tokens(resolved.max_tokens),
        ),
        "anthropic" => {
            if resolved.api_key.is_empty() {
                tracing::warn!(
                    "No API key resolved from configured secret backend or environment; LLM calls may fail"
                );
            }
            let mut client =
                desktop_assistant_llm_anthropic::AnthropicClient::new(resolved.api_key)
                    .with_model(resolved.model)
                    .with_base_url(resolved.base_url)
                    .with_temperature(resolved.temperature)
                    .with_top_p(resolved.top_p)
                    .with_max_tokens_override(resolved.max_tokens);
            if let Some(hts) = resolved.hosted_tool_search {
                client = client.with_hosted_tool_search(hts);
            }
            Arc::new(client)
        }
        "bedrock" | "aws-bedrock" => Arc::new(
            desktop_assistant_llm_bedrock::BedrockClient::new(resolved.api_key)
                .with_model(resolved.model)
                .with_base_url(resolved.base_url)
                .with_temperature(resolved.temperature)
                .with_top_p(resolved.top_p)
                .with_max_tokens(resolved.max_tokens)
                .with_aws_profile(resolved.aws_profile),
        ),
        _ => {
            if resolved.api_key.is_empty() {
                tracing::warn!(
                    "No API key resolved from configured secret backend or environment; LLM calls may fail"
                );
            }
            let mut client = desktop_assistant_llm_openai::OpenAiClient::new(resolved.api_key)
                .with_model(resolved.model)
                .with_base_url(resolved.base_url)
                .with_temperature(resolved.temperature)
                .with_top_p(resolved.top_p)
                .with_max_tokens(resolved.max_tokens);
            if let Some(hts) = resolved.hosted_tool_search {
                client = client.with_hosted_tool_search(hts);
            }
            Arc::new(client)
        }
    }
}

/// Validate a resolved connection config before building the client.
///
/// Flags the cases that definitely cannot work at request time:
/// - OpenAI / Anthropic with no API key (neither secret backend nor env).
/// - An empty/whitespace base URL (the connector constructors accept these
///   silently and then fail on the first request with a less obvious error).
///
/// Returns `Ok(())` when the config looks plausible. Returns
/// `Err(reason)` when the daemon should mark the connection unavailable
/// rather than spin up a client that will just fail every request.
fn sanity_check_resolved(resolved: &ResolvedLlmConfig) -> Result<(), String> {
    if resolved.base_url.trim().is_empty() {
        return Err("base_url is empty after resolution".to_string());
    }
    if matches!(resolved.connector.as_str(), "openai" | "anthropic")
        && resolved.api_key.trim().is_empty()
    {
        return Err(format!(
            "{} connector has no api key (check `api_key_env`, `secret`, or the {}_API_KEY env var)",
            resolved.connector,
            resolved.connector.to_ascii_uppercase()
        ));
    }
    Ok(())
}

/// Resolve + build one client, or record an unavailable reason.
fn build_one(
    id: &ConnectionId,
    conn: &ConnectionConfig,
    config: &DaemonConfig,
) -> (Option<Arc<dyn LlmClient>>, ConnectionStatus) {
    let connector_type = conn.connector_type().to_string();
    let resolved = resolve_connection_llm_config(conn, Some(&config.llm));

    if let Err(reason) = sanity_check_resolved(&resolved) {
        tracing::warn!(
            connection = %id,
            connector = %connector_type,
            "connection unavailable: {reason}"
        );
        return (
            None,
            ConnectionStatus {
                id: id.clone(),
                connector_type,
                health: ConnectionHealth::Unavailable { reason },
            },
        );
    }

    tracing::info!(
        connection = %id,
        connector = %connector_type,
        model = %resolved.model,
        base_url = %resolved.base_url,
        "building connection client"
    );
    let client = build_llm_client(resolved);
    (
        Some(client),
        ConnectionStatus {
            id: id.clone(),
            connector_type,
            health: ConnectionHealth::Ok,
        },
    )
}

/// Build a [`ConnectionRegistry`] from a loaded [`DaemonConfig`].
///
/// Each connection is built independently. A failure on one connection is
/// logged and marked unavailable; it does not abort daemon startup. If the
/// config has no `[connections]` table (legacy / first-run path), the
/// registry is built from the top-level `[llm]` block under a synthetic id
/// `default` so existing installs keep working until migration completes.
pub fn build_registry(config: &DaemonConfig) -> ConnectionRegistry {
    let mut clients: IndexMap<ConnectionId, Arc<dyn LlmClient>> = IndexMap::new();
    let mut status: IndexMap<ConnectionId, ConnectionStatus> = IndexMap::new();

    let validated = match config.validated_connections() {
        Ok(map) => Some(map),
        Err(crate::connections::ConnectionsError::Empty) => None,
        Err(err) => {
            tracing::warn!(
                "[connections] map failed validation: {err}; falling back to legacy [llm] block"
            );
            None
        }
    };

    if let Some(map) = validated {
        for (id, conn) in map.iter() {
            let (client, st) = build_one(id, conn, config);
            if let Some(c) = client {
                clients.insert(id.clone(), c);
            }
            status.insert(id.clone(), st);
        }
    } else {
        // Legacy fall-through: synthesize a "default" connection from [llm].
        // This path is the same as the migration writes out for first-run,
        // but we do it in-memory here so a freshly generated empty config
        // (or a user who deleted `[connections]`) still gets a working
        // daemon until they fix it.
        let resolved = resolve_llm_config(Some(config));
        let id = ConnectionId::new("default").expect("literal slug is valid");
        let connector_type = resolved.connector.clone();
        match sanity_check_resolved(&resolved) {
            Ok(()) => {
                tracing::info!(
                    connection = %id,
                    connector = %connector_type,
                    model = %resolved.model,
                    "building legacy default connection client"
                );
                clients.insert(id.clone(), build_llm_client(resolved));
                status.insert(
                    id.clone(),
                    ConnectionStatus {
                        id: id.clone(),
                        connector_type,
                        health: ConnectionHealth::Ok,
                    },
                );
            }
            Err(reason) => {
                tracing::warn!(
                    connection = %id,
                    "legacy default connection unavailable: {reason}"
                );
                status.insert(
                    id.clone(),
                    ConnectionStatus {
                        id,
                        connector_type,
                        health: ConnectionHealth::Unavailable { reason },
                    },
                );
            }
        }
    }

    let active = status
        .iter()
        .find(|(_, s)| matches!(s.health, ConnectionHealth::Ok))
        .map(|(id, _)| id.clone());

    if active.is_none() {
        tracing::error!(
            "no usable LLM connection available after registry build; \
             daemon will start but LLM requests will fail until configuration is fixed"
        );
    }

    ConnectionRegistry {
        clients,
        status,
        active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{
        AnthropicConnection, BedrockConnection, ConnectionConfig, ConnectionsMap, OllamaConnection,
        OpenAiConnection,
    };
    use indexmap::IndexMap;

    fn config_from_pairs(pairs: Vec<(ConnectionId, ConnectionConfig)>) -> DaemonConfig {
        // Re-insert into a raw `IndexMap<String, _>` so `DaemonConfig::validated_connections`
        // re-walks the same id-validation path the real load does.
        let _ = ConnectionsMap::from_pairs(pairs.clone()).expect("valid pairs");
        let mut raw: IndexMap<String, ConnectionConfig> = IndexMap::new();
        for (id, conn) in pairs {
            raw.insert(id.into_string(), conn);
        }
        let mut config = DaemonConfig::default();
        config.connections = raw;
        config
    }

    fn openai_with_key(key: &str) -> ConnectionConfig {
        // Force the resolver down the env-var path with a known value.
        // `api_key_env` points at a variable set by the test below.
        ConnectionConfig::OpenAi(OpenAiConnection {
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key_env: Some(key.to_string()),
            secret: None,
        })
    }

    fn ollama_local() -> ConnectionConfig {
        ConnectionConfig::Ollama(OllamaConnection {
            base_url: Some("http://localhost:11434".to_string()),
        })
    }

    fn anthropic_with_key(key: &str) -> ConnectionConfig {
        ConnectionConfig::Anthropic(AnthropicConnection {
            base_url: Some("https://api.anthropic.com".to_string()),
            api_key_env: Some(key.to_string()),
            secret: None,
        })
    }

    #[test]
    fn registry_builds_ok_for_ollama() {
        // Ollama does not require an api key; just a base_url.
        let pairs = vec![(ConnectionId::new("local").unwrap(), ollama_local())];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);

        let id = ConnectionId::new("local").unwrap();
        assert!(
            registry.get(&id).is_some(),
            "expected live client for local"
        );
        assert_eq!(registry.live_count(), 1);
        assert_eq!(registry.declared_count(), 1);

        let st = registry.status_of(&id).expect("status present");
        assert!(
            matches!(st.health, ConnectionHealth::Ok),
            "expected Ok, got {:?}",
            st.health
        );
        assert_eq!(st.connector_type, "ollama");

        assert_eq!(registry.active_connection(), Some(&id));
    }

    #[test]
    fn registry_marks_openai_without_key_unavailable() {
        // Use an env var that almost certainly does not exist. The resolver
        // falls through to empty, which sanity_check flags.
        let unused = format!("DA_TEST_OPENAI_KEY_{}", uuid::Uuid::new_v4().simple());
        // Ensure it's not set.
        // SAFETY: single-threaded test; no other code touches this unique var.
        unsafe {
            std::env::remove_var(&unused);
        }

        let pairs = vec![(
            ConnectionId::new("cloud").unwrap(),
            openai_with_key(&unused),
        )];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);

        let id = ConnectionId::new("cloud").unwrap();
        assert!(
            registry.get(&id).is_none(),
            "expected no live client for misconfigured openai"
        );
        let st = registry.status_of(&id).expect("status present");
        match &st.health {
            ConnectionHealth::Unavailable { reason } => {
                assert!(
                    reason.contains("api key"),
                    "reason should mention missing api key, got: {reason}"
                );
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert_eq!(registry.live_count(), 0);
        assert_eq!(registry.declared_count(), 1);
        assert!(registry.active_connection().is_none());
    }

    #[test]
    fn registry_mix_of_valid_and_invalid_starts_daemon() {
        // One good (ollama), one bad (openai with no key). Active must be
        // the good one; daemon must not panic / error out.
        let unused = format!("DA_TEST_BAD_KEY_{}", uuid::Uuid::new_v4().simple());
        // SAFETY: single-threaded test; unique name.
        unsafe {
            std::env::remove_var(&unused);
        }

        let bad_id = ConnectionId::new("bad").unwrap();
        let good_id = ConnectionId::new("good").unwrap();
        let pairs = vec![
            (bad_id.clone(), openai_with_key(&unused)),
            (good_id.clone(), ollama_local()),
        ];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);

        assert!(registry.get(&good_id).is_some());
        assert!(registry.get(&bad_id).is_none());
        assert_eq!(registry.live_count(), 1);
        assert_eq!(registry.declared_count(), 2);

        // Active skips the unavailable bad entry and picks the first healthy.
        assert_eq!(registry.active_connection(), Some(&good_id));

        // Both entries appear in `status()`, in declaration order.
        let statuses = registry.status();
        assert_eq!(statuses.len(), 2);
        assert_eq!(statuses[0].id, bad_id);
        assert!(matches!(
            statuses[0].health,
            ConnectionHealth::Unavailable { .. }
        ));
        assert_eq!(statuses[1].id, good_id);
        assert!(matches!(statuses[1].health, ConnectionHealth::Ok));
    }

    #[test]
    fn registry_get_returns_right_client_per_id() {
        // The registry must preserve id → client association. We can't
        // check the underlying concrete type behind `Arc<dyn LlmClient>`
        // (#44), so assert via the parallel `connector_type` field on
        // `ConnectionStatus` plus the cheap `get_default_model` shape
        // each connector reports.
        let ollama_id = ConnectionId::new("local").unwrap();
        let bedrock_id = ConnectionId::new("aws").unwrap();
        let pairs = vec![
            (ollama_id.clone(), ollama_local()),
            (
                bedrock_id.clone(),
                ConnectionConfig::Bedrock(BedrockConnection {
                    aws_profile: Some("work".to_string()),
                    region: Some("us-west-2".to_string()),
                    base_url: None,
                }),
            ),
        ];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);

        assert!(registry.get(&ollama_id).is_some(), "ollama present");
        assert!(registry.get(&bedrock_id).is_some(), "bedrock present");

        let ollama_status = registry.status_of(&ollama_id).expect("ollama status");
        assert_eq!(ollama_status.connector_type, "ollama");
        let bedrock_status = registry.status_of(&bedrock_id).expect("bedrock status");
        assert_eq!(bedrock_status.connector_type, "bedrock");

        // Asking for a non-existent id returns None.
        let missing = ConnectionId::new("nope").unwrap();
        assert!(registry.get(&missing).is_none());
    }

    #[test]
    fn registry_active_is_first_healthy_in_declaration_order() {
        // Declaration order: x (ok), y (ok). Active must be x.
        let x = ConnectionId::new("x").unwrap();
        let y = ConnectionId::new("y").unwrap();
        let pairs = vec![(x.clone(), ollama_local()), (y.clone(), ollama_local())];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);
        assert_eq!(registry.active_connection(), Some(&x));
    }

    #[test]
    fn registry_take_active_removes_client_from_live_map() {
        let id = ConnectionId::new("local").unwrap();
        let pairs = vec![(id.clone(), ollama_local())];
        let config = config_from_pairs(pairs);
        let mut registry = build_registry(&config);

        let (taken_id, _client) = registry.take_active().expect("active present");
        assert_eq!(taken_id, id);
        // Client is no longer retrievable via `get` — it's been moved out.
        assert!(registry.get(&id).is_none());
        // But the status row remains so diagnostics still show the connection.
        assert!(registry.status_of(&id).is_some());
    }

    #[test]
    fn registry_rebuild_from_picks_up_new_connections() {
        let a = ConnectionId::new("a").unwrap();
        let config_a = config_from_pairs(vec![(a.clone(), ollama_local())]);
        let mut registry = build_registry(&config_a);
        assert_eq!(registry.declared_count(), 1);
        assert_eq!(registry.active_connection(), Some(&a));

        let b = ConnectionId::new("b").unwrap();
        let config_b = config_from_pairs(vec![
            (b.clone(), ollama_local()),
            (a.clone(), ollama_local()),
        ]);
        registry.rebuild_from(&config_b);
        assert_eq!(registry.declared_count(), 2);
        // New declaration order put `b` first.
        assert_eq!(registry.active_connection(), Some(&b));
    }

    #[test]
    fn registry_legacy_fallback_when_no_connections() {
        // No [connections] at all. Resolver will default to "openai" with no
        // api key (unless OPENAI_API_KEY happens to be set in the test env).
        // We only assert that the registry builds a status row for a
        // synthetic "default" id either way — actual availability depends on
        // env.
        let config = DaemonConfig::default();
        let registry = build_registry(&config);
        assert_eq!(registry.declared_count(), 1);
        let default_id = ConnectionId::new("default").unwrap();
        assert!(registry.status_of(&default_id).is_some());
    }

    #[test]
    fn sanity_check_rejects_empty_base_url() {
        let resolved = ResolvedLlmConfig {
            connector: "openai".to_string(),
            model: "gpt".to_string(),
            base_url: "   ".to_string(),
            api_key: "present".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
            aws_profile: None,
        };
        let err = sanity_check_resolved(&resolved).unwrap_err();
        assert!(err.contains("base_url"), "got: {err}");
    }

    #[test]
    fn sanity_check_allows_bedrock_without_api_key() {
        // Bedrock auth flows through AWS credentials; empty api_key is normal.
        let resolved = ResolvedLlmConfig {
            connector: "bedrock".to_string(),
            model: "m".to_string(),
            base_url: "us-west-2".to_string(),
            api_key: String::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
            aws_profile: Some("work".to_string()),
        };
        sanity_check_resolved(&resolved).expect("bedrock without api key should pass");
    }

    #[test]
    fn sanity_check_allows_ollama_without_api_key() {
        let resolved = ResolvedLlmConfig {
            connector: "ollama".to_string(),
            model: "m".to_string(),
            base_url: "http://localhost:11434".to_string(),
            api_key: String::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
            aws_profile: None,
        };
        sanity_check_resolved(&resolved).expect("ollama without api key should pass");
    }

    #[test]
    fn anthropic_without_key_flagged() {
        let unused = format!("DA_TEST_ANTHROPIC_KEY_{}", uuid::Uuid::new_v4().simple());
        // SAFETY: unique name, single-threaded test.
        unsafe {
            std::env::remove_var(&unused);
        }
        let id = ConnectionId::new("anth").unwrap();
        let pairs = vec![(id.clone(), anthropic_with_key(&unused))];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);
        assert!(registry.get(&id).is_none());
        match &registry.status_of(&id).unwrap().health {
            ConnectionHealth::Unavailable { reason } => {
                assert!(reason.contains("api key"), "{reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn integration_multi_connection_fixture_daemon_starts() {
        // Load the multi-connection golden fixture, run `build_registry`, and
        // assert the daemon-startup invariants: at least one healthy client,
        // deterministic active id, unavailable connections surfaced without
        // aborting.
        let fixture = include_str!("../tests/fixtures/connections_migration/multi_connection.toml");
        let config: DaemonConfig = toml::from_str(fixture).expect("fixture is valid TOML");

        // The fixture relies on one env var being unset so the openai
        // connection resolves with no key. Clear it defensively.
        // SAFETY: single-threaded test; name is specific to this fixture.
        unsafe {
            std::env::remove_var("DA_ISSUE9_FIXTURE_UNSET_OPENAI_KEY");
        }

        let registry = build_registry(&config);
        assert_eq!(registry.declared_count(), 3, "three connections declared");

        let local = ConnectionId::new("local").unwrap();
        let cloud = ConnectionId::new("cloud").unwrap();
        let aws = ConnectionId::new("aws").unwrap();

        // `local` (ollama) and `aws` (bedrock) are expected healthy.
        assert!(
            registry.get(&local).is_some(),
            "local ollama should be live"
        );
        assert!(matches!(
            registry.status_of(&local).unwrap().health,
            ConnectionHealth::Ok
        ));
        assert!(registry.get(&aws).is_some(), "aws bedrock should be live");
        assert!(matches!(
            registry.status_of(&aws).unwrap().health,
            ConnectionHealth::Ok
        ));

        // `cloud` (openai) has no key; expected unavailable.
        assert!(
            registry.get(&cloud).is_none(),
            "cloud openai should be unavailable"
        );
        match &registry.status_of(&cloud).unwrap().health {
            ConnectionHealth::Unavailable { reason } => {
                assert!(reason.contains("api key"), "reason: {reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }

        // Active = first healthy in declaration order. `local` is declared
        // first, so it wins.
        assert_eq!(registry.active_connection(), Some(&local));
    }

    #[test]
    fn bedrock_connection_builds_without_api_key() {
        // Proves BedrockConnection's lack of api_key_env / secret still produces
        // a live client (auth happens via AWS SDK at request time).
        let id = ConnectionId::new("aws").unwrap();
        let pairs = vec![(
            id.clone(),
            ConnectionConfig::Bedrock(BedrockConnection {
                aws_profile: Some("work".to_string()),
                region: Some("us-west-2".to_string()),
                base_url: None,
            }),
        )];
        let config = config_from_pairs(pairs);
        let registry = build_registry(&config);
        assert!(registry.get(&id).is_some());
        assert!(matches!(
            registry.status_of(&id).unwrap().health,
            ConnectionHealth::Ok
        ));
    }
}
