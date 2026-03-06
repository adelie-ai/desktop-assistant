use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::embedding::{EmbedFn, EmbeddingClient};
use desktop_assistant_core::ports::inbound::SettingsService;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, RetryingLlmClient,
};
use desktop_assistant_core::ports::llm_profiling::MaybeProfiled;
use tracing_subscriber::EnvFilter;

mod app;
mod config;
mod settings_service;
mod store;

use crate::app::Assistant;
use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_core::service::ConversationHandler;
use desktop_assistant_dbus::conversation::DbusConversationAdapter;
use desktop_assistant_dbus::settings::DbusSettingsAdapter;
use desktop_assistant_mcp_client::config as mcp_config;
use desktop_assistant_mcp_client::executor::{BuiltinToolService, McpToolExecutor};
use desktop_assistant_ws as ws;
use settings_service::DaemonSettingsService;
use store::PersistentConversationStore;

struct WsSettingsAuth<S: SettingsService + 'static> {
    settings: Arc<S>,
}

impl<S: SettingsService + 'static> WsSettingsAuth<S> {
    fn new(settings: Arc<S>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl<S: SettingsService + 'static> ws::WsAuthValidator for WsSettingsAuth<S> {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        self.settings
            .validate_ws_jwt(token.to_string())
            .await
            .unwrap_or(false)
    }
}

struct WsBasicLogin<S: SettingsService + 'static> {
    settings: Arc<S>,
    username: String,
    mode: WsLoginMode,
}

enum WsLoginMode {
    StaticPassword(String),
    SystemPassword,
}

impl<S: SettingsService + 'static> WsBasicLogin<S> {
    fn new(settings: Arc<S>, username: String, mode: WsLoginMode) -> Self {
        Self {
            settings,
            username,
            mode,
        }
    }
}

#[async_trait]
impl<S: SettingsService + 'static> ws::WsLoginService for WsBasicLogin<S> {
    async fn authenticate_basic(&self, username: &str, password: &str) -> bool {
        if username != self.username {
            return false;
        }

        match &self.mode {
            WsLoginMode::StaticPassword(expected) => password == expected,
            WsLoginMode::SystemPassword => {
                match config::authenticate_os_user_password(username, password) {
                    Ok(valid) => valid,
                    Err(error) => {
                        tracing::warn!("system-password auth check failed: {error}");
                        false
                    }
                }
            }
        }
    }

    async fn issue_token_for_subject(&self, subject: &str) -> std::result::Result<String, String> {
        self.settings
            .generate_ws_jwt(Some(subject.to_string()))
            .await
            .map_err(|error| error.to_string())
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn is_container_environment() -> bool {
    std::env::var("container")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .is_some()
        || std::path::Path::new("/.dockerenv").exists()
        || std::path::Path::new("/run/.containerenv").exists()
}

fn resolve_ws_login_mode_decision(
    current_username: String,
    configured_username: Option<String>,
    configured_password: Option<String>,
    local_system_auth_enabled: bool,
    is_container: bool,
) -> Option<(String, WsLoginMode)> {
    if let Some(password) = configured_password {
        let username = configured_username.unwrap_or(current_username);
        return Some((username, WsLoginMode::StaticPassword(password)));
    }

    if local_system_auth_enabled && !is_container {
        return Some((current_username, WsLoginMode::SystemPassword));
    }

    None
}

fn resolve_ws_login_mode() -> Option<(String, WsLoginMode)> {
    let current_username = config::current_username();
    let configured_username = std::env::var("DESKTOP_ASSISTANT_WS_LOGIN_USERNAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let configured_password = std::env::var("DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let local_system_auth_enabled = env_bool("DESKTOP_ASSISTANT_WS_LOGIN_LOCAL_SYSTEM_AUTH", true);
    resolve_ws_login_mode_decision(
        current_username,
        configured_username,
        configured_password,
        local_system_auth_enabled,
        is_container_environment(),
    )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to install Ctrl+C handler: {e}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                let _ = stream.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Enum wrapper to dispatch between conversation store backends at runtime.
enum AnyConversationStore {
    Json(PersistentConversationStore),
    Postgres(desktop_assistant_storage::PgConversationStore),
}

impl desktop_assistant_core::ports::store::ConversationStore for AnyConversationStore {
    async fn create(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.create(conv).await,
            Self::Postgres(s) => s.create(conv).await,
        }
    }

    async fn get(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<desktop_assistant_core::domain::Conversation, CoreError> {
        match self {
            Self::Json(s) => s.get(id).await,
            Self::Postgres(s) => s.get(id).await,
        }
    }

    async fn list(&self) -> Result<Vec<desktop_assistant_core::domain::Conversation>, CoreError> {
        match self {
            Self::Json(s) => s.list().await,
            Self::Postgres(s) => s.list().await,
        }
    }

    async fn update(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.update(conv).await,
            Self::Postgres(s) => s.update(conv).await,
        }
    }

    async fn delete(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.delete(id).await,
            Self::Postgres(s) => s.delete(id).await,
        }
    }

    async fn create_summary(
        &self,
        conversation_id: &desktop_assistant_core::domain::ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        match self {
            Self::Json(s) => {
                s.create_summary(conversation_id, summary, start_ordinal, end_ordinal)
                    .await
            }
            Self::Postgres(s) => {
                s.create_summary(conversation_id, summary, start_ordinal, end_ordinal)
                    .await
            }
        }
    }

    async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.expand_summary(summary_id).await,
            Self::Postgres(s) => s.expand_summary(summary_id).await,
        }
    }
}

/// Enum wrapper to dispatch between LLM backends at runtime.
///
/// `LlmClient` uses `impl Future` returns, so it isn't dyn-compatible.
/// This enum lets `ConversationHandler` stay monomorphic while supporting
/// multiple backends.
enum AnyLlmClient {
    Anthropic(desktop_assistant_llm_anthropic::AnthropicClient),
    Bedrock(desktop_assistant_llm_bedrock::BedrockClient),
    OpenAi(desktop_assistant_llm_openai::OpenAiClient),
    Ollama(desktop_assistant_llm_ollama::OllamaClient),
}

/// Enum wrapper to dispatch between embedding backends at runtime.
///
/// Mirrors `AnyLlmClient` but for the `EmbeddingClient` trait.
/// `Unavailable` is used when the resolved connector doesn't support embeddings (e.g. Anthropic).
enum AnyEmbeddingClient {
    Bedrock(desktop_assistant_llm_bedrock::BedrockClient),
    OpenAi(desktop_assistant_llm_openai::OpenAiClient),
    Ollama(desktop_assistant_llm_ollama::OllamaClient),
    Unavailable,
}

impl EmbeddingClient for AnyEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        match self {
            Self::Bedrock(c) => c.embed(texts).await,
            Self::OpenAi(c) => c.embed(texts).await,
            Self::Ollama(c) => c.embed(texts).await,
            Self::Unavailable => Err(CoreError::Llm(
                "embeddings are not available: current connector does not support embeddings"
                    .to_string(),
            )),
        }
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        match self {
            Self::Bedrock(c) => c.model_identifier().await,
            Self::OpenAi(c) => c.model_identifier().await,
            Self::Ollama(c) => c.model_identifier().await,
            Self::Unavailable => Err(CoreError::Llm(
                "embeddings are not available: current connector does not support embeddings"
                    .to_string(),
            )),
        }
    }
}

impl LlmClient for AnyLlmClient {
    fn get_default_model(&self) -> Option<&str> {
        match self {
            Self::Anthropic(c) => c.get_default_model(),
            Self::Bedrock(c) => c.get_default_model(),
            Self::OpenAi(c) => c.get_default_model(),
            Self::Ollama(c) => c.get_default_model(),
        }
    }

    fn get_default_base_url(&self) -> Option<&str> {
        match self {
            Self::Anthropic(c) => c.get_default_base_url(),
            Self::Bedrock(c) => c.get_default_base_url(),
            Self::OpenAi(c) => c.get_default_base_url(),
            Self::Ollama(c) => c.get_default_base_url(),
        }
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Anthropic(c) => c.stream_completion(messages, tools, on_chunk).await,
            Self::Bedrock(c) => c.stream_completion(messages, tools, on_chunk).await,
            Self::OpenAi(c) => c.stream_completion(messages, tools, on_chunk).await,
            Self::Ollama(c) => c.stream_completion(messages, tools, on_chunk).await,
        }
    }

    fn supports_hosted_tool_search(&self) -> bool {
        match self {
            Self::Anthropic(c) => c.supports_hosted_tool_search(),
            Self::OpenAi(c) => c.supports_hosted_tool_search(),
            _ => false,
        }
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Anthropic(c) => {
                c.stream_completion_with_namespaces(messages, core_tools, namespaces, on_chunk)
                    .await
            }
            Self::OpenAi(c) => {
                c.stream_completion_with_namespaces(messages, core_tools, namespaces, on_chunk)
                    .await
            }
            // Bedrock/Ollama: use default flattening behavior
            _ => {
                let mut all: Vec<ToolDefinition> = core_tools.to_vec();
                for ns in namespaces {
                    all.extend(ns.tools.iter().cloned());
                }
                self.stream_completion(messages, &all, on_chunk).await
            }
        }
    }
}

/// Build an `AnyLlmClient` from a resolved LLM configuration.
fn build_llm_client(resolved: config::ResolvedLlmConfig) -> AnyLlmClient {
    match resolved.connector.as_str() {
        "ollama" => AnyLlmClient::Ollama(
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
            AnyLlmClient::Anthropic(
                desktop_assistant_llm_anthropic::AnthropicClient::new(resolved.api_key)
                    .with_model(resolved.model)
                    .with_base_url(resolved.base_url)
                    .with_temperature(resolved.temperature)
                    .with_top_p(resolved.top_p)
                    .with_max_tokens_override(resolved.max_tokens),
            )
        }
        "bedrock" | "aws-bedrock" => AnyLlmClient::Bedrock(
            desktop_assistant_llm_bedrock::BedrockClient::new(resolved.api_key)
                .with_model(resolved.model)
                .with_base_url(resolved.base_url)
                .with_temperature(resolved.temperature)
                .with_top_p(resolved.top_p)
                .with_max_tokens(resolved.max_tokens),
        ),
        _ => {
            if resolved.api_key.is_empty() {
                tracing::warn!(
                    "No API key resolved from configured secret backend or environment; LLM calls may fail"
                );
            }
            AnyLlmClient::OpenAi(
                desktop_assistant_llm_openai::OpenAiClient::new(resolved.api_key)
                    .with_model(resolved.model)
                    .with_base_url(resolved.base_url)
                    .with_temperature(resolved.temperature)
                    .with_top_p(resolved.top_p)
                    .with_max_tokens(resolved.max_tokens),
            )
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tracing::info!("desktop-assistant starting");

    // Build the LLM client from daemon.toml + KWallet (fallback to env)
    let config_path = config::default_daemon_config_path();
    let daemon_config = match config::load_daemon_config(&config_path) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(
                "failed to load daemon config at {}: {error}",
                config_path.display()
            );
            None
        }
    };

    let profiling = daemon_config
        .as_ref()
        .map(|c| c.profiling.clone())
        .unwrap_or_default();

    let resolved_llm = config::resolve_llm_config(daemon_config.as_ref());
    tracing::info!(
        "LLM connector={}, model={}, base_url={}",
        resolved_llm.connector,
        resolved_llm.model,
        resolved_llm.base_url
    );

    let llm_connector = resolved_llm.connector.clone();
    let llm_api_key = resolved_llm.api_key.clone();

    tracing::info!("using {} LLM backend", llm_connector);
    let llm: AnyLlmClient = build_llm_client(resolved_llm);

    // Build the embedding client from resolved config
    let resolved_emb = config::resolve_embeddings_config(daemon_config.as_ref());
    tracing::info!(
        "Embeddings connector={}, model={}, base_url={}, available={}, is_default={}",
        resolved_emb.connector,
        resolved_emb.model,
        resolved_emb.base_url,
        resolved_emb.available,
        resolved_emb.is_default
    );

    let embedding_client: AnyEmbeddingClient = if !resolved_emb.available {
        tracing::info!(
            "embeddings unavailable (connector={})",
            resolved_emb.connector
        );
        AnyEmbeddingClient::Unavailable
    } else {
        match resolved_emb.connector.as_str() {
            "ollama" => {
                tracing::info!("using Ollama embedding backend");
                AnyEmbeddingClient::Ollama(desktop_assistant_llm_ollama::OllamaClient::new(
                    resolved_emb.base_url.clone(),
                    resolved_emb.model.clone(),
                ))
            }
            "bedrock" | "aws-bedrock" => {
                tracing::info!("using Bedrock embedding backend");
                AnyEmbeddingClient::Bedrock(
                    desktop_assistant_llm_bedrock::BedrockClient::new(String::new())
                        .with_model(resolved_emb.model.clone())
                        .with_base_url(resolved_emb.base_url.clone()),
                )
            }
            _ => {
                tracing::info!("using OpenAI-compatible embedding backend");
                let api_key = if resolved_emb.is_default || resolved_emb.connector == llm_connector
                {
                    llm_api_key.clone()
                } else {
                    let env_key =
                        format!("{}_API_KEY", resolved_emb.connector.to_ascii_uppercase());
                    std::env::var(env_key).unwrap_or_default()
                };
                AnyEmbeddingClient::OpenAi(
                    desktop_assistant_llm_openai::OpenAiClient::new(api_key)
                        .with_model(resolved_emb.model.clone())
                        .with_base_url(resolved_emb.base_url.clone()),
                )
            }
        }
    };

    let embedding_client = Arc::new(embedding_client);

    // Resolve model identifier once at startup (includes digest for Ollama).
    let embedding_model_id: String = if matches!(
        embedding_client.as_ref(),
        AnyEmbeddingClient::Unavailable
    ) {
        resolved_emb.model.clone()
    } else {
        match embedding_client.model_identifier().await {
            Ok(id) => {
                tracing::info!("resolved embedding model identifier: {id}");
                id
            }
            Err(e) => {
                tracing::warn!(
                    "failed to resolve embedding model identifier, falling back to configured name: {e}"
                );
                resolved_emb.model.clone()
            }
        }
    };

    let embedding_fn: Option<EmbedFn> =
        if matches!(embedding_client.as_ref(), AnyEmbeddingClient::Unavailable) {
            None
        } else {
            let client = Arc::clone(&embedding_client);
            Some(Arc::new(move |texts: Vec<String>| {
                let client = Arc::clone(&client);
                Box::pin(async move { client.embed(texts).await })
            }))
        };

    // --- Database (optional) ---
    let (db_url, db_max_conns) = config::resolve_database_config(daemon_config.as_ref());
    let pg_pool = if let Some(url) = db_url {
        tracing::info!(
            "connecting to PostgreSQL (max_connections={})",
            db_max_conns
        );
        match desktop_assistant_storage::create_pool(&url, db_max_conns).await {
            Ok(pool) => {
                if let Err(e) = desktop_assistant_storage::run_migrations(&pool).await {
                    tracing::error!("failed to run database migrations: {e}");
                    return Err(e.into());
                }
                tracing::info!("database migrations applied successfully");

                // One-time JSON → Postgres migration (runs if JSON files exist)
                let conv_json = store::default_conversation_store_path();
                let data_home = conv_json.parent().unwrap_or(std::path::Path::new("."));
                let prefs_json = data_home.join("preferences.json");
                let memory_json = data_home.join("factual_memory.json");

                if conv_json.exists() || prefs_json.exists() || memory_json.exists() {
                    // Only migrate if tables are empty (first startup with Postgres)
                    if conv_json.exists()
                        && desktop_assistant_storage::is_conversations_table_empty(&pool).await
                    {
                        match desktop_assistant_storage::migrate_conversations(&conv_json, &pool)
                            .await
                        {
                            Ok(n) => tracing::info!("migrated {n} conversations from JSON"),
                            Err(e) => tracing::warn!("conversation migration failed: {e}"),
                        }
                    }
                    if (prefs_json.exists() || memory_json.exists())
                        && desktop_assistant_storage::is_knowledge_base_table_empty(&pool).await
                    {
                        match desktop_assistant_storage::migrate_knowledge(
                            &prefs_json,
                            &memory_json,
                            &pool,
                        )
                        .await
                        {
                            Ok(n) => tracing::info!("migrated {n} knowledge entries from JSON"),
                            Err(e) => tracing::warn!("knowledge migration failed: {e}"),
                        }
                    }
                }

                Some(pool)
            }
            Err(e) => {
                tracing::error!("failed to connect to PostgreSQL: {e}");
                return Err(e.into());
            }
        }
    } else {
        tracing::info!("no database URL configured; Postgres features disabled");
        None
    };

    // Invalidate embeddings from a different model so that vector-dimension
    // mismatches cannot cause search errors while the backfill is running.
    if let Some(pool) = &pg_pool
        && !matches!(embedding_client.as_ref(), AnyEmbeddingClient::Unavailable)
    {
        match desktop_assistant_storage::embedding_backfill::invalidate_stale_embeddings(
            pool,
            &embedding_model_id,
        )
        .await
        {
            Ok((kb, tools)) if kb > 0 || tools > 0 => {
                tracing::warn!(
                    "embedding model changed to {}: invalidated {kb} knowledge + {tools} tool embeddings (will re-embed in background)",
                    embedding_model_id
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("failed to invalidate stale embeddings: {e}");
            }
        }
    }

    // --- Knowledge base & tool registry stores ---
    let kb_store = pg_pool.as_ref().map(|pool| {
        Arc::new(desktop_assistant_storage::PgKnowledgeBaseStore::new(
            pool.clone(),
        ))
    });

    let tool_registry_store = pg_pool.as_ref().map(|pool| {
        Arc::new(desktop_assistant_storage::PgToolRegistryStore::new(
            pool.clone(),
        ))
    });

    // Load MCP server configuration
    let mcp_config_path = mcp_config::default_config_path();
    let mcp_configs = mcp_config::load_mcp_configs(&mcp_config_path).unwrap_or_else(|e| {
        tracing::warn!("failed to load MCP config: {e}");
        Vec::new()
    });

    // Build the MCP tool executor with builtin tools
    let mut builtin_tools = BuiltinToolService::new();
    if let Some(embed_fn) = embedding_fn {
        tracing::info!(
            "enabling built-in vector search with model={}",
            embedding_model_id
        );
        builtin_tools = builtin_tools.with_embedding(embed_fn);
    } else {
        tracing::info!("built-in vector search disabled (no embedding backend available)");
    }

    if let Some(kb) = &kb_store {
        tracing::info!("wiring knowledge base store into builtin tools");
        let kb_w = Arc::clone(kb);
        let kb_s = Arc::clone(kb);
        let kb_d = Arc::clone(kb);
        let kb_emb_model = embedding_model_id.clone();
        use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
        builtin_tools = builtin_tools.with_knowledge_base(
            Arc::new(move |entry, embedding: Option<Vec<Vec<f32>>>| {
                let store = Arc::clone(&kb_w);
                let model = if embedding.is_some() {
                    Some(kb_emb_model.clone())
                } else {
                    None
                };
                Box::pin(async move { store.write(entry, embedding, model).await })
            }),
            Arc::new(move |query, embedding, tags, limit| {
                let store = Arc::clone(&kb_s);
                Box::pin(async move { store.search(&query, embedding, tags, limit).await })
            }),
            Arc::new(move |id| {
                let store = Arc::clone(&kb_d);
                Box::pin(async move { store.delete(&id).await })
            }),
        );
    }

    if let Some(pool) = &pg_pool {
        tracing::info!("wiring database query into builtin tools");
        let pool_for_db = pool.clone();
        builtin_tools = builtin_tools.with_database(Arc::new(move |sql, limit| {
            let pool = pool_for_db.clone();
            Box::pin(async move {
                desktop_assistant_storage::execute_database_query(&pool, &sql, limit).await
            })
        }));
    }

    if let Some(tr) = &tool_registry_store {
        tracing::info!("wiring tool registry store into builtin tools");
        let tr_s = Arc::clone(tr);
        let tr_d = Arc::clone(tr);
        use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
        builtin_tools = builtin_tools.with_tool_registry(
            Arc::new(move |query, embedding, limit| {
                let store = Arc::clone(&tr_s);
                Box::pin(async move { store.search_tools(&query, embedding, limit).await })
            }),
            Arc::new(move |name| {
                let store = Arc::clone(&tr_d);
                Box::pin(async move { store.tool_definition(&name).await })
            }),
        );
    }

    let mut tool_executor = McpToolExecutor::with_builtin_tools_and_config_path(
        mcp_configs,
        builtin_tools,
        mcp_config_path,
    );
    let mcp_handle = tool_executor.control_handle();
    tool_executor
        .builtin_tools_mut()
        .set_mcp_control(mcp_handle.clone());
    if let Err(e) = tool_executor.start().await {
        tracing::warn!("failed to start MCP servers: {e}");
    }

    // Register discovered MCP tools in the tool registry (with embeddings)
    let registered_tools: Vec<(String, String)> = tool_executor.tools_by_service().await;
    if registered_tools.is_empty() {
        tracing::info!("MCP startup complete: no tools registered");
    } else {
        tracing::info!(
            "MCP startup complete: {} tool(s) registered",
            registered_tools.len()
        );
        for (i, (service, tool)) in registered_tools.iter().enumerate() {
            tracing::info!("  {}. [{}] {}", i + 1, service, tool);
        }
    }

    if let Some(tr) = &tool_registry_store {
        use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
        use desktop_assistant_core::ports::tools::ToolExecutor;

        // Register builtin tools as core (always sent to LLM)
        let builtin_defs: Vec<_> = tool_executor
            .core_tools()
            .await
            .into_iter()
            .filter(|t| t.name.starts_with("builtin_"))
            .collect();
        let builtin_embeddings = vec![None; builtin_defs.len()];
        if let Err(e) = tr
            .register_tools(builtin_defs, "builtin", true, builtin_embeddings, None)
            .await
        {
            tracing::warn!("failed to register builtin tools in registry: {e}");
        }

        // Register MCP tools as non-core (discoverable via tool_search)
        let mcp_defs: Vec<_> = tool_executor.all_mcp_tools().await;
        let mcp_embeddings = vec![None; mcp_defs.len()];
        if !mcp_defs.is_empty()
            && let Err(e) = tr
                .register_tools(mcp_defs, "mcp", false, mcp_embeddings, None)
                .await
        {
            tracing::warn!("failed to register MCP tools in registry: {e}");
        }
    }

    // Spawn background embedding backfill task
    let (backfill_shutdown_tx, backfill_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let backfill_task = if let (Some(pool), true) = (
        &pg_pool,
        !matches!(embedding_client.as_ref(), AnyEmbeddingClient::Unavailable),
    ) {
        let pool = pool.clone();
        let client = Arc::clone(&embedding_client);
        let model = embedding_model_id.clone();
        Some(tokio::spawn(async move {
            // Let tool registration and MCP connections settle.
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                _ = backfill_shutdown_rx => {
                    tracing::info!("embedding backfill cancelled before start");
                    return;
                }
            }

            tracing::info!("starting embedding backfill (model={model})");

            let embed_fn: desktop_assistant_storage::embedding_backfill::BackfillEmbedFn =
                Box::new(move |texts| {
                    let client = Arc::clone(&client);
                    Box::pin(async move { client.embed(texts).await.map_err(|e| e.to_string()) })
                });

            match desktop_assistant_storage::embedding_backfill::backfill_tool_embeddings(
                &pool, &embed_fn, &model,
            )
            .await
            {
                Ok(n) if n > 0 => tracing::info!("backfilled {n} tool embedding(s)"),
                Ok(_) => tracing::debug!("no tool embeddings to backfill"),
                Err(e) => tracing::warn!("tool embedding backfill failed: {e}"),
            }

            match desktop_assistant_storage::embedding_backfill::backfill_knowledge_embeddings(
                &pool, &embed_fn, &model,
            )
            .await
            {
                Ok(n) if n > 0 => tracing::info!("backfilled {n} knowledge embedding(s)"),
                Ok(_) => tracing::debug!("no knowledge embeddings to backfill"),
                Err(e) => tracing::warn!("knowledge embedding backfill failed: {e}"),
            }
        }))
    } else {
        drop(backfill_shutdown_rx);
        None
    };

    // Spawn background dreaming (periodic fact extraction) task
    let dreaming_enabled = daemon_config
        .as_ref()
        .map(|c| c.backend_tasks.dreaming_enabled)
        .unwrap_or(false);
    let dreaming_interval_secs = daemon_config
        .as_ref()
        .map(|c| c.backend_tasks.dreaming_interval_secs)
        .unwrap_or(3600);

    let (dreaming_shutdown_tx, dreaming_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let dreaming_task = if dreaming_enabled {
        if let (Some(pool), true) = (
            &pg_pool,
            !matches!(embedding_client.as_ref(), AnyEmbeddingClient::Unavailable),
        ) {
            let resolved_bt_llm = config::resolve_backend_tasks_llm_config(daemon_config.as_ref());
            tracing::info!(
                "dreaming LLM connector={}, model={}",
                resolved_bt_llm.connector,
                resolved_bt_llm.model
            );

            let dreaming_llm = build_llm_client(resolved_bt_llm);
            let dreaming_llm = RetryingLlmClient::new(dreaming_llm, 3);
            let dreaming_llm = MaybeProfiled::from_config(
                dreaming_llm,
                profiling.enabled,
                profiling.log_path.as_deref(),
                profiling.full_content,
            );
            let dreaming_llm = Arc::new(dreaming_llm);

            let pool = pool.clone();
            let emb_client = Arc::clone(&embedding_client);
            let emb_model = embedding_model_id.clone();

            Some(tokio::spawn(async move {
                let mut shutdown_rx = dreaming_shutdown_rx;

                // Initial delay — let startup settle.
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {}
                    _ = &mut shutdown_rx => {
                        tracing::info!("dreaming cancelled before first scan");
                        return;
                    }
                }

                let llm_fn: desktop_assistant_storage::dreaming::DreamingLlmFn =
                    Box::new(move |system_prompt, user_prompt| {
                        let llm = Arc::clone(&dreaming_llm);
                        Box::pin(async move {
                            let messages = vec![
                                Message::new(Role::System, system_prompt),
                                Message::new(Role::User, user_prompt),
                            ];
                            let response = llm
                                .stream_completion(messages, &[], Box::new(|_| true))
                                .await
                                .map_err(|e| e.to_string())?;
                            Ok(response.text)
                        })
                    });

                let embed_fn: desktop_assistant_storage::dreaming::BackfillEmbedFn =
                    Box::new(move |texts| {
                        let client = Arc::clone(&emb_client);
                        Box::pin(
                            async move { client.embed(texts).await.map_err(|e| e.to_string()) },
                        )
                    });

                loop {
                    tracing::info!("dreaming: starting scan cycle");
                    match desktop_assistant_storage::dreaming::run_dreaming_scan(
                        &pool, &llm_fn, &embed_fn, &emb_model,
                    )
                    .await
                    {
                        Ok(n) if n > 0 => tracing::info!("dreaming: wrote {n} new fact(s)"),
                        Ok(_) => tracing::debug!("dreaming: no new facts extracted"),
                        Err(e) => tracing::warn!("dreaming: scan failed: {e}"),
                    }

                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(dreaming_interval_secs)) => {}
                        _ = &mut shutdown_rx => {
                            tracing::info!("dreaming: shutdown signal received");
                            break;
                        }
                    }
                }
            }))
        } else {
            if pg_pool.is_none() {
                tracing::warn!("dreaming enabled but no database configured; dreaming disabled");
            } else {
                tracing::warn!("dreaming enabled but embeddings unavailable; dreaming disabled");
            }
            drop(dreaming_shutdown_rx);
            None
        }
    } else {
        tracing::debug!("dreaming disabled");
        drop(dreaming_shutdown_rx);
        None
    };

    // Build the conversation service with tool support
    let conversation_store: AnyConversationStore = if let Some(pool) = &pg_pool {
        tracing::info!("using PostgreSQL conversation store");
        AnyConversationStore::Postgres(desktop_assistant_storage::PgConversationStore::new(
            pool.clone(),
        ))
    } else {
        let store = PersistentConversationStore::from_default_path().map_err(|e| {
            anyhow::anyhow!("failed to initialize persistent conversation store: {e}")
        })?;
        tracing::info!(
            "using JSON conversation store at {}",
            store::default_conversation_store_path().display()
        );
        AnyConversationStore::Json(store)
    };

    let llm = RetryingLlmClient::new(llm, 3);
    let llm = MaybeProfiled::from_config(
        llm,
        profiling.enabled,
        profiling.log_path.as_deref(),
        profiling.full_content,
    );
    let mut handler = ConversationHandler::with_tools(
        conversation_store,
        llm,
        tool_executor,
        Box::new(|| uuid::Uuid::now_v7().to_string()),
    );

    // Build a separate LLM for backend tasks (title generation, context summary)
    // if [backend_tasks.llm] is configured and differs from the primary.
    let resolved_bt = config::resolve_backend_tasks_llm_config(daemon_config.as_ref());
    let resolved_primary = config::resolve_llm_config(daemon_config.as_ref());
    if resolved_bt.connector != resolved_primary.connector
        || resolved_bt.model != resolved_primary.model
    {
        tracing::info!(
            "backend-tasks LLM connector={}, model={}",
            resolved_bt.connector,
            resolved_bt.model
        );
        let bt_llm = build_llm_client(resolved_bt);
        let bt_llm = RetryingLlmClient::new(bt_llm, 3);
        let bt_llm = MaybeProfiled::from_config(
            bt_llm,
            profiling.enabled,
            profiling.log_path.as_deref(),
            profiling.full_content,
        );
        handler = handler.with_backend_llm(bt_llm);
    }

    let conversation_service = Arc::new(handler);
    let settings_service =
        Arc::new(DaemonSettingsService::new(config_path.clone()).with_mcp_control(mcp_handle));
    let dbus_service_name = std::env::var("DESKTOP_ASSISTANT_DBUS_SERVICE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "org.desktopAssistant".to_string());
    let dbus_required = env_bool("DESKTOP_ASSISTANT_DBUS_REQUIRED", true);
    tracing::info!("D-Bus well-known name={dbus_service_name}");
    tracing::info!("D-Bus required={dbus_required}");

    // Set up D-Bus connection (required by default; optional in headless/container mode).
    let dbus_connection = match zbus::connection::Builder::session() {
        Ok(builder) => match builder
            .name(dbus_service_name.as_str())
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Conversations",
                    DbusConversationAdapter::new(Arc::clone(&conversation_service)),
                )
            })
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Settings",
                    DbusSettingsAdapter::new(Arc::clone(&settings_service)),
                )
            }) {
            Ok(builder) => match builder.build().await {
                Ok(connection) => {
                    if let Some(unique_name) = connection.unique_name() {
                        tracing::info!("D-Bus service registered at {}", unique_name);
                    } else {
                        tracing::info!("D-Bus service registered");
                    }
                    Some(connection)
                }
                Err(error) => {
                    if dbus_required {
                        return Err(error.into());
                    }
                    tracing::warn!(
                        "D-Bus unavailable; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
                    );
                    None
                }
            },
            Err(error) => {
                if dbus_required {
                    return Err(error.into());
                }
                tracing::warn!(
                    "failed to configure D-Bus interface; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
                );
                None
            }
        },
        Err(error) => {
            if dbus_required {
                return Err(error.into());
            }
            tracing::warn!(
                "failed to connect to session D-Bus; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
            );
            None
        }
    };

    // WebSocket API (remote-friendly). Defaults to localhost only.
    let ws_bind = std::env::var("DESKTOP_ASSISTANT_WS_BIND")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "127.0.0.1:11339".to_string());

    let ws_addr: std::net::SocketAddr = ws_bind
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid DESKTOP_ASSISTANT_WS_BIND '{ws_bind}': {e}"))?;

    let api_handler = Arc::new(DefaultAssistantApiHandler::new(
        Arc::new(Assistant),
        Arc::clone(&conversation_service),
        Arc::clone(&settings_service),
    ));
    let ws_auth = Arc::new(WsSettingsAuth::new(Arc::clone(&settings_service)));
    let ws_login_service: Option<Arc<dyn ws::WsLoginService>> =
        resolve_ws_login_mode().map(|(username, mode)| {
            match &mode {
                WsLoginMode::StaticPassword(_) => {
                    tracing::info!("Web login enabled (env-password mode) for username={username}");
                }
                WsLoginMode::SystemPassword => {
                    tracing::info!(
                        "Web login enabled (local system-password mode) for username={username}"
                    );
                }
            }

            Arc::new(WsBasicLogin::new(
                Arc::clone(&settings_service),
                username,
                mode,
            )) as Arc<dyn ws::WsLoginService>
        });
    if ws_login_service.is_none() {
        tracing::warn!(
            "Web login disabled: set DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD or enable local auth via DESKTOP_ASSISTANT_WS_LOGIN_LOCAL_SYSTEM_AUTH=true"
        );
    }

    let (ws_shutdown_tx, ws_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let ws_task = tokio::spawn(async move {
        tracing::info!("WebSocket listening on {ws_addr} (/ws)");
        if let Err(e) = ws::serve_with_shutdown_and_login(
            api_handler,
            ws_auth,
            ws_login_service,
            ws_addr,
            async {
                let _ = ws_shutdown_rx.await;
            },
        )
        .await
        {
            tracing::error!("WebSocket server error: {e}");
        }
    });

    // Run until stopped.
    shutdown_signal().await;
    tracing::info!("shutdown signal received; stopping services");

    let _ = backfill_shutdown_tx.send(());
    if let Some(task) = backfill_task
        && let Err(e) = task.await
    {
        tracing::warn!("backfill task join error during shutdown: {e}");
    }

    let _ = dreaming_shutdown_tx.send(());
    if let Some(task) = dreaming_task
        && let Err(e) = task.await
    {
        tracing::warn!("dreaming task join error during shutdown: {e}");
    }

    let _ = ws_shutdown_tx.send(());
    if let Err(e) = ws_task.await {
        tracing::warn!("WebSocket task join error during shutdown: {e}");
    }

    drop(dbus_connection);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{WsLoginMode, resolve_ws_login_mode_decision};

    #[test]
    fn static_password_mode_uses_configured_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            Some("api-user".to_string()),
            Some("secret".to_string()),
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::StaticPassword(password))) => {
                assert_eq!(username, "api-user");
                assert_eq!(password, "secret");
            }
            _ => panic!("expected static password mode"),
        }
    }

    #[test]
    fn static_password_mode_defaults_to_current_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            None,
            Some("secret".to_string()),
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::StaticPassword(password))) => {
                assert_eq!(username, "local-user");
                assert_eq!(password, "secret");
            }
            _ => panic!("expected static password mode"),
        }
    }

    #[test]
    fn system_password_mode_ignores_configured_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            Some("other-user".to_string()),
            None,
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::SystemPassword)) => {
                assert_eq!(username, "local-user");
            }
            _ => panic!("expected system password mode"),
        }
    }

    #[test]
    fn login_mode_disabled_in_container_without_static_password() {
        let result =
            resolve_ws_login_mode_decision("local-user".to_string(), None, None, true, true);
        assert!(result.is_none());
    }

    #[test]
    fn login_mode_disabled_when_local_system_auth_is_off() {
        let result =
            resolve_ws_login_mode_decision("local-user".to_string(), None, None, false, false);
        assert!(result.is_none());
    }
}
