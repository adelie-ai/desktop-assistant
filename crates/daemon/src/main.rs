use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::embedding::{EmbedFn, EmbeddingClient};
use desktop_assistant_core::ports::inbound::SettingsService;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
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
use desktop_assistant_mcp_client::executor::{BuiltinPersistenceConfig, McpToolExecutor};
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

    let resolved_llm = config::resolve_llm_config(daemon_config.as_ref());
    tracing::info!(
        "LLM connector={}, model={}, base_url={}",
        resolved_llm.connector,
        resolved_llm.model,
        resolved_llm.base_url
    );

    let llm_connector = resolved_llm.connector.clone();
    let llm_api_key = resolved_llm.api_key.clone();

    let llm: AnyLlmClient = match resolved_llm.connector.as_str() {
        "ollama" => {
            tracing::info!("using Ollama LLM backend");
            AnyLlmClient::Ollama(desktop_assistant_llm_ollama::OllamaClient::new(
                resolved_llm.base_url,
                resolved_llm.model,
            ))
        }
        "anthropic" => {
            if resolved_llm.api_key.is_empty() {
                tracing::warn!(
                    "No API key resolved from configured secret backend or environment; LLM calls may fail"
                );
            }
            tracing::info!("using Anthropic LLM backend");
            AnyLlmClient::Anthropic(
                desktop_assistant_llm_anthropic::AnthropicClient::new(resolved_llm.api_key)
                    .with_model(resolved_llm.model)
                    .with_base_url(resolved_llm.base_url),
            )
        }
        "bedrock" | "aws-bedrock" => {
            tracing::info!("using AWS Bedrock LLM backend");
            AnyLlmClient::Bedrock(
                desktop_assistant_llm_bedrock::BedrockClient::new(resolved_llm.api_key)
                    .with_model(resolved_llm.model)
                    .with_base_url(resolved_llm.base_url),
            )
        }
        _ => {
            if resolved_llm.api_key.is_empty() {
                tracing::warn!(
                    "No API key resolved from configured secret backend or environment; LLM calls may fail"
                );
            }
            AnyLlmClient::OpenAi(
                desktop_assistant_llm_openai::OpenAiClient::new(resolved_llm.api_key)
                    .with_model(resolved_llm.model)
                    .with_base_url(resolved_llm.base_url),
            )
        }
    };

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

    // Load MCP server configuration
    let mcp_config_path = mcp_config::default_config_path();
    let mcp_configs = mcp_config::load_mcp_configs(&mcp_config_path).unwrap_or_else(|e| {
        tracing::warn!("failed to load MCP config: {e}");
        Vec::new()
    });

    // Build the MCP tool executor
    let resolved_persistence = config::resolve_persistence_config(daemon_config.as_ref());
    let builtin_persistence = if resolved_persistence.enabled {
        Some(BuiltinPersistenceConfig {
            enabled: true,
            remote_url: resolved_persistence.remote_url.clone(),
            remote_name: resolved_persistence.remote_name.clone(),
            push_on_update: resolved_persistence.push_on_update,
        })
    } else {
        None
    };

    if let Some(persistence) = &builtin_persistence {
        tracing::info!(
            remote_name = persistence.remote_name,
            push_on_update = persistence.push_on_update,
            has_remote = persistence.remote_url.is_some(),
            "built-in memory/preferences git persistence enabled"
        );
    }

    let tool_executor = if let Some(embed_fn) = embedding_fn {
        tracing::info!(
            "enabling built-in vector search for preferences/memory with model={}",
            resolved_emb.model
        );
        McpToolExecutor::new_with_embedding_and_persistence(
            mcp_configs,
            embed_fn,
            resolved_emb.model.clone(),
            builtin_persistence,
        )
    } else {
        tracing::info!("built-in vector search disabled (no embedding backend available)");
        McpToolExecutor::new_with_persistence(mcp_configs, builtin_persistence)
    };
    if let Err(e) = tool_executor.start().await {
        tracing::warn!("failed to start MCP servers: {e}");
    }

    let registered_tools = tool_executor.tools_by_service().await;
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

    // Build the conversation service with tool support
    let conversation_store = PersistentConversationStore::from_default_path()
        .map_err(|e| anyhow::anyhow!("failed to initialize persistent conversation store: {e}"))?;
    tracing::info!(
        "conversation persistence enabled at {}",
        store::default_conversation_store_path().display()
    );

    let conversation_service = Arc::new(ConversationHandler::with_tools(
        conversation_store,
        llm,
        tool_executor,
        Box::new(|| uuid::Uuid::new_v4().to_string()),
    ));
    let settings_service = Arc::new(DaemonSettingsService::new(config_path.clone()));
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

    let (ws_shutdown_tx, ws_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let ws_task = tokio::spawn(async move {
        tracing::info!("WebSocket listening on {ws_addr} (/ws)");
        if let Err(e) = ws::serve_with_shutdown(api_handler, ws_auth, ws_addr, async {
            let _ = ws_shutdown_rx.await;
        })
        .await
        {
            tracing::error!("WebSocket server error: {e}");
        }
    });

    // Run until stopped.
    shutdown_signal().await;
    tracing::info!("shutdown signal received; stopping services");

    let _ = ws_shutdown_tx.send(());
    if let Err(e) = ws_task.await {
        tracing::warn!("WebSocket task join error during shutdown: {e}");
    }

    drop(dbus_connection);

    Ok(())
}
