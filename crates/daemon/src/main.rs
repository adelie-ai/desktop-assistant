use std::sync::Arc;

use anyhow::Result;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::embedding::{EmbedFn, EmbeddingClient};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
use tracing_subscriber::EnvFilter;

mod app;
mod config;
mod settings_service;
mod store;

use desktop_assistant_core::service::ConversationHandler;
use desktop_assistant_dbus::conversation::DbusConversationAdapter;
use desktop_assistant_dbus::settings::DbusSettingsAdapter;
use desktop_assistant_mcp_client::config as mcp_config;
use desktop_assistant_mcp_client::executor::{BuiltinPersistenceConfig, McpToolExecutor};
use settings_service::DaemonSettingsService;
use store::PersistentConversationStore;

/// Enum wrapper to dispatch between LLM backends at runtime.
///
/// `LlmClient` uses `impl Future` returns, so it isn't dyn-compatible.
/// This enum lets `ConversationHandler` stay monomorphic while supporting
/// multiple backends.
enum AnyLlmClient {
    Anthropic(desktop_assistant_llm_anthropic::AnthropicClient),
    OpenAi(desktop_assistant_llm_openai::OpenAiClient),
    Ollama(desktop_assistant_llm_ollama::OllamaClient),
}

/// Enum wrapper to dispatch between embedding backends at runtime.
///
/// Mirrors `AnyLlmClient` but for the `EmbeddingClient` trait.
/// `Unavailable` is used when the resolved connector doesn't support embeddings (e.g. Anthropic).
enum AnyEmbeddingClient {
    OpenAi(desktop_assistant_llm_openai::OpenAiClient),
    Ollama(desktop_assistant_llm_ollama::OllamaClient),
    Unavailable,
}

impl EmbeddingClient for AnyEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        match self {
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
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Anthropic(c) => c.stream_completion(messages, tools, on_chunk).await,
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
    tracing::info!("D-Bus well-known name={dbus_service_name}");

    // Set up D-Bus connection
    let connection = zbus::connection::Builder::session()?
        .name(dbus_service_name.as_str())?
        .serve_at(
            "/org/desktopAssistant/Conversations",
            DbusConversationAdapter::new(Arc::clone(&conversation_service)),
        )?
        .serve_at(
            "/org/desktopAssistant/Settings",
            DbusSettingsAdapter::new(Arc::clone(&settings_service)),
        )?
        .build()
        .await?;

    tracing::info!(
        "D-Bus service registered at {}",
        connection.unique_name().unwrap()
    );

    // Run until stopped
    std::future::pending::<()>().await;

    Ok(())
}
