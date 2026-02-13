use std::sync::Arc;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod app;
mod store;

use desktop_assistant_core::service::ConversationHandler;
use desktop_assistant_dbus::conversation::DbusConversationAdapter;
use desktop_assistant_mcp_client::config;
use desktop_assistant_mcp_client::executor::McpToolExecutor;
use store::InMemoryConversationStore;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tracing::info!("desktop-assistant starting");

    // Build the LLM client from environment
    let llm = match desktop_assistant_llm_openai::OpenAiClient::from_env() {
        Ok(client) => {
            tracing::info!("OpenAI LLM client initialized");
            client
        }
        Err(e) => {
            tracing::warn!("OpenAI client not available: {e}. LLM features will fail at runtime.");
            // Create a client with a dummy key — calls will fail with auth errors
            desktop_assistant_llm_openai::OpenAiClient::new(String::new())
        }
    };

    // Load MCP server configuration
    let config_path = config::default_config_path();
    let mcp_configs = config::load_mcp_configs(&config_path).unwrap_or_else(|e| {
        tracing::warn!("failed to load MCP config: {e}");
        Vec::new()
    });

    // Build the MCP tool executor
    let tool_executor = McpToolExecutor::new(mcp_configs);
    if let Err(e) = tool_executor.start().await {
        tracing::warn!("failed to start MCP servers: {e}");
    }

    // Build the conversation service with tool support
    let conversation_service = Arc::new(ConversationHandler::with_tools(
        InMemoryConversationStore::new(),
        llm,
        tool_executor,
        Box::new(|| uuid::Uuid::new_v4().to_string()),
    ));

    // Set up D-Bus connection
    let connection = zbus::connection::Builder::session()?
        .name("org.desktopAssistant")?
        .serve_at(
            "/org/desktopAssistant/Conversations",
            DbusConversationAdapter::new(Arc::clone(&conversation_service)),
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
