use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod app;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tracing::info!("desktop-assistant starting");

    // Future: wire up core service, D-Bus adapter, and run the event loop.

    Ok(())
}
