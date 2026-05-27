//! UDS transport client for the daemon's UDS frontend. STUB — bodies
//! arrive in the implementation commit. Type signatures match the
//! final shape so tests compile against this commit and exercise the
//! real impl in the next.

use std::path::{Path, PathBuf};
use std::time::Duration;

use desktop_assistant_api_model as api;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::broadcast;

pub use api::{WsFrame, WsRequest};

#[derive(Debug, thiserror::Error)]
pub enum BridgeTransportError {
    #[error("transport error: {0}")]
    Io(#[from] std::io::Error),
    #[error("daemon error: {0}")]
    Daemon(String),
    #[error("daemon connection closed while awaiting reply for {request_id}")]
    Disconnected { request_id: String },
    #[error("daemon did not reply for {request_id} within {timeout:?}")]
    Timeout {
        request_id: String,
        timeout: Duration,
    },
    #[error("malformed daemon frame: {0}")]
    BadFrame(String),
}

#[async_trait::async_trait]
pub trait BridgeTransport: Send + Sync {
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError>;

    fn subscribe_events(&self) -> broadcast::Receiver<api::Event>;
}

#[derive(Debug, Clone)]
pub struct UdsBridgeConfig {
    pub socket_path: PathBuf,
    pub request_timeout: Duration,
    pub event_buffer: usize,
}

impl UdsBridgeConfig {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            request_timeout: Duration::from_secs(30),
            event_buffer: 256,
        }
    }
}

pub fn default_daemon_socket_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(|p| PathBuf::from(p).join("adelie").join("sock"))
}

/// UDS-backed implementation of [`BridgeTransport`]. STUB.
pub struct UdsBridgeTransport {
    _events_tx: broadcast::Sender<api::Event>,
}

impl UdsBridgeTransport {
    pub async fn connect(
        _config: UdsBridgeConfig,
        _jwt: &str,
    ) -> Result<Self, BridgeTransportError> {
        Err(BridgeTransportError::Daemon(
            "UdsBridgeTransport::connect not implemented yet".to_string(),
        ))
    }

    pub async fn connect_on_stream(
        _stream: UnixStream,
        _jwt: &str,
        _event_buffer: usize,
        _request_timeout: Duration,
    ) -> Result<Self, BridgeTransportError> {
        Err(BridgeTransportError::Daemon(
            "UdsBridgeTransport::connect_on_stream not implemented yet".to_string(),
        ))
    }

    pub fn shutdown(&self) {}
}

#[async_trait::async_trait]
impl BridgeTransport for UdsBridgeTransport {
    async fn request(
        &self,
        _command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        Err(BridgeTransportError::Daemon(
            "UdsBridgeTransport::request not implemented yet".to_string(),
        ))
    }

    fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
        self._events_tx.subscribe()
    }
}

/// Read one length-prefixed frame; matches `uds-interface`'s framing.
pub async fn read_frame<R>(reader: &mut R) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    const MAX_FRAME_LEN: u32 = 4 * 1024 * 1024;
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    if len > 0 {
        reader.read_exact(&mut body).await?;
    }
    Ok(body)
}

pub async fn write_frame<W>(writer: &mut W, body: &[u8]) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let len = body.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(body).await?;
    writer.flush().await?;
    Ok(())
}

pub fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
