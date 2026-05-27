//! Transport-agnostic dispatcher for the assistant API.
//!
//! Stub — implementation lands in the next commit. The signature below
//! is what the failing tests in `tests/dispatcher.rs` and the failing
//! UDS tests in `crates/uds-interface/tests/uds.rs` exercise.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::AssistantApiHandler;
use futures::sink::Sink;
use futures::stream::Stream;

pub use api::{WsFrame, WsRequest};

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
}

impl AuthContext {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
        }
    }

    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_string(),
        }
    }
}

pub async fn dispatch_loop<R, W>(
    _handler: Arc<dyn AssistantApiHandler>,
    _auth: AuthContext,
    _inbound: R,
    _outbound: W,
) where
    R: Stream<Item = anyhow::Result<WsRequest>> + Unpin,
    W: Sink<WsFrame> + Unpin + Send + 'static,
    W::Error: std::fmt::Debug + Send,
{
    // unimplemented; tests should fail until the implementation lands.
}
