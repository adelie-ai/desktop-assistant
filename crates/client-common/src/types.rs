//! Client-facing conversation view types.
//!
//! `ChatMessage`, `ConversationDetail`, and `ConversationSummary` (and their
//! `From<wire>` projections) moved to `api-model` (`api::client`, #377) so the
//! wasm-targeting client cores can build/convert them without this crate's
//! native transport tail. Re-exported here so existing
//! `client_common::{ChatMessage, ConversationDetail, ConversationSummary}`
//! paths are unchanged.

pub use desktop_assistant_api_model::client::{
    ChatMessage, ConversationDetail, ConversationSummary, MessageKind,
};
