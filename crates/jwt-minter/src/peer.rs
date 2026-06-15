//! Peer-credential lookup.
//!
//! The implementation was relocated to the shared `desktop-assistant-peer-cred`
//! crate (issue #407) so the UDS server can authenticate connections by
//! peer-cred and this minter can be retired. This module re-exports it so the
//! minter's existing `crate::peer::…` paths keep working unchanged until the
//! crate is deleted.

pub use desktop_assistant_peer_cred::{
    PeerIdentity, current_uid, extract_peer_identity, username_for_uid,
};
