//! Local JWT minter for desktop installs (issue #101).
//!
//! Listens on a Unix domain socket, authenticates the OS user via
//! `SO_PEERCRED`, and mints a short-lived HS256 JWT signed with the same
//! key the daemon validates against. This is *not* a real IdP — production
//! deployments use Cognito/Keycloak/etc. See
//! `docs/architecture-evolution.md` Phase 0 for context.
//!
//! Module layout:
//! - `config`: [`MintConfig`] — issuer/audience/TTL knobs + signing-key path.
//! - `peer`: `SO_PEERCRED` → UID → username via `getpwuid_r`.
//! - `group`: optional group-membership gate via `getgrouplist`.
//! - `request`: wire types and the pure `handle_request` function.
//! - `server`: the tokio loop that binds the UDS and dispatches requests.

pub mod config;
pub mod group;
pub mod peer;
pub mod request;
pub mod server;

pub use config::MintConfig;
pub use peer::PeerIdentity;
pub use request::{MintRequest, MintResponse, handle_request};
