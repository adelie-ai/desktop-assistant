//! The daemon's authorization tier: who may change how the service runs.
//!
//! Authentication answers "which subject is this connection?". This module
//! answers the next question, "what is that subject allowed to do?", and the
//! answer has exactly two levels.
//!
//! A **tenant** owns their own conversations, knowledge, scratchpads,
//! background tasks and preferences. An **administrator** additionally owns the
//! service: provider credentials, connectors and purposes, the database, the
//! WebSocket auth posture, and which child processes the daemon spawns for MCP.
//! `docs/design/multi-tenancy-boundary.md` decision 6 draws that line; decision
//! 7 sets the bar at one organization rather than hostile isolation.
//!
//! ## Two ways to be an administrator, both already at the transport
//!
//! 1. **Local is admin by construction.** On a Unix socket the kernel attests
//!    the peer uid, which the peer cannot forge. A peer uid equal to the
//!    daemon's own uid is the person who runs the daemon
//!    ([`capability_for_local_peer`]). The single-user desktop therefore needs
//!    no configuration at all, which the design record makes a hard constraint.
//! 2. **Remote is admin only by explicit allowlist** ([`AdminSubjects`], from
//!    `[authz] admin_subjects` in `daemon.toml`). The list defaults to empty and
//!    is file-only: no command writes it, because a tenant must not be able to
//!    grant themselves the capability they are being denied.
//!
//! ## One gate
//!
//! [`required_capability`] is the single source of truth for what a command
//! costs, and `dispatch_loop` is the only place that consults it. Per-service or
//! per-transport checks are the drift this design exists to prevent.
//!
//! The match in [`required_capability`] has **no wildcard arm**. That is
//! deliberate and load-bearing: a `Command` variant added later fails to
//! compile here instead of silently defaulting to permitted. Do not "fix" such
//! a build error with a `_ =>` arm.

use std::collections::BTreeSet;

use desktop_assistant_api_model as api;

/// Leading text of every authorization refusal, so a client can tell a refusal
/// from an operational error without parsing prose.
pub const REFUSAL_PREFIX: &str = "not authorized:";

/// The two capability levels. Re-exported from `api-model`, which owns it
/// because the daemon reports the caller's capability back on the wire
/// ([`api::Config::caller_capability`]); the *policy* below is this crate's.
pub use api::Capability;

/// The operator's allowlist of subjects that hold [`Capability::Admin`].
///
/// Built once at startup from `[authz] admin_subjects`. Empty by default, so a
/// daemon that was never configured grants administration to nobody over a
/// remote transport. Blank and whitespace-only entries are dropped, so a stray
/// line in `daemon.toml` cannot admit the empty subject.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdminSubjects(BTreeSet<String>);

impl AdminSubjects {
    /// Build the allowlist from configured subjects, dropping blanks and
    /// trimming surrounding whitespace.
    pub fn new<I, S>(subjects: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self(
            subjects
                .into_iter()
                .map(|s| s.as_ref().trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        )
    }

    /// Whether the allowlist names nobody - the default, and the shape a
    /// single-user desktop always has.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The capability this allowlist grants `subject`. Exact match on the
    /// authenticated subject (the JWT `sub` remotely, the peer's login name
    /// locally); an unlisted or blank subject is a tenant.
    pub fn capability_for(&self, subject: &str) -> Capability {
        let subject = subject.trim();
        if !subject.is_empty() && self.0.contains(subject) {
            Capability::Admin
        } else {
            Capability::Tenant
        }
    }
}

/// The local-peer grant: a Unix-socket peer whose kernel-attested uid equals the
/// daemon's own uid runs the daemon, so it administers the daemon.
///
/// `SO_PEERCRED` is not forgeable by the peer, so this needs no configuration
/// and no secret. Every other uid is a tenant until the allowlist says
/// otherwise.
pub fn capability_for_local_peer(peer_uid: u32, daemon_uid: u32) -> Capability {
    if peer_uid == daemon_uid {
        Capability::Admin
    } else {
        Capability::Tenant
    }
}

/// The capability a command requires.
///
/// The match below has no wildcard arm on purpose (see the module docs): a new
/// `Command` variant must be classified here explicitly, and the compiler makes
/// that mandatory rather than optional.
///
/// The split is *write* versus *read*, not command-name prefix:
///
/// - Reading connectors, models and purposes stays tenant, because those feed
///   the ordinary model picker while staying global and operator-owned
///   (design decisions 8 and 9). Writing them is administration.
/// - Reading operator settings stays tenant because the credentials they used
///   to carry are now redacted on the way out (#727) and the same values reach
///   every client through `GetConfig`. Tightening the reads needs `Config`
///   itself partitioned, which is decision 1's per-user override layer (#973).
/// - Conversation, knowledge, scratchpad and background-task commands are all
///   scoped to the calling user by `with_user_id`, so they are tenant work even
///   when they delete data. `ClearAllHistory` clears the caller's own
///   conversations, not the instance's.
pub fn required_capability(cmd: &api::Command) -> Capability {
    use Capability::{Admin, Tenant};
    match cmd {
        // Liveness and read-only aggregates.
        api::Command::Ping => Tenant,
        api::Command::GetStatus => Tenant,
        api::Command::GetConfig => Tenant,

        // The mixed command; see `config_changes_capability`.
        api::Command::SetConfig { changes } => config_changes_capability(changes),

        // Conversations: the caller's own, scoped by `with_user_id`.
        api::Command::CreateConversation { .. } => Tenant,
        api::Command::ListConversations { .. } => Tenant,
        api::Command::GetConversation { .. } => Tenant,
        api::Command::GetMessages { .. } => Tenant,
        api::Command::DeleteConversation { .. } => Tenant,
        api::Command::RenameConversation { .. } => Tenant,
        api::Command::ArchiveConversation { .. } => Tenant,
        api::Command::UnarchiveConversation { .. } => Tenant,
        api::Command::ClearAllHistory => Tenant,
        api::Command::SendMessage { .. } => Tenant,
        api::Command::SetConversationPersonality { .. } => Tenant,

        // Provider credentials and the embedding backend: operator config.
        api::Command::SetApiKey { .. } => Admin,
        api::Command::GetEmbeddingsSettings => Tenant,
        api::Command::SetEmbeddingsSettings { .. } => Admin,
        api::Command::GetConnectorDefaults { .. } => Tenant,

        // Git persistence, the database, background work, and the auth posture.
        api::Command::GetPersistenceSettings => Tenant,
        api::Command::SetPersistenceSettings { .. } => Admin,
        api::Command::GetDatabaseSettings => Tenant,
        api::Command::SetDatabaseSettings { .. } => Admin,
        api::Command::GetBackendTasksSettings => Tenant,
        api::Command::SetBackendTasksSettings { .. } => Admin,
        api::Command::GetWsAuthSettings => Tenant,
        api::Command::SetWsAuthSettings { .. } => Admin,

        // Connections and purposes: read for the model picker, write for the
        // operator (design decisions 8 and 9).
        api::Command::ListConnections => Tenant,
        api::Command::CreateConnection { .. } => Admin,
        api::Command::UpdateConnection { .. } => Admin,
        api::Command::DeleteConnection { .. } => Admin,
        api::Command::SetConnectionSecret { .. } => Admin,
        api::Command::ListAvailableModels { .. } => Tenant,
        api::Command::GetPurposes => Tenant,
        api::Command::SetPurpose { .. } => Admin,

        // Knowledge base and cost reporting: the caller's own rows.
        api::Command::GetToolUsage { .. } => Tenant,
        api::Command::ListKnowledgeEntries { .. } => Tenant,
        api::Command::GetKnowledgeEntry { .. } => Tenant,
        api::Command::SearchKnowledgeEntries { .. } => Tenant,
        api::Command::CreateKnowledgeEntry { .. } => Tenant,
        api::Command::UpdateKnowledgeEntry { .. } => Tenant,
        api::Command::DeleteKnowledgeEntry { .. } => Tenant,
        api::Command::GetKnowledgeTrashCount => Tenant,
        api::Command::EmptyKnowledgeTrash => Tenant,
        api::Command::StartKnowledgeMaintenance { .. } => Tenant,

        // MCP: every write makes the daemon spawn or reconfigure a child
        // process, with arguments and an environment the caller supplies.
        api::Command::ListMcpServers => Tenant,
        api::Command::AddMcpServer { .. } => Admin,
        api::Command::RemoveMcpServer { .. } => Admin,
        api::Command::SetMcpServerEnabled { .. } => Admin,
        api::Command::McpServerAction { .. } => Admin,
        api::Command::UpsertMcpServer { .. } => Admin,
        api::Command::SetMcpSecret { .. } => Admin,

        // Outbound OAuth service accounts are instance-wide, like connections.
        api::Command::ListServiceAccounts => Tenant,
        api::Command::UpsertServiceAccount { .. } => Admin,
        api::Command::RemoveServiceAccount { .. } => Admin,

        // Background tasks and subscriptions: per-user, per-connection.
        api::Command::ListBackgroundTasks { .. } => Tenant,
        api::Command::GetBackgroundTask { .. } => Tenant,
        api::Command::CancelBackgroundTask { .. } => Tenant,
        api::Command::GetBackgroundTaskLogs { .. } => Tenant,
        api::Command::SubscribeBackgroundTasks => Tenant,
        api::Command::UnsubscribeBackgroundTasks => Tenant,
        api::Command::SubscribeConversations { .. } => Tenant,
        api::Command::SpawnStandaloneAgent { .. } => Tenant,

        // Conversation scratchpad: the caller's own notes.
        api::Command::GetConversationScratchpad { .. } => Tenant,
        api::Command::SetScratchpadNote { .. } => Tenant,
        api::Command::DeleteScratchpadNotes { .. } => Tenant,

        // Client-side tool execution runs on the caller's own machine.
        api::Command::RegisterClientTools { .. } => Tenant,
        api::Command::ClientToolResult { .. } => Tenant,
    }
}

/// The capability a `SetConfig` requires, decided by *what it changes*.
///
/// `SetConfig` is the one mixed command: it carries daemon-global knobs
/// (embeddings backend, git persistence) alongside the seven personality traits
/// that every ordinary client writes. Splitting on the command name alone would
/// either lock a tenant out of their own personality settings or leave the
/// global knobs open, so the payload decides.
///
/// The destructuring below names every field and uses no `..` rest pattern, so a
/// new `ConfigChanges` field fails to compile here rather than defaulting to
/// tenant-writable.
fn config_changes_capability(changes: &api::ConfigChanges) -> Capability {
    let api::ConfigChanges {
        embeddings_connector,
        embeddings_model,
        embeddings_base_url,
        persistence_enabled,
        persistence_remote_url,
        persistence_remote_name,
        persistence_push_on_update,
        // Personality is per-person preference, not service configuration.
        personality_professionalism: _,
        personality_warmth: _,
        personality_directness: _,
        personality_enthusiasm: _,
        personality_humor: _,
        personality_sarcasm: _,
        personality_pretentiousness: _,
    } = changes;

    let touches_service_config = embeddings_connector.is_some()
        || embeddings_model.is_some()
        || embeddings_base_url.is_some()
        || persistence_enabled.is_some()
        || persistence_remote_url.is_some()
        || persistence_remote_name.is_some()
        || persistence_push_on_update.is_some();

    if touches_service_config {
        Capability::Admin
    } else {
        Capability::Tenant
    }
}

/// Tell the caller what it is allowed to do, on every reply that carries the
/// aggregate config (#728).
///
/// The dispatcher is the one layer that knows a connection's capability, so it
/// stamps the answer on the way out rather than threading authorization down
/// into the settings service. A client reads it to render the operator-owned
/// sections as unavailable up front, instead of discovering the boundary when a
/// write is refused.
pub(crate) fn stamp_caller_capability(result: &mut api::CommandResult, held: Capability) {
    if let api::CommandResult::Config(config) = result {
        config.caller_capability = Some(held);
    }
}

/// The [`api::Config`] to put in a `ConfigChanged` event, carrying the same
/// answer the matching result frame does.
pub(crate) fn with_caller_capability(mut config: api::Config, held: Capability) -> api::Config {
    config.caller_capability = Some(held);
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_is_the_default_capability() {
        assert_eq!(Capability::default(), Capability::Tenant);
    }

    #[test]
    fn merging_two_grants_takes_the_higher_one() {
        assert_eq!(
            Capability::Tenant.max(Capability::Admin),
            Capability::Admin,
            "an allowlist grant must survive a tenant peer-uid verdict"
        );
        assert_eq!(
            Capability::Tenant.max(Capability::Tenant),
            Capability::Tenant
        );
    }

    #[test]
    fn capability_labels_are_stable() {
        assert_eq!(Capability::Tenant.label(), "tenant");
        assert_eq!(Capability::Admin.label(), "administrator");
    }
}
