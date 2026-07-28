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
/// Delegates to [`classify`], the single exhaustive match.
///
/// The split is *write* versus *read*, not command-name prefix:
///
/// - Reading connectors, models and purposes stays tenant, because those feed
///   the ordinary model picker while staying global and operator-owned
///   (design decisions 8 and 9). Writing them is administration.
/// - Reading operator settings stays tenant because the credentials they used
///   to carry are now redacted on the way out (#727) and the same values reach
///   every client through `GetConfig`. Tightening the reads needs `Config`
///   itself partitioned, which is the read half of design decision 6 (#973).
/// - Conversation, knowledge, scratchpad and background-task commands are
///   tenant work **only where the storage path actually carries a `user_id`
///   predicate**. Verify that against the SQL before classifying a command
///   `Tenant`; do not infer it from the family it belongs to, or from a comment
///   saying the family is user-scoped. `StartKnowledgeMaintenance` sat in the
///   knowledge family and reached every tenant's rows, which is exactly how
///   this rule was learned.
pub fn required_capability(cmd: &api::Command) -> Capability {
    classify(cmd).1
}

/// The command's stable wire name (`"set_api_key"`, `"ping"`, ...), for a
/// refusal message.
///
/// Comes out of the same match as the capability, so the two cannot disagree
/// and a new variant must supply both. Reading the name off
/// `serde_json::to_value(cmd)` would be self-maintaining, but it materialises a
/// full copy of the command on the heap - including a plaintext `SetApiKey`
/// key - purely to read a tag. `commands_report_their_wire_name` pins each name
/// to the serde tag, so the hand-written strings cannot drift.
pub fn command_name(cmd: &api::Command) -> &'static str {
    classify(cmd).0
}

/// The wire name and the required capability for every command.
///
/// The match has **no wildcard arm** on purpose (see the module docs): a new
/// `Command` variant must be classified here explicitly, and the compiler makes
/// that mandatory rather than optional.
fn classify(cmd: &api::Command) -> (&'static str, Capability) {
    use Capability::{Admin, Tenant};
    match cmd {
        // Liveness and read-only aggregates.
        api::Command::Ping => ("ping", Tenant),
        api::Command::GetStatus => ("get_status", Tenant),
        api::Command::GetConfig => ("get_config", Tenant),

        // The mixed command; see `config_changes_capability`.
        api::Command::SetConfig { changes } => ("set_config", config_changes_capability(changes)),

        // Conversations: the caller's own, scoped by `with_user_id`.
        api::Command::CreateConversation { .. } => ("create_conversation", Tenant),
        api::Command::ListConversations { .. } => ("list_conversations", Tenant),
        api::Command::GetConversation { .. } => ("get_conversation", Tenant),
        api::Command::GetMessages { .. } => ("get_messages", Tenant),
        api::Command::DeleteConversation { .. } => ("delete_conversation", Tenant),
        api::Command::RenameConversation { .. } => ("rename_conversation", Tenant),
        api::Command::ArchiveConversation { .. } => ("archive_conversation", Tenant),
        api::Command::UnarchiveConversation { .. } => ("unarchive_conversation", Tenant),
        api::Command::ClearAllHistory => ("clear_all_history", Tenant),
        api::Command::SendMessage { .. } => ("send_message", Tenant),
        api::Command::SetConversationPersonality { .. } => ("set_conversation_personality", Tenant),

        // Provider credentials and the embedding backend: operator config.
        api::Command::SetApiKey { .. } => ("set_api_key", Admin),
        api::Command::GetEmbeddingsSettings => ("get_embeddings_settings", Tenant),
        api::Command::SetEmbeddingsSettings { .. } => ("set_embeddings_settings", Admin),
        api::Command::GetConnectorDefaults { .. } => ("get_connector_defaults", Tenant),

        // Git persistence, the database, background work, and the auth posture.
        api::Command::GetPersistenceSettings => ("get_persistence_settings", Tenant),
        api::Command::SetPersistenceSettings { .. } => ("set_persistence_settings", Admin),
        api::Command::GetDatabaseSettings => ("get_database_settings", Tenant),
        api::Command::SetDatabaseSettings { .. } => ("set_database_settings", Admin),
        api::Command::GetBackendTasksSettings => ("get_backend_tasks_settings", Tenant),
        api::Command::SetBackendTasksSettings { .. } => ("set_backend_tasks_settings", Admin),
        api::Command::GetWsAuthSettings => ("get_ws_auth_settings", Tenant),
        api::Command::SetWsAuthSettings { .. } => ("set_ws_auth_settings", Admin),

        // Connections and purposes: read for the model picker, write for the
        // operator (design decisions 8 and 9).
        api::Command::ListConnections => ("list_connections", Tenant),
        api::Command::CreateConnection { .. } => ("create_connection", Admin),
        api::Command::UpdateConnection { .. } => ("update_connection", Admin),
        api::Command::DeleteConnection { .. } => ("delete_connection", Admin),
        api::Command::SetConnectionSecret { .. } => ("set_connection_secret", Admin),
        api::Command::ListAvailableModels { .. } => ("list_available_models", Tenant),
        api::Command::GetPurposes => ("get_purposes", Tenant),
        api::Command::SetPurpose { .. } => ("set_purpose", Admin),

        // Knowledge base and cost reporting: the caller's own rows.
        api::Command::GetToolUsage { .. } => ("get_tool_usage", Tenant),
        api::Command::ListKnowledgeEntries { .. } => ("list_knowledge_entries", Tenant),
        api::Command::GetKnowledgeEntry { .. } => ("get_knowledge_entry", Tenant),
        api::Command::SearchKnowledgeEntries { .. } => ("search_knowledge_entries", Tenant),
        api::Command::CreateKnowledgeEntry { .. } => ("create_knowledge_entry", Tenant),
        api::Command::UpdateKnowledgeEntry { .. } => ("update_knowledge_entry", Tenant),
        api::Command::DeleteKnowledgeEntry { .. } => ("delete_knowledge_entry", Tenant),
        api::Command::GetKnowledgeTrashCount => ("get_knowledge_trash_count", Tenant),
        api::Command::EmptyKnowledgeTrash => ("empty_knowledge_trash", Tenant),
        // NOT tenant work, despite sitting in the knowledge family. Every arm
        // reaches past the caller's own rows:
        // `RecalculateEmbeddings` runs an UPDATE over `knowledge_base` with no
        // `user_id` predicate (`crates/storage/src/embedding_backfill.rs`,
        // `invalidate_all_knowledge_embeddings`), nulling every tenant's
        // vectors and re-embedding the instance through the operator's
        // provider; `Consolidation` loops over every user with active entries;
        // and `Extraction`'s archival phase widens to all users when the
        // caller's scope is the `"default"` sentinel. It backs an
        // operator-facing maintenance button, and no ordinary client calls it.
        api::Command::StartKnowledgeMaintenance { .. } => ("start_knowledge_maintenance", Admin),

        // MCP: every write makes the daemon spawn or reconfigure a child
        // process, with arguments and an environment the caller supplies.
        api::Command::ListMcpServers => ("list_mcp_servers", Tenant),
        api::Command::AddMcpServer { .. } => ("add_mcp_server", Admin),
        api::Command::RemoveMcpServer { .. } => ("remove_mcp_server", Admin),
        api::Command::SetMcpServerEnabled { .. } => ("set_mcp_server_enabled", Admin),
        api::Command::McpServerAction { action, .. } => {
            ("mcp_server_action", mcp_action_capability(action))
        }
        api::Command::UpsertMcpServer { .. } => ("upsert_mcp_server", Admin),
        api::Command::SetMcpSecret { .. } => ("set_mcp_secret", Admin),

        // Outbound OAuth service accounts are instance-wide, like connections.
        api::Command::ListServiceAccounts => ("list_service_accounts", Tenant),
        api::Command::UpsertServiceAccount { .. } => ("upsert_service_account", Admin),
        api::Command::RemoveServiceAccount { .. } => ("remove_service_account", Admin),

        // Background tasks and subscriptions: per-user, per-connection.
        api::Command::ListBackgroundTasks { .. } => ("list_background_tasks", Tenant),
        api::Command::GetBackgroundTask { .. } => ("get_background_task", Tenant),
        api::Command::CancelBackgroundTask { .. } => ("cancel_background_task", Tenant),
        api::Command::GetBackgroundTaskLogs { .. } => ("get_background_task_logs", Tenant),
        api::Command::SubscribeBackgroundTasks => ("subscribe_background_tasks", Tenant),
        api::Command::UnsubscribeBackgroundTasks => ("unsubscribe_background_tasks", Tenant),
        api::Command::SubscribeConversations { .. } => ("subscribe_conversations", Tenant),
        api::Command::SpawnStandaloneAgent { .. } => ("spawn_standalone_agent", Tenant),

        // Conversation scratchpad: the caller's own notes.
        api::Command::GetConversationScratchpad { .. } => ("get_conversation_scratchpad", Tenant),
        api::Command::SetScratchpadNote { .. } => ("set_scratchpad_note", Tenant),
        api::Command::DeleteScratchpadNotes { .. } => ("delete_scratchpad_notes", Tenant),

        // Client-side tool execution runs on the caller's own machine.
        api::Command::RegisterClientTools { .. } => ("register_client_tools", Tenant),
        api::Command::ClientToolResult { .. } => ("client_tool_result", Tenant),
    }
}

/// The capability a `SetConfig` requires, decided by *what it changes*.
///
/// Every field this command carries today writes **daemon-global** state, so
/// any non-empty change set needs the administrator capability.
///
/// The personality traits look like a personal preference and are not one, yet.
/// `SetConfig` routes them to `SettingsService::set_personality_settings`, which
/// reaches `RegistryHandle::set_personality` -> `mutate_config`: that serializes
/// the whole config over the operator's `daemon.toml` and rebuilds the
/// connection registry. `resolve_personality` then returns that one global value
/// for every conversation with no per-conversation override, so one tenant
/// writing a trait changes every other tenant's assistant, destroys the
/// operator's hand-edits and comments in `daemon.toml`, and can force a registry
/// rebuild at will.
///
/// A tenant is not left without a lever: `SetConversationPersonality` is
/// genuinely per-user (stored on the conversation, resolved against the global
/// on each send) and stays [`Capability::Tenant`]. The global value is the
/// instance default, which is operator configuration under design decision 6.
///
/// **This is staging, not a judgement that personality is an operator concern.**
/// Design decision 9 names personality and speech mode as good candidates for
/// the per-user override layer in decision 1, precisely because they are cheap
/// and personal. That layer does not exist yet, which is the only reason the
/// traits are gated here. It is tracked as #986; when it lands, a tenant's write
/// targets their own row and this arm moves to the tenant side, while writing
/// the instance default stays admin.
///
/// An empty change set writes nothing, so it costs nothing.
///
/// The destructuring below names every field and uses no `..` rest pattern, so a
/// new `ConfigChanges` field fails to compile here rather than silently
/// inheriting a classification that was never considered for it.
fn config_changes_capability(changes: &api::ConfigChanges) -> Capability {
    let api::ConfigChanges {
        embeddings_connector,
        embeddings_model,
        embeddings_base_url,
        persistence_enabled,
        persistence_remote_url,
        persistence_remote_name,
        persistence_push_on_update,
        personality_professionalism,
        personality_warmth,
        personality_directness,
        personality_enthusiasm,
        personality_humor,
        personality_sarcasm,
        personality_pretentiousness,
    } = changes;

    let touches_service_config = embeddings_connector.is_some()
        || embeddings_model.is_some()
        || embeddings_base_url.is_some()
        || persistence_enabled.is_some()
        || persistence_remote_url.is_some()
        || persistence_remote_name.is_some()
        || persistence_push_on_update.is_some()
        || personality_professionalism.is_some()
        || personality_warmth.is_some()
        || personality_directness.is_some()
        || personality_enthusiasm.is_some()
        || personality_humor.is_some()
        || personality_sarcasm.is_some()
        || personality_pretentiousness.is_some();

    if touches_service_config {
        Capability::Admin
    } else {
        Capability::Tenant
    }
}

/// The capability an `McpServerAction` requires, decided by *what it does*.
///
/// The same write-versus-read split `SetConfig` gets. `"status"` is a pure read
/// returning exactly what `ListMcpServers` returns, so gating it would be
/// inconsistent with leaving that command open to a tenant. `"start"`,
/// `"stop"` and `"restart"` spawn or kill a child process, which is
/// administration.
///
/// An unrecognized action is Admin, not Tenant. The daemon rejects it anyway,
/// but the gate must not be the place that decides an unknown verb is
/// harmless - a verb added later is administration until someone says
/// otherwise.
fn mcp_action_capability(action: &str) -> Capability {
    match action {
        "status" => Capability::Tenant,
        _ => Capability::Admin,
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
            Capability::Tenant.strongest(Capability::Admin),
            Capability::Admin,
            "an allowlist grant must survive a tenant peer-uid verdict"
        );
        assert_eq!(
            Capability::Tenant.strongest(Capability::Tenant),
            Capability::Tenant
        );
        // Fail closed: a level this build does not know never beats a real one.
        assert_eq!(
            Capability::Admin.strongest(Capability::Other("owner".to_string())),
            Capability::Admin,
            "an unrecognized level must not outrank a real grant"
        );
    }

    #[test]
    fn capability_labels_are_stable() {
        assert_eq!(Capability::Tenant.label(), "tenant");
        assert_eq!(Capability::Admin.label(), "administrator");
    }
}
