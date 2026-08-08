//! Acceptance tests for the daemon authorization tier (#728).
//!
//! Every acceptance criterion on the issue is one named test here, so a failing
//! run names the unmet requirement rather than a line number.
//!
//! The tier has two levels. A tenant runs their own conversations, knowledge
//! and preferences. An administrator additionally changes how the service runs:
//! credentials, connectors, purposes, the database, the WebSocket auth posture,
//! and which child processes the daemon spawns for MCP.
//!
//! Adding a `Command` variant makes `required_capability` fail to compile,
//! because that match has no wildcard arm. Add the new variant to
//! [`command_samples`] in the same change, so the behaviour tests below cover
//! it too.

use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_transport_dispatch::{
    AdminSubjects, AuthContext, Capability, REFUSAL_PREFIX, TransportKind,
    capability_for_local_peer, command_name, dispatch_loop, required_capability,
};
use futures::StreamExt;
use futures::channel::mpsc;

// --- test doubles -----------------------------------------------------------

/// Records the command name of everything that reaches the handler, so a test
/// can prove a refused command never got past the gate.
#[derive(Default)]
struct RecordingHandler {
    seen: Mutex<Vec<String>>,
}

impl RecordingHandler {
    fn seen(&self) -> Vec<String> {
        self.seen.lock().expect("recording handler lock").clone()
    }
}

/// The aggregate config a settings client reads. `caller_capability` is left
/// unset here on purpose: the settings service has no notion of a connection,
/// so the dispatcher is what must fill it in.
fn unstamped_config() -> api::Config {
    api::Config {
        embeddings: api::EmbeddingsSettingsView {
            connector: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            base_url: "http://localhost:11434".to_string(),
            has_api_key: false,
            available: true,
            is_default: false,
            health: api::EmbeddingHealth::Ok,
        },
        persistence: api::PersistenceSettingsView {
            enabled: false,
            remote_url: String::new(),
            remote_name: "origin".to_string(),
            push_on_update: false,
        },
        personality: api::PersonalitySettingsView::default(),
        restart_required: Vec::new(),
        caller_capability: None,
    }
}

#[async_trait::async_trait]
impl AssistantApiHandler for RecordingHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        self.seen
            .lock()
            .expect("recording handler lock")
            .push(command_key(&cmd));
        match cmd {
            api::Command::GetConfig | api::Command::SetConfig { .. } => {
                Ok(api::CommandResult::Config(unstamped_config()))
            }
            _ => Ok(api::CommandResult::Ack),
        }
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        self.seen
            .lock()
            .expect("recording handler lock")
            .push("send_message".to_string());
        Ok(())
    }
}

/// The command's serde tag (`"set_api_key"`, `"ping"`, ...) - the same name the
/// refusal message uses, so assertions read like the wire.
fn command_key(cmd: &api::Command) -> String {
    match serde_json::to_value(cmd).expect("a Command always serializes") {
        serde_json::Value::String(s) => s,
        serde_json::Value::Object(map) => map
            .keys()
            .next()
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        _ => "unknown".to_string(),
    }
}

/// Drive `commands` through the dispatcher as one connection holding
/// `capability`, and return every outbound frame plus what reached the handler.
async fn dispatch_as(
    capability: Capability,
    commands: Vec<api::Command>,
) -> (Vec<api::WsFrame>, Vec<String>) {
    let recorder = Arc::new(RecordingHandler::default());
    let handler: Arc<dyn AssistantApiHandler> = Arc::clone(&recorder) as Arc<_>;

    let requests: Vec<anyhow::Result<api::WsRequest>> = commands
        .into_iter()
        .enumerate()
        .map(|(i, command)| {
            Ok(api::WsRequest {
                id: format!("req-{i}"),
                command,
            })
        })
        .collect();
    let inbound = futures::stream::iter(requests);
    let (out_tx, out_rx) = mpsc::channel::<api::WsFrame>(1024);

    let auth = AuthContext::new("subject", TransportKind::WebSocket).with_capability(capability);
    dispatch_loop(handler, auth, inbound, out_tx).await;

    let frames = out_rx.collect::<Vec<_>>().await;
    (frames, recorder.seen())
}

/// The refusal text when the frame is an authorization refusal.
fn refusal(frame: &api::WsFrame) -> Option<&str> {
    match frame {
        api::WsFrame::Error { error, .. } if error.starts_with(REFUSAL_PREFIX) => Some(error),
        _ => None,
    }
}

/// Assert `command` is refused for a tenant and never reaches the handler.
async fn assert_refused_for_tenant(command: api::Command) {
    let key = command_key(&command);
    let (frames, seen) = dispatch_as(Capability::Tenant, vec![command]).await;
    assert_eq!(frames.len(), 1, "{key}: expected exactly one frame");
    let text = refusal(&frames[0]).unwrap_or_else(|| {
        panic!(
            "{key}: expected an authorization refusal, got {:?}",
            frames[0]
        )
    });
    assert!(
        text.contains(&key),
        "{key}: the refusal must name the command, got {text}"
    );
    assert!(
        seen.is_empty(),
        "{key}: a refused command must not reach the handler, saw {seen:?}"
    );
}

/// How many handler calls `commands` should produce when none is refused.
///
/// Two dispatcher behaviours make this differ from `commands.len()`: the
/// subscription commands are answered by the dispatcher itself and never reach
/// the handler, and `SetWsAuthSettings` is followed by a `GetConfig` read so the
/// reply can report what is still pending a restart (#686).
fn expected_handler_calls(commands: &[api::Command]) -> usize {
    commands
        .iter()
        .map(|command| match command {
            api::Command::SubscribeBackgroundTasks
            | api::Command::UnsubscribeBackgroundTasks
            | api::Command::SubscribeConversations { .. } => 0,
            api::Command::SetWsAuthSettings { .. } => 2,
            _ => 1,
        })
        .sum()
}

/// Assert `command` is NOT refused for a tenant.
async fn assert_allowed_for_tenant(command: api::Command) {
    let key = command_key(&command);
    let (frames, seen) = dispatch_as(Capability::Tenant, vec![command]).await;
    for frame in &frames {
        assert!(
            refusal(frame).is_none(),
            "{key}: must stay readable by a tenant, got {frame:?}"
        );
    }
    // "No refusal" alone is satisfied by nothing happening at all, which would
    // pass even if the dispatcher silently dropped the command. Prove it was
    // actually served.
    assert!(
        seen.contains(&key),
        "{key}: must reach the handler, not merely avoid a refusal; saw {seen:?}"
    );
    assert!(
        !frames.is_empty(),
        "{key}: must produce a reply the client can correlate"
    );
}

// --- the capability table ---------------------------------------------------

fn purpose_config() -> api::PurposeConfigView {
    api::PurposeConfigView {
        connection: "default".to_string(),
        model: "a-model".to_string(),
        effort: None,
        max_context_tokens: None,
    }
}

fn connection_config() -> api::ConnectionConfigView {
    api::ConnectionConfigView::Ollama {
        base_url: None,
        connect_timeout_secs: None,
        stream_timeout_secs: None,
        keep_warm: None,
        max_context_tokens: None,
    }
}

/// Personality-only changes: the seven traits every ordinary client writes.
fn personality_changes() -> api::ConfigChanges {
    api::ConfigChanges {
        personality_warmth: Some(api::PersonalityLevel::Often),
        ..api::ConfigChanges::default()
    }
}

/// A daemon-global change carried by the very same `SetConfig` command.
fn operator_changes() -> api::ConfigChanges {
    api::ConfigChanges {
        embeddings_base_url: Some("http://example.com:11434".to_string()),
        ..api::ConfigChanges::default()
    }
}

/// The capability table: one row per `Command` variant, pairing the variant
/// with the sample the behaviour tests drive through the dispatcher and the
/// capability it must require.
///
/// `command_table!` generates **both** the exhaustive match and the sample list
/// from these rows, so the two cannot diverge. That is the point, and it took
/// two attempts to get right:
///
/// - A separate or-pattern probe plus a hand-maintained count was defeated by
///   the shortest fix available: appending `| api::Command::NewThing` to the
///   probe satisfied the compiler while the sample list stayed at its old
///   length and the count assertion still held. A new variant sailed through
///   classified but exercised by none of the sweep tests.
/// - Generating both from one table closed that, but a row taking a general
///   `pat` could still be folded (`Ping | NewThing => (ping_sample, ...)`),
///   which also compiled green.
///
/// So a row takes a bare **identifier**, and the macro builds the pattern. One
/// variant per row is now enforced by the grammar: there is no pattern position
/// to fold into, and the only way to satisfy the build error a new variant
/// causes is to add a row - which produces its sample as a side effect.
macro_rules! command_table {
    ($( $variant:ident => ($sample:expr, $capability:expr) ),+ $(,)?) => {
        /// One sample of every `Command` variant with the capability it requires.
        fn command_samples() -> Vec<(api::Command, Capability)> {
            vec![ $( ($sample, $capability) ),+ ]
        }

        /// Never called. Its only job is to fail the build when a `Command`
        /// variant has no row above.
        ///
        /// The row grammar takes a bare variant **identifier**, not a pattern,
        /// and builds `api::Command::$variant { .. }` itself - a form rustc
        /// accepts for unit, tuple and struct variants alike. So a row cannot
        /// name two variants: `A | B => (...)` is a syntax error here, not a
        /// working shortcut. That is deliberate, and it is the hole the
        /// previous shape left open.
        #[allow(dead_code)]
        fn every_variant_is_in_the_table(cmd: &api::Command) {
            match cmd { $( api::Command::$variant { .. } => {} ),+ }
        }
    };
}

// The table below is dense; the bare names keep one row per line readable.
use desktop_assistant_transport_dispatch::Capability::{Admin, Tenant};

command_table! {
    Ping => (api::Command::Ping, Tenant),
    GetStatus => (api::Command::GetStatus, Tenant),
    GetConfig => (api::Command::GetConfig, Tenant),
    SetConfig => (api::Command::SetConfig {
                changes: personality_changes(),
            },
            Admin),
    CreateConversation => (api::Command::CreateConversation {
                title: "t".to_string(),
                tags: vec![],
            },
            Tenant),
    ListConversations => (api::Command::ListConversations {
                max_age_days: None,
                include_archived: false,
            },
            Tenant),
    GetConversation => (api::Command::GetConversation {
                id: "c".to_string(),
            },
            Tenant),
    GetMessages => (api::Command::GetMessages {
                conversation_id: "c".to_string(),
                tail: 10,
                after_count: -1,
                include_roles: vec![],
            },
            Tenant),
    DeleteConversation => (api::Command::DeleteConversation {
                id: "c".to_string(),
            },
            Tenant),
    RenameConversation => (api::Command::RenameConversation {
                id: "c".to_string(),
                title: "t".to_string(),
            },
            Tenant),
    ArchiveConversation => (api::Command::ArchiveConversation {
                id: "c".to_string(),
            },
            Tenant),
    UnarchiveConversation => (api::Command::UnarchiveConversation {
                id: "c".to_string(),
            },
            Tenant),
    ClearAllHistory => (api::Command::ClearAllHistory, Tenant),
    SendMessage => (api::Command::SendMessage {
                conversation_id: "c".to_string(),
                content: "hi".to_string(),
                override_selection: None,
                system_refinement: String::new(),
                client_context: None,
                idempotency_key: None,
                turn_id: None,
                traceparent: None,
            },
            Tenant),
    SetConversationPersonality => (api::Command::SetConversationPersonality {
                conversation_id: "c".to_string(),
                personality: api::ConversationPersonalityView::default(),
            },
            Tenant),
    SetConversationToolGate => (api::Command::SetConversationToolGate {
                conversation_id: "c".to_string(),
                disabled: true,
            },
            Tenant),
    SetApiKey => (api::Command::SetApiKey {
                api_key: "k".into(),
            },
            Admin),
    GetEmbeddingsSettings => (api::Command::GetEmbeddingsSettings, Tenant),
    SetEmbeddingsSettings => (api::Command::SetEmbeddingsSettings {
                connector: None,
                model: None,
                base_url: None,
            },
            Admin),
    GetConnectorDefaults => (api::Command::GetConnectorDefaults {
                connector: "ollama".to_string(),
            },
            Tenant),
    GetPersistenceSettings => (api::Command::GetPersistenceSettings, Tenant),
    SetPersistenceSettings => (api::Command::SetPersistenceSettings {
                enabled: true,
                remote_url: None,
                remote_name: None,
                push_on_update: false,
            },
            Admin),
    GetDatabaseSettings => (api::Command::GetDatabaseSettings, Tenant),
    SetDatabaseSettings => (api::Command::SetDatabaseSettings {
                url: api::Secret(String::new()),
                max_connections: 5,
            },
            Admin),
    GetBackendTasksSettings => (api::Command::GetBackendTasksSettings, Tenant),
    SetBackendTasksSettings => (api::Command::SetBackendTasksSettings {
                llm_connector: String::new(),
                llm_model: String::new(),
                llm_base_url: String::new(),
                dreaming_enabled: false,
                dreaming_interval_secs: 60,
                archive_after_days: 7,
            },
            Admin),
    GetWsAuthSettings => (api::Command::GetWsAuthSettings, Tenant),
    SetWsAuthSettings => (api::Command::SetWsAuthSettings {
                methods: vec!["password".to_string()],
                oidc_issuer: String::new(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            },
            Admin),
    ListConnections => (api::Command::ListConnections, Tenant),
    CreateConnection => (api::Command::CreateConnection {
                id: "c".to_string(),
                config: connection_config(),
            },
            Admin),
    UpdateConnection => (api::Command::UpdateConnection {
                id: "c".to_string(),
                config: connection_config(),
            },
            Admin),
    DeleteConnection => (api::Command::DeleteConnection {
                id: "c".to_string(),
                force: false,
            },
            Admin),
    SetConnectionSecret => (api::Command::SetConnectionSecret {
                id: "c".to_string(),
                credential: api::Secret("s".to_string()),
            },
            Admin),
    ListAvailableModels => (api::Command::ListAvailableModels {
                connection_id: None,
                refresh: false,
            },
            Tenant),
    GetPurposes => (api::Command::GetPurposes, Tenant),
    SetPurpose => (api::Command::SetPurpose {
                purpose: api::PurposeKindApi::Interactive,
                config: purpose_config(),
            },
            Admin),
    GetToolUsage => (api::Command::GetToolUsage {
                conversation_id: "c".to_string(),
            },
            Tenant),
    ListKnowledgeEntries => (api::Command::ListKnowledgeEntries {
                limit: 10,
                offset: 0,
                tag_filter: None,
            },
            Tenant),
    GetKnowledgeEntry => (api::Command::GetKnowledgeEntry {
                id: "k".to_string(),
            },
            Tenant),
    SearchKnowledgeEntries => (api::Command::SearchKnowledgeEntries {
                query: "q".to_string(),
                tag_filter: None,
                limit: 10,
            },
            Tenant),
    CreateKnowledgeEntry => (api::Command::CreateKnowledgeEntry {
                content: "c".to_string(),
                tags: vec![],
                metadata: serde_json::Value::Null,
            },
            Tenant),
    UpdateKnowledgeEntry => (api::Command::UpdateKnowledgeEntry {
                id: "k".to_string(),
                content: "c".to_string(),
                tags: vec![],
                metadata: serde_json::Value::Null,
            },
            Tenant),
    DeleteKnowledgeEntry => (api::Command::DeleteKnowledgeEntry {
                id: "k".to_string(),
            },
            Tenant),
    GetKnowledgeTrashCount => (api::Command::GetKnowledgeTrashCount, Tenant),
    EmptyKnowledgeTrash => (api::Command::EmptyKnowledgeTrash, Tenant),
    StartKnowledgeMaintenance => (api::Command::StartKnowledgeMaintenance {
                op: api::MaintenanceOp::Extraction,
            },
            Admin),
    ListMcpServers => (api::Command::ListMcpServers, Tenant),
    AddMcpServer => (api::Command::AddMcpServer {
                name: "s".to_string(),
                command: "/bin/true".to_string(),
                args: vec![],
                namespace: None,
                enabled: true,
            },
            Admin),
    RemoveMcpServer => (api::Command::RemoveMcpServer {
                name: "s".to_string(),
            },
            Admin),
    SetMcpServerEnabled => (api::Command::SetMcpServerEnabled {
                name: "s".to_string(),
                enabled: true,
            },
            Admin),
    McpServerAction => (api::Command::McpServerAction {
                action: "restart".to_string(),
                server: None,
            },
            Admin),
    UpsertMcpServer => (api::Command::UpsertMcpServer {
                config_json: "{}".to_string(),
            },
            Admin),
    SetMcpSecret => (api::Command::SetMcpSecret {
                id: "s".to_string(),
                value: api::Secret("v".to_string()),
            },
            Admin),
    ListServiceAccounts => (api::Command::ListServiceAccounts, Tenant),
    UpsertServiceAccount => (api::Command::UpsertServiceAccount {
                config_json: "{}".to_string(),
            },
            Admin),
    RemoveServiceAccount => (api::Command::RemoveServiceAccount {
                id: "a".to_string(),
            },
            Admin),
    ListBackgroundTasks => (api::Command::ListBackgroundTasks {
                include_finished: false,
                limit: None,
            },
            Tenant),
    GetBackgroundTask => (api::Command::GetBackgroundTask {
                id: "t".to_string(),
            },
            Tenant),
    CancelBackgroundTask => (api::Command::CancelBackgroundTask {
                id: "t".to_string(),
            },
            Tenant),
    GetBackgroundTaskLogs => (api::Command::GetBackgroundTaskLogs {
                id: "t".to_string(),
                after_seq: None,
                limit: None,
            },
            Tenant),
    SubscribeBackgroundTasks => (api::Command::SubscribeBackgroundTasks, Tenant),
    UnsubscribeBackgroundTasks => (api::Command::UnsubscribeBackgroundTasks, Tenant),
    SubscribeConversations => (api::Command::SubscribeConversations {
                conversation_ids: vec![],
            },
            Tenant),
    SpawnStandaloneAgent => (api::Command::SpawnStandaloneAgent {
                name: "a".to_string(),
                initial_prompt: "p".to_string(),
                override_selection: None,
                tools: None,
            },
            Tenant),
    GetConversationScratchpad => (api::Command::GetConversationScratchpad {
                conversation_id: "c".to_string(),
                max_results: None,
            },
            Tenant),
    SetScratchpadNote => (api::Command::SetScratchpadNote {
                conversation_id: "c".to_string(),
                key: "k".to_string(),
                content: "v".to_string(),
                note_type: String::new(),
                sequence: None,
                done: false,
            },
            Tenant),
    DeleteScratchpadNotes => (api::Command::DeleteScratchpadNotes {
                conversation_id: "c".to_string(),
                keys: vec![],
                all: false,
            },
            Tenant),
    RegisterClientTools => (api::Command::RegisterClientTools { tools: vec![] }, Tenant),
    ClientToolResult => (api::Command::ClientToolResult {
                task_id: api::TaskId("t".to_string()),
                tool_call_id: "tc".to_string(),
                result: Some("r".to_string()),
                error: None,
            },
            Tenant),
}

/// Classifications that depend on the command's *payload* rather than its
/// variant, so one row in the table above cannot express them.
///
/// The table carries the personality-only `SetConfig` and the process-spawning
/// `McpServerAction`; these are the other side of each split. Both also have
/// their own named tests.
fn payload_dependent_samples() -> Vec<(api::Command, Capability)> {
    vec![
        (
            api::Command::SetConfig {
                changes: operator_changes(),
            },
            Admin,
        ),
        (
            api::Command::McpServerAction {
                action: "status".to_string(),
                server: None,
            },
            Capability::Tenant,
        ),
    ]
}

/// Every sample the sweep tests drive: one per variant, plus the
/// payload-dependent cases.
fn all_samples() -> Vec<(api::Command, Capability)> {
    let mut all = command_samples();
    all.extend(payload_dependent_samples());
    all
}

fn admin_commands() -> Vec<api::Command> {
    all_samples()
        .into_iter()
        .filter(|(_, cap)| *cap == Capability::Admin)
        .map(|(cmd, _)| cmd)
        .collect()
}

// --- grant paths ------------------------------------------------------------

/// A UDS caller whose kernel-attested peer uid equals the daemon's own resolves
/// to the admin capability. This is what makes the single-user desktop need no
/// configuration.
#[test]
fn uds_peer_uid_matching_daemon_is_admin() {
    assert_eq!(capability_for_local_peer(1000, 1000), Capability::Admin);
    assert_eq!(capability_for_local_peer(0, 0), Capability::Admin);
}

/// A different local uid is a tenant, not an administrator.
#[test]
fn uds_peer_uid_differing_from_daemon_is_not_admin() {
    assert_eq!(capability_for_local_peer(1001, 1000), Capability::Tenant);
    assert_eq!(capability_for_local_peer(0, 1000), Capability::Tenant);
}

/// With the default (empty) allowlist an authenticated remote client is a
/// tenant, and is refused every admin command.
#[tokio::test]
async fn ws_subject_absent_from_admin_subjects_is_refused() {
    let allowlist = AdminSubjects::default();
    assert_eq!(allowlist.capability_for("alice"), Capability::Tenant);

    // A populated allowlist that does not name the caller is the same verdict.
    let named = AdminSubjects::new(["operator"]);
    assert_eq!(named.capability_for("alice"), Capability::Tenant);

    for command in admin_commands() {
        assert_refused_for_tenant(command).await;
    }
}

/// A subject on the allowlist is admitted, and its admin commands run.
#[tokio::test]
async fn ws_subject_in_admin_subjects_is_admitted() {
    let allowlist = AdminSubjects::new(["operator", "alice"]);
    assert_eq!(allowlist.capability_for("alice"), Capability::Admin);

    let commands = admin_commands();
    let expected = expected_handler_calls(&commands);
    let (frames, seen) = dispatch_as(allowlist.capability_for("alice"), commands).await;
    for frame in &frames {
        assert!(refusal(frame).is_none(), "admitted subject: {frame:?}");
    }
    assert_eq!(
        seen.len(),
        expected,
        "every admin command must reach the handler"
    );
}

/// Blank and whitespace-only entries never grant anything, so a stray line in
/// `daemon.toml` cannot admit the empty subject.
#[test]
fn admin_subjects_ignores_blank_entries() {
    let allowlist = AdminSubjects::new(["", "   ", " operator "]);
    assert_eq!(allowlist.capability_for(""), Capability::Tenant);
    assert_eq!(allowlist.capability_for("   "), Capability::Tenant);
    assert_eq!(allowlist.capability_for("operator"), Capability::Admin);
}

/// No command mutates the allowlist: it is file-only, so a tenant cannot grant
/// themselves the capability they are being denied.
#[test]
fn admin_subjects_cannot_be_set_over_the_wire() {
    for (command, _) in all_samples() {
        let json = serde_json::to_string(&command).expect("a Command always serializes");
        assert!(
            !json.contains("admin_subjects") && !json.contains("authz"),
            "no command may carry the admin allowlist, found it in {json}"
        );
    }
}

/// A connection with no resolved capability is a tenant. Fail closed: the
/// absence of a grant is never a grant.
#[test]
fn absent_identity_fails_closed_to_tenant() {
    assert_eq!(
        AuthContext::new("nobody", TransportKind::WebSocket).capability,
        Capability::Tenant
    );
    assert_eq!(AuthContext::anonymous().capability, Capability::Tenant);
}

// --- the closed exposures ---------------------------------------------------

#[tokio::test]
async fn set_api_key_refused_for_non_admin() {
    assert_refused_for_tenant(api::Command::SetApiKey {
        api_key: "sk-not-yours".into(),
    })
    .await;
}

#[tokio::test]
async fn add_mcp_server_refused_for_non_admin() {
    assert_refused_for_tenant(api::Command::AddMcpServer {
        name: "evil".to_string(),
        command: "/bin/sh".to_string(),
        args: vec!["-c".to_string(), "id".to_string()],
        namespace: None,
        enabled: true,
    })
    .await;
    assert_refused_for_tenant(api::Command::UpsertMcpServer {
        config_json: "{}".to_string(),
    })
    .await;
}

#[tokio::test]
async fn create_connection_refused_for_non_admin() {
    assert_refused_for_tenant(api::Command::CreateConnection {
        id: "mine".to_string(),
        config: connection_config(),
    })
    .await;
}

/// `StartKnowledgeMaintenance` sits in the knowledge family, whose other
/// commands are user-scoped, but it is not tenant work.
///
/// `RecalculateEmbeddings` reaches `invalidate_all_knowledge_embeddings`, which
/// runs `UPDATE knowledge_base SET embedding = NULL ... WHERE deleted_at IS
/// NULL` with no `user_id` predicate: one tenant nulls every tenant's vectors
/// and re-embeds the whole instance through the operator's provider. The other
/// arms are instance-wide too - consolidation loops over every user, and
/// extraction's archival phase widens to all users under the `"default"`
/// sentinel. The registry also files the task under the *caller's* user id, and
/// cancellation is owner-scoped, so an operator cannot stop it.
#[tokio::test]
async fn knowledge_maintenance_is_admin_because_it_touches_every_tenant() {
    for op in [
        api::MaintenanceOp::Extraction,
        api::MaintenanceOp::Consolidation,
        api::MaintenanceOp::RecalculateEmbeddings,
    ] {
        assert_eq!(
            required_capability(&api::Command::StartKnowledgeMaintenance { op }),
            Capability::Admin,
            "{op:?} is instance-wide work"
        );
        assert_refused_for_tenant(api::Command::StartKnowledgeMaintenance { op }).await;
    }
}

/// `McpServerAction` is split by its verb, the same way `SetConfig` is split by
/// its payload. `"status"` returns exactly what `ListMcpServers` returns, which
/// is tenant-readable, so gating it would contradict that.
#[tokio::test]
async fn mcp_status_is_a_read_and_stays_tenant() {
    assert_eq!(
        required_capability(&api::Command::McpServerAction {
            action: "status".to_string(),
            server: None,
        }),
        Capability::Tenant
    );
    assert_allowed_for_tenant(api::Command::McpServerAction {
        action: "status".to_string(),
        server: Some("fileio".to_string()),
    })
    .await;
}

/// Every other verb spawns or kills a child process. An unrecognized verb fails
/// closed to admin rather than being assumed harmless.
#[tokio::test]
async fn mcp_lifecycle_actions_require_admin() {
    for action in ["start", "stop", "restart", "a-verb-added-later"] {
        assert_eq!(
            required_capability(&api::Command::McpServerAction {
                action: action.to_string(),
                server: None,
            }),
            Capability::Admin,
            "{action} must require admin"
        );
    }
}

#[tokio::test]
async fn set_purpose_refused_for_non_admin() {
    assert_refused_for_tenant(api::Command::SetPurpose {
        purpose: api::PurposeKindApi::Interactive,
        config: purpose_config(),
    })
    .await;
}

#[tokio::test]
async fn set_ws_auth_settings_refused_for_non_admin() {
    assert_refused_for_tenant(api::Command::SetWsAuthSettings {
        methods: vec!["oidc".to_string()],
        oidc_issuer: "https://issuer.example.com".to_string(),
        oidc_auth_endpoint: String::new(),
        oidc_token_endpoint: String::new(),
        oidc_client_id: "id".to_string(),
        oidc_scopes: "openid".to_string(),
    })
    .await;
}

// --- what must keep working for a tenant ------------------------------------

#[tokio::test]
async fn list_connections_allowed_for_tenant() {
    assert_allowed_for_tenant(api::Command::ListConnections).await;
}

#[tokio::test]
async fn list_available_models_allowed_for_tenant() {
    assert_allowed_for_tenant(api::Command::ListAvailableModels {
        connection_id: None,
        refresh: false,
    })
    .await;
}

#[tokio::test]
async fn get_purposes_allowed_for_tenant() {
    assert_allowed_for_tenant(api::Command::GetPurposes).await;
}

/// Trap 1 as the issue framed it, corrected by what the code actually does.
///
/// The acceptance criterion asked that the seven personality traits stay
/// writable by a tenant, on the premise that they are a personal preference.
/// They are not, yet: `SetConfig` writes ONE global `[personality]` into the
/// operator's `daemon.toml` and every conversation without an override resolves
/// against it, so a tenant writing a trait would change every other tenant's
/// assistant. The tenant lever is the per-conversation override, which is
/// genuinely user-scoped - so this asserts the criterion's *intent* (a tenant
/// can still set their assistant's personality) against the command that
/// honours it.
///
/// Named for what it asserts, not for the criterion. A run that printed
/// `personality_traits_writable_by_tenant ... ok` would tell a reader the
/// opposite of the truth: a tenant is refused exactly those global traits, which
/// `global_personality_is_instance_configuration_and_needs_admin` pins.
#[tokio::test]
async fn tenant_sets_their_own_personality_per_conversation() {
    // The per-conversation override is the tenant's own, and stays open.
    assert_eq!(
        required_capability(&api::Command::SetConversationPersonality {
            conversation_id: "c".to_string(),
            personality: api::ConversationPersonalityView {
                warmth: Some(api::PersonalityLevel::Often),
                ..api::ConversationPersonalityView::default()
            },
        }),
        Capability::Tenant
    );
    assert_allowed_for_tenant(api::Command::SetConversationPersonality {
        conversation_id: "c".to_string(),
        personality: api::ConversationPersonalityView {
            sarcasm: Some(api::PersonalityLevel::Rarely),
            ..api::ConversationPersonalityView::default()
        },
    })
    .await;

    // An empty change set writes nothing, so it costs nothing.
    assert_eq!(
        required_capability(&api::Command::SetConfig {
            changes: api::ConfigChanges::default()
        }),
        Capability::Tenant
    );
}

/// The global `[personality]` block is the instance default, and writing it is
/// administration - it rewrites `daemon.toml`, rebuilds the connection
/// registry, and changes the disposition every other tenant gets.
#[tokio::test]
async fn global_personality_is_instance_configuration_and_needs_admin() {
    let traits = [
        api::ConfigChanges {
            personality_professionalism: Some(api::PersonalityLevel::Always),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_warmth: Some(api::PersonalityLevel::Often),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_directness: Some(api::PersonalityLevel::Often),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_enthusiasm: Some(api::PersonalityLevel::Sometimes),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_humor: Some(api::PersonalityLevel::Sometimes),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_sarcasm: Some(api::PersonalityLevel::Rarely),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            personality_pretentiousness: Some(api::PersonalityLevel::Rarely),
            ..api::ConfigChanges::default()
        },
    ];
    for changes in traits {
        assert_eq!(
            required_capability(&api::Command::SetConfig {
                changes: changes.clone()
            }),
            Capability::Admin,
            "a global personality write is instance configuration: {changes:?}"
        );
        assert_refused_for_tenant(api::Command::SetConfig { changes }).await;
    }
}

/// The other half of trap 1: a `SetConfig` that touches a daemon-global knob is
/// admin even though the command name is the same.
#[tokio::test]
async fn daemon_global_config_changes_require_admin() {
    let global = [
        api::ConfigChanges {
            embeddings_connector: Some("ollama".to_string()),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            embeddings_model: Some("m".to_string()),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            embeddings_base_url: Some("http://example.com".to_string()),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            persistence_enabled: Some(true),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            persistence_remote_url: Some("https://example.com/repo.git".into()),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            persistence_remote_name: Some("origin".to_string()),
            ..api::ConfigChanges::default()
        },
        api::ConfigChanges {
            persistence_push_on_update: Some(true),
            ..api::ConfigChanges::default()
        },
    ];
    for changes in global {
        assert_eq!(
            required_capability(&api::Command::SetConfig {
                changes: changes.clone()
            }),
            Capability::Admin,
            "a daemon-global SetConfig must need admin: {changes:?}"
        );
        assert_refused_for_tenant(api::Command::SetConfig { changes }).await;
    }

    // Mixed in one command: the daemon-global part decides.
    assert_eq!(
        required_capability(&api::Command::SetConfig {
            changes: api::ConfigChanges {
                personality_warmth: Some(api::PersonalityLevel::Often),
                embeddings_model: Some("m".to_string()),
                ..api::ConfigChanges::default()
            }
        }),
        Capability::Admin
    );
}

/// The design record's non-negotiable constraint: an empty `[authz]` section
/// plus a local UDS caller leaves every command working exactly as it does
/// today. A failure here is a wrong design, not a wrong test.
#[tokio::test]
async fn single_user_desktop_needs_no_authz_config() {
    // What a desktop daemon resolves with no configuration at all.
    let allowlist = AdminSubjects::default();
    let daemon_uid = 1000;
    let capability = capability_for_local_peer(daemon_uid, daemon_uid)
        .strongest(allowlist.capability_for("dave"));
    assert_eq!(capability, Capability::Admin);

    let commands: Vec<api::Command> = all_samples().into_iter().map(|(c, _)| c).collect();
    let expected = expected_handler_calls(&commands);

    let (frames, seen) = dispatch_as(capability, commands).await;
    for frame in &frames {
        assert!(
            refusal(frame).is_none(),
            "the single-user desktop must refuse nothing, got {frame:?}"
        );
    }
    assert_eq!(
        seen.len(),
        expected,
        "every command must still reach the handler, saw {seen:?}"
    );
}

// --- refusal behaviour ------------------------------------------------------

/// A refusal is a rendered result frame the client can display. The connection
/// stays up and keeps serving.
#[tokio::test]
async fn refused_command_returns_a_rendered_refusal_not_a_disconnect() {
    let (frames, seen) = dispatch_as(
        Capability::Tenant,
        vec![
            api::Command::SetApiKey {
                api_key: "k".into(),
            },
            api::Command::Ping,
        ],
    )
    .await;

    assert_eq!(
        frames.len(),
        2,
        "the loop must keep serving after a refusal"
    );

    match &frames[0] {
        api::WsFrame::Error { id, error, .. } => {
            assert_eq!(id, "req-0", "the refusal must correlate to the request");
            assert!(error.starts_with(REFUSAL_PREFIX), "got {error}");
        }
        other => panic!("expected a refusal frame, got {other:?}"),
    }
    assert!(
        matches!(&frames[1], api::WsFrame::Result { id, .. } if id == "req-1"),
        "the next command must still be served, got {:?}",
        frames[1]
    );
    assert_eq!(
        seen,
        vec!["ping".to_string()],
        "only the permitted command reaches the handler"
    );
}

/// The refusal names the command and the capability it needed, so a client can
/// tell the user what to do rather than showing a bare failure.
#[tokio::test]
async fn refusal_names_the_command_and_the_missing_capability() {
    let (frames, _) = dispatch_as(
        Capability::Tenant,
        vec![api::Command::SetApiKey {
            api_key: "k".into(),
        }],
    )
    .await;
    let text = refusal(&frames[0]).expect("a refusal");
    assert!(text.contains("set_api_key"), "got {text}");
    assert!(text.contains("administrator"), "got {text}");
}

/// The declared table above and the dispatcher agree for every variant.
#[tokio::test]
async fn every_declared_capability_matches_the_dispatcher_gate() {
    for (command, expected) in all_samples() {
        assert_eq!(
            required_capability(&command),
            expected,
            "{} has the wrong required capability",
            command_key(&command)
        );
    }
}

// --- discovering the capability before using it -----------------------------

/// The capability the daemon reported on a `GetConfig` reply.
async fn reported_capability(held: Capability) -> Option<api::Capability> {
    let (frames, _) = dispatch_as(held, vec![api::Command::GetConfig]).await;
    match frames.first() {
        Some(api::WsFrame::Result {
            result: api::CommandResult::Config(config),
            ..
        }) => config.caller_capability.clone(),
        other => panic!("expected a config result, got {other:?}"),
    }
}

/// An administrator is told so on the settings read it already makes, so a
/// client can render the operator sections as editable.
#[tokio::test]
async fn admin_caller_is_told_it_holds_the_admin_capability() {
    assert_eq!(
        reported_capability(Capability::Admin).await,
        Some(api::Capability::Admin)
    );
}

/// A tenant is told so too, and up front - so a settings panel marks the
/// operator sections unavailable with a reason rather than failing on submit.
#[tokio::test]
async fn tenant_caller_is_told_it_holds_only_the_tenant_capability() {
    assert_eq!(
        reported_capability(Capability::Tenant).await,
        Some(api::Capability::Tenant)
    );
}

/// The `ConfigChanged` event carries the same answer as the reply that caused
/// it, so a client re-rendering from the event does not lose the verdict.
#[tokio::test]
async fn config_changed_event_carries_the_callers_capability() {
    let (frames, _) = dispatch_as(
        Capability::Admin,
        vec![api::Command::SetConfig {
            changes: personality_changes(),
        }],
    )
    .await;
    let event = frames
        .iter()
        .find_map(|frame| match frame {
            api::WsFrame::Event {
                event: api::Event::ConfigChanged { config },
            } => Some(config.clone()),
            _ => None,
        })
        .expect("a ConfigChanged event");
    assert_eq!(event.caller_capability, Some(api::Capability::Admin));
}

// --- the refusal is an API response, not an error string --------------------

/// A caller keys on a stable code, never on English text, and knows not to
/// retry.
#[tokio::test]
async fn a_refusal_carries_the_stable_code_and_is_not_retryable() {
    let (frames, _) = dispatch_as(
        Capability::Tenant,
        vec![api::Command::SetApiKey {
            api_key: "k".into(),
        }],
    )
    .await;
    match frames.first() {
        Some(api::WsFrame::Error {
            detail: Some(detail),
            ..
        }) => {
            assert_eq!(detail.code, api::ErrorCode::NotAuthorized);
            assert_eq!(detail.code.as_str(), "not_authorized");
            assert!(!detail.retryable, "repeating cannot change the answer");
            assert!(!detail.message.is_empty(), "a person must be told why");
        }
        other => panic!("expected a classified refusal, got {other:?}"),
    }
}

/// An older client, which knows only `{id, error}`, still parses the frame -
/// the classification is an optional added field, not a new variant.
#[tokio::test]
async fn an_older_client_still_parses_a_refusal_frame() {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum LegacyWsFrame {
        Result {
            #[allow(dead_code)]
            id: String,
        },
        Error {
            id: String,
            error: String,
        },
        Event {},
    }

    let (frames, _) = dispatch_as(
        Capability::Tenant,
        vec![api::Command::SetApiKey {
            api_key: "k".into(),
        }],
    )
    .await;
    let json = serde_json::to_string(&frames[0]).expect("serialize");
    match serde_json::from_str::<LegacyWsFrame>(&json).expect("an older client must still parse") {
        LegacyWsFrame::Error { id, error } => {
            assert_eq!(id, "req-0");
            assert!(
                error.starts_with(REFUSAL_PREFIX),
                "the human string stays present and unchanged in shape: {error}"
            );
        }
        _ => panic!("expected the error variant"),
    }
}

/// Each row names a distinct command, and its sample really is that command.
///
/// The macro's grammar already makes one-variant-per-row structural, and a
/// duplicate row is an unreachable-pattern error. This is the cheap belt on top:
/// it catches a row whose *sample* was copied from a neighbour and not updated,
/// which nothing else would notice.
#[test]
fn each_row_carries_a_distinct_and_correct_sample() {
    let samples = command_samples();
    let mut counts = std::collections::BTreeMap::new();
    for (cmd, _) in &samples {
        *counts.entry(command_key(cmd)).or_insert(0usize) += 1;
    }
    let duplicated: Vec<_> = counts.iter().filter(|(_, n)| **n > 1).collect();
    assert!(
        duplicated.is_empty(),
        "a row's sample was copied from another row: {duplicated:?}"
    );
    assert_eq!(
        samples.len(),
        counts.len(),
        "every row must contribute exactly one distinct command"
    );
}

/// The hand-written wire names in `classify` match the serde tag the command
/// actually serializes as, for every variant. This is what lets the refusal
/// path read a name without materialising a copy of the command.
#[test]
fn commands_report_their_wire_name() {
    for (command, _) in all_samples() {
        assert_eq!(
            command_name(&command),
            command_key(&command),
            "the refusal name must match the wire tag"
        );
    }
}

/// Admin satisfies tenant; tenant does not satisfy admin.
#[test]
fn admin_permits_everything_a_tenant_may_do() {
    assert!(Capability::Admin.permits(&Capability::Tenant));
    assert!(Capability::Admin.permits(&Capability::Admin));
    assert!(Capability::Tenant.permits(&Capability::Tenant));
    assert!(!Capability::Tenant.permits(&Capability::Admin));
}

/// A level from a newer daemon grants nothing and satisfies nothing, in both
/// directions. Fail closed on the unknown.
#[test]
fn an_unrecognized_capability_grants_nothing() {
    let unknown = Capability::Other("owner".to_string());
    assert!(!unknown.permits(&Capability::Tenant));
    assert!(!unknown.permits(&Capability::Admin));
    assert!(!Capability::Admin.permits(&unknown));
    assert!(!Capability::Tenant.permits(&unknown));
    assert_eq!(
        Capability::Tenant.strongest(unknown.clone()),
        Capability::Tenant,
        "an unrecognized level must never outrank a real grant"
    );
}
