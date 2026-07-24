//! Application layer for protocol-neutral API handling.
//!
//! This crate maps canonical API [`desktop_assistant_api_model::Command`] values
//! to calls into the existing inbound ports in `desktop-assistant-core`.

pub mod background_tasks;
pub mod client_tools;
pub mod conversation_subs;
mod inflight;
pub mod subagent_executor;
pub mod subagent_tools;

use std::sync::Arc;

use desktop_assistant_api_model as api;
pub use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::domain::{DEFAULT_NOTE_TYPE, KnowledgeEntry, ScratchpadNote};
use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::client_tools::with_client_tools;
use desktop_assistant_core::ports::inbound::{
    AssistantService, ConnectionAvailability, ConnectionConfigPayload, ConnectionsService,
    ConversationModelSelection, ConversationService, DispatchWarning, EmbeddingHealth,
    KnowledgeMaintenanceService, KnowledgeService, PromptSelectionOverride, PurposeConfigPayload,
    SettingsService,
};
use desktop_assistant_core::ports::request_scope::RequestScope;
use desktop_assistant_core::ports::scratchpad::{
    MAX_KEYS_PER_CALL, MAX_NOTE_BYTES, MAX_RESULTS_CEILING, NewScratchpadNote, ScratchpadClearFn,
    ScratchpadDeleteManyFn, ScratchpadGetManyFn, ScratchpadListFn, ScratchpadWriteFn,
};
use desktop_assistant_core::ports::store::{IdempotencyKeyStore, TurnStateStore};
use desktop_assistant_core::ports::tool_observer::{ToolEvent, ToolObserver, with_tool_observer};
use thiserror::Error;
use tracing::warn;

use crate::background_tasks::{BackgroundTaskRegistry, TaskContext, TaskError};
use crate::client_tools::{
    ClientToolCoordinator, ClientToolResolutionError, CoordinatorClientToolPort,
    register_client_tools, resolve_client_tool_result,
};
use crate::inflight::{InFlightRegistry, InFlightTurn, TeeSink, forward_inflight};

/// Panic-safe free of a keyed turn's in-flight slot (#440).
///
/// The slot must be removed from the [`InFlightRegistry`] when the turn body
/// ends so a later same-key send falls through to completed-dedup (or runs
/// fresh). Doing that with an inline `remove` *after* `run_send_turn().await`
/// is not panic-safe: a panicking turn unwinds past the inline call and orphans
/// the slot, so the next same-key send re-attaches to a dead hub (whose
/// broadcast sender never drops) and hangs forever. Holding the removal in a
/// `Drop` guard makes it run on every exit path — normal return, `?`, or panic
/// unwind — mirroring the registry's own panic-safe `finalize`.
struct InFlightSlotGuard {
    index: Arc<InFlightRegistry>,
    user_id: String,
    conversation_id: String,
    key: String,
}

impl Drop for InFlightSlotGuard {
    fn drop(&mut self) {
        self.index
            .remove(&self.user_id, &self.conversation_id, &self.key);
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("core error: {0}")]
    Core(String),

    #[error("unsupported command")]
    Unsupported,

    /// The targeted entity is unknown to the caller. Used by background-
    /// task arms to surface both "id never existed" and "id belongs to
    /// another user" (the existence-hiding rule from #105) under a
    /// single, intentionally opaque variant. Renders to "not found" so
    /// adapters can forward the message verbatim without leaking the
    /// distinction.
    #[error("not found")]
    NotFound,

    /// The targeted task is already in a terminal state — e.g. a
    /// `Cancel` on a task the registry already marked `Failed` after
    /// the cold-restart sweep (#115). Distinct from `NotFound` so
    /// transport adapters can return a clean 409-style error to the
    /// client instead of pretending the operation succeeded.
    #[error("task is already terminal")]
    AlreadyTerminal,
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Per-request context threaded from the transport layer through the
/// handler into core services (#105).
///
/// Carries the authenticated `user_id` extracted from the JWT — the
/// daemon's stateless contract (`docs/architecture-evolution.md` rule
/// #1) keeps all request-scoped state here rather than in any handler
/// or service object. Additional per-request fields (request id,
/// tracing context, cancellation token, ...) can be added without
/// changing the handler trait's method signatures.
///
/// Construction:
/// - From a validated JWT: `RequestContext::from(&claims)`.
/// - For a local-dev / single-tenant fallback path: `RequestContext::default()`,
///   which resolves to `UserId::default()` (the schema sentinel
///   `"default"`).
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// The authenticated user_id for this request. Sourced from the
    /// validated JWT's `sub` claim per #105's mapping rule.
    pub user_id: UserId,
}

impl RequestContext {
    /// Build a context with the given user id explicitly. Used by
    /// transport adapters that don't have direct access to a `Claims`
    /// (e.g. they resolved the user id through a different path) and
    /// by tests that need to pin a known identity.
    pub fn for_user(user_id: UserId) -> Self {
        Self { user_id }
    }
}

impl From<&desktop_assistant_auth_jwt::Claims> for RequestContext {
    /// Build a `RequestContext` from a validated JWT claim set. The
    /// `sub` field is mapped to `user_id` verbatim per the phase-1
    /// rule in #105 — future revisions may consult an alternative
    /// claim, but until then `sub` IS the identity.
    fn from(claims: &desktop_assistant_auth_jwt::Claims) -> Self {
        Self {
            user_id: UserId::from(claims),
        }
    }
}

impl From<UserId> for RequestContext {
    fn from(user_id: UserId) -> Self {
        Self::for_user(user_id)
    }
}

/// Protocol-neutral handler for the assistant API.
///
/// Adapters (D-Bus, WebSocket, etc.) should depend on this trait rather than
/// reaching into core services directly.
///
/// ## Threading the request context
///
/// Each method has a companion `*_for(ctx, …)` form that takes a
/// [`RequestContext`]. Transport adapters extract the user identity
/// from the validated JWT, build a `RequestContext`, and call the
/// context-aware form. The non-`_for` methods remain as backward-
/// compatible entry points: they delegate to the `_for` form with
/// `RequestContext::default()`, which collapses to the schema
/// sentinel `"default"`. Single-tenant deployments and tests that
/// don't care about user identity continue to work unchanged (#105).
#[async_trait::async_trait]
pub trait AssistantApiHandler: Send + Sync {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult>;

    /// Context-aware variant of [`Self::handle_command`]. The default
    /// implementation installs the request's user id via
    /// [`with_user_id`] and forwards to `handle_command`. Concrete
    /// handlers don't need to override this unless they want to
    /// observe context fields beyond the user id.
    async fn handle_command_for(
        &self,
        ctx: RequestContext,
        cmd: api::Command,
    ) -> ApiResult<api::CommandResult> {
        with_user_id(ctx.user_id, self.handle_command(cmd)).await
    }

    /// Handle a streaming command.
    ///
    /// For v1 we only stream assistant response chunks for `SendMessage`.
    async fn handle_send_message(
        &self,
        conversation_id: String,
        content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()>;

    /// Context-aware variant of [`Self::handle_send_message`]. The
    /// default implementation installs the request's user id via
    /// [`with_user_id`] and forwards to `handle_send_message`.
    async fn handle_send_message_for(
        &self,
        ctx: RequestContext,
        conversation_id: String,
        content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        with_user_id(
            ctx.user_id,
            self.handle_send_message(conversation_id, content, request_id, sink),
        )
        .await
    }

    /// Handle a streaming `SendMessage` with an optional per-send model
    /// override and an optional per-request system-prompt refinement. The
    /// default implementation ignores both and forwards to
    /// `handle_send_message`; the concrete handler overrides this to thread
    /// them through.
    ///
    /// `system_refinement` (empty = none) is appended to the system prompt
    /// for this one turn only; it is never persisted, so it stays out of
    /// chat history and does not affect later turns. See
    /// [`ConversationService::send_prompt_with_override`].
    // Why allow: a streaming-send entry point carrying the conversation
    // target, two independent per-request inputs (model override + system
    // refinement), the request id, and the event sink. Bundling them solely
    // to satisfy the 7-arg lint would obscure every override of this method.
    #[allow(clippy::too_many_arguments)]
    async fn handle_send_message_with_override(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        let _ = (override_selection, system_refinement, idempotency_key);
        self.handle_send_message(conversation_id, content, request_id, sink)
            .await
    }

    /// Context-aware variant of [`Self::handle_send_message_with_override`].
    // Why allow: same shape as the method above, plus the `RequestContext`.
    #[allow(clippy::too_many_arguments)]
    async fn handle_send_message_with_override_for(
        &self,
        ctx: RequestContext,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        with_user_id(
            ctx.user_id,
            self.handle_send_message_with_override(
                conversation_id,
                content,
                override_selection,
                system_refinement,
                request_id,
                // The context-aware variant is a test/multi-tenant helper; it
                // does not carry an idempotency key (#204).
                None,
                sink,
            ),
        )
        .await
    }

    /// Subscribe to background-task events for the current task-local
    /// user id (#114). Returning `Some(receiver)` lets the transport
    /// layer spawn a forwarder that pumps `Event::Task*` frames out to
    /// a single connection until the connection drops or
    /// `UnsubscribeBackgroundTasks` arrives.
    ///
    /// The default implementation returns `None`, which tells the
    /// dispatcher to surface `SubscribeBackgroundTasks` as a clean
    /// error frame instead of acking and then never streaming
    /// anything. Handlers that wrap a `BackgroundTaskRegistry`
    /// override this method to return a real receiver.
    ///
    /// Contract: callers MUST install the per-request user id via
    /// `with_user_id` before invoking this method. The dispatcher does
    /// that as part of its per-request scope-installation discipline
    /// (#105) — handlers that read `current_user_id()` here can rely
    /// on it being set.
    async fn subscribe_user_events(&self) -> Option<tokio::sync::broadcast::Receiver<api::Event>> {
        None
    }

    /// The per-connection conversation-subscription registry (#1 live
    /// multi-client sync), or `None` when the daemon was built without it — in
    /// which case turn events reach only the connection that initiated them, as
    /// before (graceful degradation). The dispatcher registers each connection's
    /// sink here, applies `SubscribeConversations`, and fans a turn's events to
    /// every other connection viewing that conversation.
    fn conversation_subscriptions(
        &self,
    ) -> Option<Arc<crate::conversation_subs::ConversationSubscriptions>> {
        None
    }

    /// Register a `SendMessage` request as a background task and
    /// return the new task id synchronously. The body runs in the
    /// background; events stream through `sink` as before.
    ///
    /// Returning `Some(task_id)` tells the dispatcher to reply with
    /// `CommandResult::SendMessageAck { task_id }`. Returning `None`
    /// (the default) tells the dispatcher to fall back to the legacy
    /// bare-Ack flow and dispatch `handle_send_message_with_override`
    /// directly — used by tests and single-tenant deploys that don't
    /// attach a registry to the handler.
    #[allow(clippy::too_many_arguments)]
    async fn start_send_message(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        let _ = (
            conversation_id,
            content,
            override_selection,
            system_refinement,
            request_id,
            idempotency_key,
            sink,
        );
        Ok(None)
    }

    /// Called by the dispatcher when a connection closes, inside the
    /// connection's `with_session_id` scope (#261). The default is a no-op;
    /// a handler that tracks per-session state (client-local tool
    /// registrations) overrides this to evict the ending session's entries
    /// via `current_session_id()`. Handlers without per-session state (tests,
    /// single-tenant deploys) need do nothing.
    async fn on_session_end(&self) {}
}

/// Minimal sink for emitting canonical events.
///
/// Implemented by protocol adapters to forward events to connected clients.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    /// Returns `false` when the sink is no longer available (e.g. disconnected client).
    async fn emit(&self, event: api::Event) -> bool;
}

const STREAM_EVENT_BUFFER: usize = 64;

/// Forward a streaming event into the per-turn event channel from the core
/// chunk callback. The boolean return feeds the LLM adapters' cooperative
/// abort contract: `false` means "the consumer is gone, stop streaming".
///
/// DA-3: `Full` and `Closed` must not be conflated. A momentarily full
/// buffer (slow client, 64-event cap) is backpressure — the delta is
/// dropped (the final stored message converges the client) but the stream
/// stays alive. Only a closed channel means the consumer is really gone.
fn forward_stream_event(tx: &tokio::sync::mpsc::Sender<api::Event>, event: api::Event) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    match tx.try_send(event) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) => {
            tracing::debug!("stream event buffer full; dropping one delta (client will converge)");
            true
        }
        Err(TrySendError::Closed(_)) => false,
    }
}

#[cfg(test)]
mod forward_stream_event_tests {
    use super::*;

    fn delta(chunk: &str) -> api::Event {
        api::Event::AssistantDelta {
            conversation_id: "c1".into(),
            request_id: "r1".into(),
            chunk: chunk.into(),
        }
    }

    #[tokio::test]
    async fn normal_send_delivers_event_and_continues() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<api::Event>(4);
        assert!(forward_stream_event(&tx, delta("hello")));
        match rx.try_recv() {
            Ok(api::Event::AssistantDelta { chunk, .. }) => assert_eq!(chunk, "hello"),
            other => panic!("expected the delta to be delivered, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn full_channel_is_backpressure_not_disconnect() {
        // DA-3: a momentarily full event buffer must NOT be treated as a
        // client disconnect. Returning `false` makes the LLM adapter abort
        // the stream cooperatively and the truncated text gets stored as the
        // final assistant message. The chunk may be dropped (the completed
        // message converges the client), but the stream must stay alive.
        let (tx, _rx) = tokio::sync::mpsc::channel::<api::Event>(1);
        assert!(forward_stream_event(&tx, delta("fills the buffer")));
        assert!(
            forward_stream_event(&tx, delta("overflow chunk")),
            "a full channel must keep the stream alive (drop the chunk), \
             not abort the turn as if the client disconnected"
        );
    }

    #[tokio::test]
    async fn closed_channel_aborts_the_stream() {
        // The receiver is gone for good — this IS a disconnect, and the
        // adapter should stop streaming.
        let (tx, rx) = tokio::sync::mpsc::channel::<api::Event>(1);
        drop(rx);
        assert!(
            !forward_stream_event(&tx, delta("nobody listening")),
            "a closed channel means the consumer is gone; the stream must abort"
        );
    }
}

pub struct DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    assistant: Arc<A>,
    conversations: Arc<C>,
    settings: Arc<S>,
    connections: Arc<N>,
    knowledge: Arc<K>,
    /// Optional registry used to track foreground turns as
    /// [`api::TaskKind::Conversation`] background tasks (#111).
    ///
    /// When `None`, `handle_send_message_with_override` runs the turn
    /// inline as it did before #111 — this keeps single-tenant tests
    /// and pre-registry call sites working unchanged. When `Some`, the
    /// turn body spawns through the registry so the user has a Cancel
    /// button to trip and the process-manager UI has something to show.
    registry: Option<Arc<BackgroundTaskRegistry>>,
    /// Optional per-conversation scratchpad closures (#190) so clients can
    /// read/write/delete a conversation's notes. Threaded as closures (not a
    /// `dyn ScratchpadStore`, which isn't object-safe). When `None`, the
    /// scratchpad commands return a clear "not configured" error. The daemon
    /// wraps the mutating closures so each emits `Event::ScratchpadChanged`.
    scratchpad_write: Option<ScratchpadWriteFn>,
    scratchpad_get_many: Option<ScratchpadGetManyFn>,
    scratchpad_list: Option<ScratchpadListFn>,
    scratchpad_delete_many: Option<ScratchpadDeleteManyFn>,
    scratchpad_clear: Option<ScratchpadClearFn>,
    /// Optional idempotency-key store (#204). When attached, a `SendMessage`
    /// carrying an `idempotency_key` whose turn already completed replays the
    /// stored reply instead of re-running the LLM/tools (crash-safe
    /// completed-dedup). `None` (no DB, tests) makes the key a no-op so
    /// `SendMessage` behaves exactly as before.
    idempotency: Option<Arc<dyn IdempotencyKeyStore>>,
    /// In-memory index of live keyed turns (#204 phase 2). A `SendMessage`
    /// carrying an `idempotency_key` whose original is still running in this
    /// process re-attaches to that live turn (replay buffered chunks + forward
    /// live) instead of running a second turn. Always present (no DB needed);
    /// only consulted when a request carries a key.
    inflight: Arc<InFlightRegistry>,
    /// Shared client-tool coordinator + turn-state store (#107 / #234). When
    /// attached, `RegisterClientTools` / `ClientToolResult` are served and a
    /// per-turn [`CoordinatorClientToolPort`] is installed around each
    /// send-turn so the LLM can call client-local tools (suspending the turn
    /// and resuming on the client's result). `None` (the default) keeps the
    /// pre-#234 behaviour: the two commands return `Unsupported` and every
    /// tool is server-side.
    client_tools: Option<ClientToolWiring>,
    /// Optional per-connection conversation-subscription registry (#1 live
    /// multi-client sync). When attached, the dispatcher fans a turn's events to
    /// every other connection viewing that conversation. `None` keeps the prior
    /// behaviour: turn events reach only the initiating connection.
    conversation_subs: Option<Arc<crate::conversation_subs::ConversationSubscriptions>>,
    /// Optional on-demand knowledge-maintenance service (dream-cycle controls).
    /// When attached, `StartKnowledgeMaintenance` spawns the requested pass
    /// (extraction / consolidation / embedding recompute) as a tracked,
    /// cancellable background task. `None` (no DB / tests) makes the command
    /// return a clear "not configured" error. Held as a trait object (the port
    /// is `async_trait`) so no extra generic threads through the handler.
    maintenance: Option<Arc<dyn KnowledgeMaintenanceService>>,
}

/// The handler-resident client-tool dependencies (#234). Both halves are
/// daemon-lifetime singletons: one `ClientToolCoordinator` shared across all
/// turns, paired with the `TurnStateStore` the coordinator writes its
/// suspend/resolve transitions into.
#[derive(Clone)]
struct ClientToolWiring {
    coord: Arc<ClientToolCoordinator>,
    store: Arc<dyn TurnStateStore>,
}

impl<A, C, S, N, K> DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    pub fn new(
        assistant: Arc<A>,
        conversations: Arc<C>,
        settings: Arc<S>,
        connections: Arc<N>,
        knowledge: Arc<K>,
    ) -> Self {
        Self {
            assistant,
            conversations,
            settings,
            connections,
            knowledge,
            registry: None,
            scratchpad_write: None,
            scratchpad_get_many: None,
            scratchpad_list: None,
            scratchpad_delete_many: None,
            scratchpad_clear: None,
            idempotency: None,
            inflight: Arc::new(InFlightRegistry::default()),
            client_tools: None,
            conversation_subs: None,
            maintenance: None,
        }
    }

    /// Attach the shared client-tool coordinator + turn-state store (#234) so
    /// `RegisterClientTools` / `ClientToolResult` are served and send-turns
    /// offer the connection's client-local tools to the LLM (suspending the
    /// turn and resuming on the client's result). The daemon wires this in
    /// `main.rs` to one shared `ClientToolCoordinator` and an
    /// `InMemoryTurnStateStore`; callers that skip it keep the prior
    /// server-side-only behaviour.
    pub fn with_client_tool_coordinator(
        mut self,
        coord: Arc<ClientToolCoordinator>,
        store: Arc<dyn TurnStateStore>,
    ) -> Self {
        self.client_tools = Some(ClientToolWiring { coord, store });
        self
    }

    /// Attach the per-conversation scratchpad closures (#190) so the
    /// `GetConversationScratchpad` / `SetScratchpadNote` / `DeleteScratchpadNotes`
    /// commands are served. The daemon passes the same store-backed closures the
    /// builtin tools use (with mutating ones wrapped to emit `ScratchpadChanged`).
    pub fn with_scratchpad(
        mut self,
        write: ScratchpadWriteFn,
        get_many: ScratchpadGetManyFn,
        list: ScratchpadListFn,
        delete_many: ScratchpadDeleteManyFn,
        clear: ScratchpadClearFn,
    ) -> Self {
        self.scratchpad_write = Some(write);
        self.scratchpad_get_many = Some(get_many);
        self.scratchpad_list = Some(list);
        self.scratchpad_delete_many = Some(delete_many);
        self.scratchpad_clear = Some(clear);
        self
    }

    /// Attach a [`BackgroundTaskRegistry`] so foreground send-message
    /// turns register as `TaskKind::Conversation` tasks. The daemon
    /// wires this in `main.rs` to a shared registry instance; tests
    /// that don't need the registry skip this step.
    pub fn with_registry(mut self, registry: Arc<BackgroundTaskRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Attach the on-demand knowledge-maintenance service (dream-cycle controls)
    /// so `StartKnowledgeMaintenance` spawns extraction / consolidation /
    /// embedding-recompute passes as tracked, cancellable background tasks. The
    /// daemon wires this in `main.rs`, sharing the same implementation the
    /// periodic timers use; callers that skip it make the command a clear error.
    pub fn with_maintenance_service(
        mut self,
        service: Arc<dyn KnowledgeMaintenanceService>,
    ) -> Self {
        self.maintenance = Some(service);
        self
    }

    /// Broadcast a knowledge-base change to the calling user's subscribed
    /// connections so every open knowledge panel refetches live. No-op without a
    /// registry attached. Used after a manual create/update/delete so a second
    /// connected client stays in sync (maintenance passes notify from within).
    fn notify_knowledge_changed(&self) {
        if let Some(reg) = &self.registry {
            reg.notify_knowledge_changed(&desktop_assistant_core::ports::auth::current_user_id());
        }
    }

    /// Attach the per-connection conversation-subscription registry (#1) so the
    /// dispatcher can fan a turn's events to every other connection viewing that
    /// conversation (live multi-client sync). The daemon wires one shared
    /// instance in `main.rs`; callers that skip it keep turn events scoped to
    /// the initiating connection.
    pub fn with_conversation_subscriptions(
        mut self,
        subs: Arc<crate::conversation_subs::ConversationSubscriptions>,
    ) -> Self {
        self.conversation_subs = Some(subs);
        self
    }

    /// Wrap a fresh turn's event sink so each event also fans to other
    /// connections viewing the conversation (#1 live multi-client sync). A no-op
    /// when the subscription registry isn't attached. The origin — this
    /// request's connection session — is excluded from the fan-out (it receives
    /// the events through `sink` directly). Read the session here, before any
    /// spawn, while the dispatcher's per-request session scope is still active.
    /// Applied only to the fresh-turn path, never to idempotent replays, so a
    /// retried turn's stored reply is not re-broadcast to viewers.
    fn fanout_sink(&self, sink: Arc<dyn EventSink>) -> Arc<dyn EventSink> {
        match &self.conversation_subs {
            Some(subs) => Arc::new(crate::conversation_subs::FanOutSink::new(
                sink,
                Arc::clone(subs),
                desktop_assistant_core::ports::session::current_session_id()
                    .as_str()
                    .to_string(),
                // #432: capture the origin's user so fan-out only reaches this
                // user's other connections, never a different tenant that
                // subscribed to the conversation id.
                desktop_assistant_core::ports::auth::current_user_id()
                    .as_str()
                    .to_string(),
            )),
            None => sink,
        }
    }

    /// Broadcast a conversation-list change (#1) to the calling user's
    /// connections so every client's sidebar refreshes, regardless of which
    /// client made the change. No-op without a registry attached.
    fn notify_conversation_list_changed(&self, conversation_id: &str) {
        if let Some(reg) = &self.registry {
            reg.notify_conversation_list_changed(
                &desktop_assistant_core::ports::auth::current_user_id(),
                conversation_id,
            );
        }
    }

    /// Borrow the registry, if one is attached. Public so #112/#113 can
    /// reach the same instance the foreground send path uses; for #111
    /// only the foreground path itself needs it.
    pub fn registry(&self) -> Option<&Arc<BackgroundTaskRegistry>> {
        self.registry.as_ref()
    }

    /// Attach an [`IdempotencyKeyStore`] so `SendMessage`s that carry an
    /// `idempotency_key` are deduplicated: a retry of a turn that already
    /// completed replays the stored reply instead of re-running it (#204).
    /// The daemon wires this in `main.rs` when a database is available;
    /// callers without one skip it and keys become no-ops.
    pub fn with_idempotency_store(mut self, store: Arc<dyn IdempotencyKeyStore>) -> Self {
        self.idempotency = Some(store);
        self
    }

    /// Completed-dedup helper (#204). When an idempotency key is present, a
    /// store is attached, and that `(current user, conversation, key)`
    /// already holds a committed reply, replay it to `sink` and return
    /// `true` so the caller skips dispatching a fresh turn. Returns `false`
    /// (caller runs the turn normally) when there is no key, no store, or no
    /// completed row.
    async fn try_replay_idempotent(
        &self,
        conversation_id: &str,
        idempotency_key: Option<&str>,
        request_id: &str,
        content: &str,
        sink: &Arc<dyn EventSink>,
    ) -> ApiResult<bool> {
        let (Some(key), Some(store)) = (idempotency_key, self.idempotency.as_ref()) else {
            return Ok(false);
        };
        let Some(response) = store
            .lookup_completed(conversation_id, key)
            .await
            .map_err(Self::map_core_err)?
        else {
            return Ok(false);
        };
        // Echo the retry's key on the opening `UserMessageAdded` (#570) so the
        // completed-replay path matches the fresh-turn / re-attach paths.
        replay_completed_response(
            sink,
            conversation_id,
            request_id,
            content,
            Some(key.to_string()),
            response,
        )
        .await;
        Ok(true)
    }

    /// Spawn a registry task that re-attaches `sink` to a live in-flight turn:
    /// replay its buffered events, then forward live events until it ends
    /// (#204 phase 2).
    async fn spawn_reattach(
        registry: &BackgroundTaskRegistry,
        user_id: UserId,
        conversation_id: &str,
        request_id: &str,
        sink: Arc<dyn EventSink>,
        turn: Arc<InFlightTurn>,
    ) -> api::TaskId {
        let (replay, rx) = turn.snapshot_and_subscribe().await;
        let kind = api::TaskKind::Conversation {
            conversation_id: conversation_id.to_string(),
        };
        let title = format!("Conversation: {conversation_id}");
        // Restamp forwarded events with this (the re-attacher's) request id so
        // the retrying client correlates the stream like a normal turn.
        let request_id = request_id.to_string();
        registry.spawn(user_id, kind, title, move |_ctx| async move {
            forward_inflight(replay, rx, request_id, sink).await;
            Ok(())
        })
    }

    /// Spawn a registry task that replays a completed turn's stored reply to
    /// `sink` (#204 phase 1), keyed by the retry's own `request_id`. The replay
    /// opens with a `UserMessageAdded` echoing the retry's `idempotency_key`
    /// (#570) so a retrying client Case-0 dedupes its optimistic bubble.
    #[allow(clippy::too_many_arguments)]
    fn spawn_completed_replay(
        registry: &BackgroundTaskRegistry,
        user_id: UserId,
        conversation_id: &str,
        request_id: &str,
        content: String,
        echo_idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
        response: String,
    ) -> api::TaskId {
        let kind = api::TaskKind::Conversation {
            conversation_id: conversation_id.to_string(),
        };
        let title = format!("Conversation: {conversation_id}");
        let conv = conversation_id.to_string();
        let req = request_id.to_string();
        registry.spawn(user_id, kind, title, move |_ctx| async move {
            replay_completed_response(&sink, &conv, &req, &content, echo_idempotency_key, response)
                .await;
            Ok(())
        })
    }

    fn map_core_err<E: ToString>(e: E) -> ApiError {
        ApiError::Core(e.to_string())
    }

    fn normalize_optional_string(value: Option<String>) -> Option<String> {
        value.and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
    }

    async fn get_config(&self) -> ApiResult<api::Config> {
        let embeddings = self
            .settings
            .get_embeddings_settings()
            .await
            .map_err(Self::map_core_err)?;
        let persistence = self
            .settings
            .get_persistence_settings()
            .await
            .map_err(Self::map_core_err)?;
        let personality = self
            .settings
            .get_personality_settings()
            .await
            .map_err(Self::map_core_err)?;

        Ok(api::Config {
            embeddings: api::EmbeddingsSettingsView {
                connector: embeddings.connector,
                model: embeddings.model,
                base_url: embeddings.base_url,
                has_api_key: embeddings.has_api_key,
                available: embeddings.available,
                is_default: embeddings.is_default,
                health: match embeddings.health {
                    EmbeddingHealth::Disabled => api::EmbeddingHealth::Disabled,
                    EmbeddingHealth::Ok => api::EmbeddingHealth::Ok,
                    EmbeddingHealth::Unavailable { reason } => {
                        api::EmbeddingHealth::Unavailable { reason }
                    }
                    EmbeddingHealth::Unknown => api::EmbeddingHealth::Unknown,
                },
            },
            persistence: api::PersistenceSettingsView {
                enabled: persistence.enabled,
                remote_url: persistence.remote_url,
                remote_name: persistence.remote_name,
                push_on_update: persistence.push_on_update,
            },
            // `PersonalitySettingsView` is the core `Personality` type in both
            // the port and the api-model, so this is the identity conversion.
            personality,
        })
    }

    async fn set_config(&self, changes: api::ConfigChanges) -> ApiResult<api::Config> {
        let current = self.get_config().await?;
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

        let embeddings_changed = embeddings_connector.is_some()
            || embeddings_model.is_some()
            || embeddings_base_url.is_some();
        if embeddings_changed {
            let connector = if embeddings_connector.is_some() {
                Self::normalize_optional_string(embeddings_connector)
            } else if current.embeddings.is_default {
                None
            } else {
                Some(current.embeddings.connector.clone())
            };

            let model = if embeddings_model.is_some() {
                Self::normalize_optional_string(embeddings_model)
            } else {
                Some(current.embeddings.model.clone())
            };

            let base_url = if embeddings_base_url.is_some() {
                Self::normalize_optional_string(embeddings_base_url)
            } else {
                Some(current.embeddings.base_url.clone())
            };

            self.settings
                .set_embeddings_settings(connector, model, base_url)
                .await
                .map_err(Self::map_core_err)?;
        }

        let persistence_changed = persistence_enabled.is_some()
            || persistence_remote_url.is_some()
            || persistence_remote_name.is_some()
            || persistence_push_on_update.is_some();
        if persistence_changed {
            let enabled = persistence_enabled.unwrap_or(current.persistence.enabled);
            let remote_url = if persistence_remote_url.is_some() {
                Self::normalize_optional_string(persistence_remote_url)
            } else {
                Some(current.persistence.remote_url.clone())
            };
            let remote_name = if persistence_remote_name.is_some() {
                Self::normalize_optional_string(persistence_remote_name)
            } else {
                Some(current.persistence.remote_name.clone())
            };
            let push_on_update =
                persistence_push_on_update.unwrap_or(current.persistence.push_on_update);

            self.settings
                .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                .await
                .map_err(Self::map_core_err)?;
        }

        // Personality (#226): apply per-trait overrides over the current value.
        // Each `None` leaves that trait unchanged; only a present level updates
        // it. We write the whole struct back so the daemon refreshes the
        // in-memory config the next send reads.
        let personality_changed = personality_professionalism.is_some()
            || personality_warmth.is_some()
            || personality_directness.is_some()
            || personality_enthusiasm.is_some()
            || personality_humor.is_some()
            || personality_sarcasm.is_some()
            || personality_pretentiousness.is_some();
        if personality_changed {
            let mut p = current.personality;
            if let Some(v) = personality_professionalism {
                p.professionalism = v;
            }
            if let Some(v) = personality_warmth {
                p.warmth = v;
            }
            if let Some(v) = personality_directness {
                p.directness = v;
            }
            if let Some(v) = personality_enthusiasm {
                p.enthusiasm = v;
            }
            if let Some(v) = personality_humor {
                p.humor = v;
            }
            if let Some(v) = personality_sarcasm {
                p.sarcasm = v;
            }
            if let Some(v) = personality_pretentiousness {
                p.pretentiousness = v;
            }
            self.settings
                .set_personality_settings(p)
                .await
                .map_err(Self::map_core_err)?;
        }

        self.get_config().await
    }
}

// ---- Conversion helpers between api-model wire types and core port types ----

fn api_connection_config_to_core(c: api::ConnectionConfigView) -> ConnectionConfigPayload {
    match c {
        api::ConnectionConfigView::Anthropic {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::Anthropic {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::OpenAi {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::OpenAi {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::OpenRouter {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::OpenRouter {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::Azure {
            base_url,
            api_key_env,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::Azure {
            base_url,
            api_key_env,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::Google {
            base_url,
            api_key_env,
            project,
            location,
            auth_mode,
            credentials_path,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::Google {
            base_url,
            api_key_env,
            project,
            location,
            auth_mode,
            credentials_path,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::Bedrock {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfigPayload::Bedrock {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        api::ConnectionConfigView::Ollama {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        } => ConnectionConfigPayload::Ollama {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        },
    }
}

/// Inverse of [`api_connection_config_to_core`]: map the non-secret core
/// payload to its wire view. No variant of either type carries a raw secret,
/// so this conversion cannot leak one.
fn core_connection_config_to_api(c: ConnectionConfigPayload) -> api::ConnectionConfigView {
    match c {
        ConnectionConfigPayload::Anthropic {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::Anthropic {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::OpenAi {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::OpenAi {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::OpenRouter {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::OpenRouter {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::Azure {
            base_url,
            api_key_env,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::Azure {
            base_url,
            api_key_env,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::Google {
            base_url,
            api_key_env,
            project,
            location,
            auth_mode,
            credentials_path,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::Google {
            base_url,
            api_key_env,
            project,
            location,
            auth_mode,
            credentials_path,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::Bedrock {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => api::ConnectionConfigView::Bedrock {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        },
        ConnectionConfigPayload::Ollama {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        } => api::ConnectionConfigView::Ollama {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        },
    }
}

fn core_connection_to_api_view(
    v: desktop_assistant_core::ports::inbound::ConnectionView,
) -> api::ConnectionView {
    api::ConnectionView {
        id: v.id,
        connector_type: v.connector_type,
        display_label: v.display_label,
        availability: match v.availability {
            ConnectionAvailability::Ok => api::ConnectionAvailability::Ok,
            ConnectionAvailability::Unavailable { reason } => {
                api::ConnectionAvailability::Unavailable { reason }
            }
        },
        has_credentials: v.has_credentials,
        config: v.config.map(core_connection_config_to_api),
    }
}

fn core_model_listing_to_api(
    l: desktop_assistant_core::ports::inbound::ModelListing,
) -> api::ModelListing {
    api::ModelListing {
        connection_id: l.connection_id,
        connection_label: l.connection_label,
        notices: l
            .notices
            .into_iter()
            .map(core_model_listing_notice_to_api)
            .collect(),
        model: api::ModelInfoView {
            id: l.model.id,
            display_name: l.model.display_name,
            context_limit: l.model.context_limit,
            capabilities: api::ModelCapabilitiesView {
                reasoning: l.model.capabilities.reasoning,
                vision: l.model.capabilities.vision,
                tools: l.model.capabilities.tools,
                // `embedding` is derived from `kind` (the single source of
                // truth) and kept on the wire for clients that read it today.
                embedding: l.model.capabilities.is_embedding(),
                kind: core_model_kind_to_api(l.model.capabilities.kind),
            },
        },
    }
}

/// Map a connector's model-listing notice onto its wire mirror so a partial
/// listing stays visible to clients instead of dying at the daemon boundary.
/// Map the core model-kind axis onto its wire mirror (#647). `api-model`
/// deliberately does not depend on `core`, so the wire carries its own
/// [`api::ModelKindView`] and this is the one translation point.
fn core_model_kind_to_api(k: desktop_assistant_core::ports::llm::ModelKind) -> api::ModelKindView {
    use desktop_assistant_core::ports::llm::ModelKind;
    match k {
        ModelKind::Generative => api::ModelKindView::Generative,
        ModelKind::Embedding => api::ModelKindView::Embedding,
        ModelKind::Unknown => api::ModelKindView::Unknown,
    }
}

fn core_model_listing_notice_to_api(
    n: desktop_assistant_core::ports::llm::ModelListingNotice,
) -> api::ModelListingNoticeView {
    use desktop_assistant_core::ports::llm::ModelListingNoticeKind;
    api::ModelListingNoticeView {
        kind: match n.kind {
            ModelListingNoticeKind::PartialCatalog => {
                api::ModelListingNoticeKindView::PartialCatalog
            }
        },
        summary: n.summary,
        detail: n.detail,
        required_permission: n.required_permission,
    }
}

fn core_purpose_to_api(p: PurposeConfigPayload) -> api::PurposeConfigView {
    api::PurposeConfigView {
        connection: p.connection,
        model: p.model,
        effort: p.effort,
        max_context_tokens: p.max_context_tokens,
    }
}

fn api_purpose_to_core(p: api::PurposeConfigView) -> PurposeConfigPayload {
    PurposeConfigPayload {
        connection: p.connection,
        model: p.model,
        effort: p.effort,
        max_context_tokens: p.max_context_tokens,
    }
}

fn model_selection_to_view(sel: ConversationModelSelection) -> api::ConversationModelSelectionView {
    api::ConversationModelSelectionView {
        connection_id: sel.connection_id,
        model_id: sel.model_id,
        effort: sel.effort,
    }
}

fn dispatch_warning_to_api(w: DispatchWarning) -> api::ConversationWarning {
    match w {
        DispatchWarning::DanglingModelSelection {
            previous,
            fallback_to,
        } => api::ConversationWarning::DanglingModelSelection {
            previous_selection: api::ConversationModelSelectionView {
                connection_id: previous.connection_id,
                model_id: previous.model_id,
                effort: previous.effort,
            },
            fallback_to: api::ConversationModelSelectionView {
                connection_id: fallback_to.connection_id,
                model_id: fallback_to.model_id,
                effort: fallback_to.effort,
            },
        },
    }
}

/// Map a [`ClientToolResolutionError`] (#234) to the wire-level
/// [`ApiError`]. A missing/cross-user turn is reported as `NotFound` (the
/// existence-hiding rule of #105 — a cross-user probe can't tell "no such
/// turn" from "not yours"); a `tool_call_id` mismatch or a malformed payload
/// is a caller mistake reported as `Core`; storage failures surface as `Core`.
fn map_client_tool_resolution_err(e: ClientToolResolutionError) -> ApiError {
    match e {
        ClientToolResolutionError::TurnNotFound { .. } => ApiError::NotFound,
        ClientToolResolutionError::ToolCallIdMismatch { .. }
        | ClientToolResolutionError::MalformedResult(_)
        | ClientToolResolutionError::Storage(_) => ApiError::Core(e.to_string()),
    }
}

fn knowledge_entry_to_view(e: KnowledgeEntry) -> api::KnowledgeEntryView {
    api::KnowledgeEntryView {
        id: e.id,
        content: e.content,
        tags: e.tags,
        metadata: e.metadata,
        created_at: e.created_at,
        updated_at: e.updated_at,
    }
}

fn scratchpad_note_to_view(n: ScratchpadNote) -> api::ScratchpadNoteView {
    api::ScratchpadNoteView {
        id: n.id,
        key: n.key,
        content: n.content,
        note_type: n.note_type,
        sequence: n.sequence,
        done: n.done,
        updated_at: n.updated_at,
    }
}

#[async_trait::async_trait]
impl<A, C, S, N, K> AssistantApiHandler for DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    /// Evict the ending connection's client-local tool registrations (#261).
    /// Runs inside the dispatcher's `with_session_id` scope, so the
    /// coordinator's `clear_session` reads the correct session from the
    /// task-local. No-op when client tools aren't wired (no coordinator).
    async fn on_session_end(&self) {
        if let Some(wiring) = self.client_tools.as_ref() {
            wiring.coord.clear_session();
        }
    }

    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::Ping => Ok(api::CommandResult::Pong {
                value: self.assistant.ping().to_string(),
            }),

            api::Command::GetStatus => Ok(api::CommandResult::Status(api::Status {
                version: self.assistant.version().to_string(),
            })),

            api::Command::GetConfig => {
                let config = self.get_config().await?;
                Ok(api::CommandResult::Config(config))
            }

            api::Command::SetConfig { changes } => {
                let config = self.set_config(changes).await?;
                Ok(api::CommandResult::Config(config))
            }

            api::Command::CreateConversation { title, tags } => {
                let conv = self
                    .conversations
                    .create_conversation(title, tags)
                    .await
                    .map_err(Self::map_core_err)?;
                let id = conv.id.0;
                self.notify_conversation_list_changed(&id);
                Ok(api::CommandResult::ConversationId { id })
            }

            api::Command::ListConversations {
                max_age_days,
                include_archived,
            } => {
                let list = self
                    .conversations
                    .list_conversations(max_age_days, include_archived)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Conversations(
                    list.into_iter()
                        .map(|s| api::ConversationSummary {
                            id: s.id.0,
                            title: s.title,
                            message_count: s.message_count as u32,
                            updated_at: s.updated_at,
                            archived: s.archived,
                            tags: s.tags,
                        })
                        .collect(),
                ))
            }

            api::Command::GetConversation { id } => {
                let conv_id = desktop_assistant_core::domain::ConversationId::from(id.as_str());
                let conv = self
                    .conversations
                    .get_conversation(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?;
                let model_selection = self
                    .conversations
                    .get_conversation_model_selection(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?
                    .map(model_selection_to_view);
                // #227: surface the conversation's personality override
                // alongside the model selection. `None` = no override (global).
                let conversation_personality = self
                    .conversations
                    .get_conversation_personality(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?;

                Ok(api::CommandResult::Conversation(api::ConversationView {
                    id: conv.id.0,
                    title: conv.title,
                    messages: conv
                        .messages
                        .into_iter()
                        .map(|m| api::MessageView {
                            id: m.id,
                            role: format!("{:?}", m.role).to_lowercase(),
                            content: m.content,
                            idempotency_key: m.idempotency_key,
                        })
                        .collect(),
                    warnings: Vec::new(),
                    model_selection,
                    conversation_personality,
                }))
            }

            api::Command::GetMessages {
                conversation_id,
                tail,
                after_count,
                include_roles,
            } => {
                let conv = self
                    .conversations
                    .get_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        conversation_id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                let all: Vec<api::MessageView> = conv
                    .messages
                    .into_iter()
                    .map(|m| api::MessageView {
                        id: m.id,
                        role: format!("{:?}", m.role).to_lowercase(),
                        content: m.content,
                        idempotency_key: m.idempotency_key,
                    })
                    .collect();
                Ok(api::CommandResult::Messages(window_messages(
                    all,
                    tail,
                    after_count,
                    &include_roles,
                )))
            }

            api::Command::SetConversationPersonality {
                conversation_id,
                personality,
            } => {
                let conv_id =
                    desktop_assistant_core::domain::ConversationId::from(conversation_id.as_str());
                self.conversations
                    .set_conversation_personality(&conv_id, personality)
                    .await
                    .map_err(Self::map_core_err)?;
                // Echo the stored value (cleared → empty/all-None) so the
                // client confirms the write, mirroring `SetScratchpadNote`.
                let stored = self
                    .conversations
                    .get_conversation_personality(&conv_id)
                    .await
                    .map_err(Self::map_core_err)?
                    .unwrap_or_default();
                Ok(api::CommandResult::ConversationPersonality(stored))
            }

            api::Command::DeleteConversation { id } => {
                self.conversations
                    .delete_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_conversation_list_changed(&id);
                Ok(api::CommandResult::Ack)
            }

            api::Command::RenameConversation { id, title } => {
                self.conversations
                    .rename_conversation(
                        &desktop_assistant_core::domain::ConversationId::from(id.as_str()),
                        title,
                    )
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_conversation_list_changed(&id);
                Ok(api::CommandResult::Ack)
            }

            api::Command::ArchiveConversation { id } => {
                self.conversations
                    .archive_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_conversation_list_changed(&id);
                Ok(api::CommandResult::Ack)
            }

            api::Command::UnarchiveConversation { id } => {
                self.conversations
                    .unarchive_conversation(&desktop_assistant_core::domain::ConversationId::from(
                        id.as_str(),
                    ))
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_conversation_list_changed(&id);
                Ok(api::CommandResult::Ack)
            }

            api::Command::ClearAllHistory => {
                let n = self
                    .conversations
                    .clear_all_history()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Cleared { deleted_count: n })
            }

            // Settings
            api::Command::SetApiKey { api_key } => {
                self.settings
                    .set_api_key(api_key)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetEmbeddingsSettings => {
                let s = self
                    .settings
                    .get_embeddings_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::EmbeddingsSettings(
                    api::EmbeddingsSettingsView {
                        connector: s.connector,
                        model: s.model,
                        base_url: s.base_url,
                        has_api_key: s.has_api_key,
                        available: s.available,
                        is_default: s.is_default,
                        health: match s.health {
                            EmbeddingHealth::Disabled => api::EmbeddingHealth::Disabled,
                            EmbeddingHealth::Ok => api::EmbeddingHealth::Ok,
                            EmbeddingHealth::Unavailable { reason } => {
                                api::EmbeddingHealth::Unavailable { reason }
                            }
                            EmbeddingHealth::Unknown => api::EmbeddingHealth::Unknown,
                        },
                    },
                ))
            }

            api::Command::SetEmbeddingsSettings {
                connector,
                model,
                base_url,
            } => {
                self.settings
                    .set_embeddings_settings(connector, model, base_url)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetConnectorDefaults { connector } => {
                let d = self
                    .settings
                    .get_connector_defaults(connector)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::ConnectorDefaults(
                    api::ConnectorDefaultsView {
                        llm_model: d.llm_model,
                        llm_base_url: d.llm_base_url,
                        backend_llm_model: d.backend_llm_model,
                        embeddings_model: d.embeddings_model,
                        embeddings_base_url: d.embeddings_base_url,
                        embeddings_available: d.embeddings_available,
                        hosted_tool_search_available: d.hosted_tool_search_available,
                    },
                ))
            }

            api::Command::GetPersistenceSettings => {
                let p = self
                    .settings
                    .get_persistence_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::PersistenceSettings(
                    api::PersistenceSettingsView {
                        enabled: p.enabled,
                        remote_url: p.remote_url,
                        remote_name: p.remote_name,
                        push_on_update: p.push_on_update,
                    },
                ))
            }

            api::Command::SetPersistenceSettings {
                enabled,
                remote_url,
                remote_name,
                push_on_update,
            } => {
                self.settings
                    .set_persistence_settings(enabled, remote_url, remote_name, push_on_update)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // Database / backend-tasks / WS-auth settings (bridge cutover 2/7,
            // #314). Each mirrors the in-process D-Bus method of the same name:
            // the getters return the same fields the D-Bus method returns (no
            // new secret exposure — see the `Command` doc-comments), and the
            // setters apply the same `empty-string clears` normalization the
            // D-Bus methods apply before delegating to `SettingsService`.
            api::Command::GetDatabaseSettings => {
                let s = self
                    .settings
                    .get_database_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::DatabaseSettings(
                    api::DatabaseSettingsView {
                        url: s.url,
                        max_connections: s.max_connections,
                    },
                ))
            }

            api::Command::SetDatabaseSettings {
                url,
                max_connections,
            } => {
                // Mirror the D-Bus `set_database_settings`: an empty/whitespace
                // url clears the configured URL.
                self.settings
                    .set_database_settings(normalize_empty(url), max_connections)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetBackendTasksSettings => {
                let s = self
                    .settings
                    .get_backend_tasks_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::BackendTasksSettings(
                    api::BackendTasksSettingsView {
                        has_separate_llm: s.has_separate_llm,
                        llm_connector: s.llm_connector,
                        llm_model: s.llm_model,
                        llm_base_url: s.llm_base_url,
                        dreaming_enabled: s.dreaming_enabled,
                        dreaming_interval_secs: s.dreaming_interval_secs,
                        archive_after_days: s.archive_after_days,
                    },
                ))
            }

            api::Command::SetBackendTasksSettings {
                llm_connector,
                llm_model,
                llm_base_url,
                dreaming_enabled,
                dreaming_interval_secs,
                archive_after_days,
            } => {
                // Mirror the D-Bus `set_backend_tasks_settings`: an empty
                // llm_connector clears the separate LLM override; empty
                // model/base_url normalize to "unset" too.
                self.settings
                    .set_backend_tasks_settings(
                        normalize_empty(llm_connector),
                        normalize_empty(llm_model),
                        normalize_empty(llm_base_url),
                        dreaming_enabled,
                        dreaming_interval_secs,
                        archive_after_days,
                    )
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::GetWsAuthSettings => {
                let s = self
                    .settings
                    .get_ws_auth_settings()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::WsAuthSettings(
                    api::WsAuthSettingsView {
                        methods: s.methods,
                        oidc_issuer: s.oidc_issuer,
                        oidc_auth_endpoint: s.oidc_auth_endpoint,
                        oidc_token_endpoint: s.oidc_token_endpoint,
                        oidc_client_id: s.oidc_client_id,
                        oidc_scopes: s.oidc_scopes,
                    },
                ))
            }

            api::Command::SetWsAuthSettings {
                methods,
                oidc_issuer,
                oidc_auth_endpoint,
                oidc_token_endpoint,
                oidc_client_id,
                oidc_scopes,
            } => {
                // Mirror the D-Bus `set_ws_auth_settings`: pass strings through
                // verbatim (the D-Bus method does no normalization here).
                self.settings
                    .set_ws_auth_settings(
                        methods,
                        oidc_issuer,
                        oidc_auth_endpoint,
                        oidc_token_endpoint,
                        oidc_client_id,
                        oidc_scopes,
                    )
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // Knowledge base management (issue #73)
            api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            } => {
                let entries = self
                    .knowledge
                    .list_entries(limit as usize, offset as usize, tag_filter)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntries(
                    entries.into_iter().map(knowledge_entry_to_view).collect(),
                ))
            }
            api::Command::GetKnowledgeEntry { id } => {
                let entry = self
                    .knowledge
                    .get_entry(id)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntry(
                    entry.map(knowledge_entry_to_view),
                ))
            }
            api::Command::SearchKnowledgeEntries {
                query,
                tag_filter,
                limit,
            } => {
                let entries = self
                    .knowledge
                    .search_entries(query, tag_filter, limit as usize)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeEntries(
                    entries.into_iter().map(knowledge_entry_to_view).collect(),
                ))
            }
            api::Command::CreateKnowledgeEntry {
                content,
                tags,
                metadata,
            } => {
                let entry = self
                    .knowledge
                    .create_entry(content, tags, metadata)
                    .await
                    .map_err(Self::map_core_err)?;
                // Live sync: let the user's other connected panels refetch.
                self.notify_knowledge_changed();
                Ok(api::CommandResult::KnowledgeEntryWritten(
                    knowledge_entry_to_view(entry),
                ))
            }
            api::Command::UpdateKnowledgeEntry {
                id,
                content,
                tags,
                metadata,
            } => {
                let entry = self
                    .knowledge
                    .update_entry(id, content, tags, metadata)
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_knowledge_changed();
                Ok(api::CommandResult::KnowledgeEntryWritten(
                    knowledge_entry_to_view(entry),
                ))
            }
            api::Command::DeleteKnowledgeEntry { id } => {
                self.knowledge
                    .delete_entry(id)
                    .await
                    .map_err(Self::map_core_err)?;
                self.notify_knowledge_changed();
                Ok(api::CommandResult::Ack)
            }
            api::Command::GetKnowledgeTrashCount => {
                let count = self
                    .knowledge
                    .trash_count()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::KnowledgeTrashCount {
                    count: count.min(u32::MAX as usize) as u32,
                })
            }
            api::Command::EmptyKnowledgeTrash => {
                let deleted = self
                    .knowledge
                    .empty_trash()
                    .await
                    .map_err(Self::map_core_err)?;
                // Only a reap that freed something is worth a refetch; an empty
                // trash is a no-op and must not churn every connected panel.
                if deleted > 0 {
                    self.notify_knowledge_changed();
                }
                Ok(api::CommandResult::KnowledgeTrashEmptied {
                    deleted_count: deleted.min(u32::MAX as usize) as u32,
                })
            }
            api::Command::StartKnowledgeMaintenance { op } => {
                // Run the requested pass as a tracked, cancellable background
                // task via the registry, returning its id immediately — never
                // inline, so the (serial, per-connection) dispatch loop is not
                // blocked for the multi-minute LLM/embedding work. Progress and
                // completion arrive as `Task*` events; the pass itself
                // broadcasts `KnowledgeChanged` as entries land.
                let service = self.maintenance.clone().ok_or_else(|| {
                    ApiError::Core("knowledge maintenance not configured".to_string())
                })?;
                let registry = self.registry.clone().ok_or_else(|| {
                    ApiError::Core(
                        "knowledge maintenance requires the background-task registry".to_string(),
                    )
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                let name = match op {
                    api::MaintenanceOp::Extraction => "Knowledge extraction",
                    api::MaintenanceOp::Consolidation => "Knowledge consolidation",
                    api::MaintenanceOp::RecalculateEmbeddings => "Recalculate embeddings",
                };
                let task_id = registry.spawn(
                    user_id,
                    api::TaskKind::Maintenance {
                        name: name.to_string(),
                    },
                    name.to_string(),
                    move |ctx| async move {
                        let token = ctx.token.clone();
                        let result = match op {
                            api::MaintenanceOp::Extraction => service.run_extraction(token).await,
                            api::MaintenanceOp::Consolidation => {
                                service.run_consolidation(token).await
                            }
                            api::MaintenanceOp::RecalculateEmbeddings => {
                                service.recalculate_embeddings(token).await
                            }
                        };
                        match result {
                            Ok(n) => {
                                ctx.logs.append(
                                    api::LogLevel::Info,
                                    api::LogCategory::Lifecycle,
                                    format!("{name}: {n} change(s)"),
                                    None,
                                );
                                Ok(())
                            }
                            Err(e) => Err(anyhow::anyhow!("{name} failed: {e}")),
                        }
                    },
                );
                Ok(api::CommandResult::MaintenanceTaskStarted { task_id: task_id.0 })
            }

            // MCP server management
            api::Command::ListMcpServers => {
                let servers = self
                    .settings
                    .list_mcp_servers()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::McpServers(
                    servers
                        .into_iter()
                        .map(|s| api::McpServerView {
                            name: s.name,
                            command: s.command,
                            args: s.args,
                            namespace: s.namespace,
                            enabled: s.enabled,
                            status: s.status,
                            tool_count: s.tool_count,
                            transport: s.transport,
                            target: s.target,
                            detail: s.detail,
                            configure_label: s.configure_label,
                            configure_command: s.configure_command,
                            auth_kind: s.auth_kind,
                            oauth_authorized: s.oauth_authorized,
                            oauth_account: s.oauth_account,
                            oauth_account_ref: s.oauth_account_ref,
                            oauth_scopes: s.oauth_scopes,
                            oauth_client_id: s.oauth_client_id,
                            oauth_token_url: s.oauth_token_url,
                            oauth_authorize_url: s.oauth_authorize_url,
                        })
                        .collect(),
                ))
            }

            api::Command::AddMcpServer {
                name,
                command,
                args,
                namespace,
                enabled,
            } => {
                self.settings
                    .add_mcp_server(name, command, args, namespace, enabled)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::RemoveMcpServer { name } => {
                self.settings
                    .remove_mcp_server(name)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::SetMcpServerEnabled { name, enabled } => {
                self.settings
                    .set_mcp_server_enabled(name, enabled)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::McpServerAction { action, server } => {
                let servers = self
                    .settings
                    .mcp_server_action(action, server)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::McpServers(
                    servers
                        .into_iter()
                        .map(|s| api::McpServerView {
                            name: s.name,
                            command: s.command,
                            args: s.args,
                            namespace: s.namespace,
                            enabled: s.enabled,
                            status: s.status,
                            tool_count: s.tool_count,
                            transport: s.transport,
                            target: s.target,
                            detail: s.detail,
                            configure_label: s.configure_label,
                            configure_command: s.configure_command,
                            auth_kind: s.auth_kind,
                            oauth_authorized: s.oauth_authorized,
                            oauth_account: s.oauth_account,
                            oauth_account_ref: s.oauth_account_ref,
                            oauth_scopes: s.oauth_scopes,
                            oauth_client_id: s.oauth_client_id,
                            oauth_token_url: s.oauth_token_url,
                            oauth_authorize_url: s.oauth_authorize_url,
                        })
                        .collect(),
                ))
            }

            api::Command::UpsertMcpServer { config_json } => {
                self.settings
                    .upsert_mcp_server(config_json)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::SetMcpSecret { id, value } => {
                self.settings
                    .set_mcp_secret(id, value.into_inner())
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::ListServiceAccounts => {
                let accounts = self
                    .settings
                    .list_service_accounts()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::ServiceAccounts(
                    accounts
                        .into_iter()
                        .map(|a| api::ServiceAccountView {
                            id: a.id,
                            display_name: a.display_name,
                            client_id: a.client_id,
                            client_secret_ref: a.client_secret_ref,
                            authorize_url: a.authorize_url,
                            token_url: a.token_url,
                            account: a.account,
                            refresh_token_ref: a.refresh_token_ref,
                            granted_scopes: a.granted_scopes,
                            authorized: a.authorized,
                            configure_label: a.configure_label,
                            configure_command: a.configure_command,
                        })
                        .collect(),
                ))
            }

            api::Command::UpsertServiceAccount { config_json } => {
                self.settings
                    .upsert_service_account(config_json)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::RemoveServiceAccount { id } => {
                self.settings
                    .remove_service_account(id)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // Named connections (#11)
            api::Command::ListConnections => {
                let views = self
                    .connections
                    .list_connections()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Connections(
                    views.into_iter().map(core_connection_to_api_view).collect(),
                ))
            }

            api::Command::CreateConnection { id, config } => {
                self.connections
                    .create_connection(id, api_connection_config_to_core(config))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::UpdateConnection { id, config } => {
                self.connections
                    .update_connection(id, api_connection_config_to_core(config))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::DeleteConnection { id, force } => {
                self.connections
                    .delete_connection(id, force)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::SetConnectionSecret { id, credential } => {
                // Unwrap the `Secret` only here, at the point the raw value is
                // handed to the daemon's write-only secret store.
                self.connections
                    .set_connection_secret(id, credential.into_inner())
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            api::Command::ListAvailableModels {
                connection_id,
                refresh,
            } => {
                let listings = self
                    .connections
                    .list_available_models(connection_id, refresh)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Models(
                    listings
                        .into_iter()
                        .map(core_model_listing_to_api)
                        .collect(),
                ))
            }

            api::Command::GetPurposes => {
                let p = self
                    .connections
                    .get_purposes()
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Purposes(Box::new(api::PurposesView {
                    interactive: p.interactive.map(core_purpose_to_api),
                    dreaming: p.dreaming.map(core_purpose_to_api),
                    consolidation: p.consolidation.map(core_purpose_to_api),
                    embedding: p.embedding.map(core_purpose_to_api),
                    titling: p.titling.map(core_purpose_to_api),
                    voice: p.voice.map(core_purpose_to_api),
                })))
            }

            api::Command::SetPurpose { purpose, config } => {
                self.connections
                    .set_purpose(purpose, api_purpose_to_core(config))
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Ack)
            }

            // Streamed commands are handled elsewhere.
            api::Command::SendMessage { .. } => Err(ApiError::Unsupported),

            // Background-task commands (#114) — read-side arms that
            // dispatch through the in-memory registry attached to the
            // handler. Subscribe/Unsubscribe Ack here at the
            // protocol level; the transport-side forwarder that
            // actually pushes `Event::Task*` to the connection is
            // hooked up by `dispatch_loop` via
            // [`Self::subscribe_user_events`] (transport-dispatch).
            api::Command::ListBackgroundTasks {
                include_finished,
                limit,
            } => {
                let registry = self.registry.as_ref().ok_or_else(|| {
                    ApiError::Core("background task registry not attached to handler".to_string())
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                Ok(api::CommandResult::BackgroundTasks(registry.list(
                    &user_id,
                    include_finished,
                    limit,
                )))
            }
            api::Command::GetBackgroundTask { id } => {
                let registry = self.registry.as_ref().ok_or_else(|| {
                    ApiError::Core("background task registry not attached to handler".to_string())
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                registry
                    .get(&user_id, &api::TaskId(id))
                    .map(api::CommandResult::BackgroundTask)
                    .ok_or(ApiError::NotFound)
            }
            api::Command::CancelBackgroundTask { id } => {
                let registry = self.registry.as_ref().ok_or_else(|| {
                    ApiError::Core("background task registry not attached to handler".to_string())
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                match registry.cancel(&user_id, &api::TaskId(id)) {
                    Ok(()) => Ok(api::CommandResult::Ack),
                    Err(TaskError::NotFound) => Err(ApiError::NotFound),
                    Err(TaskError::AlreadyTerminal) => Err(ApiError::AlreadyTerminal),
                }
            }
            api::Command::GetBackgroundTaskLogs {
                id,
                after_seq,
                limit,
            } => {
                let registry = self.registry.as_ref().ok_or_else(|| {
                    ApiError::Core("background task registry not attached to handler".to_string())
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                match registry.logs(
                    &user_id,
                    &api::TaskId(id),
                    after_seq.unwrap_or(0),
                    limit.unwrap_or(200),
                ) {
                    Ok((entries, next_seq)) => {
                        Ok(api::CommandResult::BackgroundTaskLogs { entries, next_seq })
                    }
                    Err(TaskError::NotFound) => Err(ApiError::NotFound),
                    Err(TaskError::AlreadyTerminal) => Err(ApiError::AlreadyTerminal),
                }
            }
            // Subscribe / Unsubscribe Ack at the handler level. The
            // dispatcher inspects [`Self::subscribe_user_events`] to
            // spawn (or tear down) the forwarder that actually streams
            // `Event::Task*` frames to the connection.
            api::Command::SubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
            api::Command::UnsubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
            // Dispatcher-handled (it owns the per-connection subscription set +
            // fan-out sink, #1); Ack here for the direct-call path.
            api::Command::SubscribeConversations { .. } => Ok(api::CommandResult::Ack),

            // Standalone agent spawn (issue #113). Creates a fresh
            // conversation scoped to the calling user, registers a
            // task in the in-memory registry, and returns its id
            // synchronously — the LLM call runs in the background.
            api::Command::SpawnStandaloneAgent {
                name,
                initial_prompt,
                override_selection,
                tools,
            } => {
                let registry = self.registry.clone().ok_or_else(|| {
                    ApiError::Core("background task registry not attached to handler".to_string())
                })?;
                let user_id = desktop_assistant_core::ports::auth::current_user_id();

                // Create the conversation first so the TaskKind can
                // carry a real conversation_id. The service runs under
                // the same task-local user scope `handle_command_for`
                // installed, so the new row is owned by the requesting
                // user (#105 contract).
                let title = format!("Standalone: {name}");
                let conv = self
                    .conversations
                    .create_conversation(title.clone(), vec![])
                    .await
                    .map_err(Self::map_core_err)?;
                let conversation_id = conv.id.0.clone();

                let name_for_kind = name.clone();
                let task_id = spawn_agent_conversation(
                    registry,
                    Arc::clone(&self.conversations),
                    AgentConversationSpec {
                        user_id,
                        name,
                        title,
                        initial_prompt,
                        override_selection,
                        tools,
                        conversation_id,
                        // Standalone runs are fire-and-forget at the
                        // protocol level — only `spawn_subagent` (#112)
                        // wires a sink to pull the final text back.
                        result_sink: None,
                        // A standalone agent is not a subagent: no session-pad
                        // scope; its scratchpad ops stay on its own conversation.
                        subagent_scope: None,
                        scratchpad_write: None,
                    },
                    move |conversation_id| api::TaskKind::Standalone {
                        name: name_for_kind,
                        conversation_id,
                    },
                );
                Ok(api::CommandResult::BackgroundTaskSpawned { id: task_id.0 })
            }

            // Client-side tool execution (issue #107). The default
            // handler doesn't carry a `ClientToolCoordinator`; a
            // composition that wants client-side execution wraps this
            // handler (or constructs `DefaultAssistantApiHandler` with
            // `with_client_tool_coordinator`) and overrides these arms.
            //
            // We reject explicitly so transports surface a clean
            // "feature not enabled" instead of silently dropping the
            // command. The wrapping handler in this crate's
            // `client_tools` module is the supported path.
            // Conversation scratchpad (issue #190). User scope is already
            // installed by the dispatcher's `with_user_id`, so the closures
            // (which read `current_user_id()`) are automatically scoped to the
            // connection's user and the requested conversation — no extra
            // scoping here. Mutations emit `Event::ScratchpadChanged` from the
            // daemon-wrapped closures.
            api::Command::GetConversationScratchpad {
                conversation_id,
                max_results,
            } => {
                let list = self
                    .scratchpad_list
                    .as_ref()
                    .ok_or_else(|| ApiError::Core("scratchpad not configured".to_string()))?;
                let limit = max_results
                    .map(|n| (n as usize).min(MAX_RESULTS_CEILING))
                    .unwrap_or(MAX_RESULTS_CEILING);
                let notes = list(conversation_id, None, limit)
                    .await
                    .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Scratchpad(
                    notes.into_iter().map(scratchpad_note_to_view).collect(),
                ))
            }
            api::Command::SetScratchpadNote {
                conversation_id,
                key,
                content,
                note_type,
                sequence,
                done,
            } => {
                let write = self
                    .scratchpad_write
                    .as_ref()
                    .ok_or_else(|| ApiError::Core("scratchpad not configured".to_string()))?;
                // Bound the write like the builtin tool path: non-empty key,
                // content within the per-note byte cap — clients must not be
                // able to write unbounded notes (#190).
                let key = key.trim().to_string();
                if key.is_empty() {
                    return Err(ApiError::Core(
                        "scratchpad note key must not be empty".to_string(),
                    ));
                }
                if content.len() > MAX_NOTE_BYTES {
                    return Err(ApiError::Core(format!(
                        "scratchpad note content exceeds {MAX_NOTE_BYTES} bytes"
                    )));
                }
                let note_type = if note_type.trim().is_empty() {
                    DEFAULT_NOTE_TYPE.to_string()
                } else {
                    note_type
                };
                let saved = write(
                    conversation_id,
                    vec![NewScratchpadNote {
                        key,
                        content,
                        note_type,
                        sequence,
                        done,
                    }],
                )
                .await
                .map_err(Self::map_core_err)?;
                Ok(api::CommandResult::Scratchpad(
                    saved.into_iter().map(scratchpad_note_to_view).collect(),
                ))
            }
            api::Command::DeleteScratchpadNotes {
                conversation_id,
                keys,
                all,
            } => {
                if all {
                    let clear = self
                        .scratchpad_clear
                        .as_ref()
                        .ok_or_else(|| ApiError::Core("scratchpad not configured".to_string()))?;
                    clear(conversation_id).await.map_err(Self::map_core_err)?;
                } else {
                    // Bound the key list so one command can't issue an
                    // unbounded delete (#190).
                    if keys.len() > MAX_KEYS_PER_CALL {
                        return Err(ApiError::Core(format!(
                            "too many keys (max {MAX_KEYS_PER_CALL})"
                        )));
                    }
                    let delete_many = self
                        .scratchpad_delete_many
                        .as_ref()
                        .ok_or_else(|| ApiError::Core("scratchpad not configured".to_string()))?;
                    delete_many(conversation_id, keys)
                        .await
                        .map_err(Self::map_core_err)?;
                }
                Ok(api::CommandResult::Ack)
            }

            // Client-side tool execution (#107 / #234). Served only when the
            // handler carries a client-tool coordinator (wired by the daemon);
            // otherwise the feature is off and we reject explicitly so
            // transports surface a clean "not enabled" rather than silently
            // dropping the command. User scope is already installed by the
            // dispatcher's `with_user_id`, so the coordinator's per-user
            // registration/resolution is automatically scoped to the
            // connection's user.
            api::Command::RegisterClientTools { tools } => {
                let wiring = self.client_tools.as_ref().ok_or(ApiError::Unsupported)?;
                let count = register_client_tools(&wiring.coord, &tools).await;
                Ok(api::CommandResult::ClientToolsRegistered { count })
            }
            api::Command::ClientToolResult {
                task_id,
                tool_call_id,
                result,
                error,
            } => {
                let wiring = self.client_tools.as_ref().ok_or(ApiError::Unsupported)?;
                // Exactly one of result/error should be set; both-None is a
                // malformed result the coordinator's validator rejects, which
                // we map to a clean request error.
                let payload: Result<String, String> = match (result, error) {
                    (Some(r), _) => Ok(r),
                    (None, Some(e)) => Err(e),
                    (None, None) => {
                        return Err(ApiError::Core(
                            "client tool result must populate exactly one of `result` or `error`"
                                .to_string(),
                        ));
                    }
                };
                let user_id = desktop_assistant_core::ports::auth::current_user_id();
                resolve_client_tool_result(
                    &wiring.coord,
                    &*wiring.store,
                    user_id,
                    task_id,
                    tool_call_id,
                    payload,
                )
                .await
                .map_err(map_client_tool_resolution_err)?;
                Ok(api::CommandResult::Ack)
            }
        }
    }

    async fn handle_send_message(
        &self,
        conversation_id: String,
        content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        self.handle_send_message_with_override(
            conversation_id,
            content,
            None,
            String::new(),
            request_id,
            None,
            sink,
        )
        .await
    }

    async fn handle_send_message_with_override(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        // Completed-dedup (#204): if this key already has a committed reply,
        // replay it and skip dispatching a fresh turn. (The registry-aware
        // streaming path does the same check in `start_send_message`; this
        // covers the no-registry / direct-call path.)
        if self
            .try_replay_idempotent(
                &conversation_id,
                idempotency_key.as_deref(),
                &request_id,
                &content,
                &sink,
            )
            .await?
        {
            return Ok(());
        }
        // The raw key echoed back on `UserMessageAdded` (#570) — captured before
        // the `zip` below MOVES `idempotency_key`, and kept separate from
        // `idempotency` because the echo must fire even with no store attached.
        let echo_idempotency_key = idempotency_key.clone();
        // Pair the store with the key (both-or-neither) so the turn body
        // records the reply on completion for a future retry.
        let idempotency = self.idempotency.clone().zip(idempotency_key);

        // Fan this fresh turn's events to other connections viewing the
        // conversation (#1); no-op without the registry. After the replay
        // shortcut above, so a replayed reply is not re-broadcast.
        let sink = self.fanout_sink(sink);

        // When a registry is attached we route the turn body through
        // it so the user has a cancellable, identifiable handle on the
        // running work (#111). When no registry is attached we fall back
        // to the inline path so single-tenant tests and other
        // direct-handler callers keep working unchanged.
        if let Some(registry) = self.registry.clone() {
            self.send_message_via_registry(
                registry,
                conversation_id,
                content,
                override_selection,
                system_refinement,
                request_id,
                idempotency,
                echo_idempotency_key,
                sink,
            )
            .await
        } else {
            // No-registry direct path (single-tenant tests / non-streaming
            // callers): there is no registry `task_id` to correlate a
            // `ClientToolResult` against, so client-tool execution is off
            // here. The live daemon always attaches a registry, so this never
            // applies to the real client-tool path.
            run_send_turn(
                Arc::clone(&self.conversations),
                conversation_id,
                content,
                override_selection,
                system_refinement,
                request_id,
                idempotency,
                echo_idempotency_key,
                sink,
                tokio_util::sync::CancellationToken::new(),
                None,
                None,
            )
            .await
            .map_err(Self::map_core_err)
        }
    }

    /// Register a `SendMessage` with the attached registry so the
    /// dispatcher can return `SendMessageAck { task_id }` synchronously
    /// while the body still runs in the background (#114). The body
    /// holds the same `sink`, so streaming chunks reach the connection
    /// exactly as before — only the wire-level ack changes shape.
    ///
    /// Returns `None` when no registry is attached so callers
    /// (transport-dispatch) can fall back to the legacy
    /// `handle_send_message_with_override` + bare-`Ack` flow.
    #[allow(clippy::too_many_arguments)]
    async fn start_send_message(
        &self,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        let Some(registry) = self.registry.clone() else {
            return Ok(None);
        };
        let user_id = desktop_assistant_core::ports::auth::current_user_id();

        // Idempotency shortcuts for a keyed SendMessage (#204), tried before
        // dispatching a fresh turn.
        if let Some(key) = idempotency_key.as_deref() {
            // (1) In-flight re-attach (phase 2): the original is still running
            // in this process — re-attach to its live stream (replay buffered
            // chunks, then forward live) instead of running a second turn.
            if let Some(turn) = self.inflight.get(user_id.as_str(), &conversation_id, key) {
                let task_id = Self::spawn_reattach(
                    &registry,
                    user_id,
                    &conversation_id,
                    &request_id,
                    Arc::clone(&sink),
                    turn,
                )
                .await;
                return Ok(Some(task_id));
            }
            // (2) Completed-dedup (phase 1): the original finished — replay its
            // committed reply. A registry replay-task keeps the dispatcher's
            // `SendMessageAck { task_id }`-then-stream ordering identical to a
            // normal turn.
            if let Some(store) = self.idempotency.as_ref() {
                let completed = store
                    .lookup_completed(&conversation_id, key)
                    .await
                    .map_err(Self::map_core_err)?;
                if let Some(response) = completed {
                    let task_id = Self::spawn_completed_replay(
                        &registry,
                        user_id,
                        &conversation_id,
                        &request_id,
                        content.clone(),
                        Some(key.to_string()),
                        Arc::clone(&sink),
                        response,
                    );
                    return Ok(Some(task_id));
                }
            }
        }

        // Fresh turn. For a keyed turn, claim an in-flight slot so a concurrent
        // duplicate can re-attach; if we lose that claim to a racing same-key
        // turn, re-attach to the winner rather than running twice.
        let inflight_slot = match idempotency_key.as_deref() {
            Some(key) => match self
                .inflight
                .register(user_id.as_str(), &conversation_id, key)
            {
                Some(turn) => Some((key.to_string(), turn)),
                None => {
                    if let Some(turn) = self.inflight.get(user_id.as_str(), &conversation_id, key) {
                        let task_id = Self::spawn_reattach(
                            &registry,
                            user_id,
                            &conversation_id,
                            &request_id,
                            Arc::clone(&sink),
                            turn,
                        )
                        .await;
                        return Ok(Some(task_id));
                    }
                    None
                }
            },
            None => None,
        };

        let conversations = Arc::clone(&self.conversations);
        // The raw key echoed back on `UserMessageAdded` (#570) — captured before
        // the `zip` below MOVES `idempotency_key`, and separate from
        // `idempotency` because the echo fires even with no store attached.
        let echo_idempotency_key = idempotency_key.clone();
        // Pair the store with the key (both-or-neither) so the turn body
        // records the reply on completion for a future retry (#204).
        let idempotency = self.idempotency.clone().zip(idempotency_key);
        // Client-tool wiring (#234), if attached, so the turn body can install
        // a per-turn `CoordinatorClientToolPort` (keyed on the registry
        // `task_id`) around the LLM loop.
        let client_tools = self.client_tools.clone();
        let kind = api::TaskKind::Conversation {
            conversation_id: conversation_id.clone(),
        };
        let title = format!("Conversation: {conversation_id}");

        // Fan this fresh turn's events to other connections viewing the
        // conversation (#1); no-op without the registry. After the reattach /
        // completed-replay shortcuts above, so only a genuinely fresh turn fans.
        let sink = self.fanout_sink(sink);

        // A keyed turn emits through a `TeeSink` so its events both reach the
        // caller and feed the in-flight hub for re-attachers; an unkeyed turn
        // emits straight to the caller's sink.
        let turn_sink: Arc<dyn EventSink> = match &inflight_slot {
            Some((_, turn)) => Arc::new(TeeSink::new(Arc::clone(&sink), Arc::clone(turn))),
            None => Arc::clone(&sink),
        };

        // Panic-safe free of the in-flight slot (#440). `turn_sink` (the
        // `TeeSink`) already holds the hub for the turn's lifetime, so the
        // slot's own hub `Arc` here is redundant and dropped now; the guard
        // removes the registry entry when the body ends by ANY path — normal
        // return or a panic that would otherwise orphan the slot and strand the
        // next same-key re-attacher on a dead hub.
        let inflight_guard = inflight_slot.map(|(key, _turn)| InFlightSlotGuard {
            index: Arc::clone(&self.inflight),
            user_id: user_id.as_str().to_string(),
            conversation_id: conversation_id.clone(),
            key,
        });
        let conv_id_for_body = conversation_id.clone();
        let request_id_for_body = request_id.clone();
        // Capture every request-scoped task-local before the spawn (#305 item
        // 4). `task_local`s don't cross `tokio::spawn`, so the body re-installs
        // the whole bundle in one call — user id (#105/#154), login session
        // (#261, so the per-turn `CoordinatorClientToolPort` resolves *this*
        // connection's registered client tools rather than the unscoped
        // bucket), transport + co-location + client label (#243/#248, so the
        // tool note tags localities). Bundling them removes the
        // missed-re-install bug class (#261) at this site.
        let request_scope = RequestScope::capture();

        // `registry.spawn` is sync; the body runs on its own `tokio::spawn` so
        // we can return the new task id immediately.
        let task_id = registry.spawn(user_id, kind, title, move |ctx| async move {
            ctx.logs.append(
                api::LogLevel::Info,
                api::LogCategory::Status,
                format!("send_prompt conversation_id={conv_id_for_body}"),
                None,
            );

            // Build the per-turn client-tool port from the registry task id so
            // a client-local tool call suspends THIS turn and correlates with
            // the client's `ClientToolResult` (#234). The port shares the
            // turn's `turn_sink`, so emitted `ClientToolCall` events reach the
            // same connection the response streams to.
            let client_tool_port = client_tools.map(|wiring| {
                Arc::new(CoordinatorClientToolPort::new(
                    wiring.coord,
                    wiring.store,
                    Arc::clone(&turn_sink),
                    ctx.task_id.clone(),
                    conv_id_for_body.clone(),
                ))
                    as Arc<dyn desktop_assistant_core::ports::client_tools::ClientToolPort>
            });

            let result = request_scope
                .scope(run_send_turn(
                    conversations,
                    conv_id_for_body,
                    content,
                    override_selection,
                    system_refinement,
                    request_id_for_body,
                    idempotency,
                    echo_idempotency_key,
                    turn_sink,
                    ctx.token.clone(),
                    client_tool_port,
                    Some(ctx.clone()),
                ))
                .await;

            // Free the in-flight slot now the turn is done. `run_send_turn` has
            // returned, so its `TeeSink` (the other hub owner) is dropped here
            // too — once the guard removes the slot the broadcast closes and any
            // re-attach streams finish. Dropping the guard here keeps the normal
            // path's timing identical to the old inline `remove`; on a panic the
            // guard (a captured local) is dropped during unwind instead (#440).
            drop(inflight_guard);

            match result {
                Ok(()) => Ok(()),
                Err(desktop_assistant_core::CoreError::Cancelled) => Ok(()),
                Err(other) => Err(anyhow::Error::new(other)),
            }
        });

        Ok(Some(task_id))
    }

    /// Hand the dispatcher (or any transport-level forwarder) a
    /// `broadcast::Receiver` so it can fan `Event::Task*` frames out
    /// to a single connection. Reads the per-request user id from the
    /// task-local installed by [`handle_command_for`] (#105).
    async fn subscribe_user_events(&self) -> Option<tokio::sync::broadcast::Receiver<api::Event>> {
        let registry = self.registry.as_ref()?;
        let user_id = desktop_assistant_core::ports::auth::current_user_id();
        Some(registry.subscribe(&user_id))
    }

    fn conversation_subscriptions(
        &self,
    ) -> Option<Arc<crate::conversation_subs::ConversationSubscriptions>> {
        self.conversation_subs.clone()
    }
}

impl<A, C, S, N, K> DefaultAssistantApiHandler<A, C, S, N, K>
where
    A: AssistantService + 'static,
    C: ConversationService + 'static,
    S: SettingsService + 'static,
    N: ConnectionsService + 'static,
    K: KnowledgeService + 'static,
{
    /// Body of `handle_send_message_with_override` when a registry is
    /// attached. Spawns the turn under the registry so the in-flight
    /// task is visible to `list`/`get`/`cancel` and so the user-scoped
    /// broadcast channel sees `TaskStarted`/`TaskCompleted` events; then
    /// awaits the spawned task to preserve the pre-#111 "blocking"
    /// contract that existing transport adapters rely on.
    // Why allow: forwards the streaming-send inputs (conversation target,
    // model override, system refinement, request id, sink) plus the registry
    // handle to the spawned turn body; no meaningful struct to bundle into.
    #[allow(clippy::too_many_arguments)]
    async fn send_message_via_registry(
        &self,
        registry: Arc<BackgroundTaskRegistry>,
        conversation_id: String,
        content: String,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        request_id: String,
        idempotency: Option<(Arc<dyn IdempotencyKeyStore>, String)>,
        echo_idempotency_key: Option<String>,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        let user_id = desktop_assistant_core::ports::auth::current_user_id();
        let conversations = Arc::clone(&self.conversations);
        let client_tools = self.client_tools.clone();
        let kind = api::TaskKind::Conversation {
            conversation_id: conversation_id.clone(),
        };
        let title = format!("Conversation: {conversation_id}");

        let conv_id_for_body = conversation_id.clone();
        let request_id_for_body = request_id.clone();
        let sink_for_body = Arc::clone(&sink);
        // Capture every request-scoped task-local before the spawn (#305 item
        // 4): `task_local`s don't cross `tokio::spawn`, so the body re-installs
        // the whole bundle in one call — user id (#154), login session (#261),
        // transport + co-location + client label (#243/#248). `system_refinement`
        // is moved into the closure as an explicit value for the same reason.
        let request_scope = RequestScope::capture();

        let task_id = registry.spawn(user_id, kind, title, move |ctx| async move {
            // Lifecycle log so the UI knows the foreground turn is
            // tracked — Status category keeps it distinct from the
            // registry's own Lifecycle markers.
            ctx.logs.append(
                api::LogLevel::Info,
                api::LogCategory::Status,
                format!("send_prompt conversation_id={conv_id_for_body}"),
                None,
            );

            // Per-turn client-tool port (#234) keyed on the registry task id.
            let client_tool_port = client_tools.map(|wiring| {
                Arc::new(CoordinatorClientToolPort::new(
                    wiring.coord,
                    wiring.store,
                    Arc::clone(&sink_for_body),
                    ctx.task_id.clone(),
                    conv_id_for_body.clone(),
                ))
                    as Arc<dyn desktop_assistant_core::ports::client_tools::ClientToolPort>
            });

            let result = request_scope
                .scope(run_send_turn(
                    conversations,
                    conv_id_for_body,
                    content,
                    override_selection,
                    system_refinement,
                    request_id_for_body,
                    idempotency,
                    echo_idempotency_key,
                    sink_for_body,
                    ctx.token.clone(),
                    client_tool_port,
                    Some(ctx.clone()),
                ))
                .await;

            match result {
                Ok(()) => Ok(()),
                // Cancellation propagated cooperatively through the
                // core layer is surfaced as `CoreError::Cancelled`. We
                // want the registry to record this as `Cancelled`
                // (which it will, since the token is tripped) rather
                // than `Failed` — return `Ok` so finalize doesn't tack
                // on a misleading error string.
                Err(desktop_assistant_core::CoreError::Cancelled) => Ok(()),
                Err(other) => Err(anyhow::Error::new(other)),
            }
        });

        registry.wait(&task_id).await;
        // Surface the registry's recorded status as an `ApiResult` so
        // existing call sites (transport-dispatch) keep their existing
        // happy-path / error-path branching.
        Ok(())
    }
}

/// Shared spawn primitive used by both `SpawnStandaloneAgent` (#113)
/// and the `spawn_subagent` builtin tool (#112).
///
/// Both call sites do the same three things in the same order:
/// 1. Create a fresh conversation under the requesting user's scope so
///    the new row carries the right `user_id`.
/// 2. Register a task in the registry with a kind built from the new
///    conversation id, so the foreground UI can show, cancel, and
///    follow logs for the spawned agent.
/// 3. Drive that conversation through `send_prompt_with_override`
///    inside the task body, threading through the per-turn cancellation
///    token, the per-turn user scope, and the per-turn tool allowlist.
///
/// Returns the new `TaskId` synchronously — the body runs on the
/// current tokio runtime and the caller does NOT await its completion.
/// This is the contract the protocol relies on for `BackgroundTaskSpawned`.
///
/// The `kind_factory` lets each call site construct its own
/// `TaskKind` variant (`Standalone` vs `Subagent`) once the conversation
/// id is known. The helper deliberately doesn't take a `TaskKind`
/// directly so we can't construct one with a stale or empty
/// `conversation_id`.
/// Result slot for [`AgentConversationSpec::result_sink`]. The helper
/// writes `Ok(response)` on success, `Err("cancelled")` on cooperative
/// cancellation, and `Err(detail)` on failure — the `spawn_subagent`
/// tool body (#112) reads it to surface the child's final answer to a
/// `wait=true` caller.
pub(crate) type AgentResultSink = Arc<tokio::sync::Mutex<Option<Result<String, String>>>>;

/// Bundle of arguments for [`spawn_agent_conversation`]. Grouping them
/// in a struct keeps the helper's call sites readable and dodges the
/// `clippy::too_many_arguments` lint without sacrificing the
/// field-by-field documentation the call sites benefit from.
pub(crate) struct AgentConversationSpec {
    pub user_id: UserId,
    /// Display name for the agent; used to derive the run's log lines.
    pub name: String,
    /// Title for the registry's `TaskView`. Distinct from `name` so the
    /// caller can prepend a type tag like "Standalone: …".
    pub title: String,
    /// The user-visible first turn for this conversation.
    pub initial_prompt: String,
    /// Optional per-send override of connection + model + effort.
    pub override_selection: Option<api::SendPromptOverride>,
    /// Optional tool allowlist for this run; installed as the task-local
    /// [`desktop_assistant_core::ports::llm::current_tool_allowlist`].
    pub tools: Option<Vec<String>>,
    /// Conversation id created beforehand by the caller — the helper
    /// needs it both to construct the task kind and to send the prompt.
    pub conversation_id: String,
    /// Optional sink for the run's final assistant text. When `Some`,
    /// the helper stashes `Ok(response)` on success, `Err("cancelled")`
    /// on cooperative cancellation, and `Err(detail)` on failure. The
    /// `spawn_subagent` tool body (#112) uses this so a `wait=true`
    /// parent can pull the child's final answer out of the registry
    /// task; `SpawnStandaloneAgent` (#113) leaves it `None` because the
    /// agent run is fire-and-forget at the protocol level.
    pub result_sink: Option<AgentResultSink>,
    /// The session-pad scope (#287) a subagent run adopts: the child works in
    /// its own conversation for reasoning/history, but its scratchpad reads and
    /// writes target the session pad under `owner_todo`, snapshot-bounded by the
    /// spawn marker. `Some` for `spawn_subagent`; `None` for `SpawnStandaloneAgent`
    /// and any pre-#287 caller, leaving pad ops on the run's own conversation.
    pub subagent_scope: Option<desktop_assistant_core::ports::scratchpad_scope::SubagentScope>,
    /// Session-pad write handle (#607): when set together with `subagent_scope`,
    /// the run's final answer is written to the session pad as a `"result"` note
    /// under the child's `owner_todo`, so the parent collects it by task id.
    /// `None` for standalone runs and when the daemon has no scratchpad store.
    pub scratchpad_write: Option<ScratchpadWriteFn>,
}

/// Build a [`ToolObserver`] that mirrors the core loop's tool/MCP calls into a
/// background task's log ring. Installed around the dispatch (via
/// [`with_tool_observer`]) so the task panel shows a live feed of what the turn
/// is doing — each call's name + arguments, then its outcome — instead of an
/// empty log. The one-line `progress_hint` (driven separately by `on_status`)
/// still shows the latest humanized activity; this is the detailed timeline.
fn task_tool_observer(ctx: TaskContext) -> ToolObserver {
    Arc::new(move |event: ToolEvent| match event {
        ToolEvent::Started { name, args } => {
            let message = if args.is_empty() {
                name
            } else {
                format!("{name}  {args}")
            };
            ctx.logs.append(
                api::LogLevel::Info,
                api::LogCategory::ToolCall,
                message,
                None,
            );
        }
        ToolEvent::Finished { name, ok, output } => {
            let (level, message) = if ok {
                let msg = if output.is_empty() {
                    format!("{name} ✓")
                } else {
                    format!("{name} ✓ {output}")
                };
                (api::LogLevel::Info, msg)
            } else {
                (api::LogLevel::Warn, format!("{name} ✗ {output}"))
            };
            ctx.logs
                .append(level, api::LogCategory::ToolResult, message, None);
        }
    })
}

/// Wrap `fut` with this turn's task-tool observer when the turn is tracked by
/// a registry task (#256). Both the streaming send path (`run_send_turn`) and
/// the agent path (`spawn_agent_conversation`) need the loop's tool/MCP calls
/// mirrored into the task's log ring; this is the single place that installs
/// the observer so the two paths can't drift. When `task_ctx` is `None` (the
/// no-registry direct path) the future runs unwrapped.
///
/// The companion `progress_hint` clear is NOT done here: it lives in the
/// registry's panic-safe `finalize` (#254), so neither caller clears it inline.
async fn with_task_observer<F, T>(task_ctx: Option<TaskContext>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match task_ctx {
        Some(ctx) => with_tool_observer(task_tool_observer(ctx), fut).await,
        None => fut.await,
    }
}

pub(crate) fn spawn_agent_conversation<C>(
    registry: Arc<BackgroundTaskRegistry>,
    conversations: Arc<C>,
    spec: AgentConversationSpec,
    kind_factory: impl FnOnce(String) -> api::TaskKind,
) -> api::TaskId
where
    C: ?Sized + ConversationService + Send + Sync + 'static,
{
    let AgentConversationSpec {
        user_id,
        name,
        title,
        initial_prompt,
        override_selection,
        tools,
        conversation_id,
        result_sink,
        subagent_scope,
        scratchpad_write,
    } = spec;

    // #607: keep a copy of the child scope for the post-run result write (the
    // match below consumes `subagent_scope` into the run-body task-local nest).
    let scope_for_result = subagent_scope.clone();

    let kind = kind_factory(conversation_id.clone());
    let agent_name = name;

    // #287: pin the child's owner_todo namespace + spawn snapshot marker onto
    // the task row from the scope the dispatch loop minted. Standalone agents
    // (no subagent_scope) get the root default, leaving their row unchanged.
    let spawn_meta = subagent_scope
        .as_ref()
        .map(|s| crate::background_tasks::SpawnMeta {
            owner_todo: s.owner_todo.clone(),
            spawn_marker: Some(s.visible_before.clone()),
        })
        .unwrap_or_default();

    registry.spawn_with_meta(
        user_id.clone(),
        kind,
        title,
        spawn_meta,
        move |ctx| async move {
            ctx.logs.append(
                api::LogLevel::Info,
                api::LogCategory::Status,
                format!("agent '{agent_name}' starting prompt"),
                None,
            );

            let override_for_core = override_selection.map(|o| PromptSelectionOverride {
                connection_id: o.connection_id,
                model_id: o.model_id,
                effort: o.effort,
            });

            // Drop the chunk callback — standalone / subagent runs emit their
            // final text through the registry, not the streaming `SendMessage`
            // chunk path. The status callback, however, drives the task's
            // `progress_hint` so the task list shows what the agent is doing right
            // now (e.g. the current tool/MCP call) instead of just its title.
            let on_chunk: desktop_assistant_core::ports::llm::ChunkCallback =
                Box::new(|_chunk| true);
            let progress_ctx = ctx.clone();
            let on_status: desktop_assistant_core::ports::llm::StatusCallback =
                Box::new(move |msg| {
                    progress_ctx.set_progress_hint(Some(msg));
                });

            // Install the tool allowlist (if any) and the requesting user's
            // identity so the inner `send_prompt_with_override` call (and
            // any storage queries it triggers) observe the same scope the
            // foreground send path uses.
            let conv_id_for_send = conversation_id.clone();
            let token = ctx.token.clone();
            // Mirror this agent's tool/MCP calls into its task log so the panel
            // shows what the agent is doing, same as the foreground send path
            // (shared observer install — #256).
            let inner = with_task_observer(Some(ctx), async move {
                conversations
                    .send_prompt_with_override(
                        &desktop_assistant_core::domain::ConversationId::from(
                            conv_id_for_send.as_str(),
                        ),
                        initial_prompt,
                        override_for_core,
                        // Agent runs (standalone / subagent) carry no per-request
                        // system-prompt refinement — that's a foreground voice/chat
                        // concern, not an agent one.
                        String::new(),
                        on_chunk,
                        on_status,
                        token,
                    )
                    .await
            });
            // Compose the run's task-local scopes around `inner`, awaiting in each
            // arm so their differing future types unify at the `Result`. The #287
            // subagent scope (session pad + owner_todo + snapshot marker) is
            // installed INSIDE `with_user_id` so scratchpad ops observe the tenant
            // guard, and re-established here in the spawned body -- never read across
            // the registry's `tokio::spawn`.
            use desktop_assistant_core::ports::auth::with_user_id;
            use desktop_assistant_core::ports::llm::with_tool_allowlist;
            use desktop_assistant_core::ports::scratchpad_scope::with_subagent_scope;
            let uid = user_id.clone();
            let result = match (tools, subagent_scope) {
                (Some(tools), Some(scope)) => {
                    with_user_id(
                        uid,
                        with_tool_allowlist(tools, with_subagent_scope(scope, inner)),
                    )
                    .await
                }
                (Some(tools), None) => with_user_id(uid, with_tool_allowlist(tools, inner)).await,
                (None, Some(scope)) => with_user_id(uid, with_subagent_scope(scope, inner)).await,
                (None, None) => with_user_id(uid, inner).await,
            };

            // The "currently doing" hint is cleared by the registry's panic-safe
            // `finalize` (#254/#256), so a finished/failed agent never shows a
            // stale tool action — no in-body clear needed here.

            match result {
                Ok(outcome) => {
                    // #607: mirror the final answer to the SESSION pad as this
                    // child's "result" note under its owner_todo, so the parent
                    // collects it by task (get_subagent_status) without
                    // re-deriving from conversations. Run under the child's scope
                    // + user id so the store stamps owner_todo and targets the
                    // session conversation. Best-effort: a pad-write failure is
                    // logged, never fatal -- the sink still carries the text.
                    if let (Some(write), Some(scope)) = (&scratchpad_write, &scope_for_result) {
                        let note =
                            desktop_assistant_core::ports::scratchpad::NewScratchpadNote::new(
                                "result",
                                outcome.response.clone(),
                            );
                        let session = scope.session_conversation_id.as_str().to_string();
                        let scoped = with_subagent_scope(scope.clone(), write(session, vec![note]));
                        if let Err(e) = with_user_id(user_id.clone(), scoped).await {
                            tracing::warn!(error = %e, "subagent result pad-write failed");
                        }
                    }
                    if let Some(sink) = result_sink {
                        *sink.lock().await = Some(Ok(outcome.response));
                    }
                    Ok(())
                }
                // `Cancelled` propagated up from the cooperative core path
                // — let the registry record this as `Cancelled` via the
                // token state rather than tacking on a misleading "failed"
                // string.
                Err(desktop_assistant_core::CoreError::Cancelled) => {
                    if let Some(sink) = result_sink {
                        *sink.lock().await = Some(Err("cancelled".to_string()));
                    }
                    Ok(())
                }
                Err(other) => {
                    let detail = other.to_string();
                    if let Some(sink) = result_sink {
                        *sink.lock().await = Some(Err(detail.clone()));
                    }
                    Err(anyhow::Error::new(other))
                }
            }
        },
    )
}

/// Re-emit a stored idempotent reply (#204) as the canonical event
/// sequence a fresh turn produces — an opening `UserMessageAdded` echoing the
/// retry's `idempotency_key` (mirrors the fresh-turn emit, #570), then one
/// `AssistantDelta` carrying the full text, then `AssistantCompleted` — all
/// keyed by the *current* `request_id` so the retrying client correlates it
/// with its pending request. The leading `UserMessageAdded` lets a retrying
/// client with an optimistic bubble Case-0 dedupe rather than double-render.
/// Best-effort: a dropped sink just means the client went away.
async fn replay_completed_response(
    sink: &Arc<dyn EventSink>,
    conversation_id: &str,
    request_id: &str,
    content: &str,
    echo_idempotency_key: Option<String>,
    response: String,
) {
    let _ = sink
        .emit(api::Event::UserMessageAdded {
            conversation_id: conversation_id.to_string(),
            request_id: request_id.to_string(),
            content: content.to_string(),
            idempotency_key: echo_idempotency_key,
        })
        .await;
    let _ = sink
        .emit(api::Event::AssistantDelta {
            conversation_id: conversation_id.to_string(),
            request_id: request_id.to_string(),
            chunk: response.clone(),
        })
        .await;
    let _ = sink
        .emit(api::Event::AssistantCompleted {
            conversation_id: conversation_id.to_string(),
            request_id: request_id.to_string(),
            full_response: response,
        })
        .await;
}

/// Normalize a wire `String` into `Option<String>` for the `set_*` settings
/// commands (#314): a trimmed-empty string becomes `None` (clears the field),
/// matching the in-process D-Bus settings adapters which treat an empty string
/// as "clear this optional field". Non-empty values are trimmed and kept.
fn normalize_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Windowed-message slicing for `Command::GetMessages` (CC-5 / #361), mirroring
/// the D-Bus `get_messages` semantics so the bridge maps the two 1:1:
/// `after_count >= 0` slices from that raw index onward (a stable position
/// cursor); otherwise `tail > 0` keeps the last `tail`. The `include_roles`
/// allowlist is applied AFTER the position slice (empty = no filter), matching
/// the D-Bus method. `total_raw_count` is always the pre-slice message count so
/// the client knows how much history exists; `truncated` is set only in tail
/// mode when older messages were dropped.
fn window_messages(
    all: Vec<api::MessageView>,
    tail: i32,
    after_count: i32,
    include_roles: &[String],
) -> api::MessagesView {
    let total = all.len() as u32;
    let use_after = after_count >= 0;
    let sliced: Vec<api::MessageView> = if use_after {
        let start = (after_count as usize).min(all.len());
        all[start..].to_vec()
    } else {
        all
    };
    let filtered: Vec<api::MessageView> = sliced
        .into_iter()
        .filter(|m| include_roles.is_empty() || include_roles.contains(&m.role))
        .collect();
    let (truncated, messages) = if !use_after && tail > 0 && filtered.len() > tail as usize {
        let start = filtered.len() - tail as usize;
        (true, filtered[start..].to_vec())
    } else {
        (false, filtered)
    };
    api::MessagesView {
        total_raw_count: total,
        truncated,
        messages,
    }
}

/// Inline turn body, shared between the registry-aware and the
/// no-registry code paths. Returns `CoreError` so the caller can either
/// map it for the trait return type or propagate it into the registry's
/// task finalization.
// Why allow: this is the streaming-send body. Its arguments are the
// conversation store, the turn inputs (conversation id, content, model
// override, system refinement, request id), the event sink, and the cancel
// token — independent values with no natural struct grouping.
#[allow(clippy::too_many_arguments)]
async fn run_send_turn<C>(
    conversations: Arc<C>,
    conversation_id: String,
    content: String,
    override_selection: Option<api::SendPromptOverride>,
    system_refinement: String,
    request_id: String,
    idempotency: Option<(Arc<dyn IdempotencyKeyStore>, String)>,
    // The client-supplied key echoed back on `UserMessageAdded` (#570), raw and
    // independent of `idempotency`: the `String` in `idempotency` is present
    // only when a dedup store is *attached*, whereas the echo must fire on every
    // keyed send so the initiator can correlate its optimistic bubble even with
    // no store configured. `None` for keyless sends.
    echo_idempotency_key: Option<String>,
    sink: Arc<dyn EventSink>,
    cancellation: tokio_util::sync::CancellationToken,
    client_tool_port: Option<Arc<dyn desktop_assistant_core::ports::client_tools::ClientToolPort>>,
    // When this turn runs inside a registry task, the context lets us mirror
    // the core loop's status messages onto the task's `progress_hint` so the
    // task list shows what the turn is doing right now — e.g. the current
    // tool/MCP call (#223 status strings, surfaced per-task). `None` on the
    // no-registry direct path, which has no task to annotate.
    task_ctx: Option<TaskContext>,
) -> Result<(), desktop_assistant_core::CoreError>
where
    C: ?Sized + ConversationService + Send + Sync + 'static,
{
    let (tx, mut rx) = tokio::sync::mpsc::channel::<api::Event>(STREAM_EVENT_BUFFER);
    // Cloned before `tx` is moved into the chunk callback below; used by the
    // context-usage sink (#341) installed around the dispatch.
    let usage_tx = tx.clone();

    let sink_for_forwarder = Arc::clone(&sink);
    let forwarder = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if !sink_for_forwarder.emit(event).await {
                break;
            }
        }
    });

    // Announce the user's prompt to every client viewing this conversation —
    // including ones that did NOT initiate this turn (a voice turn, or a second
    // client on the same account) — so they render the user bubble live rather
    // than only after a reload (#1). Emitted before dispatch so it precedes the
    // assistant's chunks; the initiating client dedupes on `request_id`, or on
    // the echoed `idempotency_key` when it supplied one (#570) — it already
    // rendered the bubble optimistically.
    //
    // Captured before the echo emit MOVES `echo_idempotency_key`, and used to
    // install the persist-side task-local (#570 Phase 1b) around the FOREGROUND
    // dispatch so `send_prompt` stamps the key onto the user row.
    let persist_idempotency_key = echo_idempotency_key.clone();
    let _ = sink
        .emit(api::Event::UserMessageAdded {
            conversation_id: conversation_id.clone(),
            request_id: request_id.clone(),
            content: content.clone(),
            idempotency_key: echo_idempotency_key,
        })
        .await;

    // Bridge chunks from core callback -> canonical events.
    let conv_id_for_cb = conversation_id.clone();
    let req_id_for_cb = request_id.clone();
    let callback: desktop_assistant_core::ports::llm::ChunkCallback = Box::new(move |chunk| {
        forward_stream_event(
            &tx,
            api::Event::AssistantDelta {
                conversation_id: conv_id_for_cb.clone(),
                request_id: req_id_for_cb.clone(),
                chunk,
            },
        )
    });

    // Bridge status updates from core callback -> canonical events, and mirror
    // them onto the registry task's `progress_hint` so the task list shows the
    // current activity (e.g. "Checking your calendar events") rather than just
    // the static title.
    let status_tx = sink.clone();
    let conv_id_for_status = conversation_id.clone();
    let req_id_for_status = request_id.clone();
    let progress_ctx = task_ctx.clone();
    let on_status: desktop_assistant_core::ports::llm::StatusCallback = Box::new(move |message| {
        // `set_progress_hint` is cheap and synchronous (lock + broadcast), so
        // update the hint inline before the fire-and-forget event emit.
        if let Some(ctx) = &progress_ctx {
            ctx.set_progress_hint(Some(message.clone()));
        }
        let sink = Arc::clone(&status_tx);
        let conv_id = conv_id_for_status.clone();
        let req_id = req_id_for_status.clone();
        // Fire-and-forget: status messages are best-effort.
        tokio::spawn(async move {
            sink.emit(api::Event::AssistantStatus {
                conversation_id: conv_id,
                request_id: req_id,
                message,
            })
            .await;
        });
    });

    let override_for_core = override_selection.map(|o| PromptSelectionOverride {
        connection_id: o.connection_id,
        model_id: o.model_id,
        effort: o.effort,
    });

    let core_conv_id =
        desktop_assistant_core::domain::ConversationId::from(conversation_id.as_str());
    // Install the client's idempotency key (#570 Phase 1b) for the persist site
    // in `send_prompt` — FOREGROUND path only. Agent runs dispatch without this
    // wrap (see `spawn_agent_conversation`), so their user rows persist `None`.
    let dispatch = desktop_assistant_core::ports::llm::with_idempotency_key(
        persist_idempotency_key,
        conversations.send_prompt_with_override(
            &core_conv_id,
            content,
            override_for_core,
            system_refinement,
            callback,
            on_status,
            cancellation,
        ),
    );

    // Layer the per-turn task-locals around the dispatch. Innermost: the
    // client-tool port (#234) so the core loop can offer the connection's
    // client-local tools and suspend on a call. Outermost: a tool observer
    // (when this turn is a tracked task) so the loop's tool/MCP calls land in
    // the task's log ring for the panel's activity feed.
    // Context-usage sink (#341): the core dispatch loop reports each turn's
    // fill (used / budget tokens + compaction flag) via this sink, which we
    // forward as `Event::ContextUsage` on the same channel the chunk/status
    // callbacks use. `try_send` is synchronous and non-blocking — usage is
    // best-effort telemetry, so a full buffer simply drops the report rather
    // than stalling the turn. Token COUNTS only; no message content crosses.
    let conv_id_for_usage = conversation_id.clone();
    let req_id_for_usage = request_id.clone();
    let usage_sink: desktop_assistant_core::ports::llm::ContextUsageSink = Arc::new(
        move |usage: desktop_assistant_core::ports::llm::ContextUsage| {
            let _ = usage_tx.try_send(api::Event::ContextUsage {
                conversation_id: conv_id_for_usage.clone(),
                request_id: req_id_for_usage.clone(),
                used_tokens: usage.used_tokens,
                budget_tokens: usage.budget_tokens,
                compaction_active: usage.compaction_active,
            });
        },
    );

    let dispatched = async move {
        match client_tool_port {
            Some(port) => with_client_tools(port, dispatch).await,
            None => dispatch.await,
        }
    };
    let dispatched =
        desktop_assistant_core::ports::llm::with_context_usage_sink(usage_sink, dispatched);
    // Install this turn's task-tool observer when tracked by a registry task
    // (#256 — shared with the agent path). The companion `progress_hint` clear
    // lives in the registry's panic-safe `finalize` (#254), so there is no
    // in-body clear here.
    //
    // Box the composed task-local chain onto the heap before awaiting it: this
    // future is deeply nested (idempotency-key / client-tools / context-usage /
    // task-observer scopes around `send_prompt_with_override`) and `run_send_turn`
    // itself runs on a spawned registry task with a bounded worker stack. Keeping
    // the composition off `run_send_turn`'s own stack frame preserves the thin
    // spawned-future invariant (#205/#206) so an extra scope layer can't overflow
    // the 2 MB worker stack.
    let dispatched = with_task_observer(task_ctx.clone(), dispatched);
    let outcome = Box::pin(dispatched).await;

    if let Err(e) = forwarder.await {
        warn!("stream forwarder task failed: {e}");
    }

    match outcome {
        Ok(outcome) => {
            for w in outcome.warnings {
                let _ = sink
                    .emit(api::Event::ConversationWarningEmitted {
                        conversation_id: conversation_id.clone(),
                        warning: dispatch_warning_to_api(w),
                    })
                    .await;
            }

            let full_response = outcome.response;

            // Record the committed reply for idempotent retry (#204) before
            // emitting completion. Best-effort: a storage error is logged but
            // must not fail a turn the client has effectively received.
            if let Some((store, key)) = &idempotency {
                let recorded = store
                    .record_response(&conversation_id, key, &request_id, &full_response)
                    .await;
                if let Err(e) = recorded {
                    warn!("failed to record idempotency reply for key {key}: {e}");
                }
            }

            let _ = sink
                .emit(api::Event::AssistantCompleted {
                    conversation_id: conversation_id.clone(),
                    request_id,
                    full_response,
                })
                .await;

            if let Ok(conv) = conversations
                .get_conversation(&desktop_assistant_core::domain::ConversationId::from(
                    conversation_id.as_str(),
                ))
                .await
            {
                let _ = sink
                    .emit(api::Event::ConversationTitleChanged {
                        conversation_id,
                        title: conv.title,
                    })
                    .await;
            }

            Ok(())
        }
        Err(e) => {
            let _ = sink
                .emit(api::Event::AssistantError {
                    conversation_id,
                    request_id,
                    error: e.to_string(),
                })
                .await;
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::{
        Conversation, ConversationId, ConversationSummary, Message, Role,
    };
    use desktop_assistant_core::ports::inbound::{
        BackendTasksSettingsView, ConnectorDefaultsView, DatabaseSettingsView,
        EmbeddingsSettingsView, LlmSettingsView, ModelListing as CoreModelListing,
        PersistenceSettingsView, PersonalitySettingsView, PurposeKind,
        PurposesView as CorePurposesView,
    };
    use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct FakeKnowledge;
    impl desktop_assistant_core::ports::inbound::KnowledgeService for FakeKnowledge {
        async fn list_entries(
            &self,
            _limit: usize,
            _offset: usize,
            _tag_filter: Option<Vec<String>>,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn get_entry(&self, _id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
            Ok(None)
        }
        async fn search_entries(
            &self,
            _query: String,
            _tag_filter: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn create_entry(
            &self,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<KnowledgeEntry, CoreError> {
            let mut e = KnowledgeEntry::new("kb-test", content, tags);
            e.metadata = metadata;
            Ok(e)
        }
        async fn update_entry(
            &self,
            id: String,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<KnowledgeEntry, CoreError> {
            let mut e = KnowledgeEntry::new(id, content, tags);
            e.metadata = metadata;
            Ok(e)
        }
        async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn trash_count(&self) -> Result<usize, CoreError> {
            Ok(0)
        }
        async fn empty_trash(&self) -> Result<usize, CoreError> {
            Ok(0)
        }
    }

    struct FakeConnections;
    impl ConnectionsService for FakeConnections {
        async fn list_connections(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::ConnectionView>, CoreError>
        {
            Ok(vec![])
        }
        async fn create_connection(
            &self,
            _id: String,
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn update_connection(
            &self,
            _id: String,
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn delete_connection(&self, _id: String, _force: bool) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_available_models(
            &self,
            _connection_id: Option<String>,
            _refresh: bool,
        ) -> Result<Vec<CoreModelListing>, CoreError> {
            Ok(vec![])
        }
        async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
            Ok(CorePurposesView::default())
        }
        async fn set_purpose(
            &self,
            _purpose: PurposeKind,
            _config: PurposeConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

    struct FakeAssistant;
    impl AssistantService for FakeAssistant {
        fn version(&self) -> &str {
            "0.0.0-test"
        }
        fn ping(&self) -> &str {
            "pong"
        }
    }

    struct FakeConversations;
    #[async_trait::async_trait]
    impl ConversationService for FakeConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            let mut c = Conversation::new(id.as_str(), "t");
            c.messages.push(Message::new(Role::User, "hi"));
            Ok(c)
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            on_chunk("a".into());
            on_chunk("b".into());
            Ok("ab".into())
        }
    }

    struct FakeSettings;
    impl SettingsService for FakeSettings {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(LlmSettingsView {
                connector: "x".into(),
                model: "y".into(),
                base_url: "z".into(),
                has_api_key: false,
                temperature: None,
                top_p: None,
                max_tokens: None,
                hosted_tool_search: None,
            })
        }
        async fn set_llm_settings(
            &self,
            _connector: String,
            _model: Option<String>,
            _base_url: Option<String>,
            _temperature: Option<f64>,
            _top_p: Option<f64>,
            _max_tokens: Option<u32>,
            _hosted_tool_search: Option<bool>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
            Ok(format!(
                "jwt-for-{}",
                subject.unwrap_or_else(|| "desktop-client".to_string())
            ))
        }
        async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
            Ok(token.starts_with("jwt-for-"))
        }
        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(EmbeddingsSettingsView {
                connector: "x".into(),
                model: "y".into(),
                base_url: "z".into(),
                has_api_key: false,
                available: false,
                is_default: true,
                health: EmbeddingHealth::Disabled,
            })
        }
        async fn set_embeddings_settings(
            &self,
            _connector: Option<String>,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
            Ok(ConnectorDefaultsView {
                llm_model: "m".into(),
                llm_base_url: "u".into(),
                backend_llm_model: "bm".into(),
                embeddings_model: "em".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: false,
                hosted_tool_search_available: false,
            })
        }
        async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
            Ok(PersistenceSettingsView {
                enabled: false,
                remote_url: "".into(),
                remote_name: "origin".into(),
                push_on_update: false,
            })
        }
        async fn set_persistence_settings(
            &self,
            _enabled: bool,
            _remote_url: Option<String>,
            _remote_name: Option<String>,
            _push_on_update: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(DatabaseSettingsView {
                url: String::new(),
                max_connections: 5,
            })
        }
        async fn set_database_settings(
            &self,
            _url: Option<String>,
            _max_connections: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
            Ok(BackendTasksSettingsView {
                has_separate_llm: false,
                llm_connector: "openai".into(),
                llm_model: "gpt-5".into(),
                llm_base_url: "https://api.openai.com/v1".into(),
                dreaming_enabled: false,
                dreaming_interval_secs: 3600,
                archive_after_days: 0,
            })
        }
        async fn set_backend_tasks_settings(
            &self,
            _llm_connector: Option<String>,
            _llm_model: Option<String>,
            _llm_base_url: Option<String>,
            _dreaming_enabled: bool,
            _dreaming_interval_secs: u64,
            _archive_after_days: u32,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_mcp_servers(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn add_mcp_server(
            &self,
            _name: String,
            _command: String,
            _args: Vec<String>,
            _namespace: Option<String>,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            _name: String,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            _action: String,
            _server: Option<String>,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(vec![])
        }
        async fn get_ws_auth_settings(
            &self,
        ) -> Result<desktop_assistant_core::ports::inbound::WsAuthSettingsView, CoreError> {
            Ok(desktop_assistant_core::ports::inbound::WsAuthSettingsView {
                methods: vec![],
                oidc_issuer: String::new(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            })
        }
        async fn set_ws_auth_settings(
            &self,
            _methods: Vec<String>,
            _oidc_issuer: String,
            _oidc_auth_endpoint: String,
            _oidc_token_endpoint: String,
            _oidc_client_id: String,
            _oidc_scopes: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[derive(Clone)]
    struct SettingsState {
        llm: LlmSettingsView,
        embeddings: EmbeddingsSettingsView,
        persistence: PersistenceSettingsView,
        personality: PersonalitySettingsView,
        api_key_set: bool,
        // #314 round-trip state: database / backend-tasks / ws-auth / MCP.
        database: DatabaseSettingsView,
        backend_tasks: BackendTasksSettingsView,
        ws_auth: desktop_assistant_core::ports::inbound::WsAuthSettingsView,
        mcp_servers: Vec<desktop_assistant_core::ports::inbound::McpServerView>,
    }

    struct ConfigurableSettings {
        state: Mutex<SettingsState>,
    }

    impl ConfigurableSettings {
        fn new() -> Self {
            Self {
                state: Mutex::new(SettingsState {
                    llm: LlmSettingsView {
                        connector: "openai".into(),
                        model: "gpt-5".into(),
                        base_url: "https://api.openai.com/v1".into(),
                        has_api_key: false,
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        hosted_tool_search: None,
                    },
                    embeddings: EmbeddingsSettingsView {
                        connector: "openai".into(),
                        model: "text-embedding-3-small".into(),
                        base_url: "https://api.openai.com/v1".into(),
                        has_api_key: false,
                        available: true,
                        is_default: true,
                        health: EmbeddingHealth::Ok,
                    },
                    persistence: PersistenceSettingsView {
                        enabled: false,
                        remote_url: String::new(),
                        remote_name: "origin".into(),
                        push_on_update: true,
                    },
                    personality: PersonalitySettingsView::default(),
                    api_key_set: false,
                    database: DatabaseSettingsView {
                        url: String::new(),
                        max_connections: 5,
                    },
                    backend_tasks: BackendTasksSettingsView {
                        has_separate_llm: false,
                        llm_connector: "openai".into(),
                        llm_model: "gpt-5".into(),
                        llm_base_url: "https://api.openai.com/v1".into(),
                        dreaming_enabled: false,
                        dreaming_interval_secs: 3600,
                        archive_after_days: 0,
                    },
                    ws_auth: desktop_assistant_core::ports::inbound::WsAuthSettingsView {
                        methods: vec!["password".into()],
                        oidc_issuer: String::new(),
                        oidc_auth_endpoint: String::new(),
                        oidc_token_endpoint: String::new(),
                        oidc_client_id: String::new(),
                        oidc_scopes: String::new(),
                    },
                    mcp_servers: vec![],
                }),
            }
        }

        #[allow(dead_code)]
        fn snapshot(&self) -> SettingsState {
            self.state.lock().unwrap().clone()
        }
    }

    impl SettingsService for ConfigurableSettings {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().llm.clone())
        }

        async fn set_llm_settings(
            &self,
            connector: String,
            model: Option<String>,
            base_url: Option<String>,
            temperature: Option<f64>,
            top_p: Option<f64>,
            max_tokens: Option<u32>,
            hosted_tool_search: Option<bool>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.llm.connector = connector;
            if let Some(model) = model {
                state.llm.model = model;
            }
            if let Some(base_url) = base_url {
                state.llm.base_url = base_url;
            }
            state.llm.temperature = temperature;
            state.llm.top_p = top_p;
            state.llm.max_tokens = max_tokens;
            state.llm.hosted_tool_search = hosted_tool_search;
            Ok(())
        }

        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.api_key_set = true;
            state.llm.has_api_key = true;
            Ok(())
        }

        async fn generate_ws_jwt(&self, subject: Option<String>) -> Result<String, CoreError> {
            Ok(format!(
                "jwt-for-{}",
                subject.unwrap_or_else(|| "desktop-client".to_string())
            ))
        }

        async fn validate_ws_jwt(&self, token: String) -> Result<bool, CoreError> {
            Ok(token.starts_with("jwt-for-"))
        }

        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().embeddings.clone())
        }

        async fn set_embeddings_settings(
            &self,
            connector: Option<String>,
            model: Option<String>,
            base_url: Option<String>,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            if let Some(connector) = connector {
                state.embeddings.connector = connector;
                state.embeddings.is_default = false;
            }
            if let Some(model) = model {
                state.embeddings.model = model;
            }
            if let Some(base_url) = base_url {
                state.embeddings.base_url = base_url;
            }
            Ok(())
        }

        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
            Ok(ConnectorDefaultsView {
                llm_model: "m".into(),
                llm_base_url: "u".into(),
                backend_llm_model: "bm".into(),
                embeddings_model: "em".into(),
                embeddings_base_url: "eu".into(),
                embeddings_available: false,
                hosted_tool_search_available: false,
            })
        }

        async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().persistence.clone())
        }

        async fn set_persistence_settings(
            &self,
            enabled: bool,
            remote_url: Option<String>,
            remote_name: Option<String>,
            push_on_update: bool,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.persistence.enabled = enabled;
            if let Some(remote_url) = remote_url {
                state.persistence.remote_url = remote_url;
            }
            if let Some(remote_name) = remote_name {
                state.persistence.remote_name = remote_name;
            }
            state.persistence.push_on_update = push_on_update;
            Ok(())
        }

        async fn get_personality_settings(&self) -> Result<PersonalitySettingsView, CoreError> {
            Ok(self.state.lock().unwrap().personality)
        }

        async fn set_personality_settings(
            &self,
            personality: PersonalitySettingsView,
        ) -> Result<(), CoreError> {
            self.state.lock().unwrap().personality = personality;
            Ok(())
        }

        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().database.clone())
        }

        async fn set_database_settings(
            &self,
            url: Option<String>,
            max_connections: u32,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            // Mirror the daemon: `None` (empty url) clears the URL.
            state.database.url = url.unwrap_or_default();
            state.database.max_connections = max_connections;
            Ok(())
        }
        async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().backend_tasks.clone())
        }
        async fn set_backend_tasks_settings(
            &self,
            llm_connector: Option<String>,
            llm_model: Option<String>,
            llm_base_url: Option<String>,
            dreaming_enabled: bool,
            dreaming_interval_secs: u64,
            archive_after_days: u32,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            // Mirror the daemon: presence of a connector means a separate
            // backend-tasks LLM override; absence clears it.
            state.backend_tasks.has_separate_llm = llm_connector.is_some();
            if let Some(connector) = llm_connector {
                state.backend_tasks.llm_connector = connector;
            }
            if let Some(model) = llm_model {
                state.backend_tasks.llm_model = model;
            }
            if let Some(base_url) = llm_base_url {
                state.backend_tasks.llm_base_url = base_url;
            }
            state.backend_tasks.dreaming_enabled = dreaming_enabled;
            state.backend_tasks.dreaming_interval_secs = dreaming_interval_secs;
            state.backend_tasks.archive_after_days = archive_after_days;
            Ok(())
        }
        async fn list_mcp_servers(
            &self,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            Ok(self.state.lock().unwrap().mcp_servers.clone())
        }
        async fn add_mcp_server(
            &self,
            name: String,
            command: String,
            args: Vec<String>,
            namespace: Option<String>,
            enabled: bool,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            if state.mcp_servers.iter().any(|s| s.name == name) {
                return Err(CoreError::SystemService(format!(
                    "MCP server '{name}' already exists"
                )));
            }
            // Persist the full config so command/args/namespace round-trip on a
            // later list (#314), mirroring the daemon persisting to TOML.
            state
                .mcp_servers
                .push(desktop_assistant_core::ports::inbound::McpServerView {
                    name,
                    command,
                    args,
                    namespace,
                    enabled,
                    status: if enabled { "running" } else { "disabled" }.to_string(),
                    tool_count: 0,
                    ..Default::default()
                });
            Ok(())
        }
        async fn remove_mcp_server(&self, name: String) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            let before = state.mcp_servers.len();
            state.mcp_servers.retain(|s| s.name != name);
            if state.mcp_servers.len() == before {
                return Err(CoreError::SystemService(format!(
                    "MCP server '{name}' not found"
                )));
            }
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            name: String,
            enabled: bool,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            let server = state
                .mcp_servers
                .iter_mut()
                .find(|s| s.name == name)
                .ok_or_else(|| {
                    CoreError::SystemService(format!("MCP server '{name}' not found"))
                })?;
            server.enabled = enabled;
            server.status = if enabled { "running" } else { "disabled" }.to_string();
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            action: String,
            server: Option<String>,
        ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
            // Validate the action like the daemon, then return current status.
            match action.as_str() {
                "status" | "start" | "stop" | "restart" => {}
                other => {
                    return Err(CoreError::SystemService(format!(
                        "unknown MCP action: {other}"
                    )));
                }
            }
            let state = self.state.lock().unwrap();
            let servers = match server {
                Some(name) => state
                    .mcp_servers
                    .iter()
                    .filter(|s| s.name == name)
                    .cloned()
                    .collect(),
                None => state.mcp_servers.clone(),
            };
            Ok(servers)
        }
        async fn get_ws_auth_settings(
            &self,
        ) -> Result<desktop_assistant_core::ports::inbound::WsAuthSettingsView, CoreError> {
            Ok(self.state.lock().unwrap().ws_auth.clone())
        }
        async fn set_ws_auth_settings(
            &self,
            methods: Vec<String>,
            oidc_issuer: String,
            oidc_auth_endpoint: String,
            oidc_token_endpoint: String,
            oidc_client_id: String,
            oidc_scopes: String,
        ) -> Result<(), CoreError> {
            let mut state = self.state.lock().unwrap();
            state.ws_auth = desktop_assistant_core::ports::inbound::WsAuthSettingsView {
                methods,
                oidc_issuer,
                oidc_auth_endpoint,
                oidc_token_endpoint,
                oidc_client_id,
                oidc_scopes,
            };
            Ok(())
        }
    }

    struct CollectSink(tokio::sync::Mutex<Vec<api::Event>>);
    #[async_trait::async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, event: api::Event) -> bool {
            self.0.lock().await.push(event);
            true
        }
    }

    struct DropSink;
    #[async_trait::async_trait]
    impl EventSink for DropSink {
        async fn emit(&self, _event: api::Event) -> bool {
            false
        }
    }

    struct AbortAwareConversations {
        aborted: Arc<AtomicBool>,
    }
    #[async_trait::async_trait]
    impl ConversationService for AbortAwareConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            for _ in 0..10_000 {
                if !on_chunk("x".to_string()) {
                    self.aborted.store(true, Ordering::SeqCst);
                    return Ok("cancelled".to_string());
                }
                tokio::task::yield_now().await;
            }
            Ok("complete".to_string())
        }
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h.handle_command(api::Command::Ping).await.unwrap();
        assert_eq!(
            res,
            api::CommandResult::Pong {
                value: "pong".into()
            }
        );
    }

    // --- Conversation scratchpad commands (issue #190) --------------------

    /// In-memory scratchpad behind the five handler closures, scoped by
    /// `current_user_id()` exactly like the real `PgScratchpadStore`, so the
    /// command handlers can be exercised (incl. cross-tenant isolation) without
    /// Postgres. There is no reusable in-memory `ScratchpadStore` to borrow.
    #[allow(clippy::type_complexity)]
    fn in_memory_scratchpad() -> (
        ScratchpadWriteFn,
        ScratchpadGetManyFn,
        ScratchpadListFn,
        ScratchpadDeleteManyFn,
        ScratchpadClearFn,
    ) {
        use desktop_assistant_core::ports::auth::current_user_id;
        type Store = Arc<Mutex<Vec<(String, ScratchpadNote)>>>;
        let store: Store = Arc::new(Mutex::new(Vec::new()));

        let w = Arc::clone(&store);
        let write: ScratchpadWriteFn =
            Arc::new(move |conv: String, notes: Vec<NewScratchpadNote>| {
                let store = Arc::clone(&w);
                Box::pin(async move {
                    let user = current_user_id().as_str().to_string();
                    let mut guard = store.lock().unwrap();
                    let mut saved = Vec::new();
                    for (i, n) in notes.into_iter().enumerate() {
                        if let Some((_, existing)) = guard.iter_mut().find(|(u, e)| {
                            *u == user && e.conversation_id == conv && e.key == n.key
                        }) {
                            existing.content = n.content;
                            existing.note_type = n.note_type;
                            existing.sequence = n.sequence;
                            existing.done = n.done;
                            existing.updated_at = "t".into();
                            saved.push(existing.clone());
                        } else {
                            let mut note =
                                ScratchpadNote::new(format!("id-{i}"), &conv, &n.key, &n.content);
                            note.note_type = n.note_type;
                            note.sequence = n.sequence;
                            note.done = n.done;
                            note.updated_at = "t".into();
                            guard.push((user.clone(), note.clone()));
                            saved.push(note);
                        }
                    }
                    Ok(saved)
                })
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<Output = Result<Vec<ScratchpadNote>, CoreError>>
                                + Send,
                        >,
                    >
            });

        let g = Arc::clone(&store);
        let get_many: ScratchpadGetManyFn =
            Arc::new(move |conv: String, keys: Vec<String>, limit| {
                let store = Arc::clone(&g);
                Box::pin(async move {
                    let user = current_user_id().as_str().to_string();
                    let guard = store.lock().unwrap();
                    Ok(guard
                        .iter()
                        .filter(|(u, n)| {
                            *u == user && n.conversation_id == conv && keys.contains(&n.key)
                        })
                        .take(limit)
                        .map(|(_, n)| n.clone())
                        .collect())
                })
            });

        let l = Arc::clone(&store);
        let list: ScratchpadListFn =
            Arc::new(move |conv: String, note_type: Option<String>, limit| {
                let store = Arc::clone(&l);
                Box::pin(async move {
                    let user = current_user_id().as_str().to_string();
                    let guard = store.lock().unwrap();
                    Ok(guard
                        .iter()
                        .filter(|(u, n)| {
                            *u == user
                                && n.conversation_id == conv
                                && note_type.as_deref().is_none_or(|t| n.note_type == t)
                        })
                        .take(limit)
                        .map(|(_, n)| n.clone())
                        .collect())
                })
            });

        let d = Arc::clone(&store);
        let delete_many: ScratchpadDeleteManyFn =
            Arc::new(move |conv: String, keys: Vec<String>| {
                let store = Arc::clone(&d);
                Box::pin(async move {
                    let user = current_user_id().as_str().to_string();
                    let mut guard = store.lock().unwrap();
                    let before = guard.len();
                    guard.retain(|(u, n)| {
                        !(*u == user && n.conversation_id == conv && keys.contains(&n.key))
                    });
                    Ok((before - guard.len()) as u64)
                })
            });

        let c = Arc::clone(&store);
        let clear: ScratchpadClearFn = Arc::new(move |conv: String| {
            let store = Arc::clone(&c);
            Box::pin(async move {
                let user = current_user_id().as_str().to_string();
                let mut guard = store.lock().unwrap();
                let before = guard.len();
                guard.retain(|(u, n)| !(*u == user && n.conversation_id == conv));
                Ok((before - guard.len()) as u64)
            })
        });

        (write, get_many, list, delete_many, clear)
    }

    fn scratchpad_handler() -> DefaultAssistantApiHandler<
        FakeAssistant,
        FakeConversations,
        FakeSettings,
        FakeConnections,
        FakeKnowledge,
    > {
        let (write, get_many, list, delete_many, clear) = in_memory_scratchpad();
        DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_scratchpad(write, get_many, list, delete_many, clear)
    }

    #[tokio::test]
    async fn scratchpad_set_get_delete_roundtrip() {
        let h = scratchpad_handler();
        let ctx = || RequestContext::from(UserId::new("alice"));

        // Upsert a todo.
        let res = h
            .handle_command_for(
                ctx(),
                api::Command::SetScratchpadNote {
                    conversation_id: "c1".into(),
                    key: "t1".into(),
                    content: "wire it".into(),
                    note_type: "todo".into(),
                    sequence: Some(1),
                    done: false,
                },
            )
            .await
            .unwrap();
        let api::CommandResult::Scratchpad(saved) = res else {
            panic!("expected Scratchpad");
        };
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].note_type, "todo");
        assert_eq!(saved[0].sequence, Some(1));

        // Read it back.
        let api::CommandResult::Scratchpad(notes) = h
            .handle_command_for(
                ctx(),
                api::Command::GetConversationScratchpad {
                    conversation_id: "c1".into(),
                    max_results: None,
                },
            )
            .await
            .unwrap()
        else {
            panic!("expected Scratchpad");
        };
        assert_eq!(notes.len(), 1);
        assert!(!notes[0].done);

        // Check it off by re-writing the same key.
        h.handle_command_for(
            ctx(),
            api::Command::SetScratchpadNote {
                conversation_id: "c1".into(),
                key: "t1".into(),
                content: "wire it".into(),
                note_type: "todo".into(),
                sequence: Some(1),
                done: true,
            },
        )
        .await
        .unwrap();
        let api::CommandResult::Scratchpad(notes) = h
            .handle_command_for(
                ctx(),
                api::Command::GetConversationScratchpad {
                    conversation_id: "c1".into(),
                    max_results: None,
                },
            )
            .await
            .unwrap()
        else {
            panic!("expected Scratchpad");
        };
        assert_eq!(notes.len(), 1, "re-write upserts, not duplicates");
        assert!(notes[0].done);

        // Delete it.
        let res = h
            .handle_command_for(
                ctx(),
                api::Command::DeleteScratchpadNotes {
                    conversation_id: "c1".into(),
                    keys: vec!["t1".into()],
                    all: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(res, api::CommandResult::Ack);
        let api::CommandResult::Scratchpad(notes) = h
            .handle_command_for(
                ctx(),
                api::Command::GetConversationScratchpad {
                    conversation_id: "c1".into(),
                    max_results: None,
                },
            )
            .await
            .unwrap()
        else {
            panic!("expected Scratchpad");
        };
        assert!(notes.is_empty());
    }

    #[tokio::test]
    async fn scratchpad_commands_are_user_scoped() {
        let h = scratchpad_handler();

        // Alice writes a note to conversation c1.
        h.handle_command_for(
            RequestContext::from(UserId::new("alice")),
            api::Command::SetScratchpadNote {
                conversation_id: "c1".into(),
                key: "goal".into(),
                content: "alice secret".into(),
                note_type: String::new(),
                sequence: None,
                done: false,
            },
        )
        .await
        .unwrap();

        // Bob, asking for the same conversation_id, sees nothing.
        let api::CommandResult::Scratchpad(notes) = h
            .handle_command_for(
                RequestContext::from(UserId::new("bob")),
                api::Command::GetConversationScratchpad {
                    conversation_id: "c1".into(),
                    max_results: None,
                },
            )
            .await
            .unwrap()
        else {
            panic!("expected Scratchpad");
        };
        assert!(notes.is_empty(), "bob must not read alice's pad");

        // Bob clearing the pad must not touch alice's note.
        h.handle_command_for(
            RequestContext::from(UserId::new("bob")),
            api::Command::DeleteScratchpadNotes {
                conversation_id: "c1".into(),
                keys: vec![],
                all: true,
            },
        )
        .await
        .unwrap();
        let api::CommandResult::Scratchpad(notes) = h
            .handle_command_for(
                RequestContext::from(UserId::new("alice")),
                api::Command::GetConversationScratchpad {
                    conversation_id: "c1".into(),
                    max_results: None,
                },
            )
            .await
            .unwrap()
        else {
            panic!("expected Scratchpad");
        };
        assert_eq!(notes.len(), 1, "alice's note survives bob's clear");
        // Empty note_type defaults to the canonical default.
        assert_eq!(notes[0].note_type, DEFAULT_NOTE_TYPE);
    }

    #[tokio::test]
    async fn scratchpad_command_without_closures_errors() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );
        let res = h
            .handle_command(api::Command::GetConversationScratchpad {
                conversation_id: "c1".into(),
                max_results: None,
            })
            .await;
        assert!(
            matches!(&res, Err(ApiError::Core(m)) if m.contains("not configured")),
            "expected not-configured error, got {res:?}"
        );
    }

    #[tokio::test]
    async fn scratchpad_set_rejects_empty_key_and_oversize_content() {
        let h = scratchpad_handler();
        let empty_key = h
            .handle_command_for(
                RequestContext::from(UserId::new("alice")),
                api::Command::SetScratchpadNote {
                    conversation_id: "c1".into(),
                    key: "   ".into(),
                    content: "x".into(),
                    note_type: String::new(),
                    sequence: None,
                    done: false,
                },
            )
            .await;
        assert!(matches!(&empty_key, Err(ApiError::Core(m)) if m.contains("must not be empty")));

        let oversize = h
            .handle_command_for(
                RequestContext::from(UserId::new("alice")),
                api::Command::SetScratchpadNote {
                    conversation_id: "c1".into(),
                    key: "big".into(),
                    content: "x".repeat(MAX_NOTE_BYTES + 1),
                    note_type: String::new(),
                    sequence: None,
                    done: false,
                },
            )
            .await;
        assert!(matches!(&oversize, Err(ApiError::Core(m)) if m.contains("exceeds")));
    }

    #[tokio::test]
    async fn send_message_emits_events_and_completes() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message("c1".into(), "hi".into(), "r1".into(), sink.clone())
            .await
            .unwrap();

        let evs = sink.0.lock().await.clone();
        // A turn now opens with `UserMessageAdded` (#1) so viewers can render the
        // user bubble live, before the assistant's deltas stream in.
        assert!(matches!(evs[0], api::Event::UserMessageAdded { .. }));
        assert!(matches!(evs[1], api::Event::AssistantDelta { .. }));
        assert!(matches!(evs[2], api::Event::AssistantDelta { .. }));
        assert!(matches!(evs[3], api::Event::AssistantCompleted { .. }));
    }

    /// #570 Phase 1: a keyed `SendMessage` echoes its `idempotency_key` back on
    /// the opening `UserMessageAdded` so the initiator can correlate its
    /// optimistic bubble by exact key match. Crucially this handler has NO
    /// idempotency store attached — the echo must fire regardless, because it is
    /// a raw round-trip of the client's key, not a function of the dedup store.
    #[tokio::test]
    async fn daemon_echoes_idempotency_key_back() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );
        // No `.with_idempotency_store(..)` — proves the echo is not gated on it.
        assert!(h.idempotency.is_none(), "test asserts the store-less path");

        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        let evs = sink.0.lock().await.clone();
        let echoed = evs.iter().find_map(|e| match e {
            api::Event::UserMessageAdded {
                idempotency_key, ..
            } => Some(idempotency_key.clone()),
            _ => None,
        });
        assert_eq!(
            echoed,
            Some(Some("k1".to_string())),
            "UserMessageAdded must echo the client's idempotency_key: {evs:?}"
        );
    }

    /// #570 Phase 1 unhappy path: a keyless send (no `idempotency_key`) still
    /// opens the turn and echoes a `None` key — no error, backward-compatible
    /// with the keyless `send_prompt_full` paths.
    #[tokio::test]
    async fn daemon_tolerates_absent_idempotency_key() {
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            None,
            sink.clone(),
        )
        .await
        .unwrap();

        let evs = sink.0.lock().await.clone();
        let echoed = evs.iter().find_map(|e| match e {
            api::Event::UserMessageAdded {
                idempotency_key, ..
            } => Some(idempotency_key.clone()),
            _ => None,
        });
        assert_eq!(
            echoed,
            Some(None),
            "a keyless send emits UserMessageAdded with a None key: {evs:?}"
        );
    }

    /// A conversation service that records the task-local `current_idempotency_key`
    /// observed at `send_prompt` time, so a test can prove which dispatch paths
    /// install the key (#570 Phase 1b).
    struct RecordKeyConversations(Arc<Mutex<Vec<Option<String>>>>);
    #[async_trait::async_trait]
    impl ConversationService for RecordKeyConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            self.0
                .lock()
                .unwrap()
                .push(desktop_assistant_core::ports::llm::current_idempotency_key());
            on_chunk("ok".into());
            Ok("ok".into())
        }
    }

    /// #570 Phase 1b wiring: the FOREGROUND send path installs the client's
    /// idempotency key as the `with_idempotency_key` task-local around the core
    /// dispatch, so `send_prompt` (the sole user-message persist site) can stamp
    /// it onto the user row. A keyed send observes the key; a keyless one
    /// observes `None`.
    #[tokio::test]
    async fn foreground_send_installs_idempotency_key_for_persist() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(RecordKeyConversations(Arc::clone(&seen))),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            seen.lock().unwrap().clone(),
            vec![Some("k1".to_string())],
            "the foreground dispatch must install the client's idempotency key"
        );

        // A keyless send installs no key: the persist site stamps None.
        seen.lock().unwrap().clear();
        let sink2 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r2".into(),
            None,
            sink2.clone(),
        )
        .await
        .unwrap();
        assert_eq!(
            seen.lock().unwrap().clone(),
            vec![None],
            "a keyless foreground send installs no idempotency key"
        );
    }

    /// #570 Phase 1b wiring: an AGENT run (standalone / subagent) dispatches
    /// through `send_prompt_with_override` WITHOUT the idempotency wrap, so its
    /// user row persists a `None` key — a background agent turn is never a
    /// client-retryable send.
    #[tokio::test]
    async fn agent_run_persists_no_idempotency_key() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let conversations = Arc::new(RecordKeyConversations(Arc::clone(&seen)));
        let registry = Arc::new(crate::background_tasks::BackgroundTaskRegistry::new());

        let task_id = spawn_agent_conversation(
            Arc::clone(&registry),
            conversations,
            AgentConversationSpec {
                user_id: UserId::new("alice"),
                name: "agent".into(),
                title: "Standalone: agent".into(),
                initial_prompt: "do the thing".into(),
                override_selection: None,
                tools: None,
                conversation_id: "conv-agent".into(),
                result_sink: None,
                subagent_scope: None,
                scratchpad_write: None,
            },
            move |conversation_id| api::TaskKind::Standalone {
                name: "agent".into(),
                conversation_id,
            },
        );

        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&task_id))
            .await
            .expect("agent turn finishes");

        assert_eq!(
            seen.lock().unwrap().clone(),
            vec![None],
            "an agent run must not install an idempotency key on its persist site"
        );
    }

    /// #1 live multi-client sync: a turn fans its events to other connections
    /// viewing the conversation, and excludes the originating connection (which
    /// gets them through its own request stream).
    #[tokio::test]
    async fn turn_fans_out_to_other_subscribers_excluding_origin() {
        let subs = Arc::new(crate::conversation_subs::ConversationSubscriptions::new());

        // A viewer connection looking at c1, on a different session than the
        // sender (whose session is "unscoped" with no `with_session_id` scope).
        // The turn runs without a user scope, so the origin's user is the
        // sentinel "default" (#432); register the viewers under the same user so
        // fan-out reaches them (a different user would be filtered out — see the
        // dedicated cross-user test in conversation_subs).
        let viewer = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        subs.register("viewer-session", "default", viewer.clone());
        subs.set_subscriptions("viewer-session", vec!["c1".to_string()]);

        // A connection registered under the SENDER's own session, also viewing
        // c1 — it must NOT be fanned its own turn.
        let self_view = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        subs.register("unscoped", "default", self_view.clone());
        subs.set_subscriptions("unscoped", vec!["c1".to_string()]);

        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_conversation_subscriptions(Arc::clone(&subs));

        let origin = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message("c1".into(), "hi".into(), "r1".into(), origin.clone())
            .await
            .unwrap();

        let viewed = viewer.0.lock().await.clone();
        assert!(
            viewed
                .iter()
                .any(|e| matches!(e, api::Event::UserMessageAdded { .. })),
            "viewer of c1 must receive the user message live: {viewed:?}"
        );
        assert!(
            viewed
                .iter()
                .any(|e| matches!(e, api::Event::AssistantCompleted { .. })),
            "viewer of c1 must receive the assistant reply live: {viewed:?}"
        );
        assert!(
            self_view.0.lock().await.is_empty(),
            "a connection on the origin's own session must be excluded from the fan-out"
        );
    }

    // --- GetMessages windowing (CC-5 / #361) ------------------------------

    fn mv(id: &str, role: &str, content: &str) -> api::MessageView {
        api::MessageView {
            id: id.into(),
            role: role.into(),
            content: content.into(),
            idempotency_key: None,
        }
    }

    #[test]
    fn window_messages_tail_keeps_last_n_and_flags_truncated() {
        let all = vec![
            mv("1", "user", "a"),
            mv("2", "assistant", "b"),
            mv("3", "user", "c"),
        ];
        let w = window_messages(all, 2, -1, &[]);
        assert_eq!(w.total_raw_count, 3, "total is the full pre-slice count");
        assert!(w.truncated, "tail dropped older messages");
        assert_eq!(
            w.messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["2", "3"],
            "keeps the last `tail`, ids intact for the client cursor"
        );
    }

    #[test]
    fn window_messages_after_count_slices_from_index_and_never_truncates() {
        let all = vec![
            mv("1", "user", "a"),
            mv("2", "assistant", "b"),
            mv("3", "user", "c"),
        ];
        let w = window_messages(all, 0, 1, &[]);
        assert_eq!(w.total_raw_count, 3);
        assert!(
            !w.truncated,
            "after_count is an exact position cursor, not a truncation"
        );
        assert_eq!(
            w.messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["2", "3"]
        );
    }

    #[test]
    fn window_messages_role_filter_applies_after_slice_and_total_is_pre_slice() {
        let all = vec![
            mv("1", "user", "a"),
            mv("2", "assistant", "b"),
            mv("3", "user", "c"),
            mv("4", "tool", "d"),
        ];
        let w = window_messages(all, 0, -1, &["user".to_string()]);
        assert_eq!(w.total_raw_count, 4, "total counts every role, pre-filter");
        assert_eq!(
            w.messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["1", "3"],
            "only the allowlisted role survives"
        );
    }

    #[test]
    fn window_messages_tail_within_limit_is_not_truncated() {
        let all = vec![mv("1", "user", "a"), mv("2", "assistant", "b")];
        let w = window_messages(all, 5, -1, &[]);
        assert!(!w.truncated);
        assert_eq!(w.messages.len(), 2);
    }

    #[test]
    fn window_messages_after_count_past_end_is_empty() {
        let w = window_messages(vec![mv("1", "user", "a")], 0, 9, &[]);
        assert_eq!(w.total_raw_count, 1);
        assert!(w.messages.is_empty());
    }

    #[tokio::test]
    async fn send_message_cancels_when_sink_disconnects() {
        let aborted = Arc::new(AtomicBool::new(false));
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(AbortAwareConversations {
                aborted: Arc::clone(&aborted),
            }),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        h.handle_send_message("c1".into(), "hi".into(), "r1".into(), Arc::new(DropSink))
            .await
            .unwrap();

        assert!(aborted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn get_config_returns_aggregated_settings() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h.handle_command(api::Command::GetConfig).await.unwrap();
        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };

        assert_eq!(config.embeddings.model, "text-embedding-3-small");
        assert_eq!(config.persistence.remote_name, "origin");
    }

    #[tokio::test]
    async fn set_config_applies_changes_and_returns_updated_config() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h
            .handle_command(api::Command::SetConfig {
                changes: api::ConfigChanges {
                    embeddings_connector: Some("openai".into()),
                    embeddings_model: Some("text-embedding-3-large".into()),
                    persistence_enabled: Some(true),
                    persistence_remote_url: Some("git@example.com/repo.git".into()),
                    persistence_remote_name: Some("upstream".into()),
                    persistence_push_on_update: Some(false),
                    ..Default::default()
                },
            })
            .await
            .unwrap();

        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(config.embeddings.model, "text-embedding-3-large");
        assert_eq!(config.persistence.remote_name, "upstream");
        assert!(!config.persistence.push_on_update);
    }

    // --- #314 bridge cutover (2/7): database / backend-tasks / ws-auth ------
    //
    // Each new command gets a handler round-trip test (set then get returns the
    // written value) plus unhappy paths. The fakes mirror the daemon's
    // `empty-clears` semantics so the normalization in the handler arm is
    // exercised end-to-end.

    fn handler_with(
        settings: Arc<ConfigurableSettings>,
    ) -> DefaultAssistantApiHandler<
        FakeAssistant,
        FakeConversations,
        ConfigurableSettings,
        FakeConnections,
        FakeKnowledge,
    > {
        DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            settings,
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
    }

    #[tokio::test]
    async fn set_then_get_database_settings_round_trips() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        let ack = h
            .handle_command(api::Command::SetDatabaseSettings {
                url: "postgres://u:p@host/db".into(),
                max_connections: 12,
            })
            .await
            .unwrap();
        assert_eq!(ack, api::CommandResult::Ack);

        let res = h
            .handle_command(api::Command::GetDatabaseSettings)
            .await
            .unwrap();
        let api::CommandResult::DatabaseSettings(db) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(db.url, "postgres://u:p@host/db");
        assert_eq!(db.max_connections, 12);
    }

    #[tokio::test]
    async fn set_database_settings_empty_url_clears_it() {
        // Seed a URL, then an empty url must clear it (mirrors the D-Bus method).
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        h.handle_command(api::Command::SetDatabaseSettings {
            url: "postgres://u:p@host/db".into(),
            max_connections: 4,
        })
        .await
        .unwrap();
        h.handle_command(api::Command::SetDatabaseSettings {
            url: "   ".into(),
            max_connections: 4,
        })
        .await
        .unwrap();

        let res = h
            .handle_command(api::Command::GetDatabaseSettings)
            .await
            .unwrap();
        let api::CommandResult::DatabaseSettings(db) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(db.url, "", "empty/whitespace url clears the configured URL");
    }

    #[tokio::test]
    async fn set_then_get_backend_tasks_settings_round_trips() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        let ack = h
            .handle_command(api::Command::SetBackendTasksSettings {
                llm_connector: "ollama".into(),
                llm_model: "qwen3".into(),
                llm_base_url: "http://localhost:11434".into(),
                dreaming_enabled: true,
                dreaming_interval_secs: 1800,
                archive_after_days: 14,
            })
            .await
            .unwrap();
        assert_eq!(ack, api::CommandResult::Ack);

        let res = h
            .handle_command(api::Command::GetBackendTasksSettings)
            .await
            .unwrap();
        let api::CommandResult::BackendTasksSettings(bt) = res else {
            panic!("unexpected result variant");
        };
        assert!(bt.has_separate_llm, "a set connector means a separate LLM");
        assert_eq!(bt.llm_connector, "ollama");
        assert_eq!(bt.llm_model, "qwen3");
        assert_eq!(bt.llm_base_url, "http://localhost:11434");
        assert!(bt.dreaming_enabled);
        assert_eq!(bt.dreaming_interval_secs, 1800);
        assert_eq!(bt.archive_after_days, 14);
    }

    #[tokio::test]
    async fn set_backend_tasks_empty_connector_clears_separate_llm() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        // First set a separate LLM, then clear it with an empty connector.
        h.handle_command(api::Command::SetBackendTasksSettings {
            llm_connector: "ollama".into(),
            llm_model: "qwen3".into(),
            llm_base_url: "http://localhost:11434".into(),
            dreaming_enabled: false,
            dreaming_interval_secs: 3600,
            archive_after_days: 0,
        })
        .await
        .unwrap();
        h.handle_command(api::Command::SetBackendTasksSettings {
            llm_connector: "".into(),
            llm_model: "".into(),
            llm_base_url: "".into(),
            dreaming_enabled: false,
            dreaming_interval_secs: 3600,
            archive_after_days: 0,
        })
        .await
        .unwrap();

        let res = h
            .handle_command(api::Command::GetBackendTasksSettings)
            .await
            .unwrap();
        let api::CommandResult::BackendTasksSettings(bt) = res else {
            panic!("unexpected result variant");
        };
        assert!(
            !bt.has_separate_llm,
            "empty connector clears the separate backend-tasks LLM override"
        );
    }

    #[tokio::test]
    async fn set_then_get_ws_auth_settings_round_trips() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        let ack = h
            .handle_command(api::Command::SetWsAuthSettings {
                methods: vec!["password".into(), "oidc".into()],
                oidc_issuer: "https://issuer.example".into(),
                oidc_auth_endpoint: "https://issuer.example/authorize".into(),
                oidc_token_endpoint: "https://issuer.example/token".into(),
                oidc_client_id: "client-123".into(),
                oidc_scopes: "openid profile".into(),
            })
            .await
            .unwrap();
        assert_eq!(ack, api::CommandResult::Ack);

        let res = h
            .handle_command(api::Command::GetWsAuthSettings)
            .await
            .unwrap();
        let api::CommandResult::WsAuthSettings(ws) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(ws.methods, vec!["password".to_string(), "oidc".to_string()]);
        assert_eq!(ws.oidc_issuer, "https://issuer.example");
        assert_eq!(ws.oidc_auth_endpoint, "https://issuer.example/authorize");
        assert_eq!(ws.oidc_token_endpoint, "https://issuer.example/token");
        assert_eq!(ws.oidc_client_id, "client-123");
        assert_eq!(ws.oidc_scopes, "openid profile");
    }

    // --- #314 MCP CRUD round-trip ------------------------------------------

    #[tokio::test]
    async fn add_then_list_mcp_server_round_trips_command_args_namespace() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        let ack = h
            .handle_command(api::Command::AddMcpServer {
                name: "tasks".into(),
                command: "/usr/bin/tasks-mcp".into(),
                args: vec!["--mode".into(), "stdio".into()],
                namespace: Some("jira".into()),
                enabled: true,
            })
            .await
            .unwrap();
        assert_eq!(ack, api::CommandResult::Ack);

        let res = h
            .handle_command(api::Command::ListMcpServers)
            .await
            .unwrap();
        let api::CommandResult::McpServers(servers) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(servers.len(), 1);
        let s = &servers[0];
        assert_eq!(s.name, "tasks");
        // The crux of #314: command/args/namespace written via AddMcpServer
        // must read back on ListMcpServers.
        assert_eq!(s.command, "/usr/bin/tasks-mcp");
        assert_eq!(s.args, vec!["--mode".to_string(), "stdio".to_string()]);
        assert_eq!(s.namespace.as_deref(), Some("jira"));
        assert!(s.enabled);
    }

    #[tokio::test]
    async fn set_mcp_server_enabled_toggles_and_round_trips() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        h.handle_command(api::Command::AddMcpServer {
            name: "tasks".into(),
            command: "/usr/bin/tasks-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
        })
        .await
        .unwrap();
        h.handle_command(api::Command::SetMcpServerEnabled {
            name: "tasks".into(),
            enabled: false,
        })
        .await
        .unwrap();

        let api::CommandResult::McpServers(servers) = h
            .handle_command(api::Command::ListMcpServers)
            .await
            .unwrap()
        else {
            panic!("unexpected result variant");
        };
        assert!(!servers[0].enabled, "enabled flag must round-trip");
        assert_eq!(servers[0].status, "disabled");
    }

    #[tokio::test]
    async fn remove_then_list_mcp_server_drops_it() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        h.handle_command(api::Command::AddMcpServer {
            name: "tasks".into(),
            command: "/usr/bin/tasks-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
        })
        .await
        .unwrap();
        let ack = h
            .handle_command(api::Command::RemoveMcpServer {
                name: "tasks".into(),
            })
            .await
            .unwrap();
        assert_eq!(ack, api::CommandResult::Ack);

        let api::CommandResult::McpServers(servers) = h
            .handle_command(api::Command::ListMcpServers)
            .await
            .unwrap()
        else {
            panic!("unexpected result variant");
        };
        assert!(servers.is_empty());
    }

    #[tokio::test]
    async fn remove_unknown_mcp_server_errors() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(settings);

        let err = h
            .handle_command(api::Command::RemoveMcpServer {
                name: "does-not-exist".into(),
            })
            .await
            .expect_err("removing an unknown MCP server must error");
        assert!(
            format!("{err:?}").contains("not found"),
            "unknown server id should surface a not-found error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn set_enabled_unknown_mcp_server_errors() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(settings);

        let err = h
            .handle_command(api::Command::SetMcpServerEnabled {
                name: "does-not-exist".into(),
                enabled: true,
            })
            .await
            .expect_err("toggling an unknown MCP server must error");
        assert!(format!("{err:?}").contains("not found"), "got: {err:?}");
    }

    #[tokio::test]
    async fn mcp_server_action_unknown_action_errors() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(settings);

        let err = h
            .handle_command(api::Command::McpServerAction {
                action: "frobnicate".into(),
                server: None,
            })
            .await
            .expect_err("an unknown MCP action must error");
        assert!(
            format!("{err:?}").contains("unknown MCP action"),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn mcp_server_action_status_returns_servers() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        h.handle_command(api::Command::AddMcpServer {
            name: "tasks".into(),
            command: "/usr/bin/tasks-mcp".into(),
            args: vec!["--mode".into(), "stdio".into()],
            namespace: Some("jira".into()),
            enabled: true,
        })
        .await
        .unwrap();

        let api::CommandResult::McpServers(servers) = h
            .handle_command(api::Command::McpServerAction {
                action: "status".into(),
                server: Some("tasks".into()),
            })
            .await
            .unwrap()
        else {
            panic!("unexpected result variant");
        };
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].command, "/usr/bin/tasks-mcp");
        assert_eq!(
            servers[0].args,
            vec!["--mode".to_string(), "stdio".to_string()]
        );
        assert_eq!(servers[0].namespace.as_deref(), Some("jira"));
    }

    #[tokio::test]
    async fn add_duplicate_mcp_server_errors() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = handler_with(Arc::clone(&settings));

        h.handle_command(api::Command::AddMcpServer {
            name: "tasks".into(),
            command: "/usr/bin/tasks-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
        })
        .await
        .unwrap();
        let err = h
            .handle_command(api::Command::AddMcpServer {
                name: "tasks".into(),
                command: "/usr/bin/other".into(),
                args: vec![],
                namespace: None,
                enabled: true,
            })
            .await
            .expect_err("adding a duplicate MCP server name must error");
        assert!(
            format!("{err:?}").contains("already exists"),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn get_config_returns_default_personality() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h.handle_command(api::Command::GetConfig).await.unwrap();
        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(
            config.personality.professionalism,
            api::PersonalityLevel::Always
        );
        assert_eq!(config.personality.humor, api::PersonalityLevel::Sometimes);
    }

    #[tokio::test]
    async fn set_config_changes_personality() {
        let settings = Arc::new(ConfigurableSettings::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::clone(&settings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );

        let res = h
            .handle_command(api::Command::SetConfig {
                changes: api::ConfigChanges {
                    personality_humor: Some(api::PersonalityLevel::Never),
                    ..Default::default()
                },
            })
            .await
            .unwrap();

        let api::CommandResult::Config(config) = res else {
            panic!("unexpected result variant");
        };
        assert_eq!(config.personality.humor, api::PersonalityLevel::Never);
        // Other traits untouched.
        assert_eq!(
            config.personality.professionalism,
            api::PersonalityLevel::Always
        );
    }

    // ---- RequestContext tests (issue #105) -----------------------------

    /// A connector's model-listing notice has to survive the domain -> wire
    /// mapping, otherwise the degradation is carried all the way to the
    /// daemon boundary and dropped one step before the client sees it (#648).
    #[test]
    fn model_listing_notices_survive_the_wire_mapping() {
        use desktop_assistant_core::ports::llm::{ModelInfo, ModelListingNotice};

        let listing = CoreModelListing {
            connection_id: "bedrock".into(),
            connection_label: "bedrock (bedrock)".into(),
            model: ModelInfo::new("amazon.titan-embed-text-v2:0"),
            notices: vec![
                ModelListingNotice::partial_catalog(
                    "Inference profiles unavailable",
                    "Grant bedrock:ListInferenceProfiles",
                )
                .with_required_permission("bedrock:ListInferenceProfiles"),
            ],
        };

        let wire = core_model_listing_to_api(listing);
        let notice = wire.notices.first().expect("notice mapped to the wire");
        assert_eq!(notice.kind, api::ModelListingNoticeKindView::PartialCatalog);
        assert_eq!(notice.summary, "Inference profiles unavailable");
        assert_eq!(notice.detail, "Grant bedrock:ListInferenceProfiles");
        assert_eq!(
            notice.required_permission.as_deref(),
            Some("bedrock:ListInferenceProfiles")
        );
    }

    #[test]
    fn request_context_default_resolves_to_sentinel_user() {
        // Single-tenant deploys and unauthenticated paths default to
        // the schema sentinel so storage continues to resolve.
        let ctx = RequestContext::default();
        assert_eq!(ctx.user_id, UserId::default());
        assert_eq!(ctx.user_id.as_str(), "default");
    }

    #[test]
    fn request_context_from_claims_uses_sub_as_user_id() {
        // Phase-1 mapping rule: the JWT `sub` is the user_id verbatim.
        let claims = desktop_assistant_auth_jwt::Claims {
            iss: "test-iss".into(),
            sub: "alice".into(),
            aud: "test-aud".into(),
            exp: 0,
            iat: 0,
            nbf: 0,
            jti: "jti".into(),
        };
        let ctx = RequestContext::from(&claims);
        assert_eq!(ctx.user_id, UserId::new("alice"));
    }

    #[test]
    fn request_context_for_user_pins_explicit_identity() {
        let ctx = RequestContext::for_user(UserId::new("dave"));
        assert_eq!(ctx.user_id.as_str(), "dave");
    }

    #[tokio::test]
    async fn handle_command_for_installs_user_id_in_task_local() {
        use desktop_assistant_core::ports::auth::current_user_id;

        // A handler that records the user_id observed during dispatch.
        struct Observer {
            seen: Arc<Mutex<Option<UserId>>>,
        }
        #[async_trait::async_trait]
        impl AssistantApiHandler for Observer {
            async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
                let observed = current_user_id();
                *self.seen.lock().unwrap() = Some(observed);
                Ok(api::CommandResult::Ack)
            }
            async fn handle_send_message(
                &self,
                _conversation_id: String,
                _content: String,
                _request_id: String,
                _sink: Arc<dyn EventSink>,
            ) -> ApiResult<()> {
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let observer = Observer {
            seen: Arc::clone(&seen),
        };

        let ctx = RequestContext::for_user(UserId::new("alice"));
        let _ = observer
            .handle_command_for(ctx, api::Command::Ping)
            .await
            .unwrap();

        assert_eq!(seen.lock().unwrap().clone(), Some(UserId::new("alice")));
    }

    #[tokio::test]
    async fn handle_command_for_with_default_context_resolves_to_sentinel() {
        use desktop_assistant_core::ports::auth::current_user_id;

        struct Observer {
            seen: Arc<Mutex<Option<UserId>>>,
        }
        #[async_trait::async_trait]
        impl AssistantApiHandler for Observer {
            async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
                let observed = current_user_id();
                *self.seen.lock().unwrap() = Some(observed);
                Ok(api::CommandResult::Ack)
            }
            async fn handle_send_message(
                &self,
                _conversation_id: String,
                _content: String,
                _request_id: String,
                _sink: Arc<dyn EventSink>,
            ) -> ApiResult<()> {
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let observer = Observer {
            seen: Arc::clone(&seen),
        };

        // Boundary: a request with no explicit user context succeeds
        // and resolves to the sentinel. This is the single-tenant
        // contract — independently shippable without #103 or any
        // co-dependent PR.
        let _ = observer
            .handle_command_for(RequestContext::default(), api::Command::Ping)
            .await
            .unwrap();

        assert_eq!(
            seen.lock().unwrap().clone(),
            Some(UserId::default()),
            "default RequestContext must resolve to the schema sentinel"
        );
    }

    #[tokio::test]
    async fn handle_command_uninstrumented_path_still_resolves_to_default() {
        // The non-`_for` entry point continues to work without any
        // user context — backward-compat for existing callers and
        // tests, and the dual-mode contract for single-tenant
        // deploys (no JWT path required to keep things working).
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            Arc::new(FakeConversations),
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );
        let res = h.handle_command(api::Command::Ping).await.unwrap();
        assert_eq!(
            res,
            api::CommandResult::Pong {
                value: "pong".into()
            }
        );
    }

    // --- Idempotency-key completed-dedup (#204) ---------------------------

    /// `ConversationService` double that counts how many turns actually run
    /// and returns a fixed reply, so dedup tests can assert that a replayed
    /// retry did NOT re-invoke the LLM.
    struct CountingConversations {
        runs: Arc<std::sync::atomic::AtomicUsize>,
        reply: String,
    }
    #[async_trait::async_trait]
    impl ConversationService for CountingConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            on_chunk(self.reply.clone());
            Ok(self.reply.clone())
        }
    }

    /// In-memory `IdempotencyKeyStore` scoped by `current_user_id()` exactly
    /// like `PgIdempotencyKeyStore`, so the dedup handler logic (including
    /// per-(user, conversation, key) scoping) can be exercised without
    /// Postgres.
    #[derive(Default)]
    struct InMemoryIdempotency {
        rows: Mutex<std::collections::HashMap<(String, String, String), String>>,
    }
    #[async_trait::async_trait]
    impl IdempotencyKeyStore for InMemoryIdempotency {
        async fn lookup_completed(
            &self,
            conversation_id: &str,
            idempotency_key: &str,
        ) -> Result<Option<String>, CoreError> {
            let uid = desktop_assistant_core::ports::auth::current_user_id()
                .as_str()
                .to_string();
            Ok(self
                .rows
                .lock()
                .unwrap()
                .get(&(
                    uid,
                    conversation_id.to_string(),
                    idempotency_key.to_string(),
                ))
                .cloned())
        }
        async fn record_response(
            &self,
            conversation_id: &str,
            idempotency_key: &str,
            _request_id: &str,
            response: &str,
        ) -> Result<(), CoreError> {
            let uid = desktop_assistant_core::ports::auth::current_user_id()
                .as_str()
                .to_string();
            self.rows.lock().unwrap().insert(
                (
                    uid,
                    conversation_id.to_string(),
                    idempotency_key.to_string(),
                ),
                response.to_string(),
            );
            Ok(())
        }
    }

    fn idem_handler(
        conv: Arc<CountingConversations>,
        store: Arc<InMemoryIdempotency>,
    ) -> DefaultAssistantApiHandler<
        FakeAssistant,
        CountingConversations,
        FakeSettings,
        FakeConnections,
        FakeKnowledge,
    > {
        DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            conv,
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_idempotency_store(store)
    }

    /// Fresh key (no stored row): the turn runs once and its reply is
    /// recorded so a future retry can replay it.
    #[tokio::test]
    async fn idempotency_new_key_runs_turn_and_records_reply() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "answer".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(runs.load(Ordering::SeqCst), 1, "a fresh key runs the turn");
        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
            Some("answer"),
            "the committed reply is recorded under the key"
        );
    }

    /// A key whose turn already completed replays the stored reply and does
    /// NOT re-run the LLM — the crash-safe completed-dedup win.
    #[tokio::test]
    async fn idempotency_completed_key_replays_without_rerunning() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "fresh".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        store
            .record_response("c1", "k1", "orig-req", "stored answer")
            .await
            .unwrap();
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "retry-req".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            0,
            "a completed key must not re-run the turn"
        );
        let evs = sink.0.lock().await.clone();
        assert!(
            matches!(&evs[0], api::Event::UserMessageAdded { request_id, idempotency_key, .. }
                if request_id == "retry-req" && idempotency_key.as_deref() == Some("k1")),
            "replay opens with a UserMessageAdded echoing the retry's request id and key, got {:?}",
            evs.first()
        );
        assert!(
            matches!(&evs[1], api::Event::AssistantDelta { request_id, chunk, .. }
                if request_id == "retry-req" && chunk == "stored answer"),
            "replay emits the stored reply as a delta keyed by the retry's request id, got {:?}",
            evs.get(1)
        );
        assert!(
            matches!(&evs[2], api::Event::AssistantCompleted { request_id, full_response, .. }
                if request_id == "retry-req" && full_response == "stored answer"),
            "replay completes with the stored reply, got {:?}",
            evs.get(2)
        );
    }

    /// The completed-replay retry path echoes the retry's `idempotency_key` on
    /// its opening `UserMessageAdded`, matching the fresh-turn and in-flight
    /// re-attach paths so a retrying client with an optimistic bubble Case-0
    /// dedupes it (#570) rather than rendering a second bubble.
    #[tokio::test]
    async fn completed_replay_echoes_idempotency_key() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "fresh".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        store
            .record_response("c1", "k1", "orig-req", "stored answer")
            .await
            .unwrap();
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "retry-req".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        let evs = sink.0.lock().await.clone();
        let first = evs.first().expect("replay emits at least one event");
        assert!(
            matches!(first, api::Event::UserMessageAdded { conversation_id, request_id, content, idempotency_key }
                if conversation_id == "c1"
                    && request_id == "retry-req"
                    && content == "hi"
                    && idempotency_key.as_deref() == Some("k1")),
            "completed replay must echo the retry's idempotency_key on UserMessageAdded, got {first:?}"
        );
    }

    /// No key: the turn runs and nothing is recorded (backward compat).
    #[tokio::test]
    async fn idempotency_absent_key_runs_and_records_nothing() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "x".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            None,
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(runs.load(Ordering::SeqCst), 1, "no key still runs the turn");
        assert!(
            store.rows.lock().unwrap().is_empty(),
            "a turn with no key records nothing"
        );
    }

    /// Dedup is scoped to the conversation: the same key in a different
    /// conversation is not a hit.
    #[tokio::test]
    async fn idempotency_scoped_to_conversation() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "x".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        store
            .record_response("conv-A", "k1", "o", "stored")
            .await
            .unwrap();
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "conv-B".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "the same key in a different conversation is not a dedup hit"
        );
    }

    /// Dedup is scoped to the user: another user's identical key is not a
    /// hit (cross-tenant isolation).
    #[tokio::test]
    async fn idempotency_scoped_to_user() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "x".into(),
        });
        let store = Arc::new(InMemoryIdempotency::default());
        with_user_id(
            UserId::new("alice"),
            store.record_response("c1", "k1", "o", "alice-reply"),
        )
        .await
        .unwrap();
        let h = idem_handler(conv, Arc::clone(&store));
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        with_user_id(
            UserId::new("bob"),
            h.handle_send_message_with_override(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r1".into(),
                Some("k1".into()),
                sink.clone(),
            ),
        )
        .await
        .unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "another user's identical key must not dedup"
        );
    }

    /// With no store attached, a key is a harmless no-op: the turn runs.
    #[tokio::test]
    async fn idempotency_no_store_is_noop() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(CountingConversations {
            runs: Arc::clone(&runs),
            reply: "x".into(),
        });
        // Note: no `.with_idempotency_store(...)`.
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            conv,
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        );
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r1".into(),
            Some("k1".into()),
            sink.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "a key with no store attached runs the turn as normal"
        );
    }

    /// `ConversationService` double that emits one chunk, blocks until
    /// released, then emits a second chunk and completes — so a second
    /// same-key request can arrive while the turn is provably in flight.
    struct GatedConversations {
        runs: Arc<std::sync::atomic::AtomicUsize>,
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl ConversationService for GatedConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            on_chunk("part1 ".to_string());
            self.release.notified().await;
            on_chunk("part2".to_string());
            Ok("part1 part2".to_string())
        }
    }

    /// #204 phase 2: a second `SendMessage` with the same key while the first
    /// is still running in this process re-attaches to the live turn (replay +
    /// live) instead of running it again. The turn runs exactly once and both
    /// callers see completion; the re-attacher receives chunks emitted after it
    /// joined.
    #[tokio::test]
    async fn idempotency_inflight_reattach_does_not_rerun() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let conv = Arc::new(GatedConversations {
            runs: Arc::clone(&runs),
            release: Arc::clone(&release),
        });
        let registry = Arc::new(crate::background_tasks::BackgroundTaskRegistry::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            conv,
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_registry(Arc::clone(&registry));

        let sink1 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        let sink2 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        // Request 1 registers the in-flight slot (synchronously) and spawns the
        // turn, which blocks after its first chunk.
        let t1 = h
            .start_send_message(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r1".into(),
                Some("k1".into()),
                sink1.clone(),
            )
            .await
            .unwrap()
            .expect("task 1");
        // Request 2, same key while #1 is in flight, must re-attach (not rerun).
        let t2 = h
            .start_send_message(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r2".into(),
                Some("k1".into()),
                sink2.clone(),
            )
            .await
            .unwrap()
            .expect("task 2");

        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&t1))
            .await
            .expect("turn finishes");
        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&t2))
            .await
            .expect("re-attach finishes");

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "the turn must run exactly once for two same-key requests"
        );

        let ev1 = sink1.0.lock().await.clone();
        let ev2 = sink2.0.lock().await.clone();
        assert!(
            ev1.iter()
                .any(|e| matches!(e, api::Event::AssistantCompleted { .. })),
            "original caller completes"
        );
        assert!(
            ev2.iter().any(|e| matches!(
                e,
                api::Event::AssistantCompleted { request_id, .. } if request_id == "r2"
            )),
            "re-attached caller completes, with events restamped to ITS request id (r2), \
             not the original turn's (r1) — so a request_id-correlating client matches them"
        );
        let live2: String = ev2
            .iter()
            .filter_map(|e| match e {
                api::Event::AssistantDelta { chunk, .. } => Some(chunk.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            live2.contains("part2"),
            "re-attacher receives chunks emitted after it joined: {live2:?}"
        );
    }

    // ----------------------------------------------------------------
    // #440: concurrency / failure edges for the in-flight slot and
    // idempotency-on-failure.
    // ----------------------------------------------------------------

    /// Selects how [`ModeConversations::send_prompt`] behaves on its first
    /// call. Both modes recover on the second call so a test can prove a
    /// *fresh* re-run (not a re-attach to the failed turn) actually completes.
    #[derive(Clone, Copy)]
    enum ConvMode {
        /// First turn panics mid-flight; later turns succeed.
        PanicThenOk,
        /// First turn returns `Err`; later turns succeed.
        FailThenOk,
    }

    /// `ConversationService` double whose first turn fails (panics or `Err`,
    /// per [`ConvMode`]) and whose subsequent turns emit `"recovered"` and
    /// complete. `runs` counts every `send_prompt` entry so a test can assert
    /// how many turns actually executed.
    struct ModeConversations {
        runs: Arc<std::sync::atomic::AtomicUsize>,
        mode: ConvMode,
    }
    #[async_trait::async_trait]
    impl ConversationService for ModeConversations {
        async fn create_conversation(
            &self,
            title: String,
            _tags: Vec<String>,
        ) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            let n = self.runs.fetch_add(1, Ordering::SeqCst) + 1;
            if n == 1 {
                match self.mode {
                    ConvMode::PanicThenOk => panic!("first keyed turn panics before completing"),
                    ConvMode::FailThenOk => return Err(CoreError::Llm("first turn failed".into())),
                }
            }
            on_chunk("recovered".to_string());
            Ok("recovered".to_string())
        }
    }

    /// #440 HIGH (`lib.rs:2403-2405`): the in-flight slot is freed inside the
    /// turn body *after* `run_send_turn().await`, so a panic skips the inline
    /// removal. Without a panic-safe free, the slot is orphaned and a later
    /// same-key send re-attaches to a dead hub (its broadcast sender never
    /// drops) and hangs forever instead of running fresh. This test drives a
    /// keyed send whose first turn panics, then a second same-key send, and
    /// asserts the second runs a *fresh* turn to completion.
    #[tokio::test]
    async fn panicking_turn_frees_inflight_slot() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(ModeConversations {
            runs: Arc::clone(&runs),
            mode: ConvMode::PanicThenOk,
        });
        let registry = Arc::new(crate::background_tasks::BackgroundTaskRegistry::new());
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            conv,
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_registry(Arc::clone(&registry));

        // Request 1: a keyed send whose turn panics mid-flight.
        let sink1 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        let t1 = h
            .start_send_message(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r1".into(),
                Some("k1".into()),
                sink1.clone(),
            )
            .await
            .unwrap()
            .expect("task 1");
        // The panicking turn still finalizes (as Failed) — the registry runs
        // the body in a child task and catches the panic (#171). The slot must
        // be freed by the time the task is terminal.
        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&t1))
            .await
            .expect("panicking turn finalizes, not hangs");

        // Request 2: same key, after the first turn died. It must NOT re-attach
        // to the orphaned (dead) hub — it must claim a fresh slot and run.
        let sink2 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        let t2 = h
            .start_send_message(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r2".into(),
                Some("k1".into()),
                sink2.clone(),
            )
            .await
            .unwrap()
            .expect("task 2");
        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&t2))
            .await
            .expect(
                "second same-key send must run fresh and complete — a hang here means it \
                 re-attached to the panicked turn's orphaned in-flight slot (dead hub)",
            );

        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "the second same-key send must run a FRESH turn (not re-attach to the dead hub)"
        );
        let ev2 = sink2.0.lock().await.clone();
        assert!(
            ev2.iter().any(|e| matches!(
                e,
                api::Event::AssistantCompleted { request_id, full_response, .. }
                    if request_id == "r2" && full_response == "recovered"
            )),
            "the fresh retry completes with its own reply, got {ev2:?}"
        );
    }

    /// #440 HIGH (`lib.rs:2297-2311` + `inflight.rs:198-212`): two *concurrent*
    /// same-key sends must run the turn exactly once — the loser re-attaches to
    /// the winner's live turn. Unlike the existing sequential test
    /// (`idempotency_inflight_reattach_does_not_rerun`), both requests race
    /// through `start_send_message` behind a barrier, so this exercises the
    /// atomic check-and-insert in `InFlightRegistry::register`.
    #[tokio::test]
    async fn two_concurrent_sends_same_key_run_once() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let conv = Arc::new(GatedConversations {
            runs: Arc::clone(&runs),
            release: Arc::clone(&release),
        });
        let registry = Arc::new(crate::background_tasks::BackgroundTaskRegistry::new());
        let h = Arc::new(
            DefaultAssistantApiHandler::new(
                Arc::new(FakeAssistant),
                conv,
                Arc::new(FakeSettings),
                Arc::new(FakeConnections),
                Arc::new(FakeKnowledge),
            )
            .with_registry(Arc::clone(&registry)),
        );

        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut handles = Vec::new();
        for req in ["ra", "rb"] {
            let h = Arc::clone(&h);
            let barrier = Arc::clone(&barrier);
            handles.push(tokio::spawn(async move {
                let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
                // Line the two calls up as closely as the runtime allows so
                // they genuinely race on the in-flight registration.
                barrier.wait().await;
                let task_id = h
                    .start_send_message(
                        "c1".into(),
                        "hi".into(),
                        None,
                        String::new(),
                        req.to_string(),
                        Some("shared-key".into()),
                        sink.clone(),
                    )
                    .await
                    .unwrap()
                    .expect("task id");
                (task_id, sink)
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.expect("send task joins"));
        }

        // Exactly one turn is blocked on the winner's gate; wake it.
        release.notify_one();

        for (task_id, _) in &results {
            tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(task_id))
                .await
                .expect("both the winner and the re-attacher finish");
        }

        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "two concurrent same-key sends must run the turn exactly once"
        );
        for (_, sink) in &results {
            let evs = sink.0.lock().await.clone();
            assert!(
                evs.iter()
                    .any(|e| matches!(e, api::Event::AssistantCompleted { .. })),
                "both the winner and the re-attacher must complete, got {evs:?}"
            );
        }
    }

    /// #440 (`lib.rs:3031`): `record_response` runs only on the `Ok` branch, so
    /// a *failed* keyed turn records nothing and a retry re-runs from scratch
    /// (rather than replaying a phantom success). Uses the no-registry direct
    /// path so the failure surfaces as an `Err` return.
    #[tokio::test]
    async fn failed_keyed_turn_records_nothing_and_retry_reruns() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let conv = Arc::new(ModeConversations {
            runs: Arc::clone(&runs),
            mode: ConvMode::FailThenOk,
        });
        let store = Arc::new(InMemoryIdempotency::default());
        let store_dyn: Arc<dyn desktop_assistant_core::ports::store::IdempotencyKeyStore> =
            store.clone();
        let h = DefaultAssistantApiHandler::new(
            Arc::new(FakeAssistant),
            conv,
            Arc::new(FakeSettings),
            Arc::new(FakeConnections),
            Arc::new(FakeKnowledge),
        )
        .with_idempotency_store(store_dyn);

        // First attempt: the turn fails, so the handler returns Err.
        let sink1 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        let first = h
            .handle_send_message_with_override(
                "c1".into(),
                "hi".into(),
                None,
                String::new(),
                "r1".into(),
                Some("k1".into()),
                sink1.clone(),
            )
            .await;
        assert!(first.is_err(), "a failed turn surfaces as Err");
        assert_eq!(runs.load(Ordering::SeqCst), 1, "the first turn ran");
        assert!(
            store.lookup_completed("c1", "k1").await.unwrap().is_none(),
            "a failed keyed turn must record NOTHING for the key"
        );

        // Retry with the same key: nothing was recorded, so it re-runs (does
        // not replay), and this time succeeds and records the reply.
        let sink2 = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));
        h.handle_send_message_with_override(
            "c1".into(),
            "hi".into(),
            None,
            String::new(),
            "r2".into(),
            Some("k1".into()),
            sink2.clone(),
        )
        .await
        .expect("retry succeeds");
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "the retry of a failed key must run a fresh turn, not replay"
        );
        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
            Some("recovered"),
            "the successful retry records its reply under the key"
        );
    }
}
