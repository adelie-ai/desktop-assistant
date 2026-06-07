//! Transport liveness timeouts (#221).
//!
//! The socket transports (UDS / WebSocket) used to wait forever: a command
//! awaited its response `oneshot` with no deadline, the event fan-out only
//! noticed a *closed* connection (never a silent-but-open one), and the WS had
//! no keepalive — so a server that accepted the connection and then went quiet
//! would hang the client indefinitely. These bounds turn every "wait forever"
//! into a bounded wait that surfaces a clear error.
//!
//! They are deliberately generous: an LLM turn can stream for a long time, and
//! the orchestrator now emits periodic `AssistantStatus` while it works, so a
//! healthy turn keeps resetting the stall clock. The values are exposed here
//! (rather than buried as literals in each transport) so they can be tuned in
//! one place and referenced by tests.

use std::time::Duration;

/// How long a single command waits for its `Result`/`Error` frame before the
/// dispatch is abandoned and a transport error is returned. A request that
/// times out is removed from the pending map so the slot can't leak.
///
/// This bounds the *acknowledgement* of a command, not a streamed turn: even a
/// long LLM turn acks promptly (the daemon replies with `SendMessageAck` and
/// streams the body as events), so 30s is comfortably above any healthy ack
/// latency while still catching a wedged server.
pub const DISPATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum gap between events on the signal stream before the connection is
/// treated as stalled (open but silent). Any received event resets the clock,
/// and the orchestrator emits periodic `AssistantStatus`, so a live connection
/// stays well under this even mid-turn. On expiry the fan-out surfaces a
/// terminal `Disconnected` so a waiting client errors instead of hanging.
pub const EVENT_STALL_TIMEOUT: Duration = Duration::from_secs(90);

/// How often the WebSocket client sends a keepalive `Ping`. A dead-but-open
/// TCP socket (e.g. a silently-dropped NAT mapping) is detected when the ping
/// write fails or no traffic — including the matching `Pong` — arrives within
/// [`EVENT_STALL_TIMEOUT`]. Well under the stall window so several pings ride
/// inside one stall budget.
pub const WS_PING_INTERVAL: Duration = Duration::from_secs(20);

/// First delay before the [`Connector`](crate::Connector) retries connecting
/// after the transport drops (#246). Each subsequent failure doubles the delay
/// up to [`RECONNECT_BACKOFF_MAX`]; a small jitter is added so a fleet of
/// clients reconnecting after a single daemon restart doesn't thunder in
/// lockstep.
pub const RECONNECT_BACKOFF_INITIAL: Duration = Duration::from_millis(500);

/// Cap on the [`Connector`](crate::Connector) reconnect backoff (#246). The
/// delay grows exponentially from [`RECONNECT_BACKOFF_INITIAL`] but never
/// exceeds this, so a long outage settles into a steady ~30s retry cadence
/// rather than spinning tightly or backing off into the minutes.
pub const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(30);
