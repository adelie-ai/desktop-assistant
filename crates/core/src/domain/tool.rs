use serde::{Deserialize, Serialize};

/// Where a tool physically executes (issue #243).
///
/// MCP servers and built-in tools run on the **daemon's** host; tools a
/// connection registers via `RegisterClientTools` run on the **registering
/// client's** machine. The same capability (e.g. a terminal) can exist in
/// both places, so the service needs to know the locality of each tool the
/// LLM can call in order to route the work to the right machine and to
/// describe it accurately in the per-turn tool note.
///
/// ## Co-location and the phase boundary
///
/// In the common single-machine setup the client and daemon are the *same*
/// host, so a `Client` tool and a `Server` tool are physically co-located
/// and the distinction collapses to "this machine". This PR infers
/// co-location from the connection's [`TransportKind`] (a local transport ⇒
/// same machine); a later phase will replace that heuristic with an explicit
/// per-machine system-id handshake (clients send their id; the server
/// compares it to its own, preferring `/etc/machine-id` when suitable). The
/// `host` / `id` fields are populated by hostname today precisely so that
/// future id can slot in without reshaping this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "where")]
pub enum ToolLocality {
    /// Runs on the daemon's host. Used for MCP servers and built-in tools.
    /// `host` is the daemon's self-identity label (hostname today; a stable
    /// machine-id in the follow-up phase).
    Server { host: String },
    /// Runs on a connected client's machine. Used for tools registered via
    /// `RegisterClientTools`. `id` is the connection/client identity and
    /// `label` is a human-friendly device name for the tool note.
    Client { id: String, label: String },
}

impl ToolLocality {
    /// Convenience constructor for a server-side (daemon-host) locality.
    pub fn server(host: impl Into<String>) -> Self {
        Self::Server { host: host.into() }
    }

    /// Convenience constructor for a client-side (registering-machine) locality.
    pub fn client(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self::Client {
            id: id.into(),
            label: label.into(),
        }
    }

    /// True for a server-side (daemon-host) tool.
    pub fn is_server(&self) -> bool {
        matches!(self, Self::Server { .. })
    }

    /// True for a client-side (registering-machine) tool.
    pub fn is_client(&self) -> bool {
        matches!(self, Self::Client { .. })
    }
}

/// How a connection reaches the daemon, used to infer tool co-location
/// (issue #243).
///
/// The signal is deliberately coarse: a local transport (Unix-domain socket
/// or D-Bus) can only be reached from the daemon's own machine, so a client
/// on such a transport and the daemon are the *same* host — a `Client` tool
/// and a `Server` tool are co-located and the locality distinction collapses.
/// A WebSocket connection can be remote, so `Server` (daemon host) and
/// `Client` (the WS peer's machine) are treated as distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// Unix-domain socket: same machine as the daemon ⇒ co-located.
    Uds,
    /// D-Bus: same machine as the daemon ⇒ co-located.
    Dbus,
    /// WebSocket: potentially remote ⇒ server and client are distinct.
    WebSocket,
}

impl TransportKind {
    /// Whether a client on this transport is co-located with the daemon
    /// (same machine). True for the local transports (UDS / D-Bus), false
    /// for WebSocket, which may terminate on a different host.
    pub fn is_co_located(self) -> bool {
        matches!(self, Self::Uds | Self::Dbus)
    }
}

/// Definition of a tool that can be called by the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A request from the LLM to call a specific tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// A named group of tools for deferred loading via hosted tool search.
///
/// When using OpenAI's hosted tool search, tools within a namespace are
/// sent with `defer_loading: true` so the model can discover them on demand
/// instead of having them all in the active context window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolNamespace {
    pub name: String,
    pub description: String,
    pub tools: Vec<ToolDefinition>,
}

impl ToolNamespace {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tools,
        }
    }
}

/// The result of executing a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_creation() {
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"]
        });
        let tool = ToolDefinition::new("read_file", "Read a file from disk", params.clone());
        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description, "Read a file from disk");
        assert_eq!(tool.parameters, params);
    }

    #[test]
    fn tool_definition_serialization_roundtrip() {
        let tool = ToolDefinition::new(
            "write_file",
            "Write content to a file",
            serde_json::json!({"type": "object"}),
        );
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, tool);
    }

    #[test]
    fn tool_call_creation() {
        let call = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/test.txt"}"#);
        assert_eq!(call.id, "call-1");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments, r#"{"path": "/tmp/test.txt"}"#);
    }

    #[test]
    fn tool_call_serialization_roundtrip() {
        let call = ToolCall::new("call-2", "write_file", r#"{"path": "/tmp/out.txt"}"#);
        let json = serde_json::to_string(&call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, call);
    }

    #[test]
    fn tool_result_creation() {
        let result = ToolResult::new("call-1", "file contents here");
        assert_eq!(result.tool_call_id, "call-1");
        assert_eq!(result.content, "file contents here");
    }

    #[test]
    fn tool_result_serialization_roundtrip() {
        let result = ToolResult::new("call-1", "success");
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, result);
    }

    #[test]
    fn tool_definition_clone() {
        let tool = ToolDefinition::new("test", "desc", serde_json::json!({}));
        let cloned = tool.clone();
        assert_eq!(tool, cloned);
    }

    #[test]
    fn tool_call_clone() {
        let call = ToolCall::new("id", "name", "args");
        let cloned = call.clone();
        assert_eq!(call, cloned);
    }

    #[test]
    fn tool_namespace_creation() {
        let tools = vec![ToolDefinition::new("t1", "desc1", serde_json::json!({}))];
        let ns = ToolNamespace::new("my_ns", "A namespace", tools.clone());
        assert_eq!(ns.name, "my_ns");
        assert_eq!(ns.description, "A namespace");
        assert_eq!(ns.tools, tools);
    }

    #[test]
    fn tool_namespace_serialization_roundtrip() {
        let ns = ToolNamespace::new(
            "test_ns",
            "Test namespace",
            vec![ToolDefinition::new(
                "t1",
                "desc",
                serde_json::json!({"type": "object"}),
            )],
        );
        let json = serde_json::to_string(&ns).unwrap();
        let deserialized: ToolNamespace = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ns);
    }

    // --- ToolLocality / TransportKind (issue #243) -------------------------

    #[test]
    fn tool_locality_constructors_and_predicates() {
        let server = ToolLocality::server("daemon-host");
        assert!(server.is_server());
        assert!(!server.is_client());
        assert_eq!(
            server,
            ToolLocality::Server {
                host: "daemon-host".into()
            }
        );

        let client = ToolLocality::client("conn-7", "Dave's laptop");
        assert!(client.is_client());
        assert!(!client.is_server());
        assert_eq!(
            client,
            ToolLocality::Client {
                id: "conn-7".into(),
                label: "Dave's laptop".into()
            }
        );
    }

    #[test]
    fn tool_locality_serde_round_trip() {
        for loc in [
            ToolLocality::server("host-a"),
            ToolLocality::client("id-1", "Laptop"),
        ] {
            let json = serde_json::to_string(&loc).unwrap();
            let parsed: ToolLocality = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, loc);
        }
    }

    #[test]
    fn transport_kind_co_location() {
        // Local transports collapse client/server to the same machine.
        assert!(TransportKind::Uds.is_co_located());
        assert!(TransportKind::Dbus.is_co_located());
        // WebSocket may be remote, so locality is distinct.
        assert!(!TransportKind::WebSocket.is_co_located());
    }
}
