use serde::{Deserialize, Serialize};

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
}
