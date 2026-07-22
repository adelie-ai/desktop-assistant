//! Tool conversion: domain [`ToolDefinition`]s into `chat/completions`
//! `tools[]`, plus the top-level tool-schema sanitizer.

use serde::{Deserialize, Serialize};

use desktop_assistant_core::domain::ToolDefinition;

/// A `chat/completions` tool entry: `{type:"function", function:{...}}`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatTool {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function name, description, and parameter schema.
    pub function: ChatToolFunction,
}

/// The `function` object of a [`ChatTool`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatToolFunction {
    /// The tool/function name.
    pub name: String,
    /// A human-readable description shown to the model.
    pub description: String,
    /// The JSON Schema for the tool's parameters, passed through
    /// [`sanitize_tool_schema`].
    pub parameters: serde_json::Value,
}

/// Convert domain [`ToolDefinition`]s into `chat/completions` `tools[]`.
///
/// Each tool becomes `{type:"function", function:{name, description,
/// parameters}}`, with the parameter schema passed through
/// [`sanitize_tool_schema`] so a single MCP schema carrying a top-level
/// composite keyword cannot 400 the whole turn.
pub fn to_chat_tools(tools: &[ToolDefinition]) -> Vec<ChatTool> {
    tools
        .iter()
        .map(|t| ChatTool {
            tool_type: "function".to_string(),
            function: ChatToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: sanitize_tool_schema(&t.parameters),
            },
        })
        .collect()
}

/// Strip the composite keywords strict OpenAI-compatible backends reject at the
/// **top level** of a tool parameters schema.
///
/// One misbehaving MCP schema otherwise 400s the entire turn -- every other
/// tool goes down with the one offender (the Bedrock #214/#67 failure class,
/// and OpenRouter's long tail proxies to equally strict backends). Behavior
/// mirrors `llm-bedrock`'s `sanitize_tool_schema` exactly:
///
/// - Acts only on a JSON **object** schema; any other value (`true`, a string,
///   `null`, ...) is returned untouched, never wrapped.
/// - Removes top-level `oneOf`, `anyOf`, `allOf` only. `not` is left alone.
/// - Does **not** recurse into `properties.*` or anywhere else -- nested
///   composites inside property subschemas are legal and must be preserved.
/// - If stripping left the object without a `type`, injects `"type":"object"`
///   so the result is still a valid object schema. A clean schema that
///   legitimately omits `type` is returned exactly as given.
pub fn sanitize_tool_schema(schema: &serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(map) = schema else {
        // Non-object schema -- nothing to strip, and we must not wrap it.
        return schema.clone();
    };
    let mut map = map.clone();

    let mut removed_any = false;
    for key in ["oneOf", "anyOf", "allOf"] {
        if map.remove(key).is_some() {
            removed_any = true;
        }
    }

    // Only ensure a `type` when we actually altered the schema and left it
    // without one -- a clean schema that omits `type` is left as-is.
    if removed_any && !map.contains_key("type") {
        map.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
    }

    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- to_chat_tools ---------------------------------------------------

    #[test]
    fn to_chat_tools_produces_function_shape() {
        let tools = vec![ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
        )];
        let out = to_chat_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tool_type, "function");
        assert_eq!(out[0].function.name, "read_file");
        assert_eq!(out[0].function.description, "Read a file");
        assert_eq!(
            out[0].function.parameters["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn to_chat_tools_serializes_to_expected_json() {
        let tools = vec![ToolDefinition::new(
            "t",
            "d",
            serde_json::json!({"type":"object"}),
        )];
        let json = serde_json::to_value(&to_chat_tools(&tools)[0]).expect("serialize");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "t");
        assert_eq!(json["function"]["description"], "d");
        assert_eq!(json["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn empty_tools_yields_empty_vec() {
        assert!(to_chat_tools(&[]).is_empty());
    }

    #[test]
    fn to_chat_tools_sanitizes_schema_top_level_composite() {
        // End-to-end: a top-level oneOf must not reach the wire schema.
        let tools = vec![ToolDefinition::new(
            "terminal_execute",
            "run",
            serde_json::json!({
                "type":"object",
                "properties":{"cmd":{"type":"string"}},
                "oneOf":[{"required":["cmd"]}],
            }),
        )];
        let out = to_chat_tools(&tools);
        assert!(
            out[0].function.parameters.get("oneOf").is_none(),
            "oneOf must be stripped before it reaches the request"
        );
        assert_eq!(out[0].function.parameters["type"], "object");
        assert!(out[0].function.parameters.get("properties").is_some());
    }

    // --- sanitize_tool_schema -------------------------------------------

    #[test]
    fn schema_strips_top_level_one_of() {
        let got = sanitize_tool_schema(&serde_json::json!({
            "type":"object",
            "description":"a tool",
            "properties":{"x":{"type":"string"}},
            "required":["x"],
            "oneOf":[{"required":["x"]}],
        }));
        assert!(got.get("oneOf").is_none(), "oneOf must be stripped");
        assert_eq!(got["type"], "object");
        assert_eq!(got["description"], "a tool");
        assert_eq!(got["properties"]["x"]["type"], "string");
        assert_eq!(got["required"], serde_json::json!(["x"]));
    }

    #[test]
    fn schema_strips_top_level_any_of() {
        let got = sanitize_tool_schema(&serde_json::json!({
            "type":"object",
            "anyOf":[{"type":"object"},{"type":"null"}],
        }));
        assert!(got.get("anyOf").is_none(), "anyOf must be stripped");
        assert_eq!(got["type"], "object");
    }

    #[test]
    fn schema_strips_top_level_all_of() {
        let got = sanitize_tool_schema(&serde_json::json!({
            "type":"object",
            "allOf":[{"required":["a"]},{"required":["b"]}],
        }));
        assert!(got.get("allOf").is_none(), "allOf must be stripped");
        assert_eq!(got["type"], "object");
    }

    #[test]
    fn schema_injects_type_when_missing_after_stripping() {
        let got = sanitize_tool_schema(&serde_json::json!({
            "oneOf":[{"type":"object"},{"type":"string"}],
        }));
        assert!(got.get("oneOf").is_none());
        assert_eq!(got["type"], "object", "missing type must default to object");
    }

    #[test]
    fn schema_clean_object_is_unchanged() {
        let clean = serde_json::json!({
            "type":"object",
            "properties":{"a":{"type":"integer"}},
        });
        assert_eq!(sanitize_tool_schema(&clean), clean);
    }

    #[test]
    fn schema_no_type_without_composite_is_not_injected() {
        // We only inject `type` when we actually stripped something.
        let no_type = serde_json::json!({
            "properties":{"a":{"type":"integer"}},
        });
        assert_eq!(sanitize_tool_schema(&no_type), no_type);
    }

    #[test]
    fn schema_does_not_recurse_into_properties() {
        let got = sanitize_tool_schema(&serde_json::json!({
            "type":"object",
            "properties":{
                "foo":{"anyOf":[{"type":"string"},{"type":"null"}]},
            },
        }));
        assert_eq!(
            got["properties"]["foo"]["anyOf"],
            serde_json::json!([{"type":"string"},{"type":"null"}]),
            "nested anyOf must be preserved"
        );
    }

    #[test]
    fn schema_non_object_values_pass_through() {
        assert_eq!(
            sanitize_tool_schema(&serde_json::json!(true)),
            serde_json::json!(true)
        );
        assert_eq!(
            sanitize_tool_schema(&serde_json::json!("a string")),
            serde_json::json!("a string")
        );
        assert_eq!(
            sanitize_tool_schema(&serde_json::Value::Null),
            serde_json::Value::Null
        );
    }
}
