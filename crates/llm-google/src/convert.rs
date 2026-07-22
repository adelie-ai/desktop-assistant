//! Domain `Message`/`ToolDefinition` -> Gemini `generateContent` mapping.
//!
//! - System messages are concatenated into a single `systemInstruction`.
//! - User text -> `contents[{role:"user", parts:[{text}]}]`.
//! - Assistant text -> `contents[{role:"model", parts:[{text}]}]`.
//! - Assistant tool calls -> `parts:[{functionCall:{name, args}}]` (args
//!   parsed from the domain arguments string into a JSON object).
//! - Tool results -> `contents[{role:"user", parts:[{functionResponse:{name,
//!   response}}]}]`, `response` a JSON object; consecutive tool results merge
//!   into one `user` turn (the Bedrock message-merging pattern).
//!
//! Gemini correlates a `functionResponse` to its call by **name**, but the
//! domain `Message::tool_result` carries only a `tool_call_id`, not the tool
//! name. We recover the name by scanning the assistant `tool_calls` seen
//! earlier in the same history (id -> name), so the reverse mapping is exact.

use std::collections::HashMap;

use desktop_assistant_core::domain::{Message, Role, ToolDefinition};
use serde_json::Value;

use crate::schema::sanitize_tool_schema;
use crate::wire::{Content, FunctionDeclaration, Part, SystemInstruction, Tool};

/// Gemini turn-role literals.
pub(crate) const ROLE_USER: &str = "user";
pub(crate) const ROLE_MODEL: &str = "model";

/// Convert a domain message history into a Gemini `systemInstruction` plus the
/// turn `contents`. See the module docs for the mapping rules.
pub fn convert_messages(messages: &[Message]) -> (Option<SystemInstruction>, Vec<Content>) {
    let mut system_parts: Vec<Part> = Vec::new();
    let mut contents: Vec<Content> = Vec::new();
    // id -> function name, accumulated from assistant tool_calls so a later
    // tool-result message can recover the name Gemini's functionResponse needs.
    let mut id_to_name: HashMap<String, String> = HashMap::new();

    for msg in messages {
        match msg.role {
            Role::System => system_parts.push(Part::text(msg.content.clone())),
            Role::User => push_user_part(&mut contents, Part::text(msg.content.clone())),
            Role::Assistant => {
                let mut parts: Vec<Part> = Vec::new();
                if !msg.content.is_empty() {
                    parts.push(Part::text(msg.content.clone()));
                }
                for tc in &msg.tool_calls {
                    id_to_name.insert(tc.id.clone(), tc.name.clone());
                    parts.push(Part::function_call(
                        tc.name.clone(),
                        parse_args(&tc.arguments),
                    ));
                }
                // A model turn with neither text nor calls still needs a part so
                // role alternation is preserved.
                if parts.is_empty() {
                    parts.push(Part::text(String::new()));
                }
                contents.push(Content {
                    role: Some(ROLE_MODEL.to_string()),
                    parts,
                });
            }
            Role::Tool => {
                let name = msg
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_name.get(id))
                    .cloned()
                    .unwrap_or_default();
                let response = tool_result_response(&msg.content);
                push_user_part(&mut contents, Part::function_response(name, response));
            }
        }
    }

    let system = (!system_parts.is_empty()).then_some(SystemInstruction {
        parts: system_parts,
    });
    (system, contents)
}

/// Append `part` to the last `user` turn if there is one, else start a new
/// `user` turn. This merges consecutive tool results (and consecutive user
/// text) into a single turn, matching Gemini's expectation that a set of tool
/// results is delivered as one `user` content.
fn push_user_part(contents: &mut Vec<Content>, part: Part) {
    if let Some(last) = contents.last_mut()
        && last.role.as_deref() == Some(ROLE_USER)
    {
        last.parts.push(part);
    } else {
        contents.push(Content {
            role: Some(ROLE_USER.to_string()),
            parts: vec![part],
        });
    }
}

/// Parse a tool-call `arguments` string into a JSON **object**. Non-object or
/// unparseable arguments become an empty object (Gemini requires `args` to be
/// an object).
fn parse_args(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}))
}

/// Build the `tools` array from tool definitions, sanitizing each schema into
/// Gemini's OpenAPI subset. Returns an empty vec when there are no tools (the
/// caller omits the field). All declarations are flattened under a single
/// `functionDeclarations` entry, per Gemini's schema.
pub fn build_tools(tools: &[ToolDefinition]) -> Vec<Tool> {
    if tools.is_empty() {
        return Vec::new();
    }
    vec![Tool {
        function_declarations: tools.iter().map(function_declaration).collect(),
    }]
}

/// Build a single sanitized `FunctionDeclaration`. `parameters` is omitted
/// when the schema is an empty object (a no-argument tool), which Gemini
/// prefers over an empty `{}` schema.
pub fn function_declaration(tool: &ToolDefinition) -> FunctionDeclaration {
    let sanitized = sanitize_tool_schema(tool.parameters.clone());
    let is_empty = matches!(&sanitized, Value::Object(m) if m.is_empty()) || sanitized.is_null();
    FunctionDeclaration {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: (!is_empty).then_some(sanitized),
    }
}

/// Coerce a tool-result string into the JSON **object** Gemini requires for
/// `functionResponse.response`. A result that already parses as a JSON object
/// is used as-is; anything else (a bare string, array, number, or unparseable
/// text) is wrapped as `{ "result": <content> }`.
pub fn tool_result_response(content: &str) -> Value {
    match serde_json::from_str::<Value>(content) {
        Ok(v @ Value::Object(_)) => v,
        _ => serde_json::json!({ "result": content }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::{Role, ToolCall};
    use serde_json::json;

    #[test]
    fn system_message_becomes_system_instruction() {
        let messages = vec![
            Message::new(Role::System, "you are helpful"),
            Message::new(Role::User, "hi"),
        ];
        let (system, contents) = convert_messages(&messages);
        let system = system.expect("system instruction present");
        assert_eq!(system.parts.len(), 1);
        assert_eq!(system.parts[0].text.as_deref(), Some("you are helpful"));
        // System is not in the turn contents.
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some(ROLE_USER));
    }

    #[test]
    fn multiple_system_messages_concatenate_into_parts() {
        let messages = vec![
            Message::new(Role::System, "main instruction"),
            Message::new(Role::System, "context summary"),
            Message::new(Role::User, "hi"),
        ];
        let (system, _) = convert_messages(&messages);
        let system = system.expect("system instruction present");
        assert_eq!(system.parts.len(), 2);
        assert_eq!(system.parts[0].text.as_deref(), Some("main instruction"));
        assert_eq!(system.parts[1].text.as_deref(), Some("context summary"));
    }

    #[test]
    fn no_system_message_yields_none() {
        let (system, _) = convert_messages(&[Message::new(Role::User, "hi")]);
        assert!(system.is_none());
    }

    #[test]
    fn user_and_assistant_text_roles() {
        let messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];
        let (_, contents) = convert_messages(&messages);
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role.as_deref(), Some(ROLE_USER));
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hello"));
        assert_eq!(contents[1].role.as_deref(), Some(ROLE_MODEL));
        assert_eq!(contents[1].parts[0].text.as_deref(), Some("hi there"));
    }

    #[test]
    fn assistant_tool_call_becomes_function_call_with_object_args() {
        let calls = vec![ToolCall::new("c1", "get_weather", r#"{"city":"NYC"}"#)];
        let messages = vec![Message::assistant_with_tool_calls(calls)];
        let (_, contents) = convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some(ROLE_MODEL));
        let fc = contents[0].parts[0]
            .function_call
            .as_ref()
            .expect("functionCall part");
        assert_eq!(fc.name, "get_weather");
        assert_eq!(
            fc.args,
            json!({"city": "NYC"}),
            "args must be a JSON object"
        );
    }

    #[test]
    fn tool_result_round_trips_name_from_assistant_call() {
        // The functionResponse must carry the function NAME, recovered from the
        // preceding assistant tool_call by id.
        let calls = vec![ToolCall::new("c1", "get_weather", r#"{"city":"NYC"}"#)];
        let messages = vec![
            Message::assistant_with_tool_calls(calls),
            Message::tool_result("c1", r#"{"temp":72}"#),
        ];
        let (_, contents) = convert_messages(&messages);
        assert_eq!(contents.len(), 2);
        let fr = contents[1].parts[0]
            .function_response
            .as_ref()
            .expect("functionResponse part");
        assert_eq!(fr.name, "get_weather", "name recovered from the call id");
        assert_eq!(fr.response, json!({"temp": 72}));
        assert_eq!(contents[1].role.as_deref(), Some(ROLE_USER));
    }

    #[test]
    fn consecutive_tool_results_merge_into_one_user_turn() {
        let calls = vec![
            ToolCall::new("c1", "tool_a", "{}"),
            ToolCall::new("c2", "tool_b", "{}"),
        ];
        let messages = vec![
            Message::assistant_with_tool_calls(calls),
            Message::tool_result("c1", "result-a"),
            Message::tool_result("c2", "result-b"),
        ];
        let (_, contents) = convert_messages(&messages);
        // One assistant turn + one merged user turn with both responses.
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[1].role.as_deref(), Some(ROLE_USER));
        assert_eq!(
            contents[1].parts.len(),
            2,
            "both tool results merge into one user turn"
        );
        assert_eq!(
            contents[1].parts[0]
                .function_response
                .as_ref()
                .unwrap()
                .name,
            "tool_a"
        );
        assert_eq!(
            contents[1].parts[1]
                .function_response
                .as_ref()
                .unwrap()
                .name,
            "tool_b"
        );
    }

    #[test]
    fn tool_result_wraps_non_object_string() {
        assert_eq!(
            tool_result_response("plain text"),
            json!({"result": "plain text"})
        );
        assert_eq!(
            tool_result_response("[1,2,3]"),
            json!({"result": "[1,2,3]"})
        );
        assert_eq!(tool_result_response("42"), json!({"result": "42"}));
    }

    #[test]
    fn tool_result_uses_object_json_as_is() {
        assert_eq!(
            tool_result_response(r#"{"ok": true, "n": 5}"#),
            json!({"ok": true, "n": 5})
        );
    }

    #[test]
    fn assistant_tool_call_with_malformed_args_defaults_to_empty_object() {
        let calls = vec![ToolCall::new("c1", "noop", "not json")];
        let (_, contents) = convert_messages(&[Message::assistant_with_tool_calls(calls)]);
        let fc = contents[0].parts[0].function_call.as_ref().unwrap();
        assert_eq!(
            fc.args,
            json!({}),
            "unparseable args become an empty object"
        );
    }

    #[test]
    fn build_tools_empty_when_no_tools() {
        assert!(build_tools(&[]).is_empty());
    }

    #[test]
    fn build_tools_flattens_into_one_function_declarations_entry() {
        let tools = vec![
            ToolDefinition::new("a", "tool a", json!({"type": "object"})),
            ToolDefinition::new("b", "tool b", json!({"type": "object"})),
        ];
        let built = build_tools(&tools);
        assert_eq!(built.len(), 1, "all tools flatten under one entry");
        assert_eq!(built[0].function_declarations.len(), 2);
        assert_eq!(built[0].function_declarations[0].name, "a");
        assert_eq!(built[0].function_declarations[1].name, "b");
    }

    #[test]
    fn function_declaration_sanitizes_schema() {
        // A schema with a top-level composite must be sanitized in the decl.
        let tool = ToolDefinition::new(
            "t",
            "desc",
            json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "oneOf": [{"required": ["x"]}],
            }),
        );
        let decl = function_declaration(&tool);
        let params = decl.parameters.expect("parameters present");
        assert!(
            params.get("oneOf").is_none(),
            "oneOf must be sanitized away"
        );
        assert_eq!(params["type"], "object");
    }

    #[test]
    fn function_declaration_omits_empty_parameters() {
        let tool = ToolDefinition::new("t", "desc", json!({}));
        let decl = function_declaration(&tool);
        assert!(
            decl.parameters.is_none(),
            "an empty schema yields no `parameters` field"
        );
    }

    #[test]
    fn empty_message_history_yields_no_contents() {
        let (system, contents) = convert_messages(&[]);
        assert!(system.is_none());
        assert!(contents.is_empty());
    }
}
