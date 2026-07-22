//! Message conversion: domain [`Message`]s into OpenAI `chat/completions`
//! `messages[]`, the prompt-cache breakpoint helper, and the tool-argument
//! sanitizer used on the history path.

use serde::{Deserialize, Serialize};

use desktop_assistant_core::domain::{Message, Role};

/// A single entry in the `chat/completions` `messages[]` array.
///
/// `content` is optional so an assistant turn that carries only `tool_calls`
/// serializes with `content` omitted (OpenAI accepts a null/absent content in
/// that case). A `tool` result message sets [`Self::tool_call_id`]; an
/// assistant tool-call turn sets [`Self::tool_calls`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// `"system"` | `"user"` | `"assistant"` | `"tool"`.
    pub role: String,
    /// Message text, or the multi-part content array once a cache breakpoint
    /// is stamped on it (see [`mark_system_cache_breakpoint`]). Omitted from
    /// the wire when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatContent>,
    /// Tool calls requested by an assistant turn. Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    /// The id of the tool call this `tool`-role message answers. Omitted for
    /// non-tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// The `content` field of a [`ChatMessage`]: either a plain string or the
/// multi-part content-array form (used to attach a `cache_control` marker).
///
/// Serialized untagged so `Text` becomes a JSON string and `Parts` a JSON
/// array, matching the two shapes the API accepts.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum ChatContent {
    /// Plain-string content (the common case).
    Text(String),
    /// Multi-part content: a `[{type:"text", text, cache_control?}]` array.
    Parts(Vec<ChatContentPart>),
}

/// One element of a multi-part [`ChatContent::Parts`] array.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatContentPart {
    /// The part type; always `"text"` for the parts this crate produces.
    #[serde(rename = "type")]
    pub part_type: String,
    /// The text of this part.
    pub text: String,
    /// Optional cache breakpoint marker (`{type:"ephemeral"}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// The `cache_control` marker attached to a content part to request prompt
/// caching. OpenRouter normalizes this per routed upstream; Azure ignores it
/// (its caching is automatic).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CacheControl {
    /// The cache type; only `"ephemeral"` is used.
    #[serde(rename = "type")]
    pub cache_type: String,
}

impl CacheControl {
    /// The `{"type":"ephemeral"}` marker.
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// A tool call as it appears inside an assistant [`ChatMessage`] (the history
/// path back to the API), shaped `{id, type:"function", function:{name,
/// arguments}}`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatToolCall {
    /// The provider-assigned tool-call id, echoed by the matching `tool`
    /// result message's `tool_call_id`.
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub call_type: String,
    /// The called function's name and (stringified JSON) arguments.
    pub function: ChatFunctionCall,
}

/// The `function` object of a [`ChatToolCall`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatFunctionCall {
    /// The tool/function name.
    pub name: String,
    /// The call arguments as a JSON string (sanitized on the history path).
    pub arguments: String,
}

/// Convert domain [`Message`]s into `chat/completions` `messages[]`.
///
/// Mapping:
/// - [`Role::System`] -> `{role:"system", content:<text>}`
/// - [`Role::User`] -> `{role:"user", content:<text>}`
/// - [`Role::Assistant`] -> `{role:"assistant", content:<text?>, tool_calls?}`;
///   `content` is omitted when empty (a tool-call-only turn), and each domain
///   tool call becomes a `{id, type:"function", function:{name, arguments}}`
///   entry with its arguments passed through [`sanitize_tool_arguments`] so the
///   `{"":{}}` gpt-oss garbage never rides back into the request.
/// - [`Role::Tool`] -> `{role:"tool", tool_call_id:<id>, content:<text>}`.
///   A tool message without a `tool_call_id` is dropped, since the API rejects
///   a `tool` message that answers no call.
pub fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.role {
            Role::System => out.push(ChatMessage {
                role: "system".to_string(),
                content: Some(ChatContent::Text(msg.content.clone())),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),
            Role::User => out.push(ChatMessage {
                role: "user".to_string(),
                content: Some(ChatContent::Text(msg.content.clone())),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),
            Role::Assistant => {
                let tool_calls = msg
                    .tool_calls
                    .iter()
                    .map(|tc| ChatToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: ChatFunctionCall {
                            name: tc.name.clone(),
                            arguments: sanitize_tool_arguments(&tc.arguments),
                        },
                    })
                    .collect::<Vec<_>>();
                // Omit content on a tool-call-only turn; keep it otherwise.
                let content = if msg.content.is_empty() {
                    None
                } else {
                    Some(ChatContent::Text(msg.content.clone()))
                };
                out.push(ChatMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                });
            }
            Role::Tool => {
                let Some(call_id) = &msg.tool_call_id else {
                    // A tool result with no id answers no call; drop it rather
                    // than emit a message the API will reject.
                    continue;
                };
                out.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(ChatContent::Text(msg.content.clone())),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id.clone()),
                });
            }
        }
    }
    out
}

/// Normalize a tool-call `arguments` JSON string on the history path.
///
/// gpt-oss (which OpenRouter routes) emits `{"":{}}` -- an object with a single
/// empty-string key -- for no-argument calls; re-sending that echoed history
/// 400s every subsequent turn. This drops empty-string keys, and coerces
/// anything that is not a JSON object (including unparseable input) to an empty
/// object `{}` so the field is always a valid arguments object. Well-formed
/// arguments pass through unchanged (modulo key re-serialization). Mirrors
/// `llm-bedrock`'s `sanitize_tool_input`.
pub fn sanitize_tool_arguments(arguments: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        // Not valid JSON at all -> represent "no arguments" as `{}`.
        Err(_) => return "{}".to_string(),
    };
    match value {
        serde_json::Value::Object(map) => {
            let cleaned: serde_json::Map<String, serde_json::Value> =
                map.into_iter().filter(|(k, _)| !k.is_empty()).collect();
            serde_json::Value::Object(cleaned).to_string()
        }
        // A non-object arguments payload is not valid; coerce to `{}`.
        _ => "{}".to_string(),
    }
}

/// Stamp an ephemeral `cache_control` marker on the **last** system message,
/// converting its content to the multi-part content-array form.
///
/// The system prompt is the only safe cache breakpoint: the tool list is
/// dynamic (runtime tool search mutates it), so marking tools or later messages
/// would thrash the cache (see the Anthropic connector's breakpoint note).
/// Whether to call this is the connector's decision -- OpenRouter marks the
/// system block; Azure does not (its caching is automatic).
///
/// - `content` = a string -> becomes `[{type:"text", text, cache_control}]`.
/// - `content` = parts -> the last part gains the marker.
/// - `content` = absent, or no system message present -> no-op.
pub fn mark_system_cache_breakpoint(messages: &mut [ChatMessage]) {
    let Some(system) = messages.iter_mut().rev().find(|m| m.role == "system") else {
        return;
    };
    match system.content.take() {
        Some(ChatContent::Text(text)) => {
            system.content = Some(ChatContent::Parts(vec![ChatContentPart {
                part_type: "text".to_string(),
                text,
                cache_control: Some(CacheControl::ephemeral()),
            }]));
        }
        Some(ChatContent::Parts(mut parts)) => {
            if let Some(last) = parts.last_mut() {
                last.cache_control = Some(CacheControl::ephemeral());
            }
            system.content = Some(ChatContent::Parts(parts));
        }
        // No content to cache; leave the message untouched.
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::ToolCall;

    fn text_of(msg: &ChatMessage) -> &str {
        match msg.content.as_ref().expect("content present") {
            ChatContent::Text(t) => t,
            ChatContent::Parts(_) => panic!("expected text content, got parts"),
        }
    }

    // --- to_chat_messages ------------------------------------------------

    #[test]
    fn system_message_maps_to_system_role() {
        let out = to_chat_messages(&[Message::new(Role::System, "you are helpful")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "system");
        assert_eq!(text_of(&out[0]), "you are helpful");
        assert!(out[0].tool_calls.is_empty());
        assert!(out[0].tool_call_id.is_none());
    }

    #[test]
    fn user_message_maps_to_user_role() {
        let out = to_chat_messages(&[Message::new(Role::User, "hello")]);
        assert_eq!(out[0].role, "user");
        assert_eq!(text_of(&out[0]), "hello");
    }

    #[test]
    fn assistant_text_only_has_no_tool_calls() {
        let out = to_chat_messages(&[Message::new(Role::Assistant, "sure thing")]);
        assert_eq!(out[0].role, "assistant");
        assert_eq!(text_of(&out[0]), "sure thing");
        assert!(out[0].tool_calls.is_empty());
    }

    #[test]
    fn assistant_tool_call_only_omits_content() {
        let calls = vec![ToolCall::new("call_1", "read_file", r#"{"path":"/a"}"#)];
        let out = to_chat_messages(&[Message::assistant_with_tool_calls(calls)]);
        assert_eq!(out[0].role, "assistant");
        assert!(
            out[0].content.is_none(),
            "tool-call-only turn must omit content"
        );
        assert_eq!(out[0].tool_calls.len(), 1);
        let tc = &out[0].tool_calls[0];
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.call_type, "function");
        assert_eq!(tc.function.name, "read_file");
        assert_eq!(tc.function.arguments, r#"{"path":"/a"}"#);
    }

    #[test]
    fn assistant_with_text_and_tool_calls_keeps_both() {
        let mut msg = Message::new(Role::Assistant, "let me check");
        msg.tool_calls = vec![ToolCall::new("c1", "lookup", "{}")];
        let out = to_chat_messages(&[msg]);
        assert_eq!(text_of(&out[0]), "let me check");
        assert_eq!(out[0].tool_calls.len(), 1);
    }

    #[test]
    fn tool_result_maps_to_tool_role_with_call_id() {
        let out = to_chat_messages(&[Message::tool_result("call_1", "file contents")]);
        assert_eq!(out[0].role, "tool");
        assert_eq!(out[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(text_of(&out[0]), "file contents");
    }

    #[test]
    fn tool_result_without_call_id_is_dropped() {
        // A tool message that answers no call is rejected by the API; drop it.
        let mut msg = Message::new(Role::Tool, "orphan");
        msg.tool_call_id = None;
        let out = to_chat_messages(&[msg]);
        assert!(out.is_empty(), "orphan tool message must be dropped");
    }

    #[test]
    fn mixed_history_preserves_order_and_roles() {
        let calls = vec![ToolCall::new("c1", "search", r#"{"q":"x"}"#)];
        let history = vec![
            Message::new(Role::System, "sys"),
            Message::new(Role::User, "find x"),
            Message::assistant_with_tool_calls(calls),
            Message::tool_result("c1", "found"),
            Message::new(Role::Assistant, "here it is"),
        ];
        let out = to_chat_messages(&history);
        let roles: Vec<&str> = out.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "assistant"]
        );
        assert_eq!(out[3].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn empty_messages_yields_empty_vec() {
        assert!(to_chat_messages(&[]).is_empty());
    }

    #[test]
    fn assistant_tool_call_arguments_are_sanitized() {
        // The empty-key gpt-oss garbage must not ride back into the request.
        let calls = vec![ToolCall::new("c1", "noargs", r#"{"":{}}"#)];
        let out = to_chat_messages(&[Message::assistant_with_tool_calls(calls)]);
        assert_eq!(out[0].tool_calls[0].function.arguments, "{}");
    }

    #[test]
    fn chat_message_serializes_to_expected_shape() {
        let calls = vec![ToolCall::new("call_1", "do_it", r#"{"n":1}"#)];
        let out = to_chat_messages(&[Message::assistant_with_tool_calls(calls)]);
        let json = serde_json::to_value(&out[0]).expect("serialize");
        assert_eq!(json["role"], "assistant");
        assert!(json.get("content").is_none(), "content must be omitted");
        assert_eq!(json["tool_calls"][0]["id"], "call_1");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "do_it");
        assert_eq!(json["tool_calls"][0]["function"]["arguments"], r#"{"n":1}"#);
    }

    #[test]
    fn user_content_serializes_as_plain_string() {
        let out = to_chat_messages(&[Message::new(Role::User, "hi")]);
        let json = serde_json::to_value(&out[0]).expect("serialize");
        assert_eq!(json["content"], "hi", "text content must be a JSON string");
    }

    // --- sanitize_tool_arguments -----------------------------------------

    #[test]
    fn sanitize_args_strips_empty_key_garbage() {
        assert_eq!(sanitize_tool_arguments(r#"{"":{}}"#), "{}");
    }

    #[test]
    fn sanitize_args_preserves_real_arguments() {
        let got = sanitize_tool_arguments(r#"{"content":"note","key":"goal"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&got).expect("valid json");
        assert_eq!(parsed, serde_json::json!({"content":"note","key":"goal"}));
    }

    #[test]
    fn sanitize_args_drops_only_the_empty_key() {
        let got = sanitize_tool_arguments(r#"{"":1,"real":2}"#);
        let parsed: serde_json::Value = serde_json::from_str(&got).expect("valid json");
        assert_eq!(parsed, serde_json::json!({"real":2}));
    }

    #[test]
    fn sanitize_args_coerces_non_object_to_empty_object() {
        assert_eq!(sanitize_tool_arguments("null"), "{}");
        assert_eq!(sanitize_tool_arguments(r#""oops""#), "{}");
        assert_eq!(sanitize_tool_arguments("[1,2]"), "{}");
        assert_eq!(sanitize_tool_arguments("42"), "{}");
    }

    #[test]
    fn sanitize_args_coerces_invalid_json_to_empty_object() {
        assert_eq!(sanitize_tool_arguments("not json at all"), "{}");
        assert_eq!(sanitize_tool_arguments(""), "{}");
    }

    // --- mark_system_cache_breakpoint ------------------------------------

    #[test]
    fn cache_breakpoint_marks_last_system_message() {
        let mut msgs = to_chat_messages(&[
            Message::new(Role::System, "sys prompt"),
            Message::new(Role::User, "hi"),
        ]);
        mark_system_cache_breakpoint(&mut msgs);
        let json = serde_json::to_value(&msgs[0]).expect("serialize");
        // Content is now the multi-part array with an ephemeral marker.
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "sys prompt");
        assert_eq!(json["content"][0]["cache_control"]["type"], "ephemeral");
        // The user message is untouched (still a plain string).
        let user = serde_json::to_value(&msgs[1]).expect("serialize");
        assert_eq!(user["content"], "hi");
    }

    #[test]
    fn cache_breakpoint_marks_only_the_last_system_message() {
        let mut msgs = to_chat_messages(&[
            Message::new(Role::System, "first"),
            Message::new(Role::System, "second"),
        ]);
        mark_system_cache_breakpoint(&mut msgs);
        // First stays a plain string...
        assert_eq!(
            msgs[0].content,
            Some(ChatContent::Text("first".to_string()))
        );
        // ...only the last becomes a marked parts array.
        match &msgs[1].content {
            Some(ChatContent::Parts(parts)) => {
                assert_eq!(parts[0].cache_control, Some(CacheControl::ephemeral()));
            }
            other => panic!("expected marked parts, got {other:?}"),
        }
    }

    #[test]
    fn cache_breakpoint_noop_without_system_message() {
        let mut msgs = to_chat_messages(&[Message::new(Role::User, "hi")]);
        let before = msgs.clone();
        mark_system_cache_breakpoint(&mut msgs);
        assert_eq!(msgs, before, "no system message -> no change");
    }

    #[test]
    fn cache_breakpoint_marks_last_part_when_already_parts() {
        let mut msgs = vec![ChatMessage {
            role: "system".to_string(),
            content: Some(ChatContent::Parts(vec![
                ChatContentPart {
                    part_type: "text".to_string(),
                    text: "a".to_string(),
                    cache_control: None,
                },
                ChatContentPart {
                    part_type: "text".to_string(),
                    text: "b".to_string(),
                    cache_control: None,
                },
            ])),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }];
        mark_system_cache_breakpoint(&mut msgs);
        match &msgs[0].content {
            Some(ChatContent::Parts(parts)) => {
                assert!(parts[0].cache_control.is_none(), "first part untouched");
                assert_eq!(parts[1].cache_control, Some(CacheControl::ephemeral()));
            }
            other => panic!("expected parts, got {other:?}"),
        }
    }
}
