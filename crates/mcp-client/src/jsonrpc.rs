use serde::{Deserialize, Serialize};

/// The reserved property MCP puts protocol metadata under, on the `params` of a
/// request.
///
/// A JSON-RPC message over a pipe has no headers, so this is the only place a
/// caller can put a W3C trace context. The Streamable HTTP transport has real
/// headers and uses them instead.
pub const META_FIELD: &str = "_meta";

/// The key inside [`META_FIELD`] that carries the W3C `traceparent`.
pub const TRACEPARENT_KEY: &str = "traceparent";

/// Add `traceparent` to a request's `params`, keeping whatever was there.
///
/// Three cases, and each is deliberate:
///
/// - No `params` at all: one is created holding only the trace context. `params`
///   is optional in JSON-RPC and an object is what every MCP method takes, so
///   this is additive rather than a change of shape.
/// - `params` is an object: `_meta.traceparent` is set, and any other key the
///   caller put under `_meta` is left alone.
/// - `params` is anything else: returned untouched. No MCP method takes a
///   non-object `params`, and corrupting one to carry telemetry would trade a
///   working call for a trace.
pub fn with_traceparent(
    params: Option<serde_json::Value>,
    traceparent: &str,
) -> Option<serde_json::Value> {
    let mut params = params.unwrap_or_else(|| serde_json::json!({}));
    let serde_json::Value::Object(fields) = &mut params else {
        return Some(params);
    };
    let meta = fields
        .entry(META_FIELD)
        .or_insert_with(|| serde_json::json!({}));
    if let serde_json::Value::Object(meta) = meta {
        meta.insert(
            TRACEPARENT_KEY.to_string(),
            serde_json::Value::String(traceparent.to_string()),
        );
    }
    Some(params)
}

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: 1,
            method: "initialize".into(),
            params: Some(serde_json::json!({"key": "value"})),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"initialize\""));
        assert!(json.contains("\"params\""));
    }

    #[test]
    fn request_without_params_omits_field() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: 1,
            method: "test".into(),
            params: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("params"));
    }

    #[test]
    fn request_roundtrip() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: 42,
            method: "tools/list".into(),
            params: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 42);
        assert_eq!(parsed.method, "tools/list");
    }

    #[test]
    fn response_with_result() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, Some(serde_json::Value::Number(1.into())));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn response_with_error() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    #[test]
    fn response_with_null_id() {
        let json = r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        // serde deserializes JSON null into Option as None
        assert!(resp.id.is_none());
    }

    #[test]
    fn error_with_data() {
        let json = r#"{"code":-32000,"message":"Custom error","data":{"detail":"extra info"}}"#;
        let err: JsonRpcError = serde_json::from_str(json).unwrap();
        assert_eq!(err.code, -32000);
        assert!(err.data.is_some());
    }

    const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn trace_context_is_added_to_existing_params() {
        let params = with_traceparent(
            Some(serde_json::json!({"name": "echo", "arguments": {"text": "hi"}})),
            TRACEPARENT,
        )
        .expect("params survive");
        assert_eq!(params["_meta"]["traceparent"], TRACEPARENT);
        assert_eq!(
            params["name"], "echo",
            "the method's own params must be untouched"
        );
        assert_eq!(params["arguments"]["text"], "hi");
    }

    #[test]
    fn trace_context_is_added_where_there_were_no_params() {
        let params = with_traceparent(None, TRACEPARENT).expect("params are created");
        assert_eq!(params["_meta"]["traceparent"], TRACEPARENT);
        assert_eq!(
            params.as_object().expect("an object").len(),
            1,
            "nothing but the metadata may be invented"
        );
    }

    #[test]
    fn trace_context_keeps_the_callers_own_meta() {
        let params = with_traceparent(
            Some(serde_json::json!({"_meta": {"progressToken": 7}})),
            TRACEPARENT,
        )
        .expect("params survive");
        assert_eq!(params["_meta"]["traceparent"], TRACEPARENT);
        assert_eq!(
            params["_meta"]["progressToken"], 7,
            "another key under `_meta` belongs to the caller and must survive"
        );
    }

    #[test]
    fn a_non_object_params_is_left_alone() {
        // No MCP method takes one, and corrupting a call to carry telemetry
        // would trade a working request for a trace.
        let params = with_traceparent(Some(serde_json::json!([1, 2, 3])), TRACEPARENT)
            .expect("params survive");
        assert_eq!(params, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn error_without_data() {
        let err = JsonRpcError {
            code: -32600,
            message: "Invalid Request".into(),
            data: None,
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(!json.contains("data"));
    }
}
