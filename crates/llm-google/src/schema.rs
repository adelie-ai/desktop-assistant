//! Gemini-specific tool-parameter schema sanitizer.
//!
//! Gemini accepts an OpenAPI-3.0 *subset* and is stricter than Bedrock's
//! Converse: on top of the composite keywords (`oneOf`/`anyOf`/`allOf`) it
//! also rejects `$schema`, `$ref`/`$defs`/`definitions`, `additionalProperties`,
//! and unrecognized `format` values — and it rejects them at *any* depth, not
//! just the top level. One bad MCP tool schema 400s the entire turn (taking
//! every other tool down with it), so this guard runs on the request path
//! before the schema is serialized into a `functionDeclaration`.
//!
//! Reimplemented (not shared) using `llm-bedrock`'s `sanitize_tool_schema` as
//! a guide: Bedrock strips only the top-level composites; Gemini's subset is
//! narrower, so this version recurses and strips a larger key set.

use serde_json::{Map, Value};

/// Keys Gemini rejects anywhere in a parameter schema. Stripped recursively.
const FORBIDDEN_KEYS: &[&str] = &[
    "$schema",
    "$ref",
    "$defs",
    "definitions",
    "additionalProperties",
    "oneOf",
    "anyOf",
    "allOf",
    "patternProperties",
];

/// Composite keys whose removal means the surrounding object lost its only
/// structural shape and must be given an explicit `type`.
const COMPOSITE_KEYS: &[&str] = &["oneOf", "anyOf", "allOf"];

/// `format` values Gemini's OpenAPI subset accepts. Any other `format` is
/// dropped (the underlying `type` is kept), since an unrecognized value 400s.
const ALLOWED_FORMATS: &[&str] = &[
    "date-time",
    "date",
    "time",
    "int32",
    "int64",
    "float",
    "double",
];

/// Sanitize a tool `parameters` JSON Schema into Gemini's OpenAPI subset.
///
/// - Recurses through `properties.*`, `items`, and array members, stripping
///   [`FORBIDDEN_KEYS`] at every level.
/// - Drops a `format` whose value is not in [`ALLOWED_FORMATS`], keeping the
///   `type`.
/// - Ensures a `type` on any object that has `properties` or that just lost a
///   composite keyword, so the result is always a well-typed object schema.
/// - Non-object schemas (`true`/`false`/string/number/null) pass through
///   untouched — there is nothing to strip and we must not wrap them.
pub fn sanitize_tool_schema(schema: Value) -> Value {
    let Value::Object(map) = schema else {
        return schema;
    };
    Value::Object(sanitize_object(map))
}

fn sanitize_object(mut map: Map<String, Value>) -> Map<String, Value> {
    let mut removed_composite = false;
    for key in FORBIDDEN_KEYS {
        if map.remove(*key).is_some() && COMPOSITE_KEYS.contains(key) {
            removed_composite = true;
        }
    }

    // Drop an unsupported `format`, keeping the field's `type`.
    if let Some(Value::String(fmt)) = map.get("format")
        && !ALLOWED_FORMATS.contains(&fmt.as_str())
    {
        map.remove("format");
    }

    // Recurse into every remaining value so nested schemas are cleaned too.
    for value in map.values_mut() {
        let taken = std::mem::replace(value, Value::Null);
        *value = sanitize_value(taken);
    }

    // Ensure a `type` when the object is structurally an object but lacks one,
    // or when stripping a composite left it shapeless.
    if (removed_composite || map.contains_key("properties")) && !map.contains_key("type") {
        map.insert("type".to_string(), Value::String("object".to_string()));
    }

    map
}

fn sanitize_value(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(sanitize_object(map)),
        Value::Array(items) => Value::Array(items.into_iter().map(sanitize_value).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_top_level_composites() {
        for key in ["oneOf", "anyOf", "allOf"] {
            let got = sanitize_tool_schema(json!({
                "type": "object",
                key: [{"type": "object"}],
                "properties": {"x": {"type": "string"}},
            }));
            assert!(got.get(key).is_none(), "{key} must be stripped");
            assert_eq!(got["type"], "object");
            assert_eq!(got["properties"]["x"]["type"], "string");
        }
    }

    #[test]
    fn adds_type_when_missing_after_stripping_composite() {
        let got = sanitize_tool_schema(json!({
            "oneOf": [{"type": "object"}, {"type": "string"}],
        }));
        assert!(got.get("oneOf").is_none());
        assert_eq!(got["type"], "object", "missing type must default to object");
    }

    #[test]
    fn strips_dollar_schema_ref_and_additional_properties() {
        let got = sanitize_tool_schema(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": false,
            "$ref": "#/$defs/Foo",
            "$defs": {"Foo": {"type": "string"}},
            "properties": {"a": {"type": "integer"}},
        }));
        assert!(got.get("$schema").is_none(), "$schema must be stripped");
        assert!(got.get("$ref").is_none(), "$ref must be stripped");
        assert!(got.get("$defs").is_none(), "$defs must be stripped");
        assert!(
            got.get("additionalProperties").is_none(),
            "additionalProperties must be stripped"
        );
        assert_eq!(got["type"], "object");
        assert_eq!(got["properties"]["a"]["type"], "integer");
    }

    #[test]
    fn strips_forbidden_keys_recursively() {
        // Unlike the Bedrock sanitizer (top-level only), Gemini rejects these
        // at any depth, so nested `additionalProperties` / `$ref` must go too.
        let got = sanitize_tool_schema(json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {"deep": {"type": "string", "$ref": "#/x"}},
                },
            },
        }));
        assert!(
            got["properties"]["nested"]
                .get("additionalProperties")
                .is_none(),
            "nested additionalProperties must be stripped"
        );
        assert!(
            got["properties"]["nested"]["properties"]["deep"]
                .get("$ref")
                .is_none(),
            "deeply-nested $ref must be stripped"
        );
        assert_eq!(
            got["properties"]["nested"]["properties"]["deep"]["type"],
            "string"
        );
    }

    #[test]
    fn drops_unsupported_format_but_keeps_supported() {
        let got = sanitize_tool_schema(json!({
            "type": "object",
            "properties": {
                "u": {"type": "string", "format": "uri"},
                "t": {"type": "string", "format": "date-time"},
                "n": {"type": "integer", "format": "int64"},
            },
        }));
        assert!(
            got["properties"]["u"].get("format").is_none(),
            "unsupported format `uri` must be dropped"
        );
        assert_eq!(got["properties"]["u"]["type"], "string");
        assert_eq!(
            got["properties"]["t"]["format"], "date-time",
            "supported format must be preserved"
        );
        assert_eq!(got["properties"]["n"]["format"], "int64");
    }

    #[test]
    fn clean_object_schema_is_unchanged() {
        let clean = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        });
        assert_eq!(sanitize_tool_schema(clean.clone()), clean);
    }

    #[test]
    fn object_with_properties_but_no_type_gets_type() {
        // Gemini requires object schemas to declare their type.
        let got = sanitize_tool_schema(json!({
            "properties": {"a": {"type": "string"}},
        }));
        assert_eq!(got["type"], "object");
    }

    #[test]
    fn non_object_values_pass_through() {
        assert_eq!(sanitize_tool_schema(json!(true)), json!(true));
        assert_eq!(sanitize_tool_schema(json!("a string")), json!("a string"));
        assert_eq!(sanitize_tool_schema(Value::Null), Value::Null);
    }

    #[test]
    fn empty_schema_passes_through_without_type() {
        // An empty object with no properties and no composite is a legal
        // "any" schema; we don't force a type onto it.
        let got = sanitize_tool_schema(json!({}));
        assert_eq!(got, json!({}));
    }
}
