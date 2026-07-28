//! How a failure or an operational decline becomes a wire frame.
//!
//! The `Command` / `CommandResult` contract is the product, so an error is an
//! API response with a shape, not a string a user interface happens to render.
//! Every frame this module builds carries [`api::ErrorDetail`] when the daemon
//! can classify the outcome honestly, and omits it when it cannot.
//!
//! Two rules from the engineering standards drive the shapes here:
//!
//! - **8.2** - an operational decline is a normal outcome, not a failure. An
//!   authorization refusal and a missing entity are declines: they are reported
//!   at info level and marked `retryable: false`, because repeating the request
//!   cannot change the answer.
//! - **8.3** - the business outcome rides in the payload. The transport frame
//!   already distinguishes success from failure, so the detail carries what the
//!   split cannot: a stable code, a developer-facing description, a
//!   user-facing message, and the retry verdict.
//!
//! ## What is not classified yet
//!
//! [`api_error_frame`] classifies the structural [`ApiError`] variants. It
//! leaves [`ApiError::Core`] unclassified, because that variant carries a
//! `CoreError` already flattened to a string, and matching on an error's
//! `Display` output is exactly what this project forbids. Classifying those
//! declines - the config-load refusal and the in-use connection refusal among
//! them - needs `CoreError`'s variants carried across the `ApiError` boundary
//! first (#972). Until then an unclassified frame is honest about being
//! unclassified rather than mislabelled.
//!
//! [`ApiError::InvalidInput`] is the one narrow exception (#804, #895): a
//! rules-based refusal of a URL a client supplied (a connection `base_url`,
//! a remote MCP endpoint) needed a legible, classified refusal immediately,
//! rather than waiting on #972's broader reclassification of the
//! pre-existing message-only variants above. It carries its own stable code
//! and user-facing message end to end from `CoreError::InvalidInput`, so
//! this module does not have to guess one from a string.

use desktop_assistant_api_model as api;
use desktop_assistant_application::ApiError;

use crate::authz::Capability;

/// Build the frame for a refused command (#728).
///
/// The description names the command and the capability it needed, so a log
/// line explains itself. It never carries the command's payload, so a refused
/// credential write cannot echo the credential.
pub(crate) fn refusal_frame(
    id: String,
    command: &api::Command,
    required: &Capability,
    held: &Capability,
) -> api::WsFrame {
    let name = crate::authz::command_name(command);
    api::WsFrame::declined(
        id,
        api::ErrorDetail {
            code: api::ErrorCode::NotAuthorized,
            description: format!(
                "{} '{name}' requires the {} capability; this connection holds {}",
                crate::authz::REFUSAL_PREFIX,
                required.label(),
                held.label()
            ),
            message: format!(
                "Only a daemon administrator can do that. This connection is a {}.",
                held.label()
            ),
            retryable: false,
        },
    )
}

/// Build the frame for a handler error, classifying it where the error type
/// carries enough structure to do so honestly.
/// A classified `Unsupported` decline whose description names the specific
/// feature, rather than the generic "unsupported command".
///
/// The caller knows which capability was missing; the `ApiError` variant does
/// not carry that, and every shipped client renders this string verbatim.
pub(crate) fn unsupported_frame(id: String, description: &str, message: &str) -> api::WsFrame {
    api::WsFrame::declined(
        id,
        api::ErrorDetail {
            code: api::ErrorCode::Unsupported,
            description: description.to_string(),
            message: message.to_string(),
            retryable: false,
        },
    )
}

pub(crate) fn api_error_frame(id: String, error: ApiError) -> api::WsFrame {
    match error {
        // Already flattened to a string upstream; see the module docs.
        ApiError::Core(message) => api::WsFrame::error(id, message),
        ApiError::Unsupported => api::WsFrame::declined(
            id,
            api::ErrorDetail {
                code: api::ErrorCode::Unsupported,
                description: ApiError::Unsupported.to_string(),
                message: "This daemon does not support that operation.".to_string(),
                retryable: false,
            },
        ),
        ApiError::NotFound => api::WsFrame::declined(
            id,
            api::ErrorDetail {
                code: api::ErrorCode::NotFound,
                description: ApiError::NotFound.to_string(),
                message: "That item no longer exists.".to_string(),
                retryable: false,
            },
        ),
        ApiError::AlreadyTerminal => api::WsFrame::declined(
            id,
            api::ErrorDetail {
                code: api::ErrorCode::AlreadyTerminal,
                description: ApiError::AlreadyTerminal.to_string(),
                message: "That task has already finished.".to_string(),
                retryable: false,
            },
        ),
        ApiError::InvalidInput {
            code,
            description,
            message,
        } => api::WsFrame::declined(
            id,
            api::ErrorDetail {
                code: api::ErrorCode::Other(code),
                description,
                message,
                retryable: false,
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detail(frame: &api::WsFrame) -> api::ErrorDetail {
        match frame {
            api::WsFrame::Error {
                detail: Some(detail),
                ..
            } => detail.clone(),
            other => panic!("expected a classified error frame, got {other:?}"),
        }
    }

    #[test]
    fn a_refusal_is_not_retryable_and_carries_the_stable_code() {
        let frame = refusal_frame(
            "r1".to_string(),
            &api::Command::SetApiKey {
                api_key: "sk-secret-value".into(),
            },
            &Capability::Admin,
            &Capability::Tenant,
        );
        let detail = detail(&frame);
        assert_eq!(detail.code, api::ErrorCode::NotAuthorized);
        assert_eq!(detail.code.as_str(), "not_authorized");
        assert!(!detail.retryable);
        assert!(detail.description.contains("set_api_key"), "{detail:?}");
        assert!(
            !detail.description.contains("sk-secret-value")
                && !detail.message.contains("sk-secret-value"),
            "a refusal must not echo the payload: {detail:?}"
        );
    }

    #[test]
    fn the_error_string_repeats_the_description_for_older_clients() {
        let frame = refusal_frame(
            "r1".to_string(),
            &api::Command::Ping,
            &Capability::Admin,
            &Capability::Tenant,
        );
        match &frame {
            api::WsFrame::Error { error, detail, .. } => {
                assert_eq!(error, &detail.as_ref().expect("detail").description);
            }
            other => panic!("expected an error frame, got {other:?}"),
        }
    }

    #[test]
    fn structural_api_errors_are_classified() {
        assert_eq!(
            detail(&api_error_frame("i".to_string(), ApiError::Unsupported)).code,
            api::ErrorCode::Unsupported
        );
        assert_eq!(
            detail(&api_error_frame("i".to_string(), ApiError::NotFound)).code,
            api::ErrorCode::NotFound
        );
        assert_eq!(
            detail(&api_error_frame("i".to_string(), ApiError::AlreadyTerminal)).code,
            api::ErrorCode::AlreadyTerminal
        );
    }

    /// #804 / #895: a rules-based URL refusal reaches the wire as a
    /// classified, not-retryable decline, carrying its own stable code
    /// and user-facing message rather than the generic unclassified string.
    #[test]
    fn an_invalid_input_refusal_is_classified_with_its_own_stable_code() {
        let frame = api_error_frame(
            "i".to_string(),
            ApiError::InvalidInput {
                code: "url_insecure_scheme".to_string(),
                description: "connection base_url refused: ...".to_string(),
                message: "Use https:// instead.".to_string(),
            },
        );
        let detail = detail(&frame);
        assert_eq!(
            detail.code,
            api::ErrorCode::Other("url_insecure_scheme".to_string())
        );
        assert_eq!(detail.message, "Use https:// instead.");
        assert!(!detail.retryable, "the same URL will always be refused");
    }

    #[test]
    fn a_flattened_core_error_is_reported_unclassified_not_mislabelled() {
        let frame = api_error_frame("i".to_string(), ApiError::Core("boom".to_string()));
        match frame {
            api::WsFrame::Error { error, detail, .. } => {
                assert_eq!(error, "boom", "the message stays verbatim");
                assert!(detail.is_none(), "guessing a code would be dishonest");
            }
            other => panic!("expected an error frame, got {other:?}"),
        }
    }
}
