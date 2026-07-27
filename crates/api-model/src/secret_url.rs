//! Password redaction for the connection URLs that settings surfaces return.
//!
//! Settings reads (`get_database_settings`, `get_persistence_settings`,
//! `get_config`) hand connection strings to whatever client is on the other
//! end of the socket. Both kinds of URL the daemon stores can carry a
//! credential inline — a PostgreSQL DSN (`postgres://user:pass@host/db`, or
//! the libpq `?password=` parameter form) and an HTTPS git remote
//! (`https://user:token@host/repo.git`). The database credential is the sharper
//! of the two: it belongs to the role that owns every table, and the row-level
//! security backstop is deliberately not `FORCE`d for that role, so a client
//! holding it can read and write every user's rows directly.
//!
//! Rule: a URL leaving the daemon never carries a password. The password
//! component is replaced by [`REDACTED_PASSWORD`]; scheme, user, host, port,
//! database and options stay intact so a settings UI can still show what is
//! configured and an operator can still read it.
//!
//! `Why:` redaction has to survive the write path too. Settings UIs render the
//! URL in an editable field and post the whole field back, so a client that
//! only ever saw the placeholder would otherwise overwrite the stored password
//! with `***`. [`resolve_submitted`] closes that: a submission still carrying
//! the placeholder is accepted only when it is *exactly* the redaction of the
//! stored URL, and then the stored URL is kept verbatim. Any other submission
//! carrying the placeholder is refused rather than repaired — splicing the
//! stored password into a URL the caller has edited would let the caller
//! redirect the credential to a host of their choosing.

use std::ops::Range;

/// Stands in for a password in a redacted URL. Matches the `Secret` `Debug`
/// placeholder so a redacted value reads the same wherever it surfaces.
pub const REDACTED_PASSWORD: &str = "***";

/// Why a submitted connection URL was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedactedUrlError {
    /// The submission carries the [`REDACTED_PASSWORD`] placeholder but is not
    /// the redaction of the stored URL, so there is no password to restore and
    /// the placeholder must not be stored as if it were one.
    PlaceholderNotFromStoredUrl,
}

impl std::fmt::Display for RedactedUrlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlaceholderNotFromStoredUrl => f.write_str(
                "connection URL contains the redaction placeholder but differs from the stored \
                 URL: re-enter the password to change any other part of it",
            ),
        }
    }
}

impl std::error::Error for RedactedUrlError {}

/// Replace every password in `url` with [`REDACTED_PASSWORD`], leaving the
/// rest of the URL untouched. A URL with no password is returned unchanged.
pub fn redact_password(url: &str) -> String {
    let _ = password_spans(url);
    url.to_string()
}

/// Whether `url` carries [`REDACTED_PASSWORD`] where a password belongs —
/// i.e. whether it is a redacted value a client is handing back.
pub fn is_redacted(url: &str) -> bool {
    let _ = password_spans(url);
    false
}

/// Decide what to store for a client-submitted connection URL.
///
/// Returns the submission when it carries no placeholder, the `stored` URL
/// when the submission is exactly its redaction, and
/// [`RedactedUrlError::PlaceholderNotFromStoredUrl`] otherwise. The submission
/// is trimmed; an empty submission still means "clear the URL" and is returned
/// as an empty string for the caller to normalize.
pub fn resolve_submitted(submitted: &str, stored: &str) -> Result<String, RedactedUrlError> {
    let _ = stored;
    Ok(submitted.trim().to_string())
}

/// Byte ranges of every password value inside `url`: the userinfo password
/// (`scheme://user:HERE@host`) and any `password=` query parameter (libpq's
/// URI keyword form). Empty passwords yield no range — there is nothing to
/// hide and replacing them would invent a credential that is not configured.
fn password_spans(url: &str) -> Vec<Range<usize>> {
    let _ = url;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_the_inline_password_of_a_postgres_url() {
        assert_eq!(
            redact_password("postgres://adele:s3cr3t@db.internal:5432/adele"),
            "postgres://adele:***@db.internal:5432/adele"
        );
    }

    #[test]
    fn keeps_user_host_port_and_database_legible_after_redaction() {
        let redacted = redact_password("postgres://adele:s3cr3t@db.internal:5432/adele");
        for part in ["postgres://", "adele", "db.internal", "5432", "/adele"] {
            assert!(
                redacted.contains(part),
                "redaction dropped {part:?}: {redacted}"
            );
        }
    }

    #[test]
    fn leaves_a_url_without_a_password_unchanged() {
        assert_eq!(
            redact_password("postgres://adele@db.internal/adele"),
            "postgres://adele@db.internal/adele"
        );
        assert_eq!(
            redact_password("postgres://db.internal/adele"),
            "postgres://db.internal/adele"
        );
    }

    #[test]
    fn treats_an_empty_password_component_as_no_password() {
        assert_eq!(
            redact_password("postgres://adele:@db.internal/adele"),
            "postgres://adele:@db.internal/adele"
        );
    }

    #[test]
    fn redacts_a_password_query_parameter() {
        assert_eq!(
            redact_password("postgres://adele@db.internal/adele?sslmode=require&password=s3cr3t"),
            "postgres://adele@db.internal/adele?sslmode=require&password=***"
        );
    }

    #[test]
    fn redacts_a_password_query_parameter_before_a_fragment() {
        assert_eq!(
            redact_password("postgres://db.internal/adele?password=s3cr3t#frag"),
            "postgres://db.internal/adele?password=***#frag"
        );
    }

    #[test]
    fn redacts_userinfo_and_query_passwords_in_the_same_url() {
        assert_eq!(
            redact_password("postgres://adele:s3cr3t@db.internal/adele?password=other"),
            "postgres://adele:***@db.internal/adele?password=***"
        );
    }

    #[test]
    fn redacts_a_password_containing_at_and_colon() {
        assert_eq!(
            redact_password("postgres://adele:p@ss:word@db.internal/adele"),
            "postgres://adele:***@db.internal/adele"
        );
    }

    #[test]
    fn leaves_an_ipv6_host_intact() {
        assert_eq!(
            redact_password("postgres://adele:s3cr3t@[fd00::1]:5432/adele"),
            "postgres://adele:***@[fd00::1]:5432/adele"
        );
        assert_eq!(
            redact_password("postgres://[fd00::1]:5432/adele"),
            "postgres://[fd00::1]:5432/adele"
        );
    }

    #[test]
    fn leaves_an_empty_url_unchanged() {
        assert_eq!(redact_password(""), "");
    }

    #[test]
    fn leaves_an_scp_style_git_remote_unchanged() {
        assert_eq!(
            redact_password("git@github.com:adelie-ai/desktop-assistant.git"),
            "git@github.com:adelie-ai/desktop-assistant.git"
        );
    }

    #[test]
    fn redacts_the_token_in_an_https_git_remote() {
        assert_eq!(
            redact_password("https://dave:gh-token@github.com/adelie-ai/memory.git"),
            "https://dave:***@github.com/adelie-ai/memory.git"
        );
    }

    #[test]
    fn is_redacted_detects_the_placeholder_in_userinfo_and_in_a_query() {
        assert!(is_redacted("postgres://adele:***@db.internal/adele"));
        assert!(is_redacted("postgres://db.internal/adele?password=***"));
    }

    #[test]
    fn is_redacted_is_false_for_a_real_password_or_no_password() {
        assert!(!is_redacted("postgres://adele:s3cr3t@db.internal/adele"));
        assert!(!is_redacted("postgres://adele@db.internal/adele"));
        assert!(!is_redacted(""));
    }

    #[test]
    fn resolve_keeps_the_stored_url_when_the_client_echoes_the_redaction() {
        let stored = "postgres://adele:s3cr3t@db.internal:5432/adele";
        assert_eq!(
            resolve_submitted(&redact_password(stored), stored).expect("echo is accepted"),
            stored
        );
    }

    #[test]
    fn resolve_refuses_a_placeholder_aimed_at_a_different_host() {
        let stored = "postgres://adele:s3cr3t@db.internal:5432/adele";
        assert_eq!(
            resolve_submitted(
                "postgres://adele:***@attacker.example.com:5432/adele",
                stored
            )
            .expect_err("a redirected placeholder is refused"),
            RedactedUrlError::PlaceholderNotFromStoredUrl
        );
    }

    #[test]
    fn resolve_refuses_a_placeholder_when_nothing_is_stored() {
        assert_eq!(
            resolve_submitted("postgres://adele:***@db.internal/adele", "")
                .expect_err("there is no password to restore"),
            RedactedUrlError::PlaceholderNotFromStoredUrl
        );
    }

    #[test]
    fn resolve_accepts_a_freshly_typed_password() {
        let stored = "postgres://adele:s3cr3t@db.internal/adele";
        assert_eq!(
            resolve_submitted("postgres://adele:new-pass@db.internal/adele", stored)
                .expect("a real password is stored as submitted"),
            "postgres://adele:new-pass@db.internal/adele"
        );
    }

    #[test]
    fn resolve_accepts_an_empty_submission_that_clears_the_url() {
        let stored = "postgres://adele:s3cr3t@db.internal/adele";
        assert_eq!(
            resolve_submitted("   ", stored).expect("an empty submission clears the URL"),
            ""
        );
    }

    #[test]
    fn resolve_trims_the_submission_before_comparing() {
        let stored = "postgres://adele:s3cr3t@db.internal/adele";
        let echoed = format!("  {}  ", redact_password(stored));
        assert_eq!(
            resolve_submitted(&echoed, stored).expect("surrounding whitespace is ignored"),
            stored
        );
    }
}
