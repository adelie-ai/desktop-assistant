//! Shared scheme + SSRF policy for a remote URL that arrives from a client
//! payload (#804, #895): a remote MCP endpoint's `url`, and a connection's
//! `base_url`. Two call sites, one defect — a URL is used to attach a bearer
//! token or an API key without ever being validated — so this module gives
//! both call sites one rule instead of two ad hoc checks.
//!
//! ## The rule
//!
//! TLS is required. Plain `http://` is accepted only when the host stays on
//! a network the operator already controls: loopback, an RFC1918 private
//! range, or a bare (dot-free) hostname — the shape of a Kubernetes short
//! Service name (`ollama`, `postgres`) or a LAN `/etc/hosts` entry.
//! `docs/remote-brain-setup.md` documents exactly this: the shipped k8s
//! manifests reach Ollama at `http://ollama:11434` over the in-cluster
//! network, and a blanket TLS-only rule would refuse a working install.
//!
//! Regardless of scheme, a destination in the link-local range (which
//! includes the cloud metadata address `169.254.169.254` shared by AWS,
//! Azure, and GCP's legacy metadata path), the unspecified address
//! (`0.0.0.0` / `::`), or the literal GCP metadata hostname is always
//! refused. That check runs first, so an operator who types `https://` to
//! one of these is not saved by having gotten the scheme right.
//!
//! ## What this does not do
//!
//! No DNS resolution: a hostname is judged on its literal shape, not on
//! where it currently resolves. A dotted, public-looking hostname needs
//! `https` like any other public name even if a private DNS zone happens to
//! resolve it to a private address today — catching a name deliberately
//! rebound at resolve time is full egress filtering, which is
//! disproportionate to what two config-validation call sites need.
//!
//! ## What this deliberately does not unify with
//!
//! `oauth.rs`'s `validate_endpoint_url` keeps its own, stricter, loopback-
//! literal-only rule. It validates a third-party identity provider's
//! authorize/token endpoint, which has no legitimate in-cluster or LAN
//! deployment the way a self-hosted MCP server or LLM connector does —
//! widening it to match this policy would loosen a check that is already
//! correctly strict for what it guards.
//!
//! ## When this runs
//!
//! See the callers: `HttpTransport::new` (this crate) validates at connect
//! time — the backstop for a URL that reached the transport by any route,
//! including a config file edited by hand. `crates/daemon/src/api_surface.rs`
//! and `crates/daemon/src/settings_service.rs` additionally validate at
//! write time, so an operator typing a bad URL into a settings panel gets a
//! legible refusal immediately instead of a failure the next time something
//! tries to connect.

/// Why a candidate remote URL was refused. Each variant is an independently
/// testable rule, not a prose string a caller has to pattern-match:
/// [`Self::code`] gives the stable, machine-readable identifier a wire
/// adapter carries forward as `api::ErrorCode::Other` without inventing a
/// second classification scheme, and [`Self::user_message`] gives text fit
/// to show the person who supplied the URL.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UrlPolicyError {
    /// `raw` could not be parsed as an absolute URL at all — including the
    /// common case of a scheme left off entirely (`"mcp.example.com/api"`).
    #[error("{url:?} is not a valid URL ({reason})")]
    Malformed { url: String, reason: String },

    /// `raw` parsed, but named a scheme other than `http`/`https`.
    #[error("{url:?} names scheme {scheme:?}; only http and https are accepted")]
    SchemeNotAllowed { url: String, scheme: String },

    /// `raw` used `http://` to a host that is not loopback, a private
    /// network address, or a bare hostname — i.e. a host reachable over the
    /// open network, where the request's bearer token or API key would
    /// travel in the clear.
    #[error(
        "{url:?} uses http:// to a host that is not loopback, a private network address, or a \
         bare hostname; TLS is required unless the destination stays on a network the operator \
         already controls"
    )]
    InsecureScheme { url: String },

    /// `raw` names a link-local address (including the cloud metadata IP),
    /// an unspecified address, or the GCP metadata hostname. Refused
    /// regardless of scheme.
    #[error("{url:?} names a link-local, unspecified, or cloud-metadata address")]
    BlockedTarget { url: String },
}

impl UrlPolicyError {
    /// Stable, machine-readable identifier for this refusal. Snake-case, so
    /// a wire adapter can carry it forward verbatim as
    /// `api::ErrorCode::Other(code.to_string())`.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Malformed { .. } => "url_malformed",
            Self::SchemeNotAllowed { .. } => "url_scheme_not_allowed",
            Self::InsecureScheme { .. } => "url_insecure_scheme",
            Self::BlockedTarget { .. } => "url_target_blocked",
        }
    }

    /// Text fit to show the person who supplied the URL: what is wrong, and
    /// what to do about it. Short, direct sentences — no jargon, one idea
    /// each.
    pub fn user_message(&self) -> String {
        match self {
            Self::Malformed { .. } => {
                "This is not a valid URL. Check the address and try again.".to_string()
            }
            Self::SchemeNotAllowed { scheme, .. } => {
                format!("This address uses '{scheme}://'. Use 'https://' instead.")
            }
            Self::InsecureScheme { .. } => "This address uses 'http://' to a public host. Use \
                 'https://', or point at localhost, a private network address, or an internal \
                 hostname."
                .to_string(),
            Self::BlockedTarget { .. } => "This address is not allowed. It names a link-local \
                 or cloud metadata address."
                .to_string(),
        }
    }
}

/// Validate a URL arriving from a client payload before it is stored or
/// dialed. See the module docs for the rule and why it is shaped this way.
///
/// Not yet implemented: this commit is the spec (#804, #895) — the failing
/// tests below define the rule; the next commit implements it.
pub fn validate_remote_url(raw: &str) -> Result<(), UrlPolicyError> {
    let _ = raw;
    unimplemented!("url_policy::validate_remote_url: see the following implementation commit")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- accepted: the shapes that must keep working -----

    #[test]
    fn accepts_a_legitimate_https_url_to_a_public_host() {
        validate_remote_url("https://mcp.example.com/api").expect("https to a public host");
    }

    #[test]
    fn accepts_http_to_ipv4_loopback() {
        validate_remote_url("http://127.0.0.1:8080/mcp").expect("http to loopback");
    }

    #[test]
    fn accepts_http_to_ipv6_loopback() {
        validate_remote_url("http://[::1]:8080/mcp").expect("http to ipv6 loopback");
    }

    #[test]
    fn accepts_http_to_localhost_hostname() {
        validate_remote_url("http://localhost:8080/mcp").expect("http to localhost");
    }

    /// The load-bearing case: `deploy/k8s/base/daemon.toml` and
    /// `docs/remote-brain-setup.md` reach Ollama at `http://ollama:11434` —
    /// an in-cluster Kubernetes Service name, which is a bare hostname.
    #[test]
    fn accepts_http_to_a_bare_in_cluster_service_name() {
        validate_remote_url("http://ollama:11434").expect("http to a bare k8s service name");
    }

    #[test]
    fn accepts_http_to_an_rfc1918_private_address() {
        validate_remote_url("http://192.168.1.50:8000/v1").expect("http to a LAN address");
    }

    #[test]
    fn accepts_ipv4_mapped_ipv6_loopback_over_http() {
        validate_remote_url("http://[::ffff:127.0.0.1]/mcp")
            .expect("an ipv4-mapped ipv6 loopback literal is still loopback");
    }

    // ----- refused: insecure scheme -----

    #[test]
    fn rejects_http_to_a_public_dotted_host() {
        let err = validate_remote_url("http://evil.example.com/mcp").unwrap_err();
        assert_eq!(err.code(), "url_insecure_scheme");
    }

    // ----- refused: blocked target, regardless of scheme -----

    #[test]
    fn rejects_link_local_address_over_http() {
        let err = validate_remote_url("http://169.254.1.1/").unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_cloud_metadata_ipv4_address_even_over_https() {
        // The scheme check alone would accept this; the blocked-target check
        // must run first regardless of scheme (see the module docs).
        let err = validate_remote_url("https://169.254.169.254/latest/meta-data/").unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_gcp_metadata_hostname() {
        let err = validate_remote_url("http://metadata.google.internal/computeMetadata/v1/")
            .unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_unspecified_ipv4_address() {
        let err = validate_remote_url("http://0.0.0.0:8080/").unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_unspecified_ipv6_address() {
        let err = validate_remote_url("http://[::]:8080/").unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_ipv4_mapped_ipv6_link_local_address() {
        // Bypass check: a metadata address dressed up as an ipv4-mapped
        // ipv6 literal must still be caught.
        let err = validate_remote_url("http://[::ffff:169.254.169.254]/").unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    // ----- refused: malformed / missing scheme / disallowed scheme -----

    #[test]
    fn rejects_a_malformed_url() {
        let err = validate_remote_url("not a url at all").unwrap_err();
        assert_eq!(err.code(), "url_malformed");
    }

    #[test]
    fn rejects_a_url_with_no_scheme() {
        let err = validate_remote_url("mcp.example.com/api").unwrap_err();
        assert_eq!(err.code(), "url_malformed");
    }

    #[test]
    fn rejects_a_disallowed_scheme() {
        let err = validate_remote_url("ftp://example.com/mcp").unwrap_err();
        assert_eq!(err.code(), "url_scheme_not_allowed");
    }

    // ----- the refusal is structured, not a prose string -----

    #[test]
    fn each_error_carries_a_stable_machine_readable_code() {
        let cases: &[(&str, &str)] = &[
            ("not a url", "url_malformed"),
            ("ftp://example.com/", "url_scheme_not_allowed"),
            ("http://evil.example.com/", "url_insecure_scheme"),
            ("http://169.254.169.254/", "url_target_blocked"),
        ];
        for (input, expected_code) in cases {
            let err = validate_remote_url(input).unwrap_err();
            assert_eq!(err.code(), *expected_code, "input: {input}");
            assert!(
                !err.user_message().is_empty(),
                "every refusal must carry user-facing text: {input}"
            );
        }
    }
}
