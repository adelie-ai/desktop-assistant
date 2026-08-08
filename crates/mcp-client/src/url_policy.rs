//! Shared scheme + SSRF policy for a remote URL that arrives from a client
//! payload (#804, #895): a remote MCP endpoint's `url`, a named connection's
//! `base_url`, and (since the #804/#895 adversarial review) the legacy
//! `[embeddings]` and backend-tasks `base_url` settings. Four call sites,
//! one defect — a URL is used to attach a bearer token or an API key
//! without ever being validated — so this module gives all four one rule
//! instead of separate ad hoc checks. A fifth, `set_persistence_settings`'s
//! git `remote_url`, is deliberately not one of them; see #991.
//!
//! ## The rule
//!
//! TLS is required. Plain `http://` is accepted only when the host stays on
//! a network the operator already controls: loopback, an RFC1918 IPv4 or
//! IPv6 ULA (`fc00::/7`) private range unconditionally, and CGNAT
//! (`100.64.0.0/10`) or a bare (dot-free) hostname only when the request
//! carries no credential. `docs/k8s-deployment.md` documents the
//! load-bearing case: the shipped k8s manifests reach Ollama, which carries
//! no credential, at `http://ollama:11434` over the in-cluster network, and
//! a blanket TLS-only rule would refuse a working install.
//!
//! Regardless of scheme, a destination in the link-local range (which
//! includes the cloud metadata address `169.254.169.254` shared by AWS,
//! Azure, and GCP's legacy metadata path), the unspecified address
//! (`0.0.0.0` / `::`), a known cloud-metadata hostname alias, or a known
//! cloud-metadata IPv4 literal outside the link-local range (Alibaba Cloud's
//! ECS metadata service lives inside the CGNAT range instead) is always
//! refused. That check runs first and is not exempted by anything below it,
//! so an operator who types `https://` to one of these, or reaches it
//! through the bare-hostname or CGNAT exemption, is not saved by either.
//!
//! ## Why the bare-hostname and CGNAT exemptions are credential-gated, and
//! RFC1918/ULA are not
//!
//! A bare hostname and a CGNAT address are both reached through a mechanism
//! the operator does not fully control, which an RFC1918 or ULA literal is
//! not:
//!
//! - A dot-free name is whatever the resolver decides: search-domain append
//!   (DHCP option 15/119, attacker-settable on a hostile LAN), LLMNR (an
//!   unauthenticated multicast query any LAN peer can answer), and, for a
//!   single-label name that happens to also be a public TLD with an apex
//!   record, public DNS.
//! - CGNAT (`100.64.0.0/10`, RFC 6598) is explicitly *shared* address space
//!   - carriers assign it to many unrelated subscribers on the same tethered
//!     or LTE network, unlike RFC1918 or ULA, which are never routed on a
//!     shared public-facing network. A credentialed plain-HTTP request to a
//!     CGNAT literal can land on another subscriber's equipment, not just
//!     the operator's own.
//!
//! A request that carries no credential has nothing an adversarial answer or
//! neighbour can steal, so `http://ollama:11434` (Ollama has no credential
//! concept at all) stays permitted regardless of how `ollama` resolves, and
//! a credential-free request to a Tailscale/CGNAT address keeps working. A
//! request that does carry one — a remote MCP server's bearer token, a
//! connection's API key — needs the same scrutiny a private IP literal
//! gets: a bare name or a CGNAT literal is refused, and the operator points
//! it at an RFC1918/ULA/loopback address or `https://` instead.
//! `RequestCredential` is the caller's declaration of which case this is;
//! get it right, because this module cannot infer it from the URL string
//! alone.
//!
//! ## What this does not do
//!
//! No DNS resolution: a hostname is judged on its literal shape, not on
//! where it currently resolves. A dotted, public-looking hostname needs
//! `https` like any other public name even if a private DNS zone happens to
//! resolve it to a private address today — catching a name deliberately
//! rebound at resolve time is full egress filtering, which is
//! disproportionate to what a handful of config-validation call sites need.
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
//! ## What this deliberately leaves uniform across connectors (recorded, not
//! silent)
//!
//! #895's own sketch asked for a *per-connector* split: block loopback/
//! private ranges for a hosted provider (Anthropic, Azure, Google,
//! OpenRouter) while still permitting them for a self-hosted one (Ollama).
//! This module does not implement that split, deliberately. Two of the
//! "hosted" connector types have their own legitimate private-network
//! deployment documented in this codebase: `openai` is the connector type
//! for both the real hosted OpenAI API *and* any self-hosted OpenAI-
//! compatible server (vLLM, LM Studio, llama.cpp) — there is no separate
//! connector type for the latter — and `BedrockConnection::base_url`'s own
//! doc comment names a private-endpoint (VPC PrivateLink) proxy as a
//! supported deployment. A per-connector split that is correct for both of
//! those and still closes the gap for Anthropic/Azure/Google/OpenRouter is
//! a real design job (base rule 7.4: compare candidate designs before
//! committing), not a one-line addition to a review-fix pass — tracked as
//! #992. Until then,
//! an admin who can create or update a connection can point a hosted
//! connector at a private address; the threat model this repo's authz tier
//! defends against is an honest mistake or a compromised admin credential,
//! not a hostile admin (`docs/design/multi-tenancy-boundary.md` decision 7),
//! so this is a recorded gap, not a silent one.
//!
//! ## When this runs
//!
//! See the callers: `HttpTransport::new` (this crate) validates at connect
//! time — the backstop for a URL that reached the transport by any route,
//! including a config file edited by hand. `crates/daemon/src/api_surface.rs`
//! (`create_connection`/`update_connection`), `crates/daemon/src/settings_service.rs`
//! (`upsert_mcp_server`), and `crates/daemon/src/config/views.rs`
//! (`set_embeddings_settings`, `set_backend_tasks_settings`) additionally
//! validate at write time, so an operator typing a bad URL into a settings
//! panel gets a legible refusal immediately instead of a failure the next
//! time something tries to connect.

use std::net::{Ipv4Addr, Ipv6Addr};

use url::{Host, Url};

/// Whether the request that will use this URL carries a credential — a
/// bearer token, an API key header, an OAuth access token, or a signed
/// request built from one. Declared by the caller, who knows its own
/// connector or transport; see the module docs for why this narrows the
/// bare-hostname exemption rather than the private-IP-literal ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestCredential {
    /// No credential travels with this request — e.g. a self-hosted Ollama
    /// connection, or an MCP server with neither `auth_bearer_secret` nor
    /// an OAuth block set. A bare hostname carries nothing to steal.
    None,
    /// A credential is attached to every request through this URL.
    Attached,
}

/// Hostname aliases cloud providers commonly map — via a preloaded
/// `/etc/hosts` entry baked into the image, or a documented DNS-search-
/// domain convention — to their instance-metadata service, in addition to
/// the canonical link-local IP `169.254.169.254` (covered separately by
/// the link-local range check). Refused unconditionally, the same as that
/// range: a name on this list is never "a network the operator controls"
/// no matter how it resolves, so it is checked before, and regardless of,
/// the bare-hostname exemption below.
const BLOCKED_METADATA_HOSTNAMES: &[&str] = &[
    "metadata.google.internal", // GCP: canonical
    "metadata",                 // GCP: the short alias GCE Linux images preload into /etc/hosts
    "instance-data",            // AWS: a documented alias for the IMDS on some AMIs
];

/// Cloud-metadata IPv4 literals outside the link-local range this policy
/// already blocks unconditionally: Alibaba Cloud ECS's metadata service,
/// which lives inside `100.64.0.0/10` (CGNAT / RFC 6598) rather than
/// `169.254.0.0/16`. CGNAT is otherwise a credential-gated private-range
/// exemption ([`is_private_target`]) - these two addresses are refused
/// unconditionally instead, the same as link-local, because nothing inside
/// that range should ever answer as a metadata service, credentialed or
/// not.
const BLOCKED_METADATA_IPV4: [Ipv4Addr; 2] = [
    Ipv4Addr::new(100, 100, 100, 200), // Alibaba Cloud ECS: primary
    Ipv4Addr::new(100, 100, 100, 100), // Alibaba Cloud ECS: secondary
];

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
    /// network address, or (when no credential is attached) a bare
    /// hostname — i.e. a host reachable over the open network, where an
    /// attached credential would travel in the clear, or an unattached
    /// request would resolve by a mechanism the operator does not control.
    #[error(
        "{url:?} uses http:// to a host that is not loopback, a private network address, or a \
         bare hostname permitted for this request; TLS is required unless the destination stays \
         on a network the operator already controls"
    )]
    InsecureScheme { url: String },

    /// `raw` names a link-local address (including the cloud metadata IP),
    /// an unspecified address, or a known cloud-metadata hostname alias.
    /// Refused regardless of scheme.
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
/// dialed. See the module docs for the rule, `credential`, and why it is
/// shaped this way.
pub fn validate_remote_url(raw: &str, credential: RequestCredential) -> Result<(), UrlPolicyError> {
    let url = Url::parse(raw).map_err(|e| UrlPolicyError::Malformed {
        url: raw.to_string(),
        reason: e.to_string(),
    })?;

    // The SSRF floor: evaluated first, unconditionally, and nothing below
    // it can exempt a match. A typo'd scheme, or an otherwise-permitted
    // bare hostname, must not "save" a request to a blocked destination.
    if is_blocked_target(&url) {
        return Err(UrlPolicyError::BlockedTarget {
            url: raw.to_string(),
        });
    }

    match url.scheme() {
        "https" => Ok(()),
        "http" if is_private_target(&url, credential) => Ok(()),
        "http" => Err(UrlPolicyError::InsecureScheme {
            url: raw.to_string(),
        }),
        other => Err(UrlPolicyError::SchemeNotAllowed {
            url: raw.to_string(),
            scheme: other.to_string(),
        }),
    }
}

/// A destination this policy always refuses, regardless of scheme or
/// [`RequestCredential`]: the link-local range (host to the cloud-metadata
/// address `169.254.169.254` shared by AWS, Azure, and GCP's legacy
/// metadata path lives here), the unspecified address, a known
/// cloud-metadata hostname alias ([`BLOCKED_METADATA_HOSTNAMES`]), or a
/// known cloud-metadata IPv4 literal outside the link-local range
/// ([`BLOCKED_METADATA_IPV4`]).
fn is_blocked_target(url: &Url) -> bool {
    match url.host() {
        Some(Host::Ipv4(ip)) => {
            ip.is_link_local() || ip.is_unspecified() || BLOCKED_METADATA_IPV4.contains(&ip)
        }
        Some(Host::Ipv6(ip)) => match embedded_ipv4(&ip) {
            Some(v4) => {
                v4.is_link_local() || v4.is_unspecified() || BLOCKED_METADATA_IPV4.contains(&v4)
            }
            None => is_ipv6_link_local(&ip) || ip.is_unspecified(),
        },
        Some(Host::Domain(name)) => {
            let name = name.strip_suffix('.').unwrap_or(name);
            BLOCKED_METADATA_HOSTNAMES
                .iter()
                .any(|blocked| name.eq_ignore_ascii_case(blocked))
        }
        None => false,
    }
}

/// A destination plain `http://` may reach without sending an attached
/// credential onto the open network: loopback, an RFC1918 IPv4 range, an
/// IPv6 ULA (`fc00::/7`) range, or a bare (dot-free) hostname / CGNAT
/// address (`100.64.0.0/10`) — but only the first three are unconditional.
///
/// CGNAT and the bare-hostname arm are both gated on
/// [`RequestCredential::None`], for the same underlying reason stated in the
/// module docs, not two different ones: both are shared or resolved by a
/// mechanism outside the operator's control. A bare hostname resolves by
/// search-domain append, LLMNR, or public DNS; CGNAT space is explicitly
/// *shared* across a carrier's subscribers (RFC 6598's whole purpose,
/// unlike RFC1918 or ULA, which are not routed on any shared public-facing
/// network) - on a tethered or LTE link, the daemon's own address and an
/// unrelated subscriber's are both inside `100.64.0.0/10`, so a credentialed
/// plain-HTTP request there can land on someone else's equipment. Loopback,
/// RFC1918, and ULA carry no such risk: none of them is a shared or
/// resolved destination, so they stay exempt regardless of credential. A
/// credential-free request to a CGNAT/Tailscale address (`http://ollama`'s
/// shape, just by IP instead of by name) keeps working either way.
fn is_private_target(url: &Url, credential: RequestCredential) -> bool {
    match url.host() {
        Some(Host::Ipv4(ip)) => {
            ip.is_loopback()
                || ip.is_private()
                || (credential == RequestCredential::None && is_cgnat(ip))
        }
        Some(Host::Ipv6(ip)) => match embedded_ipv4(&ip) {
            Some(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || (credential == RequestCredential::None && is_cgnat(v4))
            }
            None => ip.is_loopback() || is_ipv6_ula(&ip),
        },
        Some(Host::Domain(name)) => {
            let name = name.strip_suffix('.').unwrap_or(name);
            name.eq_ignore_ascii_case("localhost")
                || (credential == RequestCredential::None && !name.contains('.'))
        }
        None => false,
    }
}

/// `100.64.0.0/10` (RFC 6598, Shared Address Space / CGNAT). Not covered by
/// `Ipv4Addr::is_private`, which is RFC1918 only.
fn is_cgnat(ip: Ipv4Addr) -> bool {
    (u32::from(ip) & 0xffc0_0000) == 0x6440_0000
}

/// `fe80::/10`. `std::net::Ipv6Addr` has no stable link-local predicate, so
/// this checks the range directly rather than reaching for an unstable one.
fn is_ipv6_link_local(ip: &Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

/// `fc00::/7` (RFC 4193, Unique Local Address) — IPv6's RFC1918 equivalent.
fn is_ipv6_ula(ip: &Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

/// Extract an IPv4 address embedded in an IPv6 literal, covering the three
/// ways one can hide there: IPv4-mapped (`::ffff:a.b.c.d`), the deprecated
/// IPv4-compatible form (`::a.b.c.d`), and the NAT64 well-known prefix
/// (`64:ff9b::a.b.c.d`, RFC 6052). Each is a literal re-encoding of the same
/// address, not a resolution step, so unwrapping them is still "judge the
/// literal shape" (see the module docs) — it closes an encoding-level
/// bypass, not a DNS one.
fn embedded_ipv4(ip: &Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return Some(v4);
    }
    let segments = ip.segments();
    // IPv4-compatible (deprecated; RFC 4291 section 2.5.5.1): the top 96
    // bits are zero. Excludes `::` and `::1` (unspecified/loopback), which
    // share that prefix but are not IPv4-compatible addresses.
    if segments[0..5] == [0, 0, 0, 0, 0] && segments[5] == 0 {
        let v4 = (u32::from(segments[6]) << 16) | u32::from(segments[7]);
        if v4 > 1 {
            return Some(Ipv4Addr::from(v4));
        }
    }
    // NAT64 well-known prefix (RFC 6052): 64:ff9b::/96.
    if segments[0] == 0x64 && segments[1] == 0xff9b && segments[2..6] == [0, 0, 0, 0] {
        let v4 = (u32::from(segments[6]) << 16) | u32::from(segments[7]);
        return Some(Ipv4Addr::from(v4));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: RequestCredential = RequestCredential::None;
    const ATTACHED: RequestCredential = RequestCredential::Attached;

    // ----- accepted: the shapes that must keep working -----

    #[test]
    fn accepts_a_legitimate_https_url_to_a_public_host() {
        validate_remote_url("https://mcp.example.com/api", ATTACHED)
            .expect("https to a public host");
    }

    #[test]
    fn accepts_http_to_ipv4_loopback_with_a_credential_attached() {
        validate_remote_url("http://127.0.0.1:8080/mcp", ATTACHED)
            .expect("http to loopback, even with a credential attached");
    }

    #[test]
    fn accepts_http_to_ipv6_loopback_with_a_credential_attached() {
        validate_remote_url("http://[::1]:8080/mcp", ATTACHED)
            .expect("http to ipv6 loopback, even with a credential attached");
    }

    #[test]
    fn accepts_http_to_localhost_hostname_with_a_credential_attached() {
        validate_remote_url("http://localhost:8080/mcp", ATTACHED)
            .expect("http to localhost, even with a credential attached");
    }

    /// The load-bearing case: `deploy/k8s/base/daemon.toml` and
    /// `docs/k8s-deployment.md` reach Ollama, which has no credential
    /// concept, at `http://ollama:11434` — an in-cluster Kubernetes Service
    /// name, which is a bare hostname.
    #[test]
    fn accepts_http_to_a_bare_in_cluster_service_name_with_no_credential() {
        validate_remote_url("http://ollama:11434", NONE)
            .expect("http to a bare k8s service name, when nothing is attached to steal");
    }

    #[test]
    fn accepts_http_to_an_rfc1918_private_address_with_a_credential_attached() {
        validate_remote_url("http://192.168.1.50:8000/v1", ATTACHED)
            .expect("http to a LAN address, even with a credential attached");
    }

    /// F3 (review): CGNAT space is shared across a carrier's subscribers
    /// (unlike RFC1918/ULA), so - mirroring the bare-hostname arm exactly -
    /// it is only exempt when the request carries nothing an unrelated
    /// subscriber's equipment could steal.
    #[test]
    fn accepts_http_to_a_cgnat_address_with_no_credential() {
        // 100.64.0.0/10 is the range Tailscale addresses live in.
        validate_remote_url("http://100.100.100.5:8080/v1", NONE)
            .expect("http to a CGNAT/Tailscale address is fine when nothing is attached to steal");
    }

    #[test]
    fn rejects_http_to_a_cgnat_address_with_a_credential_attached() {
        let err = validate_remote_url("http://100.100.100.5:8080/v1", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_insecure_scheme");
    }

    #[test]
    fn accepts_http_to_an_ipv6_ula_address_with_a_credential_attached() {
        validate_remote_url("http://[fd12:3456:789a::1]:8080/v1", ATTACHED)
            .expect("http to an ipv6 ULA address, even with a credential attached");
    }

    #[test]
    fn accepts_ipv4_mapped_ipv6_loopback_over_http() {
        validate_remote_url("http://[::ffff:127.0.0.1]/mcp", ATTACHED)
            .expect("an ipv4-mapped ipv6 loopback literal is still loopback");
    }

    // ----- refused: insecure scheme -----

    #[test]
    fn rejects_http_to_a_public_dotted_host() {
        let err = validate_remote_url("http://evil.example.com/mcp", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_insecure_scheme");
    }

    /// F1 (review): a bare hostname is not "a network the operator
    /// controls" when a credential travels with the request — it is
    /// resolved by search-domain append, LLMNR, or public DNS, none of
    /// which the operator authenticated.
    #[test]
    fn rejects_a_bare_hostname_over_http_when_a_credential_is_attached() {
        let err = validate_remote_url("http://homeassistant-mcp:8080/mcp", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_insecure_scheme");
    }

    // ----- refused: blocked target, regardless of scheme or credential -----

    #[test]
    fn rejects_link_local_address_over_http() {
        let err = validate_remote_url("http://169.254.1.1/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_cloud_metadata_ipv4_address_even_over_https() {
        // The scheme check alone would accept this; the blocked-target check
        // must run first regardless of scheme (see the module docs).
        let err =
            validate_remote_url("https://169.254.169.254/latest/meta-data/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_gcp_metadata_hostname() {
        let err = validate_remote_url(
            "http://metadata.google.internal/computeMetadata/v1/",
            ATTACHED,
        )
        .unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// F1 (review, the demonstrated case): GCE Linux images preload the
    /// bare alias `metadata` for the metadata IP in `/etc/hosts`. Without
    /// this on the blocklist, the bare-hostname exemption (proven above,
    /// for a request that carries no credential) would let it straight
    /// through — the blocklist has to be a floor even the no-credential
    /// case cannot cross.
    #[test]
    fn rejects_the_gce_metadata_bare_alias_even_with_no_credential_attached() {
        let err = validate_remote_url(
            "http://metadata/computeMetadata/v1/instance/service-accounts/default/token",
            NONE,
        )
        .unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// AWS's equally dot-free metadata alias, resolved through the instance's
    /// DNS search domain on some AMIs.
    #[test]
    fn rejects_the_aws_instance_data_bare_alias_even_with_no_credential_attached() {
        let err = validate_remote_url("http://instance-data/latest/meta-data/", NONE).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// F3 (review, the demonstrated case): Alibaba Cloud ECS's metadata
    /// service lives inside the CGNAT range (`100.64.0.0/10`), not the
    /// link-local range the other cloud metadata IPs share, so it needs its
    /// own entry on the floor. Demonstrated accepted before this fix, with a
    /// credential attached, over plain http.
    #[test]
    fn rejects_alibaba_metadata_primary_address_even_with_no_credential_attached() {
        let err = validate_remote_url(
            "http://100.100.100.200/latest/meta-data/ram/security-credentials/",
            NONE,
        )
        .unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_alibaba_metadata_secondary_address_even_over_https() {
        // The scheme check alone would accept this; the blocked-target check
        // must run first regardless of scheme (see the module docs).
        let err = validate_remote_url("https://100.100.100.100/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// The CGNAT exemption is credential-gated, but the Alibaba metadata
    /// addresses must stay refused even in the permitted (no-credential)
    /// case - they are on the unconditional floor, not the gated exemption.
    #[test]
    fn rejects_alibaba_metadata_address_regardless_of_the_cgnat_exemption() {
        let err =
            validate_remote_url("http://100.100.100.200/latest/meta-data/", NONE).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_unspecified_ipv4_address() {
        let err = validate_remote_url("http://0.0.0.0:8080/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_unspecified_ipv6_address() {
        let err = validate_remote_url("http://[::]:8080/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    #[test]
    fn rejects_ipv4_mapped_ipv6_link_local_address() {
        // Bypass check: a metadata address dressed up as an ipv4-mapped
        // ipv6 literal must still be caught.
        let err = validate_remote_url("http://[::ffff:169.254.169.254]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// A genuine (non-mapped) IPv6 link-local literal — `fe80::/10` has no
    /// stable stdlib predicate, so `is_ipv6_link_local`'s hand-rolled bitmask
    /// needs its own direct coverage, not just the ipv4-mapped-form tests.
    #[test]
    fn rejects_a_genuine_ipv6_link_local_address() {
        let err = validate_remote_url("http://[fe80::1]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// The upper edge of `fe80::/10` (`fe80::` through `febf:ffff:...`):
    /// confirms the bitmask compares the right width, not just one bit
    /// pattern.
    #[test]
    fn rejects_an_ipv6_link_local_address_at_the_top_of_the_range() {
        let err = validate_remote_url("http://[febf::1]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// Just outside `fe80::/10` on the low side (`fe7f::`) must NOT be
    /// treated as link-local — proves the mask, not just a substring match
    /// on "fe8".
    #[test]
    fn accepts_an_ipv6_address_just_below_the_link_local_range() {
        validate_remote_url("https://[fe7f::1]/", ATTACHED)
            .expect("fe7f::/16 is outside fe80::/10 and must not be treated as link-local");
    }

    /// IPv4-mapped IPv6 is otherwise only tested for loopback and
    /// link-local; a mapped private address exercises the same
    /// `embedded_ipv4` branch inside `is_private_target`.
    #[test]
    fn accepts_ipv4_mapped_ipv6_private_address_over_http() {
        validate_remote_url("http://[::ffff:192.168.1.1]/mcp", ATTACHED)
            .expect("an ipv4-mapped ipv6 private literal is still private");
    }

    /// ...and a mapped unspecified address exercises the same branch inside
    /// `is_blocked_target`.
    #[test]
    fn rejects_ipv4_mapped_ipv6_unspecified_address() {
        let err = validate_remote_url("http://[::ffff:0.0.0.0]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// F4 (review): the deprecated IPv4-compatible form must not dodge the
    /// link-local/metadata check either.
    #[test]
    fn rejects_ipv4_compatible_ipv6_link_local_address() {
        let err = validate_remote_url("http://[::169.254.169.254]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// F4 (review): nor the NAT64 well-known-prefix form.
    #[test]
    fn rejects_nat64_embedded_link_local_address() {
        let err = validate_remote_url("http://[64:ff9b::a9fe:a9fe]/", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    /// F4 (review): a trailing dot (the DNS root label, `name.` == `name`)
    /// must not dodge the metadata-hostname check.
    #[test]
    fn rejects_gcp_metadata_hostname_with_a_trailing_dot() {
        let err = validate_remote_url("https://metadata.google.internal./", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_target_blocked");
    }

    // ----- refused: malformed / missing scheme / disallowed scheme -----

    #[test]
    fn rejects_a_malformed_url() {
        let err = validate_remote_url("not a url at all", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_malformed");
    }

    #[test]
    fn rejects_a_url_with_no_scheme() {
        let err = validate_remote_url("mcp.example.com/api", ATTACHED).unwrap_err();
        assert_eq!(err.code(), "url_malformed");
    }

    #[test]
    fn rejects_a_disallowed_scheme() {
        let err = validate_remote_url("ftp://example.com/mcp", ATTACHED).unwrap_err();
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
            let err = validate_remote_url(input, ATTACHED).unwrap_err();
            assert_eq!(err.code(), *expected_code, "input: {input}");
            assert!(
                !err.user_message().is_empty(),
                "every refusal must carry user-facing text: {input}"
            );
        }
    }
}
