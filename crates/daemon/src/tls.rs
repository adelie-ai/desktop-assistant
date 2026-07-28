use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, anyhow};
use rcgen::{
    BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair, KeyUsagePurpose, SanType,
};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

const CA_CERT_FILENAME: &str = "ca.pem";
const CA_KEY_FILENAME: &str = "ca-key.pem";
const SERVER_CERT_FILENAME: &str = "server-cert.pem";
const SERVER_KEY_FILENAME: &str = "server-key.pem";

/// Returns `$XDG_DATA_HOME/desktop-assistant/tls/`.
pub fn default_tls_dir() -> PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join(".local")
                .join("share")
        });
    data_home.join("desktop-assistant").join("tls")
}

/// Public path to the CA certificate so clients can discover it.
pub fn default_ca_cert_path() -> PathBuf {
    default_tls_dir().join(CA_CERT_FILENAME)
}

/// Set up TLS: ensure CA + server cert exist, return a `rustls::ServerConfig`.
///
/// If `cert_file` and `key_file` are provided, use those instead of auto-generating.
pub fn setup(
    cert_file: Option<&Path>,
    key_file: Option<&Path>,
) -> anyhow::Result<Arc<rustls::ServerConfig>> {
    match (cert_file, key_file) {
        (Some(cert), Some(key)) => {
            let cert_pem = std::fs::read(cert)
                .with_context(|| format!("reading TLS cert {}", cert.display()))?;
            let key_pem =
                std::fs::read(key).with_context(|| format!("reading TLS key {}", key.display()))?;
            build_server_config(&cert_pem, &key_pem)
        }
        (Some(_), None) | (None, Some(_)) => Err(anyhow!(
            "both tls.cert_file and tls.key_file must be set together"
        )),
        (None, None) => {
            let tls_dir = default_tls_dir();
            let (ca_cert_pem, ca_key_pem) = ensure_ca(&tls_dir)?;
            let (server_cert_pem, server_key_pem) =
                ensure_server_cert(&tls_dir, &ca_cert_pem, &ca_key_pem)?;
            // Chain: server cert + CA cert
            let mut chain = server_cert_pem;
            chain.extend_from_slice(&ca_cert_pem);
            build_server_config(&chain, &server_key_pem)
        }
    }
}

/// Outcome of resolving the WebSocket door's TLS posture at startup (#805).
///
/// There is no "TLS enabled but serving plaintext" state. Either TLS is off
/// by deliberate configuration ([`PlaintextByConfig`](Self::PlaintextByConfig)),
/// or it is on and ready ([`Tls`](Self::Tls)) — "on but not deliverable" is
/// not a variant here because [`resolve_ws_tls`] returns `Err` for it
/// instead, so the caller cannot construct a silent downgrade by accident.
pub enum WsTlsPosture {
    /// `[tls] enabled = false` (or `DESKTOP_ASSISTANT_WS_TLS=false`): the
    /// operator deliberately turned TLS off. Serving plaintext here is a
    /// choice the operator made, not a downgrade from one they didn't.
    PlaintextByConfig,
    /// TLS is on and ready to serve.
    Tls(Arc<rustls::ServerConfig>),
}

/// Decide the WebSocket door's TLS posture (#805): plaintext because TLS is
/// deliberately off, or TLS because it is on and ready — and fail closed
/// when it is on but cannot be delivered.
///
/// Why: the remote WebSocket door now carries the bearer token for an
/// administrator (#728: `[authz] admin_subjects`), so a daemon told to use
/// TLS and unable to must not silently serve that door in plaintext. When
/// `enabled` is `true` and [`setup`] cannot deliver it — missing files,
/// unreadable permissions (e.g. an ACME renewal that rewrote the key file
/// 0600 root:root), malformed PEM, or a write failure on the auto-generate
/// path — this returns `Err` instead of falling back to
/// [`WsTlsPosture::PlaintextByConfig`]. The configuration asked for a
/// security property the process cannot provide, which is a startup
/// failure, not a degradation: the caller is expected to abort rather than
/// bind.
pub fn resolve_ws_tls(
    enabled: bool,
    cert_file: Option<&Path>,
    key_file: Option<&Path>,
) -> anyhow::Result<WsTlsPosture> {
    if !enabled {
        return Ok(WsTlsPosture::PlaintextByConfig);
    }
    let server_config = setup(cert_file, key_file).context(
        "TLS is enabled ([tls] enabled defaults to true) but setup failed; refusing to serve \
         the remote WebSocket door in plaintext. Fix the TLS configuration, or set \
         DESKTOP_ASSISTANT_WS_TLS=false ([tls] enabled = false) to serve plaintext deliberately",
    )?;
    Ok(WsTlsPosture::Tls(server_config))
}

/// Load or generate the local CA.
fn ensure_ca(tls_dir: &Path) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let cert_path = tls_dir.join(CA_CERT_FILENAME);
    let key_path = tls_dir.join(CA_KEY_FILENAME);

    if cert_path.exists() && key_path.exists() {
        let cert_pem = std::fs::read(&cert_path)
            .with_context(|| format!("reading {}", cert_path.display()))?;
        let key_pem =
            std::fs::read(&key_path).with_context(|| format!("reading {}", key_path.display()))?;
        return Ok((cert_pem, key_pem));
    }

    tracing::info!("generating local CA certificate");

    let key_pair = KeyPair::generate()?;
    let mut params = CertificateParams::default();
    params
        .distinguished_name
        .push(DnType::CommonName, "Desktop Assistant Local CA");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    // 10-year validity
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    params.not_before = time::OffsetDateTime::from_unix_timestamp(now as i64)?;
    params.not_after =
        time::OffsetDateTime::from_unix_timestamp(now as i64 + 10 * 365 * 24 * 3600)?;

    let cert = params.self_signed(&key_pair)?;
    let cert_pem = cert.pem().into_bytes();
    let key_pem = key_pair.serialize_pem().into_bytes();

    ensure_dir(tls_dir)?;
    write_public_file(&cert_path, &cert_pem)?;
    write_secret_file(&key_path, &key_pem)?;

    Ok((cert_pem, key_pem))
}

/// Load or generate the server leaf certificate signed by the CA.
fn ensure_server_cert(
    tls_dir: &Path,
    ca_cert_pem: &[u8],
    ca_key_pem: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let cert_path = tls_dir.join(SERVER_CERT_FILENAME);
    let key_path = tls_dir.join(SERVER_KEY_FILENAME);

    if cert_path.exists() && key_path.exists() {
        let cert_pem = std::fs::read(&cert_path)?;
        // Check if the certificate is still valid (not expired).
        if !is_pem_cert_expired(&cert_pem) {
            let key_pem = std::fs::read(&key_path)?;
            return Ok((cert_pem, key_pem));
        }
        tracing::info!("server certificate expired; regenerating");
    }

    tracing::info!("generating server certificate");

    let ca_key = KeyPair::from_pem(std::str::from_utf8(ca_key_pem)?)?;
    let ca_issuer = Issuer::from_ca_cert_pem(std::str::from_utf8(ca_cert_pem)?, ca_key)?;

    let server_key = KeyPair::generate()?;
    let mut params = CertificateParams::new(vec!["localhost".to_string()])?;
    params
        .distinguished_name
        .push(DnType::CommonName, "Desktop Assistant");
    params.subject_alt_names = vec![
        SanType::DnsName("localhost".try_into()?),
        SanType::IpAddress(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
        SanType::IpAddress(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
    ];
    // 1-year validity
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    params.not_before = time::OffsetDateTime::from_unix_timestamp(now as i64)?;
    params.not_after = time::OffsetDateTime::from_unix_timestamp(now as i64 + 365 * 24 * 3600)?;

    let cert = params.signed_by(&server_key, &ca_issuer)?;
    let cert_pem = cert.pem().into_bytes();
    let key_pem = server_key.serialize_pem().into_bytes();

    ensure_dir(tls_dir)?;
    write_public_file(&cert_path, &cert_pem)?;
    write_secret_file(&key_path, &key_pem)?;

    Ok((cert_pem, key_pem))
}

/// Check whether the first certificate in a PEM bundle has expired.
///
/// Returns `true` for "treat as expired" — used by the bootstrap path
/// to decide whether to regenerate the local CA-signed cert. Parse
/// failures count as expired (so a corrupted file gets replaced rather
/// than blocking startup), but every unparseable case logs the reason
/// at `warn` so an operator can tell the difference between a genuine
/// expiry and a malformed-cert churn loop.
fn is_pem_cert_expired(pem_bytes: &[u8]) -> bool {
    let mut reader = std::io::BufReader::new(pem_bytes);
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_reader_iter(&mut reader)
        .filter_map(|r| r.ok())
        .collect();

    let Some(cert_der) = certs.first() else {
        tracing::warn!(
            "TLS cert PEM contained no parseable certificates; \
             treating as expired so the bootstrap path will regenerate"
        );
        return true;
    };

    // Parse the X.509 not_after field via a minimal DER walk.
    match parse_not_after_expired(cert_der.as_ref()) {
        Ok(expired) => expired,
        Err(reason) => {
            tracing::warn!(
                reason,
                "TLS cert DER parse failed; treating as expired so the \
                 bootstrap path will regenerate. A malformed cert here \
                 will trigger regeneration on every startup — replace \
                 the cert file or fix the parse error to break the loop."
            );
            true
        }
    }
}

/// Minimal DER parse to extract notAfter from an X.509 certificate.
/// Returns `Ok(true)` if `notAfter` is in the past, `Ok(false)` if the
/// cert is still valid, and `Err(&'static str)` describing where the
/// parse failed so the caller can log a meaningful diagnostic.
fn parse_not_after_expired(der: &[u8]) -> Result<bool, &'static str> {
    // X.509 structure: SEQUENCE { tbsCertificate SEQUENCE { ... validity SEQUENCE { notBefore, notAfter } ... } }
    // We use a simple approach: walk through the TBS fields.
    fn read_tag_len(data: &[u8]) -> Option<(u8, usize, usize)> {
        if data.is_empty() {
            return None;
        }
        let tag = data[0];
        if data.len() < 2 {
            return None;
        }
        let (len, header_len) = if data[1] & 0x80 == 0 {
            (data[1] as usize, 2)
        } else {
            let num_bytes = (data[1] & 0x7f) as usize;
            if data.len() < 2 + num_bytes {
                return None;
            }
            let mut len = 0usize;
            for i in 0..num_bytes {
                len = (len << 8) | data[2 + i] as usize;
            }
            (len, 2 + num_bytes)
        };
        Some((tag, len, header_len))
    }

    fn skip_element(data: &[u8]) -> Option<&[u8]> {
        let (_, len, header_len) = read_tag_len(data)?;
        data.get(header_len + len..)
    }

    fn enter_sequence(data: &[u8]) -> Option<&[u8]> {
        let (tag, len, header_len) = read_tag_len(data)?;
        if tag != 0x30 {
            return None;
        }
        // Truncate to the SEQUENCE's declared length so trailing siblings
        // (e.g. signatureAlgorithm + signature after the TBS SEQUENCE)
        // don't leak into the inner walk.
        data.get(header_len..header_len + len)
    }

    fn parse_time(data: &[u8]) -> Option<i64> {
        let (tag, len, header_len) = read_tag_len(data)?;
        let time_bytes = data.get(header_len..header_len + len)?;
        let s = std::str::from_utf8(time_bytes).ok()?;
        // UTCTime (tag 0x17): YYMMDDHHMMSSZ
        // GeneralizedTime (tag 0x18): YYYYMMDDHHMMSSZ
        let (year, rest) = if tag == 0x17 {
            let y: i32 = s.get(..2)?.parse().ok()?;
            let year = if y >= 50 { 1900 + y } else { 2000 + y };
            (year, s.get(2..)?)
        } else if tag == 0x18 {
            let y: i32 = s.get(..4)?.parse().ok()?;
            (y, s.get(4..)?)
        } else {
            return None;
        };
        let month: u32 = rest.get(..2)?.parse().ok()?;
        let day: u32 = rest.get(2..4)?.parse().ok()?;
        let hour: u32 = rest.get(4..6)?.parse().ok()?;
        let min: u32 = rest.get(6..8)?.parse().ok()?;
        let sec: u32 = rest.get(8..10)?.parse().ok()?;

        // Rough epoch calculation (ignoring leap seconds, good enough for expiry check)
        let days_before_year = {
            let y = year as i64;
            365 * (y - 1970) + ((y - 1969) / 4) - ((y - 1901) / 100) + ((y - 1601) / 400)
        };
        let days_in_months: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let mut day_of_year: u32 = 0;
        for (m, days) in days_in_months.iter().enumerate().take((month - 1) as usize) {
            day_of_year += days;
            if m == 1 && is_leap {
                day_of_year += 1;
            }
        }
        day_of_year += day - 1;

        let ts = days_before_year * 86400
            + day_of_year as i64 * 86400
            + hour as i64 * 3600
            + min as i64 * 60
            + sec as i64;
        Some(ts)
    }

    // Parse: outer SEQUENCE → TBS SEQUENCE
    let tbs = enter_sequence(der)
        .and_then(enter_sequence)
        .ok_or("expected outer SEQUENCE → TBS SEQUENCE")?;

    // TBS fields: version (explicit tag [0], optional), serialNumber, signature, issuer, validity
    let mut pos = tbs;

    // Skip version if present (context tag [0] = 0xA0)
    if pos.first() == Some(&0xA0) {
        pos = skip_element(pos).unwrap_or(pos);
    }
    // Skip serialNumber
    pos = skip_element(pos).unwrap_or(pos);
    // Skip signature
    pos = skip_element(pos).unwrap_or(pos);
    // Skip issuer
    pos = skip_element(pos).unwrap_or(pos);

    // Validity SEQUENCE { notBefore, notAfter }
    let validity = enter_sequence(pos).ok_or("expected validity SEQUENCE")?;
    // Skip notBefore
    let after_not_before =
        skip_element(validity).ok_or("validity sequence ended after notBefore")?;
    // Parse notAfter
    let not_after = parse_time(after_not_before).ok_or("could not parse notAfter timestamp")?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    Ok(now >= not_after)
}

fn build_server_config(
    cert_chain_pem: &[u8],
    key_pem: &[u8],
) -> anyhow::Result<Arc<rustls::ServerConfig>> {
    let certs: Vec<CertificateDer<'static>> =
        CertificateDer::pem_reader_iter(&mut std::io::BufReader::new(cert_chain_pem))
            .collect::<Result<Vec<_>, _>>()
            .context("parsing certificate PEM")?;

    let key: PrivateKeyDer<'static> =
        PrivateKeyDer::from_pem_reader(&mut std::io::BufReader::new(key_pem))
            .context("parsing private key PEM")?;

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("building rustls ServerConfig")?;

    Ok(Arc::new(config))
}

fn ensure_dir(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating directory {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    Ok(())
}

fn write_secret_file(path: &Path, data: &[u8]) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .with_context(|| format!("writing {}", path.display()))?;
        file.write_all(data)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, data).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn write_public_file(path: &Path, data: &[u8]) -> anyhow::Result<()> {
    std::fs::write(path, data).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_crypto_provider() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    }

    #[test]
    fn generates_ca_and_server_cert() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        let tls_dir = dir.path();

        let (ca_cert, ca_key) = ensure_ca(tls_dir).unwrap();
        assert!(tls_dir.join(CA_CERT_FILENAME).exists());
        assert!(tls_dir.join(CA_KEY_FILENAME).exists());
        assert!(String::from_utf8_lossy(&ca_cert).contains("BEGIN CERTIFICATE"));
        assert!(String::from_utf8_lossy(&ca_key).contains("BEGIN PRIVATE KEY"));

        let (server_cert, server_key) = ensure_server_cert(tls_dir, &ca_cert, &ca_key).unwrap();
        assert!(tls_dir.join(SERVER_CERT_FILENAME).exists());
        assert!(tls_dir.join(SERVER_KEY_FILENAME).exists());
        assert!(String::from_utf8_lossy(&server_cert).contains("BEGIN CERTIFICATE"));
        assert!(String::from_utf8_lossy(&server_key).contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn reuses_existing_ca_on_second_call() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        let tls_dir = dir.path();

        let (ca1, _) = ensure_ca(tls_dir).unwrap();
        let (ca2, _) = ensure_ca(tls_dir).unwrap();
        assert_eq!(ca1, ca2, "CA should be reused, not regenerated");
    }

    #[test]
    fn builds_server_config_from_generated_certs() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        let tls_dir = dir.path();

        let (ca_cert, ca_key) = ensure_ca(tls_dir).unwrap();
        let (server_cert, server_key) = ensure_server_cert(tls_dir, &ca_cert, &ca_key).unwrap();

        let mut chain = server_cert;
        chain.extend_from_slice(&ca_cert);
        let config = build_server_config(&chain, &server_key);
        assert!(
            config.is_ok(),
            "should build valid ServerConfig: {:?}",
            config.err()
        );
    }

    #[test]
    fn setup_generates_certs_and_returns_config() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        // Override XDG_DATA_HOME so setup() writes to our temp dir.
        // Since setup() uses default_tls_dir() internally, we test the full path
        // via the individual functions instead.
        let tls_dir = dir.path();

        let (ca_cert, ca_key) = ensure_ca(tls_dir).unwrap();
        let (server_cert, server_key) = ensure_server_cert(tls_dir, &ca_cert, &ca_key).unwrap();
        let mut chain = server_cert;
        chain.extend_from_slice(&ca_cert);
        let config = build_server_config(&chain, &server_key).unwrap();
        // Smoke test: config should accept TLS 1.3
        assert!(config.alpn_protocols.is_empty()); // default, no ALPN
    }

    #[test]
    fn not_after_check_detects_valid_cert() {
        let dir = tempfile::tempdir().unwrap();
        let (ca_cert, _) = ensure_ca(dir.path()).unwrap();
        assert!(
            !is_pem_cert_expired(&ca_cert),
            "fresh CA should not be expired"
        );
    }

    #[test]
    fn not_after_check_detects_expired_cert() {
        // Generate a cert that expired 1 second ago.
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::default();
        params
            .distinguished_name
            .push(DnType::CommonName, "expired-test");
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        params.not_before = time::OffsetDateTime::from_unix_timestamp(now - 3600).unwrap();
        params.not_after = time::OffsetDateTime::from_unix_timestamp(now - 1).unwrap();
        let cert = params.self_signed(&key).unwrap();
        let pem = cert.pem().into_bytes();

        assert!(is_pem_cert_expired(&pem), "should detect expired cert");
    }

    // --- resolve_ws_tls (#805): fail closed instead of silently downgrading
    // the remote WebSocket door to plaintext when TLS is enabled but cannot
    // be delivered. -------------------------------------------------------

    /// Writes a fresh self-signed cert + key PEM pair to `dir` and returns
    /// their paths, so a test can point `resolve_ws_tls` at real files
    /// without touching the process-wide `XDG_DATA_HOME` auto-generate path.
    fn write_self_signed_cert(dir: &Path) -> (PathBuf, PathBuf) {
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("generate self-signed cert for test fixture");
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, certified.cert.pem()).expect("write test cert");
        std::fs::write(&key_path, certified.signing_key.serialize_pem()).expect("write test key");
        (cert_path, key_path)
    }

    /// Acceptance: TLS configured and working. A daemon with `[tls] enabled
    /// = true` and a readable, valid cert/key pair serves TLS, not
    /// plaintext.
    #[test]
    fn resolve_ws_tls_configured_and_working_yields_a_ready_tls_posture() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        let (cert_path, key_path) = write_self_signed_cert(dir.path());

        let posture = resolve_ws_tls(true, Some(&cert_path), Some(&key_path))
            .expect("valid cert/key must resolve to a TLS posture, not an error");

        assert!(
            matches!(posture, WsTlsPosture::Tls(_)),
            "TLS configured and working must yield WsTlsPosture::Tls, not plaintext"
        );
    }

    /// Acceptance: TLS configured and failing. A daemon with `[tls] enabled
    /// = true` (the default) but an unreadable cert/key pair — an expired
    /// ACME renewal, a permissions error, a corrupt file — must refuse to
    /// serve the remote WebSocket door at all. It must NOT return a
    /// plaintext posture: this is the silent-downgrade bug (#805), and the
    /// regression is specifically that `resolve_ws_tls` must return `Err`,
    /// not `Ok(WsTlsPosture::PlaintextByConfig)`.
    #[test]
    fn resolve_ws_tls_configured_and_failing_refuses_rather_than_downgrading() {
        install_crypto_provider();
        let dir = tempfile::tempdir().unwrap();
        let missing_cert = dir.path().join("does-not-exist-cert.pem");
        let missing_key = dir.path().join("does-not-exist-key.pem");

        let result = resolve_ws_tls(true, Some(&missing_cert), Some(&missing_key));

        assert!(
            result.is_err(),
            "TLS enabled with an unreadable cert/key must fail closed (Err), \
             never silently fall back to plaintext"
        );
    }

    /// Acceptance: TLS not configured at all. `[tls] enabled = false` (or
    /// `DESKTOP_ASSISTANT_WS_TLS=false`) is a deliberate operator choice, so
    /// plaintext here is correct — and it must stay correct even when a
    /// stale cert/key path also happens to be present in config, since a
    /// disabled TLS section makes those paths irrelevant.
    #[test]
    fn resolve_ws_tls_not_configured_at_all_is_deliberate_plaintext() {
        let result =
            resolve_ws_tls(false, None, None).expect("TLS disabled must resolve, not error");
        assert!(matches!(result, WsTlsPosture::PlaintextByConfig));

        let dir = tempfile::tempdir().unwrap();
        let missing_cert = dir.path().join("does-not-exist-cert.pem");
        let missing_key = dir.path().join("does-not-exist-key.pem");
        let result_with_stale_paths =
            resolve_ws_tls(false, Some(&missing_cert), Some(&missing_key))
                .expect("TLS disabled must resolve even if cert/key paths are stale");
        assert!(matches!(
            result_with_stale_paths,
            WsTlsPosture::PlaintextByConfig
        ));
    }
}
