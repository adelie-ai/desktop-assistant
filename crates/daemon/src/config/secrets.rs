//! Secret-store backends for connection API keys.
//!
//! Extracted from `config.rs` (#41). Each backend reads + writes a
//! single string value keyed by `(service, account)`. The `auto`
//! backend tries the file store first (cheapest), then systemd
//! credentials, then the system Secret Service, then KWallet.
//!
//! The Secret Service backend talks D-Bus in-process via keyring-core's
//! zbus store (registered at daemon startup). Older daemons shelled out to
//! the `secret-tool` CLI, which keyed items by the `account` attribute;
//! keyring-core keys by `username`, so reads transparently migrate any
//! legacy `account`-keyed items to the current scheme.
//!
//! Schema-side helpers (`SecretConfig`, `default_secret_account`,
//! `resolve_secret_account`, etc.) stay in `super` because they are
//! also called from settings setters and views unrelated to the
//! backend I/O. This module reaches them via `super::…`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use anyhow::anyhow;
use keyring_core::Entry;

use super::SecretConfig;

pub(super) fn read_secret_from_backend(secret: &SecretConfig, connector: &str) -> Option<String> {
    match secret.backend.trim().to_lowercase().as_str() {
        "auto" => read_auto_secret(secret, connector),
        "systemd" | "systemd-credentials" => read_systemd_credential(secret, connector),
        "keyring" | "libsecret" => read_keyring_secret(secret, connector),
        "kwallet" => read_kwallet_secret(secret, connector),
        other => {
            tracing::warn!("unsupported secret backend '{}', falling back", other);
            None
        }
    }
}

fn read_auto_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let account = super::resolve_secret_account(secret, connector);
    if let Some(value) = read_common_file_secret(&account) {
        return Some(value);
    }

    if let Some(value) = read_systemd_credential(secret, connector) {
        return Some(value);
    }

    if let Some(value) = read_keyring_secret(secret, connector) {
        return Some(value);
    }

    read_kwallet_secret(secret, connector)
}

pub(super) fn read_common_file_secret(account: &str) -> Option<String> {
    let path = super::common_secret_file_path(account);
    let value = std::fs::read_to_string(path).ok()?;
    sanitize_secret_value(&value)
}

fn read_systemd_credential(secret: &SecretConfig, connector: &str) -> Option<String> {
    let credentials_dir = std::env::var_os("CREDENTIALS_DIRECTORY")?;
    let account = super::resolve_secret_account(secret, connector);
    let path = PathBuf::from(credentials_dir).join(account);

    let value = std::fs::read_to_string(path).ok()?;
    sanitize_secret_value(&value)
}

fn read_keyring_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let service = secret
        .service
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(super::default_secret_service);
    let account = super::resolve_secret_account(secret, connector);

    run_keyring_blocking(|| {
        // Fast path: the current scheme stores the account under the Secret
        // Service `username` attribute (keyring-core's `user` parameter).
        match Entry::new(&service, &account).and_then(|entry| entry.get_password()) {
            Ok(value) => return sanitize_secret_value(&value),
            Err(keyring_core::Error::NoEntry) => {} // fall through to the legacy lookup
            Err(keyring_core::Error::NoDefaultStore) => return None, // headless: no Secret Service
            Err(error) => {
                tracing::debug!("keyring read failed: {error}");
                return None;
            }
        }

        // Back-compat: secrets written by older daemons (via `secret-tool`)
        // live under the `account` attribute. Read them, then migrate to the
        // current scheme so this slower path runs at most once per secret.
        read_and_migrate_legacy_secret(&service, &account)
    })
}

/// Read a secret stored under the legacy `secret-tool` attribute scheme
/// (`service` + `account`) and, on success, rewrite it under the current
/// scheme (`service` + `username`) so subsequent reads hit the fast path.
/// The rewrite/cleanup is best-effort — the caller still gets the value even
/// if migration fails.
///
/// Note: this path can only be exercised against a real Secret Service. The
/// in-memory mock store used by unit tests has no attributes, so `search`
/// finds nothing there and this returns `None`.
fn read_and_migrate_legacy_secret(service: &str, account: &str) -> Option<String> {
    let spec = HashMap::from([("service", service), ("account", account)]);
    let legacy = Entry::search(&spec).ok()?.into_iter().next()?;
    let value = sanitize_secret_value(&legacy.get_password().ok()?)?;

    match Entry::new(service, account) {
        Ok(entry) if entry.set_password(&value).is_ok() => {
            if let Err(error) = legacy.delete_credential() {
                tracing::debug!("failed to delete migrated legacy keyring secret: {error}");
            }
        }
        Ok(_) => tracing::debug!("failed to migrate legacy keyring secret to current scheme"),
        Err(error) => tracing::debug!("failed to build entry for legacy migration: {error}"),
    }

    Some(value)
}

/// Run a blocking Secret Service operation without starving the async runtime.
///
/// keyring-core's Secret Service store drives D-Bus over zbus's *blocking*
/// API, which must not run on an async worker thread (it can stall the
/// runtime). On the daemon's multi-threaded runtime we hand the work to
/// `block_in_place`, which relocates other tasks onto a sibling worker; off a
/// runtime (sync tests) or on a current-thread runtime we just run inline.
fn run_keyring_blocking<T>(operation: impl FnOnce() -> T) -> T {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(operation)
        }
        _ => operation(),
    }
}

pub(super) fn write_secret_to_backend(
    secret: &SecretConfig,
    value: &str,
    connector: &str,
) -> anyhow::Result<()> {
    match secret.backend.trim().to_lowercase().as_str() {
        "auto" => write_auto_secret(secret, value, connector),
        "systemd" | "systemd-credentials" => Err(anyhow!(
            "systemd credentials backend is read-only; configure credentials via systemd and use SetLlmSettings only"
        )),
        "keyring" | "libsecret" => write_keyring_secret(secret, value, connector),
        "kwallet" => write_kwallet_secret(secret, value, connector),
        other => Err(anyhow!("unsupported secret backend '{other}'")),
    }
}

fn write_auto_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let account = super::resolve_secret_account(secret, connector);
    write_common_file_secret(&account, value)
}

pub(super) fn write_common_file_secret(account: &str, value: &str) -> anyhow::Result<()> {
    let dir = super::default_secret_store_dir();
    std::fs::create_dir_all(&dir).map_err(|error| {
        anyhow!(
            "failed to create secret store directory {}: {error}",
            dir.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }

    let path = super::common_secret_file_path(account);

    // Write the secret file with restricted permissions atomically to avoid a
    // TOCTOU window where the file is world-readable before chmod.
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| anyhow!("failed to write secret file {}: {error}", path.display()))?;
        file.write_all(value.as_bytes())
            .map_err(|error| anyhow!("failed to write secret file {}: {error}", path.display()))?;
    }

    #[cfg(not(unix))]
    {
        std::fs::write(&path, value)
            .map_err(|error| anyhow!("failed to write secret file {}: {error}", path.display()))?;
    }

    Ok(())
}

fn write_keyring_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let service = secret
        .service
        .clone()
        .filter(|candidate| !candidate.trim().is_empty())
        .unwrap_or_else(super::default_secret_service);
    let account = super::resolve_secret_account(secret, connector);

    run_keyring_blocking(|| {
        // Stores under the current scheme (Secret Service `username` attribute).
        // Any legacy `account`-keyed item is cleaned up the next time the secret
        // is read (see `read_and_migrate_legacy_secret`); the daemon reads every
        // configured secret at startup, so that happens before user-driven writes.
        let entry = Entry::new(&service, &account)
            .map_err(|error| anyhow!("failed to initialize keyring entry: {error}"))?;
        entry
            .set_password(value)
            .map_err(|error| anyhow!("failed to write keyring secret: {error}"))
    })
}

pub(super) fn sanitize_secret_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    if is_placeholder_secret_value(trimmed) {
        tracing::warn!("ignoring placeholder-like secret value from backend");
        return None;
    }

    Some(trimmed.to_string())
}

pub(super) fn is_placeholder_secret_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();

    value.contains('*')
        || normalized.starts_with("file-store")
        || normalized.starts_with("secret-store")
        || normalized.contains("write-only")
        || normalized.contains("leave blank")
}

pub(super) fn bucket_secret_len(len: usize) -> &'static str {
    match len {
        0 => "0",
        1..=15 => "<16",
        16..=31 => "16-31",
        32..=47 => "32-47",
        48..=79 => "48-79",
        _ => ">=80",
    }
}

pub(super) fn redacted_secret_audit(value: &str) -> (usize, String) {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let trimmed = value.trim();
    let mut hash = FNV_OFFSET_BASIS;
    for byte in trimmed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    (trimmed.len(), format!("fnv1a64:{hash:016x}"))
}

fn write_kwallet_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let entry = super::resolve_wallet_entry(secret, connector);
    let attempts = [
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-w".to_string(),
            value.to_string(),
            entry.clone(),
            secret.wallet.clone(),
        ],
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-e".to_string(),
            entry,
            "-w".to_string(),
            value.to_string(),
            secret.wallet.clone(),
        ],
    ];

    let mut last_error = String::from("unknown kwallet error");
    for args in attempts {
        let output = Command::new("kwallet-query").args(args).output();

        match output {
            Ok(result) if result.status.success() => return Ok(()),
            Ok(result) => {
                let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
                let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
                last_error = if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    "kwallet-query returned non-zero exit status".to_string()
                };
            }
            Err(error) => {
                last_error = error.to_string();
            }
        }
    }

    Err(anyhow!("failed to write KWallet secret: {last_error}"))
}

fn read_kwallet_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let entry = super::resolve_wallet_entry(secret, connector);
    let output = Command::new("kwallet-query")
        .arg("-f")
        .arg(&secret.folder)
        .arg("-r")
        .arg(&entry)
        .arg(&secret.wallet)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout);
    sanitize_secret_value(value.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    // keyring-core's default store is process-global, so register the in-memory
    // mock store once for the whole test binary. Each test uses a unique service
    // name so the shared store can't cross-contaminate across parallel tests.
    //
    // These cover the current-scheme wiring (build/set/get via the
    // `keyring`/`libsecret` backend). The legacy `account`-attribute migration
    // in `read_and_migrate_legacy_secret` can only be exercised against a real
    // Secret Service: the mock store has no attributes, so its `search` cannot
    // model a `secret-tool`-written item.
    fn with_mock_store() {
        use std::sync::Once;
        static MOCK_STORE: Once = Once::new();
        MOCK_STORE.call_once(|| {
            keyring_core::set_default_store(keyring_core::mock::Store::new().unwrap());
        });
    }

    fn keyring_config(service: &str) -> SecretConfig {
        SecretConfig {
            backend: "keyring".to_string(),
            service: Some(service.to_string()),
            account: Some("api_key".to_string()),
            wallet: "kdewallet".to_string(),
            folder: "desktop-assistant".to_string(),
            entry: None,
        }
    }

    #[test]
    fn keyring_backend_round_trips_secret() {
        with_mock_store();
        let secret = keyring_config("test-roundtrip.desktopAssistant");
        write_secret_to_backend(&secret, "sk-live-roundtrip", "openai").unwrap();
        assert_eq!(
            read_secret_from_backend(&secret, "openai"),
            Some("sk-live-roundtrip".to_string())
        );
    }

    #[test]
    fn keyring_backend_returns_none_when_absent() {
        with_mock_store();
        let secret = keyring_config("test-absent.desktopAssistant");
        assert_eq!(read_secret_from_backend(&secret, "openai"), None);
    }

    #[test]
    fn keyring_backend_read_rejects_placeholder_value() {
        with_mock_store();
        // A placeholder that slipped past the UI must be filtered on read by
        // sanitize_secret_value rather than handed back as a real key.
        let secret = keyring_config("test-placeholder.desktopAssistant");
        write_secret_to_backend(&secret, "file-store-openai-key", "openai").unwrap();
        assert_eq!(read_secret_from_backend(&secret, "openai"), None);
    }
}
