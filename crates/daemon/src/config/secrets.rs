//! Secret-store backends for connection API keys.
//!
//! Extracted from `config.rs` (#41). Each backend reads + writes a
//! single string value keyed by `(service, account)`. The `auto`
//! backend tries the file store first (cheapest), then systemd
//! credentials, then libsecret/keyring, then KWallet.
//!
//! Schema-side helpers (`SecretConfig`, `default_secret_account`,
//! `resolve_secret_account`, etc.) stay in `super` because they are
//! also called from settings setters and views unrelated to the
//! backend I/O. This module reaches them via `super::…`.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::anyhow;
use keyring::Entry;

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

    if let Some(value) = read_secret_tool_secret(&service, &account) {
        return Some(value);
    }

    let entry = Entry::new(&service, &account).ok()?;
    let value = entry.get_password().ok()?;
    sanitize_secret_value(&value)
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

    if command_exists("secret-tool") {
        write_secret_tool_secret(&service, &account, value)?;
        return Ok(());
    }

    let entry = Entry::new(&service, &account)
        .map_err(|error| anyhow!("failed to initialize keyring entry: {error}"))?;
    entry
        .set_password(value)
        .map_err(|error| anyhow!("failed to write keyring secret: {error}"))
}

fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn read_secret_tool_secret(service: &str, account: &str) -> Option<String> {
    let output = Command::new("secret-tool")
        .arg("lookup")
        .arg("service")
        .arg(service)
        .arg("account")
        .arg(account)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout);
    sanitize_secret_value(value.as_ref())
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

fn write_secret_tool_secret(service: &str, account: &str, value: &str) -> anyhow::Result<()> {
    let mut child = Command::new("secret-tool")
        .arg("store")
        .arg("--label")
        .arg("Desktop Assistant API Key")
        .arg("service")
        .arg(service)
        .arg("account")
        .arg(account)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| anyhow!("failed to invoke secret-tool: {error}"))?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write as _;
        stdin
            .write_all(value.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .map_err(|error| anyhow!("failed to write secret-tool stdin: {error}"))?;
    } else {
        return Err(anyhow!("failed to open secret-tool stdin"));
    }

    let output = child
        .wait_with_output()
        .map_err(|error| anyhow!("failed waiting for secret-tool: {error}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            "secret-tool returned non-zero exit status".to_string()
        };
        Err(anyhow!("failed to write secret-tool secret: {detail}"))
    }
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
