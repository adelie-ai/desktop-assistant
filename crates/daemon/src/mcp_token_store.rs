//! Keyring-backed persistence for remote-MCP OAuth tokens (#455).
//!
//! The `mcp-client` crate is deliberately agnostic about *where* OAuth tokens
//! live — it only knows the [`TokenStore`] trait. This module supplies the
//! daemon's concrete backend: the system Secret Service (via keyring-core,
//! already registered at daemon startup), storing each account's `TokenSet` as
//! JSON under one entry.
//!
//! It is a **best-effort cache**: when there is no Secret Service (headless) it
//! degrades to a silent no-op and the provider keeps tokens in memory. Swapping
//! the backend out later (e.g. for a server-side store when this functionality
//! is factored out of the desktop daemon) means implementing `TokenStore`
//! elsewhere and injecting it — no change to the transport or provider.

use std::sync::Arc;

use desktop_assistant_mcp_client::oauth::{OAuthError, TokenSet, TokenStore};
use keyring_core::{CredentialStore, Entry};

/// Secret Service "service" attribute under which per-account MCP OAuth tokens
/// are stored. The account key (usually an email) is the entry's "username".
const KEYRING_SERVICE: &str = "desktop-assistant-mcp-oauth";

/// A [`TokenStore`] backed by keyring-core.
///
/// In production `store` is `None`, so entries are built from the process-global
/// default store (the Secret Service, registered at daemon startup). Tests inject
/// a dedicated store so they never touch — or race — the global default.
#[derive(Default)]
pub struct KeyringTokenStore {
    store: Option<Arc<CredentialStore>>,
}

impl KeyringTokenStore {
    /// Use the process-global default credential store.
    pub fn new() -> Self {
        Self { store: None }
    }

    /// Build a keyring entry for `key`, either from the injected store or the
    /// process-global default.
    fn entry(&self, key: &str) -> Result<Entry, keyring_core::Error> {
        match &self.store {
            Some(store) => store.build(KEYRING_SERVICE, key, None),
            None => Entry::new(KEYRING_SERVICE, key),
        }
    }
}

impl TokenStore for KeyringTokenStore {
    fn load(&self, key: &str) -> Result<Option<TokenSet>, OAuthError> {
        run_keyring_blocking(
            || match self.entry(key).and_then(|entry| entry.get_password()) {
                Ok(json) => {
                    let tokens = serde_json::from_str(&json).map_err(|error| {
                        OAuthError::Store(format!("corrupt stored token for '{key}': {error}"))
                    })?;
                    Ok(Some(tokens))
                }
                // No token cached yet, or no Secret Service at all (headless).
                Err(keyring_core::Error::NoEntry | keyring_core::Error::NoDefaultStore) => Ok(None),
                Err(error) => Err(OAuthError::Store(format!(
                    "keyring read failed for '{key}': {error}"
                ))),
            },
        )
    }

    fn save(&self, key: &str, token: &TokenSet) -> Result<(), OAuthError> {
        let json = serde_json::to_string(token)
            .map_err(|error| OAuthError::Store(format!("failed to serialize token: {error}")))?;
        run_keyring_blocking(|| {
            let entry = match self.entry(key) {
                Ok(entry) => entry,
                // Headless: no Secret Service ⇒ silently skip persistence.
                Err(keyring_core::Error::NoDefaultStore) => return Ok(()),
                Err(error) => {
                    return Err(OAuthError::Store(format!(
                        "failed to init keyring entry for '{key}': {error}"
                    )));
                }
            };
            match entry.set_password(&json) {
                Ok(()) => Ok(()),
                Err(keyring_core::Error::NoDefaultStore) => Ok(()),
                Err(error) => Err(OAuthError::Store(format!(
                    "keyring write failed for '{key}': {error}"
                ))),
            }
        })
    }
}

/// Run a blocking Secret Service call without starving the async runtime.
///
/// Mirrors the helper in `config::secrets` (kept local so this module stays
/// self-contained and easy to lift out). keyring-core's Secret Service store
/// drives D-Bus over zbus's *blocking* API, which must not run directly on an
/// async worker. On the multi-threaded runtime `block_in_place` relocates other
/// tasks; off a runtime or on a current-thread one, run inline.
fn run_keyring_blocking<T>(operation: impl FnOnce() -> T) -> T {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(operation)
        }
        _ => operation(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store backed by a *dedicated* mock — deliberately NOT the process
    /// global, so this test can't race the `config::secrets` keyring tests that
    /// share the global default store.
    fn mock_store() -> KeyringTokenStore {
        KeyringTokenStore {
            store: Some(keyring_core::mock::Store::new().unwrap()),
        }
    }

    fn sample_token() -> TokenSet {
        TokenSet {
            access_token: "at".into(),
            refresh_token: Some("rt".into()),
            expires_at: None,
            token_type: "Bearer".into(),
            scope: Some("calendar".into()),
        }
    }

    #[test]
    fn keyring_store_roundtrips_a_token() {
        let store = mock_store();
        // Missing key ⇒ None.
        assert!(store.load("nobody@example.com").unwrap().is_none());

        store.save("dave@example.com", &sample_token()).unwrap();
        let loaded = store.load("dave@example.com").unwrap().unwrap();
        assert_eq!(loaded.access_token, "at");
        assert_eq!(loaded.refresh_token.as_deref(), Some("rt"));
        assert_eq!(loaded.scope.as_deref(), Some("calendar"));
    }

    #[test]
    fn keyring_store_overwrites_on_resave() {
        let store = mock_store();
        store.save("acct", &sample_token()).unwrap();
        let mut updated = sample_token();
        updated.access_token = "at2".into();
        store.save("acct", &updated).unwrap();
        assert_eq!(store.load("acct").unwrap().unwrap().access_token, "at2");
    }
}
