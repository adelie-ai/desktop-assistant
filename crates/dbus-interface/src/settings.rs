use std::sync::Arc;

use desktop_assistant_core::ports::inbound::SettingsService;
use zbus::{fdo, interface};

/// D-Bus adapter for assistant settings.
///
/// Exposes non-sensitive settings values and a write-only API key method.
pub struct DbusSettingsAdapter<S: SettingsService + 'static> {
    service: Arc<S>,
}

impl<S: SettingsService + 'static> DbusSettingsAdapter<S> {
    pub fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

#[interface(name = "org.desktopAssistant.Settings")]
impl<S: SettingsService + 'static> DbusSettingsAdapter<S> {
    /// Return non-sensitive LLM settings and whether an API key is available.
    async fn get_llm_settings(&self) -> fdo::Result<(String, String, String, bool)> {
        let settings = self
            .service
            .get_llm_settings()
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;

        Ok((
            settings.connector,
            settings.model,
            settings.base_url,
            settings.has_api_key,
        ))
    }

    /// Update non-sensitive LLM settings.
    async fn set_llm_settings(
        &self,
        connector: &str,
        model: &str,
        base_url: &str,
    ) -> fdo::Result<()> {
        let model = if model.trim().is_empty() {
            None
        } else {
            Some(model.to_string())
        };

        let base_url = if base_url.trim().is_empty() {
            None
        } else {
            Some(base_url.to_string())
        };

        self.service
            .set_llm_settings(connector.to_string(), model, base_url)
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    /// Write API key to configured secret backend.
    ///
    /// This is intentionally write-only; there is no D-Bus method to read back secrets.
    async fn set_api_key(&self, api_key: &str) -> fdo::Result<()> {
        self.service
            .set_api_key(api_key.to_string())
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::ports::inbound::{LlmSettingsView, SettingsService};

    struct FakeSettingsService;

    impl SettingsService for FakeSettingsService {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(LlmSettingsView {
                connector: "openai".to_string(),
                model: "gpt-4o".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                has_api_key: true,
            })
        }

        async fn set_llm_settings(
            &self,
            _connector: String,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }

        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[test]
    fn adapter_construction() {
        let service = Arc::new(FakeSettingsService);
        let _adapter = DbusSettingsAdapter::new(service);
    }
}
