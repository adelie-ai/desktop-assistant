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

    /// Return resolved embeddings settings.
    ///
    /// Returns: (connector, model, base_url, has_api_key, available, is_default)
    async fn get_embeddings_settings(
        &self,
    ) -> fdo::Result<(String, String, String, bool, bool, bool)> {
        let settings = self
            .service
            .get_embeddings_settings()
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;

        Ok((
            settings.connector,
            settings.model,
            settings.base_url,
            settings.has_api_key,
            settings.available,
            settings.is_default,
        ))
    }

    /// Update embeddings settings. Empty connector clears override (reverts to LLM default).
    async fn set_embeddings_settings(
        &self,
        connector: &str,
        model: &str,
        base_url: &str,
    ) -> fdo::Result<()> {
        let connector = if connector.trim().is_empty() {
            None
        } else {
            Some(connector.to_string())
        };

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
            .set_embeddings_settings(connector, model, base_url)
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    /// Return connector defaults.
    ///
    /// Returns: (llm_model, llm_base_url, embeddings_model, embeddings_base_url, embeddings_available)
    async fn get_connector_defaults(
        &self,
        connector: &str,
    ) -> fdo::Result<(String, String, String, String, bool)> {
        let defaults = self
            .service
            .get_connector_defaults(connector.to_string())
            .await
            .map_err(|e| fdo::Error::Failed(e.to_string()))?;

        Ok((
            defaults.llm_model,
            defaults.llm_base_url,
            defaults.embeddings_model,
            defaults.embeddings_base_url,
            defaults.embeddings_available,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::ports::inbound::{
        ConnectorDefaultsView, EmbeddingsSettingsView, LlmSettingsView, SettingsService,
    };

    struct FakeSettingsService;

    impl SettingsService for FakeSettingsService {
        async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
            Ok(LlmSettingsView {
                connector: "openai".to_string(),
                model: "gpt-5.2".to_string(),
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

        async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
            Ok(EmbeddingsSettingsView {
                connector: "ollama".to_string(),
                model: "nomic-embed-text".to_string(),
                base_url: "http://localhost:11434".to_string(),
                has_api_key: false,
                available: true,
                is_default: true,
            })
        }

        async fn set_embeddings_settings(
            &self,
            _connector: Option<String>,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
            Ok(ConnectorDefaultsView {
                llm_model: "gpt-5.2".to_string(),
                llm_base_url: "https://api.openai.com/v1".to_string(),
                embeddings_model: "text-embedding-3-small".to_string(),
                embeddings_base_url: "https://api.openai.com/v1".to_string(),
                embeddings_available: true,
            })
        }
    }

    #[test]
    fn adapter_construction() {
        let service = Arc::new(FakeSettingsService);
        let _adapter = DbusSettingsAdapter::new(service);
    }
}
