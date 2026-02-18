use std::path::PathBuf;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::inbound::{
    ConnectorDefaultsView, EmbeddingsSettingsView, LlmSettingsView, SettingsService,
};

use crate::config;

pub struct DaemonSettingsService {
    config_path: PathBuf,
}

impl DaemonSettingsService {
    pub fn new(config_path: PathBuf) -> Self {
        Self { config_path }
    }
}

impl SettingsService for DaemonSettingsService {
    async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
        let view = config::get_llm_settings_view(&self.config_path)
            .map_err(|error| CoreError::SystemService(error.to_string()))?;

        Ok(LlmSettingsView {
            connector: view.connector,
            model: view.model,
            base_url: view.base_url,
            has_api_key: view.has_api_key,
        })
    }

    async fn set_llm_settings(
        &self,
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<(), CoreError> {
        config::set_llm_settings(
            &self.config_path,
            &connector,
            model.as_deref(),
            base_url.as_deref(),
        )
        .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn set_api_key(&self, api_key: String) -> Result<(), CoreError> {
        config::set_api_key(&self.config_path, &api_key)
            .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
        let view = config::get_embeddings_settings_view(&self.config_path)
            .map_err(|error| CoreError::SystemService(error.to_string()))?;

        Ok(EmbeddingsSettingsView {
            connector: view.connector,
            model: view.model,
            base_url: view.base_url,
            has_api_key: view.has_api_key,
            available: view.available,
            is_default: view.is_default,
        })
    }

    async fn set_embeddings_settings(
        &self,
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<(), CoreError> {
        config::set_embeddings_settings(
            &self.config_path,
            connector.as_deref(),
            model.as_deref(),
            base_url.as_deref(),
        )
        .map_err(|error| CoreError::SystemService(error.to_string()))
    }

    async fn get_connector_defaults(
        &self,
        connector: String,
    ) -> Result<ConnectorDefaultsView, CoreError> {
        let defaults = config::get_connector_defaults(&connector);
        Ok(ConnectorDefaultsView {
            llm_model: defaults.llm_model,
            llm_base_url: defaults.llm_base_url,
            embeddings_model: defaults.embeddings_model,
            embeddings_base_url: defaults.embeddings_base_url,
            embeddings_available: defaults.embeddings_available,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_service_constructs() {
        let service = DaemonSettingsService::new(PathBuf::from("/tmp/desktop-assistant-test.toml"));
        assert!(service.config_path.ends_with("desktop-assistant-test.toml"));
    }
}
