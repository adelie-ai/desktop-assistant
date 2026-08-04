//! The AWS SDK clients for one Bedrock connection, built once and shared.
//!
//! Bedrock is reached through more than one SDK client: `bedrock-runtime` for
//! inference, and `bedrock` for the control plane that lists models. Each
//! backend needs the client its own API surface runs on, and the connector
//! needs a runtime client of its own for embeddings until `InvokeBackend`
//! takes them.
//!
//! They all share one value, rather than holding a lazy cell each, because
//! building a client resolves credentials: a cell per holder loads the AWS
//! credential chain - environment, profile file, SSO, instance role - once per
//! holder instead of once per connection.

use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrock::Client as BedrockControlClient;
use aws_sdk_bedrockruntime::Client;
use desktop_assistant_core::CoreError;
use tokio::sync::OnceCell;

use crate::{aws_profile_exists, region_from_base_url, static_credentials_from_api_key};

/// Lazily built AWS SDK clients for one connection's credentials and region.
pub(crate) struct SdkClients {
    /// Region, or a `bedrock-runtime` endpoint the region is read out of.
    base_url: String,
    /// Static credentials in `ACCESS_KEY_ID:SECRET[:SESSION_TOKEN]` form. An
    /// empty or unparseable value falls back to the AWS credential chain.
    api_key: String,
    /// Named AWS profile, if the connection sets one.
    aws_profile: Option<String>,
    /// Test-only control-plane endpoint. `None` in production, where the
    /// endpoint is derived from the region.
    control_endpoint_override: Option<String>,
    /// Test-only runtime endpoint. `None` in production.
    runtime_endpoint_override: Option<String>,
    /// The resolved credentials and region, built once and read by every
    /// client below.
    ///
    /// Its own cell, and not a step inside each client's cell, because
    /// resolving it is the expensive half: the AWS credential chain reads a
    /// profile file, and on an instance role or an SSO source it makes a
    /// network call. A cell per client pays that once per client.
    shared_config: OnceCell<aws_config::SdkConfig>,
    runtime: OnceCell<Client>,
    control: OnceCell<BedrockControlClient>,
}

impl SdkClients {
    pub(crate) fn new(
        api_key: String,
        base_url: String,
        aws_profile: Option<String>,
        control_endpoint_override: Option<String>,
        runtime_endpoint_override: Option<String>,
    ) -> Self {
        Self {
            base_url,
            api_key,
            aws_profile,
            control_endpoint_override,
            runtime_endpoint_override,
            shared_config: OnceCell::new(),
            runtime: OnceCell::new(),
            control: OnceCell::new(),
        }
    }

    /// The `bedrock-runtime` client, for inference.
    pub(crate) async fn runtime(&self) -> Result<&Client, CoreError> {
        self.runtime
            .get_or_try_init(|| async {
                let shared_config = self.shared_config().await;
                let Some(endpoint) = self.runtime_endpoint_override.as_ref() else {
                    return Ok(Client::new(shared_config));
                };
                let config = aws_sdk_bedrockruntime::config::Builder::from(shared_config)
                    .endpoint_url(endpoint)
                    .build();
                Ok(Client::from_conf(config))
            })
            .await
    }

    /// The `bedrock` control-plane client, for model listing.
    pub(crate) async fn control(&self) -> Result<&BedrockControlClient, CoreError> {
        self.control
            .get_or_try_init(|| async {
                let shared_config = self.shared_config().await;
                let Some(endpoint) = self.control_endpoint_override.as_ref() else {
                    return Ok(BedrockControlClient::new(shared_config));
                };
                let config = aws_sdk_bedrock::config::Builder::from(shared_config)
                    .endpoint_url(endpoint)
                    .build();
                Ok(BedrockControlClient::from_conf(config))
            })
            .await
    }

    /// The resolved credentials and region, resolved on first use.
    async fn shared_config(&self) -> &aws_config::SdkConfig {
        self.shared_config
            .get_or_init(|| self.load_shared_config())
            .await
    }

    async fn load_shared_config(&self) -> aws_config::SdkConfig {
        let mut loader = aws_config::defaults(BehaviorVersion::latest());

        let effective_profile = self
            .aws_profile
            .clone()
            .or_else(|| aws_profile_exists("adele").then(|| "adele".to_string()));

        if let Some(ref profile) = effective_profile {
            tracing::info!(aws_profile = %profile, "using AWS profile");
            loader = loader.profile_name(profile);
        }

        if let Some(region) = region_from_base_url(&self.base_url) {
            loader = loader.region(Region::new(region));
        }

        if let Some(credentials) = static_credentials_from_api_key(&self.api_key) {
            loader = loader.credentials_provider(credentials);
        } else if !self.api_key.trim().is_empty() {
            tracing::debug!(
                "llm.bedrock.api_key is set but not parseable as static credentials; falling back to AWS credential chain"
            );
        }

        loader.load().await
    }
}
