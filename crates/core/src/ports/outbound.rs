/// Outbound port for system service interaction.
///
/// The core uses this trait to call into system-level services
/// (e.g. system D-Bus) without depending on a concrete implementation.
pub trait SystemServiceClient: Send + Sync {
    /// Retrieves the hostname from the system.
    fn hostname(
        &self,
    ) -> impl std::future::Future<Output = Result<String, crate::CoreError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSystemClient;

    impl SystemServiceClient for FakeSystemClient {
        async fn hostname(&self) -> Result<String, crate::CoreError> {
            Ok("test-host".to_string())
        }
    }

    #[tokio::test]
    async fn fake_system_client_returns_hostname() {
        let client = FakeSystemClient;
        let hostname = client.hostname().await.unwrap();
        assert_eq!(hostname, "test-host");
    }
}
