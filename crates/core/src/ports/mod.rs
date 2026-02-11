/// Inbound ports — trait interfaces that adapters (e.g. D-Bus) call into.
pub mod inbound;

/// Outbound ports — trait interfaces the core uses to reach external services.
pub mod outbound;

#[cfg(test)]
mod tests {
    #[test]
    fn ports_modules_are_accessible() {
        // Validates that the port sub-modules compile and are reachable.
        let _ = std::any::type_name::<dyn super::inbound::AssistantService>();
        // SystemServiceClient uses impl Future, so it's not dyn-compatible.
        // We verify it exists by naming a concrete impl in its own test module.
        fn _assert_trait_exists<T: super::outbound::SystemServiceClient>() {}
    }
}
