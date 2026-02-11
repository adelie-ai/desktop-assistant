pub mod ports;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("system service error: {0}")]
    SystemService(String),
}

#[cfg(test)]
mod tests {
    #[test]
    fn core_crate_loads() {
        // Validates that the core crate compiles and its module tree is reachable.
        assert_eq!(1, 1);
    }
}
