//! The `[Recall]` width a deployment configured (#1124).
//!
//! Its own integration binary because the width is process-wide: the daemon
//! installs it once, at startup, before the first turn runs. A unit test inside
//! the library's own test binary would set it for every other test in that
//! process, so the setting is proven here instead.

use desktop_assistant_core::recall::{
    DEFAULT_MAX_RECALL_ENTRIES, max_recall_entries, set_max_recall_entries,
};

/// Acceptance (#1124): the width is configurable. One test function, not
/// several, because the setting is process-wide and separate test functions
/// would run in an order nothing states.
#[test]
fn a_configured_recall_width_replaces_the_default_for_the_process() {
    assert_eq!(
        max_recall_entries(),
        DEFAULT_MAX_RECALL_ENTRIES,
        "a process where nothing was configured renders the derived default"
    );

    assert_eq!(
        set_max_recall_entries(12),
        12,
        "a width the block can honestly render takes effect as stated"
    );
    assert_eq!(max_recall_entries(), 12);

    assert_eq!(
        set_max_recall_entries(30),
        12,
        "the width is read once at startup, so a second install keeps the live value"
    );
    assert_eq!(max_recall_entries(), 12);
}
