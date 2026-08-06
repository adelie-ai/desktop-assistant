//! Who asked for a knowledge row to be destroyed (issue #1122).
//!
//! A knowledge row is removed for good on several paths: a person's delete
//! from a client panel, the model's own delete tool, the periodic trash sweep,
//! and the reap inside a consolidation transaction. A deployment must be able
//! to allow the first and refuse the rest, and the difference between them is
//! not the code path - it is whether a person asked.
//!
//! So the initiator travels as ambient context, the way the tenant identity
//! does in [`crate::ports::auth`]. A handler that receives a delete command
//! from a client control installs [`DeleteInitiator::Person`] with
//! [`with_delete_initiator`] before it calls the store. The storage adapter
//! reads it with [`current_delete_initiator`] when it composes the statement.
//!
//! ## The default is machine
//!
//! An unset slot reads as [`DeleteInitiator::Machine`]. Every path that nobody
//! has marked is therefore subject to the safety flag, including a path added
//! after this module was written. A new handler that forgets to install the
//! scope loses the ability to destroy a row, which is the safe direction.
//!
//! ## This is not the tenant
//!
//! [`crate::ports::auth::UserId`] says whose data a statement may touch. It is
//! set on every path, including the background sweep, so it cannot say whether
//! a person asked for anything. The two are separate on purpose.
//!
//! ## Temporary
//!
//! This exists because the model still holds a destructive verb and a wrong
//! decision cannot be undone. It is removed once deletion is a human verb by
//! construction (#893) and a retired entry can be restored (#710).

/// Who asked for a knowledge row to be destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeleteInitiator {
    /// A person named this entry through a client control - the knowledge
    /// panel's delete, or emptying the trash. A request to be forgotten is
    /// this, and it always erases.
    Person,
    /// Nobody asked. A maintenance pass, or the model acting on its own
    /// judgement inside a turn. This is the default, so an unmarked path is
    /// treated as the model's own decision rather than a person's.
    #[default]
    Machine,
}

impl DeleteInitiator {
    /// Stable spelling for logs and refusal messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Machine => "machine",
        }
    }

    /// Did a person ask for this?
    pub const fn is_person(self) -> bool {
        matches!(self, Self::Person)
    }
}

tokio::task_local! {
    /// Who asked for the delete now in progress. Installed by the handler that
    /// receives a person's delete command; read by the storage adapter when it
    /// composes the statement.
    ///
    /// An unset slot reads as [`DeleteInitiator::Machine`], so a background
    /// worker, a tool call, and a path that has not been considered all fall
    /// under the safety flag without having to say so.
    static DELETE_INITIATOR: DeleteInitiator;
}

/// Run `fut` with `initiator` installed as the current delete initiator.
///
/// Install it at the boundary where a person's intent arrives, not deeper: the
/// point of the value is that it records who asked, and a call site inside the
/// storage layer can only record which function was called.
pub async fn with_delete_initiator<F, T>(initiator: DeleteInitiator, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    DELETE_INITIATOR.scope(initiator, fut).await
}

/// Who asked for the delete now in progress, or [`DeleteInitiator::Machine`]
/// when no scope is installed.
///
/// Safe to call from any async context - it never panics and never blocks.
pub fn current_delete_initiator() -> DeleteInitiator {
    DELETE_INITIATOR.try_with(|i| *i).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unscoped_delete_is_machine_initiated() {
        assert_eq!(current_delete_initiator(), DeleteInitiator::Machine);
    }

    #[tokio::test]
    async fn a_scoped_delete_reports_the_installed_initiator() {
        with_delete_initiator(DeleteInitiator::Person, async {
            assert_eq!(current_delete_initiator(), DeleteInitiator::Person);
        })
        .await;
    }

    #[tokio::test]
    async fn the_scope_ends_with_the_future() {
        with_delete_initiator(DeleteInitiator::Person, async {}).await;
        assert_eq!(current_delete_initiator(), DeleteInitiator::Machine);
    }

    #[tokio::test]
    async fn a_spawned_task_does_not_inherit_a_person_scope() {
        // Task-locals do not cross `spawn`, so work handed to another task
        // loses the person's authority rather than carrying it silently.
        let inner = with_delete_initiator(DeleteInitiator::Person, async {
            tokio::spawn(async { current_delete_initiator() })
                .await
                .expect("join")
        })
        .await;
        assert_eq!(inner, DeleteInitiator::Machine);
    }

    #[test]
    fn the_spellings_are_stable() {
        assert_eq!(DeleteInitiator::Person.as_str(), "person");
        assert_eq!(DeleteInitiator::Machine.as_str(), "machine");
        assert!(DeleteInitiator::Person.is_person());
        assert!(!DeleteInitiator::Machine.is_person());
    }
}
