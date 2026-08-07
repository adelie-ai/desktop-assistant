//! What the model reads this round, where that differs from what is stored.
//!
//! Context management keeps a prompt under the model's input-token budget by
//! replacing the content of individual messages: a large tool result becomes a
//! short notice, and a completed step's results become pointers to the note
//! that distilled them. The replacement belongs to the round. The record it
//! replaces belongs to the conversation.
//!
//! `Conversation::messages` is that record - the durable transcript, the
//! observation layer the knowledge base and the scratchpad both cite, and what
//! a user reads back when they ask what a tool returned. The turn writes the
//! list to storage when it ends, and the store reconciles by ordinal slot, so
//! a replacement written into the list overwrites the stored row and the
//! original is gone. A projection holds the two apart: the model reads the
//! replacement, storage keeps the original.
//!
//! One projection lives for one turn. Entries are keyed by message id, so
//! appending to the transcript mid-turn never invalidates one.
//!
//! A turn starts by seeding its own projection from the DECISIONS earlier turns
//! recorded on the rows - which notes a result was distilled into
//! (`Message::distilled_into`). That is how the saving outlives the turn that
//! made it without anything writing a replacement to storage: the pointer is
//! rebuilt from the keys, so the row keeps the output and the model still reads
//! the pointer. A decision whose notes are gone rebuilds nothing, and the turn
//! reads the stored output. See the `crates/core/src/planning.rs` module
//! header.

use std::collections::HashMap;

use crate::domain::Message;

/// The round's replacement content, by message id. Empty means the model reads
/// the stored transcript unchanged.
#[derive(Debug, Default, Clone)]
pub(crate) struct ContextProjection {
    replaced: HashMap<String, String>,
}

impl ContextProjection {
    /// The content the model reads for `msg`: the replacement when one is
    /// recorded, the stored content otherwise.
    pub(crate) fn content<'a>(&'a self, msg: &'a Message) -> &'a str {
        self.replaced
            .get(&msg.id)
            .map_or(msg.content.as_str(), String::as_str)
    }

    /// Whether the round already reads something else for `msg`.
    pub(crate) fn is_replaced(&self, msg: &Message) -> bool {
        self.replaced.contains_key(&msg.id)
    }

    /// Read `msg` as `content` for the rest of the turn.
    pub(crate) fn replace(&mut self, msg: &Message, content: String) {
        self.replaced.insert(msg.id.clone(), content);
    }

    /// How many messages the round reads differently.
    pub(crate) fn replaced_count(&self) -> usize {
        self.replaced.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    #[test]
    fn an_empty_projection_reads_the_stored_content() {
        let projection = ContextProjection::default();
        let msg = Message::new(Role::Tool, "the raw output");
        assert_eq!(projection.content(&msg), "the raw output");
        assert_eq!(projection.replaced_count(), 0);
    }

    #[test]
    fn a_replacement_leaves_the_stored_message_untouched() {
        let mut projection = ContextProjection::default();
        let msg = Message::new(Role::Tool, "the raw output");
        projection.replace(&msg, "a short notice".to_string());

        assert_eq!(projection.content(&msg), "a short notice");
        assert_eq!(
            msg.content, "the raw output",
            "the projection must not write to the message it projects"
        );
        assert!(projection.is_replaced(&msg));
        assert_eq!(projection.replaced_count(), 1);
    }

    #[test]
    fn a_projection_is_keyed_by_message_and_not_by_content() {
        // Two messages with identical content compare equal (`Message` excludes
        // the id from `PartialEq`), so a content-keyed projection would replace
        // both. Only the one that was projected may change.
        let mut projection = ContextProjection::default();
        let first = Message::new(Role::Tool, "same text");
        let second = Message::new(Role::Tool, "same text");
        projection.replace(&first, "notice".to_string());

        assert_eq!(projection.content(&first), "notice");
        assert_eq!(projection.content(&second), "same text");
    }
}
