//! What the round's tool block carries, and what stops it growing (#1212).
//!
//! ## The measurement this exists to answer
//!
//! One production turn carried 99 tool schemas - about 23.7k estimated tokens,
//! 17.9% of the budget - in front of a 254-character prompt, before the turn
//! had done anything. It then grew: a tool search activated ten more, nothing
//! ever retired one, and the round budget is 200. Nobody decided on 99. It is
//! what accumulated while every new capability added its tools to a set that
//! only ever admitted.
//!
//! ## An index, not a payload
//!
//! A tool schema costs roughly 250 estimated tokens. A tool *name* costs about
//! ten. So the block carries the schemas the turn needs and the names of
//! everything else, and a name is enough: the model can call it, and the round's
//! table ([`crate::tool_routing::ToolRouter`]) routes it whether or not its
//! schema was in the block. This is the law the knowledge base already follows,
//! applied to the tool surface.
//!
//! ## The core set, and the rule that decides membership
//!
//! **A tool is core when the model needs it to find, or to keep, what the rest
//! of the turn depends on. Everything else is discovered.**
//!
//! That rule admits exactly two groups, and the next capability added has an
//! obvious answer rather than a default of "advertise it":
//!
//! - **The daemon's own built-ins.** Discovery itself (`builtin_tool_search`),
//!   the assistant's memory (`builtin_knowledge_base_*`), its working notes
//!   (`builtin_scratchpad_*`), its skills (`builtin_skill_*`) and the handful of
//!   daemon faculties with no index behind them. Deferring these would mean
//!   searching for the search tool. They are bounded by
//!   [`CORE_TOOL_CEILING`], which is what stops this set from becoming the next
//!   99.
//! - **The turn loop's own control surface** (`begin_step`, `complete_step`,
//!   `promote_plan_to_skill`). No search indexes them, because no executor owns
//!   them - the loop runs them itself. Core by necessity, not by choice.
//!
//! Everything reached through a connection - the daemon's MCP fleet, and the
//! tools the connected client registers - is discovered. The fleet already was.
//! The client's tools were not, and that is where the 99 came from: #538 moved
//! a fleet of servers into the client process, from the daemon side where
//! deferral applies to the client side where it did not.
//!
//! ## The client's own slice, and why it is a cap rather than a list
//!
//! A connected client registers whatever it happens to host, so the daemon
//! cannot know which of its tools a session actually needs. What it can do is
//! bound the cost: [`MAX_CLIENT_TOOLS_IN_BLOCK`] of them keep their schemas and
//! the rest are named. The slice is taken in the order the connection
//! registered them, which is the only priority signal that crosses the wire
//! today - a client that wants a tool schema'd registers it early.
//!
//! **Deferral is conditional on discovery actually being offered.** A turn with
//! no `builtin_tool_search` in its block advertises every client tool in full,
//! because a name nothing can look up is a name the model cannot use.
//!
//! ## The bound and the lifetime
//!
//! [`ActivationLedger`] holds what a turn's tool searches activated.
//!
//! - **The lifetime is the turn.** A new turn builds a new ledger, so nothing
//!   carries over. This is where the epic's eviction argument lands for tools:
//!   the turn is the scope, exactly as it is for the provenance gate.
//! - **Under the bound it only ever appends.** Round N's tool block stays a
//!   byte-identical prefix of round N+1's, which is what lets a provider's
//!   prompt cache serve the block instead of reprocessing it. A set that
//!   reordered between rounds would throw that away for nothing.
//! - **At the bound, the longest-unused activation is retired.** Refusing
//!   instead would strand a turn that needs a capability it has not used yet,
//!   and the bound is what makes the growth finite. Retirement moves the block,
//!   so it is deliberately the exception rather than a per-round sweep.
//! - **An activation used in the current round is never retired**, so a tool the
//!   model is working with cannot be taken away mid-round. When every entry was
//!   used this round the activation is refused and the turn keeps its set.

use crate::domain::ToolDefinition;
use crate::tool_routing::ToolConnection;

/// The daemon's own discovery tool, by the name the model calls it under.
///
/// Named here rather than at each use because it is the condition deferral
/// depends on: a name the model cannot look up is a name it cannot use, so a
/// round that does not offer this tool advertises everything in full.
pub const DISCOVERY_TOOL: &str = "builtin_tool_search";

/// The most tools the always-advertised built-in core may hold.
///
/// Not a configuration value: it is the assertion that keeps the core a
/// decision. The set stood at 22 when this was written, so the ceiling leaves
/// room to add a faculty and none to drift back to 99. A change that pushes
/// past it fails a test by name, and the answer is either the membership rule
/// in this module's header or a deliberate, recorded move of the ceiling.
pub const CORE_TOOL_CEILING: usize = 32;

/// The most tools one connected client may put in the round's block, with the
/// rest named and reachable by name.
///
/// Eight schemas cost roughly 2k estimated tokens against the roughly 19k the
/// measured connection's 77 tools cost, and eight covers the handful a session
/// uses without looking anything up - the voice client's `say_this`, a
/// screenshot, a notification.
pub const MAX_CLIENT_TOOLS_IN_BLOCK: usize = 8;

/// The most tools one turn may hold activated at once.
///
/// A tool search returns at most ten rows from the registry and ten from the
/// device, so this is two searches' worth: enough that an ordinary turn never
/// meets the bound, and roughly 6k estimated tokens if one does.
pub const MAX_ACTIVATED_TOOLS: usize = 24;

/// What one attempt to activate a tool did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activated {
    /// It is now in the block. `retired` names the provider name of the
    /// activation the bound pushed out, where one had to go.
    Admitted {
        /// The activation retired to make room, by provider name.
        retired: Option<String>,
    },
    /// The ledger already held it; its last use was refreshed and nothing else
    /// changed.
    AlreadyHeld,
    /// The ledger is full and every activation in it was used this round, so
    /// nothing could be retired without taking a tool away mid-round.
    Refused,
}

/// One activated tool, and when the model last called it.
#[derive(Debug, Clone)]
struct Activation {
    connection: ToolConnection,
    def: ToolDefinition,
    /// The round this tool was last called in, or the round it was activated
    /// in when the model has not called it yet. A just-activated tool is
    /// therefore never the one retired.
    last_used_round: usize,
}

/// The tools one turn's searches activated: bounded, append-only under the
/// bound, and emptied by the turn ending.
///
/// See the module header for why each of those three holds.
#[derive(Debug)]
pub struct ActivationLedger {
    entries: Vec<Activation>,
    bound: usize,
}

impl Default for ActivationLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationLedger {
    /// An empty ledger bounded by [`MAX_ACTIVATED_TOOLS`].
    pub fn new() -> Self {
        Self::with_bound(MAX_ACTIVATED_TOOLS)
    }

    /// An empty ledger with an explicit bound, so a test can reach the bound
    /// without activating two dozen tools.
    pub fn with_bound(bound: usize) -> Self {
        Self {
            entries: Vec::new(),
            bound,
        }
    }

    /// The bound this ledger enforces.
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// How many activations it holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the tool this provider name names is already activated.
    pub fn holds(&self, provider_name: &str) -> bool {
        self.entries.iter().any(|e| e.def.name == provider_name)
    }

    /// Put `def` in the block for the rest of the turn, retiring the
    /// longest-unused activation if the bound demands it.
    pub fn activate(
        &mut self,
        connection: ToolConnection,
        def: ToolDefinition,
        round: usize,
    ) -> Activated {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.def.name == def.name) {
            // One tool is one entry, on whichever surface it arrived by. The
            // refresh is what keeps the retirement order honest: a tool the
            // turn keeps reaching for must not read as unused.
            entry.last_used_round = round;
            return Activated::AlreadyHeld;
        }

        let mut retired = None;
        if self.entries.len() >= self.bound {
            // The longest unused, and the earliest of those when several tie,
            // so the choice is deterministic rather than whatever the scan
            // happened to meet first.
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used_round, *index))
                .map(|(index, entry)| (index, entry.last_used_round));
            match victim {
                // Never one this round is working with: a tool taken away
                // mid-round is a schema the model just read and can no longer
                // call.
                Some((index, last_used)) if last_used < round => {
                    retired = Some(self.entries.remove(index).def.name);
                }
                _ => return Activated::Refused,
            }
        }

        self.entries.push(Activation {
            connection,
            def,
            last_used_round: round,
        });
        Activated::Admitted { retired }
    }

    /// Note that the model called this tool in `round`, so the ledger retires
    /// it last.
    pub fn mark_used(&mut self, provider_name: &str, round: usize) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.def.name == provider_name)
        {
            entry.last_used_round = round;
        }
    }

    /// Every activation, in the order the block offers them: the connection
    /// that runs it and the definition the model reads.
    pub fn offers(&self) -> impl Iterator<Item = (&ToolConnection, &ToolDefinition)> {
        self.entries.iter().map(|e| (&e.connection, &e.def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(name, "a tool", serde_json::json!({"type": "object"}))
    }

    fn registry() -> ToolConnection {
        ToolConnection::daemon_registry()
    }

    fn names(ledger: &ActivationLedger) -> Vec<String> {
        ledger.offers().map(|(_, d)| d.name.clone()).collect()
    }

    #[test]
    fn under_the_bound_an_activation_only_appends_and_retires_nothing() {
        // The property a provider's prompt cache depends on: round N's tool
        // block is a byte-identical prefix of round N+1's, so the cached
        // prefix survives instead of being rebuilt.
        let mut ledger = ActivationLedger::with_bound(4);
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            assert_eq!(
                ledger.activate(registry(), tool(name), i),
                Activated::Admitted { retired: None },
                "under the bound nothing is retired"
            );
        }
        assert_eq!(names(&ledger), vec!["a", "b", "c"]);
    }

    #[test]
    fn an_activation_past_the_bound_retires_the_longest_unused_tool() {
        let mut ledger = ActivationLedger::with_bound(3);
        ledger.activate(registry(), tool("a"), 0);
        ledger.activate(registry(), tool("b"), 0);
        ledger.activate(registry(), tool("c"), 0);
        // Every entry is used, at a different round, and the one used longest
        // ago is deliberately not the one added first - otherwise "longest
        // unused" and "oldest" would be the same answer and this would not
        // tell them apart.
        ledger.mark_used("a", 3);
        ledger.mark_used("b", 1);
        ledger.mark_used("c", 2);

        assert_eq!(
            ledger.activate(registry(), tool("d"), 4),
            Activated::Admitted {
                retired: Some("b".to_string())
            },
            "the bound retires the tool unused longest, not the one added first"
        );
        assert_eq!(names(&ledger), vec!["a", "c", "d"]);
        assert_eq!(ledger.len(), 3, "the bound is a ceiling, not a suggestion");
    }

    #[test]
    fn an_activation_the_current_round_used_is_never_retired() {
        // A tool the model is working with in this very round must not vanish
        // from under it, so a full ledger whose every entry is in use refuses
        // the new activation rather than taking one away.
        let mut ledger = ActivationLedger::with_bound(2);
        ledger.activate(registry(), tool("a"), 0);
        ledger.activate(registry(), tool("b"), 0);
        ledger.mark_used("a", 5);
        ledger.mark_used("b", 5);

        assert_eq!(
            ledger.activate(registry(), tool("c"), 5),
            Activated::Refused,
            "nothing may be retired while every activation is in use this round"
        );
        assert_eq!(names(&ledger), vec!["a", "b"]);
    }

    #[test]
    fn activating_a_tool_already_held_refreshes_its_use_rather_than_duplicating_it() {
        let mut ledger = ActivationLedger::with_bound(2);
        ledger.activate(registry(), tool("a"), 0);
        ledger.activate(registry(), tool("b"), 0);

        assert_eq!(
            ledger.activate(registry(), tool("a"), 4),
            Activated::AlreadyHeld,
            "one tool is one entry"
        );
        assert_eq!(names(&ledger), vec!["a", "b"], "and it keeps its place");

        // The refresh is what makes the retirement order right: `a` was just
        // re-activated, so `b` is now the longest unused.
        assert_eq!(
            ledger.activate(registry(), tool("c"), 5),
            Activated::Admitted {
                retired: Some("b".to_string())
            },
        );
        assert_eq!(names(&ledger), vec!["a", "c"]);
    }
}
