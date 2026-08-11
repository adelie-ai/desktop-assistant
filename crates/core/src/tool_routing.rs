//! The turn's tool table: one key, one host, one route (#1216).
//!
//! A tool's identity is the pair `(capability name, host)`. The name says what
//! the tool does. The host says which machine runs it. The name never carries
//! the host, so a host that is renamed, added or removed changes no capability
//! name, and what a turn learned about a tool on one machine still applies to
//! the same tool on another.
//!
//! ## Why this module exists
//!
//! The advertised tool set and the dispatch route used to be two lookups over
//! the same `name` string against two different tables, with opposite
//! precedence: the merge that built the advertised set preferred the daemon's
//! definition, and dispatch preferred the client's executor. On a name both
//! sides offered, the model read the daemon's schema and the call ran on the
//! client. Nothing arbitrated between them, so nothing could notice.
//!
//! [`ToolRouter`] is that arbiter, and it is the only one. Both questions -
//! "which definition does the model see" and "which host runs the call" - are
//! answered by one private `preferred` over the same table, so an answer that
//! differed between them cannot be built.
//!
//! ## The three orders over one name
//!
//! Three things claim a name in a turn, and they are ranked here once rather
//! than at each call site:
//!
//! 1. [`ToolSource::CoreLoop`] - the loop's own control surface (`begin_step`,
//!    `complete_step`, `promote_plan_to_skill`). The loop intercepts these by
//!    name before any executor, so a hosted tool of the same name could never
//!    have run. Ranking it here means the model is not shown a schema for a
//!    tool that cannot run either.
//! 2. A daemon-hosted capability.
//! 3. A device-hosted capability (one the connected client registered).
//!
//! Two and three are ranked by [`RoutingPolicy`], not by which was inserted
//! first.
//!
//! ## How the model names a host
//!
//! When one capability name has more than one host, the advertised definition
//! grows a [`HOST_ARGUMENT`] property whose values are the host tokens
//! ([`crate::domain::ToolLocality::host_token`]) - the same vocabulary a
//! tool-search hit reports in `runs_on`. The model states one when the task is
//! about a machine ("read that file on the laptop") and omits it otherwise, and
//! the harness picks by policy. The argument is routing metadata, so the loop
//! removes it before the tool runs: it is not part of any tool's own schema.

use crate::domain::{ToolDefinition, ToolLocality};

/// The argument that carries the host of a call, on a capability that more
/// than one host offers.
///
/// Reserved rather than plain `host` because a tool may legitimately take a
/// `host` argument of its own (an HTTP client, a database tool), and routing
/// metadata that collided with a real parameter would be read as one.
pub const HOST_ARGUMENT: &str = "__host";

/// How the harness picks a host when the model names none.
///
/// One variant today, named rather than implied: the rule is a decision the
/// product makes, and a reader who wants to know what it is should find it
/// written down instead of deducing it from the order of two pushes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    /// Prefer the host co-located with the turn, which is the daemon's own
    /// host: the daemon is the process that runs the loop, so a daemon-hosted
    /// call stays inside it, while a device-hosted call leaves the process,
    /// crosses a socket, and depends on a client that may disconnect
    /// mid-turn.
    ///
    /// This holds when the client is on the same machine as the daemon, where
    /// the two hosts are one machine and the choice is between two routes to
    /// it rather than between two machines. It also holds when the client is
    /// remote: the model is then told both hosts exist, and states the one it
    /// means.
    PreferCoLocated,
}

impl RoutingPolicy {
    /// The policy's name, for the log line that reports how a collision was
    /// resolved.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreferCoLocated => "prefer-co-located",
        }
    }
}

/// What claims a name in the turn's tool table.
///
/// Not a third answer to "where does this run" - that is the entry's
/// [`ToolLocality`]. This says whether the turn loop runs the call itself or
/// hands it to a host's executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSource {
    /// The turn loop's own control surface. Intercepted by the loop; no
    /// executor sees it.
    CoreLoop,
    /// A capability a host offers, run by that host's executor.
    Hosted,
}

/// One entry in the turn's tool table: a definition, the host that runs it,
/// and what claims the name.
#[derive(Debug, Clone)]
pub struct RoutedTool {
    definition: ToolDefinition,
    host: ToolLocality,
    source: ToolSource,
}

impl RoutedTool {
    /// The capability name, which is the table's key together with
    /// [`RoutedTool::host`].
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// The definition the model is shown for this (name, host) pair.
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// The machine that runs this call.
    pub fn host(&self) -> &ToolLocality {
        &self.host
    }

    /// Whether the turn loop runs this call itself rather than handing it to
    /// an executor.
    pub fn is_core_loop(&self) -> bool {
        self.source == ToolSource::CoreLoop
    }
}

/// What the table says about a name the model called.
#[derive(Debug)]
pub enum Route<'a> {
    /// The name resolves to exactly one entry: its definition was the one
    /// advertised for the stated (or policy-chosen) host, and its host runs
    /// the call.
    Found(&'a RoutedTool),
    /// The model named a host that does not offer this capability.
    UnknownHost {
        /// What the model asked for.
        asked: String,
        /// The host tokens this capability does offer.
        available: Vec<&'static str>,
    },
    /// The turn's table holds no entry for this name. The caller decides what
    /// that means; the turn loop hands it to the daemon executor, whose
    /// routing table outlives the turn and holds tools this turn never
    /// advertised.
    Unrouted,
}

/// The turn's tool table.
///
/// Built once per round from the sets that offer tools, then asked twice: once
/// for the definitions the model is shown, and once per tool call for the host
/// that runs it.
#[derive(Debug)]
pub struct ToolRouter {
    policy: RoutingPolicy,
    daemon_host: String,
    device_label: String,
    entries: Vec<RoutedTool>,
}

impl ToolRouter {
    /// A table with no entries, holding the labels its hosts are named by.
    ///
    /// `daemon_host` is the daemon's self-identity label and `device_label` is
    /// the connected client's, empty when the client reported none. Both are
    /// names, not keys: the addressing token is [`ToolLocality::host_token`],
    /// so renaming either changes nothing a caller must know.
    pub fn new(
        policy: RoutingPolicy,
        daemon_host: impl Into<String>,
        device_label: impl Into<String>,
    ) -> Self {
        Self {
            policy,
            daemon_host: daemon_host.into(),
            device_label: device_label.into(),
            entries: Vec::new(),
        }
    }

    /// The policy this table resolves an unstated host by.
    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    /// Offer capabilities hosted on the daemon's own machine: built-ins, the
    /// MCP fleet the daemon spawned, and anything a tool search activated.
    pub fn offer_daemon_tools(&mut self, defs: &[ToolDefinition]) {
        for def in defs {
            let host = ToolLocality::server(&self.daemon_host);
            self.insert(def.clone(), host, ToolSource::Hosted);
        }
    }

    /// Offer capabilities hosted on the connected client's machine.
    pub fn offer_device_tools(&mut self, defs: &[ToolDefinition]) {
        for def in defs {
            let host = ToolLocality::client(&self.device_label, &self.device_label);
            self.insert(def.clone(), host, ToolSource::Hosted);
        }
    }

    /// Offer one of the turn loop's own control tools.
    pub fn offer_core_loop_tool(&mut self, def: ToolDefinition) {
        let host = ToolLocality::server(&self.daemon_host);
        self.insert(def, host, ToolSource::CoreLoop);
    }

    /// Drop every entry whose name `keep` rejects.
    pub fn retain<F: Fn(&str) -> bool>(&mut self, keep: F) {
        self.entries.retain(|e| keep(e.name()));
    }

    /// The definitions the model is shown: one per capability name, in the
    /// order the names were first offered, each from the host this table would
    /// route the name to.
    pub fn advertised_definitions(&self) -> Vec<ToolDefinition> {
        unimplemented!("#1216: the advertised set is chosen by policy")
    }

    /// The entry that answers for `name`, with the host the model stated (as
    /// it wrote it) or `None` when it stated none.
    pub fn resolve(&self, name: &str, host: Option<&str>) -> Route<'_> {
        // Placeholder: today's dispatch, which prefers the device entry
        // whatever the model was shown. Replaced by a resolution that shares
        // `preferred` with the advertised set.
        let _ = host;
        let mut found: Option<&RoutedTool> = None;
        for entry in self.entries.iter().filter(|e| e.name() == name) {
            if found.is_none() || entry.host().is_client() {
                found = Some(entry);
            }
        }
        found.map_or(Route::Unrouted, Route::Found)
    }

    /// Add an entry, keeping the first definition offered for a
    /// (name, host) pair.
    fn insert(&mut self, definition: ToolDefinition, host: ToolLocality, source: ToolSource) {
        let duplicate = self
            .entries
            .iter()
            .any(|e| e.name() == definition.name && e.host.host_token() == host.host_token());
        if !duplicate {
            self.entries.push(RoutedTool {
                definition,
                host,
                source,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str, description: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            description,
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )
    }

    /// A table where the same capability is offered by both hosts, with the
    /// device entry inserted **first** so lookup order and policy disagree.
    fn collided_table() -> ToolRouter {
        let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
        router.offer_device_tools(&[def("read_file", "DEVICE read_file")]);
        router.offer_daemon_tools(&[def("read_file", "DAEMON read_file")]);
        router
    }

    /// #1216 AC4: with no host stated, the host is chosen by the written
    /// policy. The device entry is offered first here, so a resolution that
    /// followed insertion order would answer the device.
    #[test]
    fn unstated_host_routes_by_the_prefer_co_located_policy_not_lookup_order() {
        let router = collided_table();
        assert_eq!(
            router.policy().as_str(),
            "prefer-co-located",
            "the policy must be named in the code, not implied by lookup order"
        );
        let Route::Found(routed) = router.resolve("read_file", None) else {
            panic!("a collided capability must resolve");
        };
        assert_eq!(
            routed.host().host_token(),
            "daemon",
            "the policy prefers the co-located host, whichever entry was offered first"
        );
    }

    /// #1216 AC4, the same rule from the other side: reversing the order the
    /// two hosts are offered in changes nothing.
    #[test]
    fn offering_the_hosts_in_either_order_resolves_to_the_same_host() {
        let daemon_first = {
            let mut r = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
            r.offer_daemon_tools(&[def("read_file", "DAEMON read_file")]);
            r.offer_device_tools(&[def("read_file", "DEVICE read_file")]);
            r
        };
        let device_first = collided_table();
        let token_of = |r: &ToolRouter| match r.resolve("read_file", None) {
            Route::Found(routed) => routed.host().host_token(),
            other => panic!("a collided capability must resolve, got {other:?}"),
        };
        assert_eq!(token_of(&daemon_first), token_of(&device_first));
    }

    /// #1216 AC2: whatever picks the schema picks the runner. Every advertised
    /// definition must be the one the route resolves to.
    #[test]
    fn every_advertised_definition_comes_from_the_host_the_route_resolves_to() {
        let router = collided_table();
        for advertised in router.advertised_definitions() {
            let Route::Found(routed) = router.resolve(&advertised.name, None) else {
                panic!("an advertised name must resolve: {}", advertised.name);
            };
            assert_eq!(
                routed.definition().description,
                advertised.description,
                "the schema shown for {} came from a host the call would not reach",
                advertised.name
            );
        }
    }

    /// #1216 AC1: a capability two hosts offer is addressable as two things -
    /// the advertised definition carries the host argument, and its values are
    /// the tokens of the hosts that actually offer it.
    #[test]
    fn a_capability_on_two_hosts_advertises_the_host_argument_with_both_tokens() {
        let router = collided_table();
        let advertised = router.advertised_definitions();
        let read_file = advertised
            .iter()
            .find(|d| d.name == "read_file")
            .expect("the collided capability is advertised");
        let values = read_file.parameters["properties"][HOST_ARGUMENT]["enum"]
            .as_array()
            .expect("the host argument offers the hosts as an enum");
        let mut tokens: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
        tokens.sort_unstable();
        assert_eq!(tokens, vec!["daemon", "device"]);
    }

    /// #1216 AC1: a capability only one host offers gains no host argument -
    /// there is no choice to state, and a field the model cannot use costs
    /// context on every round.
    #[test]
    fn a_capability_on_one_host_advertises_no_host_argument() {
        let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
        router.offer_daemon_tools(&[def("web_read", "read a page")]);
        let advertised = router.advertised_definitions();
        let web_read = advertised
            .iter()
            .find(|d| d.name == "web_read")
            .expect("advertised");
        assert!(
            web_read.parameters["properties"]
                .get(HOST_ARGUMENT)
                .is_none(),
            "a single-host capability must not advertise a host to choose"
        );
    }

    /// #1216 AC1: naming a host reaches that host's definition, on both sides.
    #[test]
    fn naming_a_host_resolves_to_that_hosts_entry() {
        let router = collided_table();
        let device = match router.resolve("read_file", Some("device")) {
            Route::Found(routed) => routed.definition().description.clone(),
            other => panic!("device must be reachable by name, got {other:?}"),
        };
        let daemon = match router.resolve("read_file", Some("daemon")) {
            Route::Found(routed) => routed.definition().description.clone(),
            other => panic!("daemon must be reachable by name, got {other:?}"),
        };
        assert_eq!(device, "DEVICE read_file");
        assert_eq!(daemon, "DAEMON read_file");
    }

    /// A host token the capability does not offer is refused, and the refusal
    /// names the ones it does - the model can correct itself from the answer.
    #[test]
    fn a_host_the_capability_does_not_offer_is_refused_and_names_the_ones_it_does() {
        let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
        router.offer_daemon_tools(&[def("web_read", "read a page")]);
        match router.resolve("web_read", Some("device")) {
            Route::UnknownHost { asked, available } => {
                assert_eq!(asked, "device");
                assert_eq!(available, vec!["daemon"]);
            }
            other => panic!("an unavailable host must be refused, got {other:?}"),
        }
    }

    /// `remote-service` is a daemon-issued call: it says what the tool reaches,
    /// not where the call is made. A model that states the `runs_on` value it
    /// read in a search hit must reach the daemon entry.
    #[test]
    fn remote_service_names_the_daemon_host() {
        let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
        router.offer_daemon_tools(&[def("calendar_list", "a hosted calendar")]);
        assert!(
            matches!(
                router.resolve("calendar_list", Some("remote-service")),
                Route::Found(_)
            ),
            "remote-service is issued from the daemon"
        );
    }

    /// The third order over one name: the loop's own control surface wins, so
    /// the model is never shown a hosted schema under a name the loop
    /// intercepts before any executor.
    #[test]
    fn a_core_loop_name_wins_over_a_hosted_tool_of_the_same_name() {
        let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, "daemon-host", "laptop");
        router.offer_daemon_tools(&[def("begin_step", "an MCP server's own begin_step")]);
        router.offer_core_loop_tool(def("begin_step", "the loop's step control"));
        let Route::Found(routed) = router.resolve("begin_step", None) else {
            panic!("begin_step must resolve");
        };
        assert!(routed.is_core_loop(), "the loop's control surface wins");
        let advertised = router.advertised_definitions();
        let entries: Vec<&ToolDefinition> = advertised
            .iter()
            .filter(|d| d.name == "begin_step")
            .collect();
        assert_eq!(entries.len(), 1, "one entry per name, never two");
        assert_eq!(entries[0].description, "the loop's step control");
    }

    /// #1216 AC5: a host's name is a label, never part of the key. Renaming
    /// both hosts changes no capability name and no host token, so a burn, a
    /// skill or a use-log row that names a tool still names it.
    #[test]
    fn renaming_a_host_leaves_every_capability_name_unchanged() {
        let build = |daemon: &str, device: &str| {
            let mut router = ToolRouter::new(RoutingPolicy::PreferCoLocated, daemon, device);
            router.offer_daemon_tools(&[def("read_file", "DAEMON read_file")]);
            router.offer_device_tools(&[
                def("read_file", "DEVICE read_file"),
                def("take_screenshot", "DEVICE screenshot"),
            ]);
            router
        };
        let before = build("daemon-host", "laptop");
        let after = build("adele-1", "workshop-pc");

        let names = |r: &ToolRouter| -> Vec<String> {
            r.advertised_definitions()
                .into_iter()
                .map(|d| d.name)
                .collect()
        };
        assert_eq!(names(&before), names(&after));

        let tokens = |r: &ToolRouter| -> Vec<String> {
            r.advertised_definitions()
                .into_iter()
                .map(|d| d.parameters["properties"][HOST_ARGUMENT]["enum"].to_string())
                .collect()
        };
        assert_eq!(
            tokens(&before),
            tokens(&after),
            "the token that addresses a host must not change when its label does"
        );
    }

    /// A name the turn never offered is not routed here. The turn loop hands
    /// it to the daemon executor, whose table outlives the turn.
    #[test]
    fn a_name_the_turn_never_offered_is_unrouted() {
        let router = collided_table();
        assert!(matches!(router.resolve("nothing", None), Route::Unrouted));
    }
}
