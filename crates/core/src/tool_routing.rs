//! Composed tool names, and the one table that resolves them (#1216).
//!
//! Every name the model is offered is unique across the whole set, namespace
//! included. Names are a human surface - they are read, typed into a
//! configuration, quoted in a skill and stored in a lesson - so they are made
//! unique **by construction** rather than disambiguated at dispatch. Because
//! they are unique there is nothing to resolve between: a composed name
//! identifies exactly one tool, and there is no override behaviour anywhere.
//!
//! ## The connection is the unit of locality
//!
//! Every tool belongs to exactly one connection - a client device, an MCP
//! server, or the daemon's own built-ins - and is addressed as
//! (connection, tool name). A coarse daemon/device axis cannot express "invoke
//! this on *that* connection", and one production turn reported six devices, so
//! "the device" identifies nothing. The composed name derives from the pair,
//! which is why uniqueness is a consequence of connections being distinct
//! rather than a rule enforced on top of them.
//!
//! ## The prefix is temporary, and stays removable
//!
//! It is a uniqueness device for the daemon's own bookkeeping that happens to
//! surface in the advertised name. It is not how the model is told about
//! topology, and it is not the answer to locality - a structural field beside
//! the tool is, and this buys time for it.
//!
//! **So nothing derives locality by parsing a name.** Location is read from the
//! entry's connection, here, today, and from that field tomorrow. If any layer
//! started reading the prefix, removing it would stop being a rename and become
//! a behaviour change, and the stopgap would be permanent by accident. The one
//! thing that may look at the prefix is [`strip_location`], which removes it.
//!
//! ## How a name composes
//!
//! The location is the root namespace, and a provider's namespace nests
//! beneath it:
//!
//! ```text
//! daemon built-in              daemon_<tool>
//! daemon MCP server "fileio"   daemon_fileio__<tool>
//! client built-in              client_<tool>
//! client MCP server "fileio"   client_fileio__<tool>
//! ```
//!
//! The provider names itself only *within* its root. [`compose_name`] applies
//! the root, so a client connection configured as `daemon` composes to
//! `client_daemon__<tool>` and a daemon server configured as `client` composes
//! to `daemon_client__<tool>`.
//!
//! **This is a security property, not a formatting rule.** A name that escaped
//! its root would be presented to the model as hosted somewhere it is not, and
//! the model chooses what to trust a tool with partly from where it runs. The
//! rule is symmetric and has no trusted side: a daemon server's namespace comes
//! from the daemon's own configuration and *feels* trustworthy, and treating it
//! as trustworthy is exactly how this would rot. Same code path, same
//! sanitising, both roots.
//!
//! ## Why this module exists
//!
//! The advertised tool set and the dispatch route used to be two lookups over
//! the same bare `name` against two different tables, with opposite precedence:
//! the merge that built the advertised set preferred the daemon's definition,
//! and dispatch preferred the client's executor. On a name both sides offered,
//! the model read the daemon's schema and the call ran on the client.
//!
//! Unique names remove the contest, but single resolution is the property that
//! must hold either way, so [`ToolRouter`] answers both questions - which
//! definition the model sees, and which host runs the call - from one table.
//!
//! ## What the rest of the system sees
//!
//! The composed name is what the **model** reads and writes, and nothing else.
//! Execution, the negative-memory digest (#1126), the provenance gate and the
//! caller's allowlist all use the provider's own name, which [`RoutedTool`]
//! carries beside the composed one. A prefixed name in a learning key would
//! fragment what the assistant learns per host - a tool that burned the user on
//! the daemon would teach nothing about the same tool on their laptop, and
//! nobody would see it happen.

use crate::domain::{ToolDefinition, ToolNamespace};

/// Longest provider segment a composed name will carry. A provider names
/// itself, and the name is rendered into every round's tool block, so the
/// length is bounded here rather than trusted.
const MAX_SEGMENT_CHARS: usize = 120;

/// The connection a tool belongs to: a client device, an MCP server, or the
/// daemon's own built-ins.
///
/// This is the unit of locality, and half of a tool's address. A coarse
/// daemon/device axis cannot express "invoke this on *that* connection" - one
/// production turn reported six devices, so "the device" identifies nothing -
/// so the daemon addresses a tool as (connection, tool name) and the table is
/// keyed on the pair.
///
/// `namespace` is the connection's own configured name, chosen by a person and
/// already unique within a host's configuration. The daemon's built-ins and a
/// client's own built-ins have none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolConnection {
    location: ToolLocation,
    namespace: Option<String>,
}

impl ToolConnection {
    /// The daemon's own built-ins, which are one connection with no namespace.
    pub fn daemon_builtins() -> Self {
        Self {
            location: ToolLocation::Daemon,
            namespace: None,
        }
    }

    /// One MCP server the daemon reaches, named as its configuration names it.
    pub fn daemon_server(namespace: impl Into<String>) -> Self {
        Self {
            location: ToolLocation::Daemon,
            namespace: Some(namespace.into()),
        }
    }

    /// The daemon's MCP registry, for a tool the turn cannot attribute to a
    /// named server - it activated it by name and the server list is not in
    /// view. It is a distinct connection so that such a tool colliding with a
    /// built-in is still reported as the fault it is, and it never reaches a
    /// name: only the location root does.
    pub fn daemon_registry() -> Self {
        Self {
            location: ToolLocation::Daemon,
            namespace: Some("registry".to_string()),
        }
    }

    /// The connected client device, and whatever it hosts in its own process.
    pub fn client_device() -> Self {
        Self {
            location: ToolLocation::Client,
            namespace: None,
        }
    }

    /// Where this connection's tools run. Read from the table, never parsed
    /// from a name.
    pub fn location(&self) -> ToolLocation {
        self.location
    }

    /// The connection's configured name, where it has one.
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// How the connection is named in a log line, a duplicate report, and the
    /// `server` axis of the per-connection schema cost (#1212).
    ///
    /// Bounded by the operator's own configuration - the daemon's built-ins,
    /// the client's, and one value per configured MCP server - which is what
    /// makes it safe as a metric label where a conversation id would not be.
    pub fn label(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{}:{ns}", self.location.as_str()),
            None => format!("{}:built-ins", self.location.as_str()),
        }
    }
}

/// Where a tool runs, as the root namespace of its name.
///
/// Two values, because a name has one root. This is not
/// [`crate::domain::ToolLocality`], which carries the machine labels the
/// per-turn tool note renders, and not either `ToolRunner`, which answer what a
/// tool reaches and which side a span measured. This is the first segment of a
/// name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolLocation {
    /// Runs on the daemon: a built-in, or an MCP server the daemon reaches.
    Daemon,
    /// Runs on the connected client's own machine.
    Client,
}

impl ToolLocation {
    /// The root every name of this location begins with.
    pub const fn root(self) -> &'static str {
        match self {
            Self::Daemon => "daemon_",
            Self::Client => "client_",
        }
    }

    /// The location's name, for a log line or a message to the model.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Daemon => "daemon",
            Self::Client => "client",
        }
    }

    /// Every location, so a caller that must consider all of them cannot miss
    /// one when a third appears.
    const ALL: [Self; 2] = [Self::Daemon, Self::Client];
}

/// Compose the name the model is offered: this code's root, then the provider's
/// own name beneath it.
///
/// The provider segment is sanitised rather than trusted - it keeps only the
/// characters a tool name may carry, and cannot begin or end with a separator,
/// so it can neither climb out of its root nor read as one. `None` when nothing
/// legible survives, which the caller reports as a refused offer rather than
/// advertising an unnamed tool.
pub fn compose_name(location: ToolLocation, provider_name: &str) -> Option<String> {
    let segment = sanitize_segment(provider_name)?;
    Some(format!("{}{segment}", location.root()))
}

/// The provider's own name, with the location root removed.
///
/// A name carrying no root is returned unchanged: the model may call a tool it
/// learned in an earlier turn, and the daemon executor's routing table outlives
/// the turn. Strip exactly once - a client tool whose provider named it
/// `daemon_read` composes to `client_daemon_read`, and a second strip would
/// read the provider's own word as a root.
pub fn strip_location(name: &str) -> &str {
    ToolLocation::ALL
        .into_iter()
        .find_map(|location| name.strip_prefix(location.root()))
        .unwrap_or(name)
}

/// Keep only what a tool name may carry, and refuse a segment with nothing
/// legible left.
///
/// A provider names itself, and both sides are equally untrusted here: a client
/// is a separate process the daemon does not control, and a daemon server's
/// namespace is a string in a configuration file. So this drops anything that
/// is not a tool-name character, trims the separators from both ends - a
/// leading one would read as a root's own separator - and bounds the length,
/// because the name is rendered into every round's tool block.
///
/// It cannot climb out: the root is applied by [`compose_name`] afterwards, and
/// nothing this returns can precede it.
fn sanitize_segment(raw: &str) -> Option<String> {
    let kept: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(MAX_SEGMENT_CHARS)
        .collect();
    let trimmed = kept.trim_matches(|c| c == '_' || c == '-');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Which surface offered a tool, which decides where its schema is shown -
/// never which of two claimants to a name wins, because there are never two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSurface {
    /// The turn loop's own control surface (`begin_step`, `complete_step`,
    /// `promote_plan_to_skill`). The loop runs these itself, before any
    /// executor, so they have no location and take no root.
    CoreLoop,
    /// Advertised in the round's tool block, with its schema.
    Block,
    /// Reached through the provider's own hosted tool search. In the table so
    /// it can be routed and counted for uniqueness; its schema travels in the
    /// namespaces rather than the block.
    Deferred,
    /// Advertised as a name and nothing else (#1212): the tool note says it
    /// exists, no schema is shown anywhere, and the table routes a call to it.
    ///
    /// Distinct from [`ToolSurface::Deferred`] because the difference decides
    /// whether the model has read the schema. A deferred tool's schema reaches
    /// the model through the provider's own tool search; a named one's reaches
    /// nothing, so a first call to it is a guess from the name, and the loop
    /// treats it as one.
    Named,
}

impl ToolSurface {
    /// Whether the round's tool block carries this tool's schema.
    pub(crate) fn in_block(self) -> bool {
        matches!(self, Self::CoreLoop | Self::Block)
    }
}

/// One entry: the name the model reads, the name everything else uses, and
/// where it runs.
#[derive(Debug, Clone)]
pub struct RoutedTool {
    advertised: ToolDefinition,
    provider_name: String,
    connection: Option<ToolConnection>,
    surface: ToolSurface,
}

impl RoutedTool {
    /// The composed name, which is the table's key and the only name the model
    /// ever sees.
    pub fn advertised_name(&self) -> &str {
        &self.advertised.name
    }

    /// The definition the model is shown, under its composed name.
    pub fn definition(&self) -> &ToolDefinition {
        &self.advertised
    }

    /// The provider's own name: what the executor is asked for, what a lesson
    /// is keyed on, what the provenance gate classifies, and what a caller's
    /// allowlist names. Never the composed name.
    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// The connection that runs it, and the other half of its address. `None`
    /// for the turn loop's own control surface, which runs in the loop and
    /// reaches no executor.
    ///
    /// This is where a caller learns the location. Nothing reads the name's
    /// prefix to decide it: the prefix is a uniqueness device that happens to
    /// be visible, and the day a structural field replaces it, taking it out
    /// must stay a rename rather than a behaviour change.
    pub fn connection(&self) -> Option<&ToolConnection> {
        self.connection.as_ref()
    }

    /// Whether the turn loop runs this call itself rather than handing it to an
    /// executor.
    pub fn is_core_loop(&self) -> bool {
        self.surface == ToolSurface::CoreLoop
    }

    /// Whether the model was offered this tool's name with no schema anywhere
    /// (#1212), so a call to it was written from the name alone.
    pub fn is_named_only(&self) -> bool {
        self.surface == ToolSurface::Named
    }

    /// Whether this round's tool block already carries this tool's schema.
    pub fn is_in_block(&self) -> bool {
        self.surface.in_block()
    }

    /// Whether it runs on the connected client's machine, read from the
    /// connection rather than from the name.
    pub fn is_client(&self) -> bool {
        self.connection
            .as_ref()
            .is_some_and(|c| c.location() == ToolLocation::Client)
    }

    /// The telemetry label for the side that ran the call, derived from the
    /// route rather than computed a second time. A span that disagreed with the
    /// route would describe a turn that did not happen.
    pub(crate) fn telemetry_runner(&self) -> crate::telemetry::ToolRunner {
        if self.is_client() {
            crate::telemetry::ToolRunner::Client
        } else {
            crate::telemetry::ToolRunner::Server
        }
    }
}

/// Two tools that claimed one composed name. A fault in configuration, not a
/// case with semantics: it is refused and reported so a person can rename one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateName {
    /// The composed name both claimed.
    pub name: String,
    /// The claimant already in the table, as `<location>:<provider name>`.
    pub held_by: String,
    /// The claimant that was refused, in the same form.
    pub refused: String,
}

/// What the table says about a name the model called.
#[derive(Debug)]
pub enum Route<'a> {
    /// The name resolves to exactly one tool.
    Found(&'a RoutedTool),
    /// The table holds no entry. The turn loop hands it to the daemon
    /// executor, whose routing table outlives the turn and holds tools this
    /// turn never offered.
    Unrouted,
}

/// The turn's tool table.
///
/// Built once per round from the surfaces that offer tools, then asked twice:
/// once for the definitions the model is shown, and once per tool call for the
/// tool that answers the name. One table, so the two cannot disagree.
#[derive(Debug, Default)]
pub struct ToolRouter {
    entries: Vec<RoutedTool>,
    duplicates: Vec<DuplicateName>,
}

impl ToolRouter {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Offer a connection's capabilities, with their schemas in the round's
    /// tool block.
    pub fn offer(&mut self, connection: &ToolConnection, defs: &[ToolDefinition]) {
        self.admit(connection, defs, ToolSurface::Block);
    }

    /// Offer a connection's capabilities as deferred: the model reaches them
    /// through the provider's own tool search, so their schemas travel in the
    /// namespaces rather than the block. They are in the table all the same, so
    /// they can be routed and so they count for uniqueness.
    pub fn offer_deferred(&mut self, connection: &ToolConnection, defs: &[ToolDefinition]) {
        self.admit(connection, defs, ToolSurface::Deferred);
    }

    /// Offer a connection's capabilities as names only: the tool note says they
    /// exist, no schema is shown, and a call to one is routed all the same.
    ///
    /// This is what keeps a large connection off the bill (#1212). A schema
    /// costs roughly 250 estimated tokens and a name about ten, so the model
    /// keeps the recognition surface at a fortieth of the price and pays for a
    /// body only when it uses one.
    pub fn offer_named(&mut self, connection: &ToolConnection, defs: &[ToolDefinition]) {
        self.admit(connection, defs, ToolSurface::Named);
    }

    /// Offer one of the turn loop's own control tools, which has no location
    /// and takes no root: the loop runs it itself and no executor sees it.
    pub fn offer_core_loop_tool(&mut self, def: ToolDefinition) {
        let provider_name = def.name.clone();
        self.insert(def, provider_name, None, ToolSurface::CoreLoop);
    }

    /// The tool a connection offers under `tool_name` - the address the daemon
    /// invokes by when it means a particular connection, rather than the
    /// composed name the model writes.
    ///
    /// The same table answers both, so the two can never name different tools.
    pub fn resolve_on(&self, connection: &ToolConnection, tool_name: &str) -> Route<'_> {
        self.entries
            .iter()
            .find(|e| e.connection.as_ref() == Some(connection) && e.provider_name == tool_name)
            .map_or(Route::Unrouted, Route::Found)
    }

    /// Whether this round's block already carries the schema `connection`
    /// offers under `tool_name`.
    ///
    /// What activation asks before spending a slot (#1212): a tool already in
    /// the block gains nothing from being activated, because the offer is a
    /// no-op, and the ledger row would bound out a capability the turn does not
    /// yet have.
    pub fn advertises(&self, connection: &ToolConnection, tool_name: &str) -> bool {
        matches!(self.resolve_on(connection, tool_name), Route::Found(entry) if entry.is_in_block())
    }

    /// Drop every entry whose provider name `keep` rejects.
    ///
    /// By provider name, because a caller's allowlist names tools as their
    /// providers name them - a location is this turn's fact about where a tool
    /// ran, not part of what the caller was permitted.
    pub fn retain<F: Fn(&str) -> bool>(&mut self, keep: F) {
        self.entries.retain(|e| keep(e.provider_name()));
    }

    /// The definitions the model is shown, in the order they were offered, each
    /// under its composed name.
    pub fn advertised_definitions(&self) -> Vec<ToolDefinition> {
        self.entries
            .iter()
            .filter(|e| e.surface.in_block())
            .map(|e| e.advertised.clone())
            .collect()
    }

    /// The composed names of every tool offered as a name and nothing else
    /// (#1212), in the order they were offered.
    ///
    /// What the tool note lists so the model knows they exist. Read from the
    /// table, so a name the note gives is a name the table routes.
    pub fn named_only_names(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.surface == ToolSurface::Named)
            .map(|e| e.advertised.name.clone())
            .collect()
    }

    /// What each connection's advertised schemas cost this round, by the label
    /// an operator reads (#1212).
    ///
    /// One aggregate figure says the tools cost 23.7k and names nothing to
    /// drop; this says which connection to look at. Only what the block
    /// actually carries is counted, because that is what was paid for.
    pub fn advertised_cost_by_connection(
        &self,
        cost: &dyn Fn(&ToolDefinition) -> u64,
    ) -> Vec<(String, u64)> {
        let mut totals: Vec<(String, u64)> = Vec::new();
        for entry in self.entries.iter().filter(|e| e.surface.in_block()) {
            let label = entry
                .connection
                .as_ref()
                .map_or_else(|| "core-loop".to_string(), ToolConnection::label);
            let spent = cost(&entry.advertised);
            match totals.iter_mut().find(|(name, _)| name == &label) {
                Some((_, total)) => *total = total.saturating_add(spent),
                None => totals.push((label, spent)),
            }
        }
        totals
    }

    /// The deferred namespaces as the model may be offered them: every tool
    /// still reached that way, under its composed name, and an empty namespace
    /// dropped.
    pub fn offered_namespaces(&self, namespaces: &[ToolNamespace]) -> Vec<ToolNamespace> {
        namespaces
            .iter()
            .filter_map(|ns| {
                let tools: Vec<ToolDefinition> = ns
                    .tools
                    .iter()
                    .filter_map(|t| {
                        self.entries
                            .iter()
                            .find(|e| {
                                e.surface == ToolSurface::Deferred && e.provider_name == t.name
                            })
                            .map(|e| e.advertised.clone())
                    })
                    .collect();
                (!tools.is_empty())
                    .then(|| ToolNamespace::new(ns.name.clone(), ns.description.clone(), tools))
            })
            .collect()
    }

    /// The tool a composed name names. One lookup, and the same table the
    /// advertised set was built from.
    pub fn resolve(&self, name: &str) -> Route<'_> {
        self.entries
            .iter()
            .find(|e| e.advertised_name() == name)
            .map_or(Route::Unrouted, Route::Found)
    }

    /// The advertised names of everything one location runs, for the per-turn
    /// tool note. Read from the table, never by looking at the names.
    pub fn advertised_names_at(&self, location: ToolLocation) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| {
                e.connection
                    .as_ref()
                    .is_some_and(|c| c.location() == location)
            })
            .map(|e| e.advertised.name.clone())
            .collect()
    }

    /// The names two tools both claimed, which were refused. Empty in a healthy
    /// configuration.
    pub fn duplicates(&self) -> &[DuplicateName] {
        &self.duplicates
    }

    fn admit(
        &mut self,
        connection: &ToolConnection,
        defs: &[ToolDefinition],
        surface: ToolSurface,
    ) {
        for def in defs {
            let Some(name) = compose_name(connection.location(), &def.name) else {
                tracing::warn!(
                    connection = %connection.label(),
                    "a tool whose name carries nothing a name may carry cannot be offered"
                );
                continue;
            };
            let mut advertised = def.clone();
            advertised.name = name;
            self.insert(
                advertised,
                def.name.clone(),
                Some(connection.clone()),
                surface,
            );
        }
    }

    /// Admit one entry, or refuse it as a duplicate.
    ///
    /// A composed name is claimed by exactly one tool, and uniqueness follows
    /// from connections being distinct rather than being enforced on top of
    /// them: the name derives from the connection and the tool.
    ///
    /// The one case that is not two claimants is one tool offered on two
    /// surfaces - a connection's tool is in the deferred fleet and may also be
    /// activated into the block - so an offer matching an entry's connection
    /// and tool name is the same tool arriving twice, and the block offer is
    /// kept because it is the one whose schema the model reads. Anything else
    /// is a fault: refused, recorded with both claimants, never silently
    /// resolved.
    ///
    /// **A promotion moves the entry to the end** rather than upgrading it
    /// where it stands (#1294). The block's order is what a round-to-round
    /// comparison reads, and a schema that appeared in the middle of it would
    /// shift every tool behind it - so a tool the round put in the block takes
    /// a position after everything already advertised, and the round before
    /// stays a prefix of this one.
    fn insert(
        &mut self,
        advertised: ToolDefinition,
        provider_name: String,
        connection: Option<ToolConnection>,
        surface: ToolSurface,
    ) {
        let claimant = |connection: Option<&ToolConnection>, provider_name: &str| {
            format!(
                "{}/{provider_name}",
                connection.map_or("core-loop".to_string(), ToolConnection::label)
            )
        };
        if let Some(position) = self
            .entries
            .iter()
            .position(|e| e.advertised.name == advertised.name)
        {
            let held = &self.entries[position];
            if held.connection == connection && held.provider_name == provider_name {
                // One tool, two surfaces. Keep the one the model can read: an
                // offer that puts the schema in the block wins over one that
                // leaves it out, whichever arrived first.
                if !held.surface.in_block() && surface.in_block() {
                    let mut promoted = self.entries.remove(position);
                    promoted.surface = surface;
                    promoted.advertised = advertised;
                    self.entries.push(promoted);
                }
                return;
            }
            let held_by = claimant(held.connection.as_ref(), &held.provider_name);
            self.duplicates.push(DuplicateName {
                name: advertised.name.clone(),
                held_by,
                refused: claimant(connection.as_ref(), &provider_name),
            });
            return;
        }
        self.entries.push(RoutedTool {
            advertised,
            provider_name,
            connection,
            surface,
        });
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

    fn advertised_names(router: &ToolRouter) -> Vec<String> {
        router
            .advertised_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect()
    }

    /// #1216: the name is the location root, then the provider's namespace
    /// where it has one, then the tool. All four rows of the ticket's table.
    #[test]
    fn a_name_composes_as_location_root_then_provider_namespace_then_tool() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::daemon_builtins(),
            &[def("builtin_tool_search", "find tools")],
        );
        router.offer(
            &ToolConnection::daemon_server("fileio"),
            &[def("fileio__read_file", "read a file")],
        );
        router.offer(
            &ToolConnection::client_device(),
            &[
                def("take_screenshot", "client built-in"),
                def("fileio__read_file", "client MCP server"),
            ],
        );
        assert_eq!(
            advertised_names(&router),
            vec![
                "daemon_builtin_tool_search".to_string(),
                "daemon_fileio__read_file".to_string(),
                "client_take_screenshot".to_string(),
                "client_fileio__read_file".to_string(),
            ]
        );
    }

    /// #1216, a security property: the composing code applies the root and a
    /// connection can never place itself. Both roots, because neither side is
    /// trusted - a daemon server's namespace comes from the daemon's own
    /// configuration, and treating that as trustworthy is how the rule rots.
    #[test]
    fn a_connections_namespace_cannot_escape_its_root() {
        let hostile = [
            "daemon__read_file",
            "client__read_file",
            "__read_file",
            "read_file__",
            "_daemon_read_file",
            "read\nfile",
            "../../read_file",
        ];
        for provider_name in hostile {
            let client = compose_name(ToolLocation::Client, provider_name)
                .unwrap_or_else(|| panic!("{provider_name} composes"));
            assert!(
                client.starts_with("client_"),
                "a client connection escaped its root: {provider_name} -> {client}"
            );
            let daemon = compose_name(ToolLocation::Daemon, provider_name)
                .unwrap_or_else(|| panic!("{provider_name} composes"));
            assert!(
                daemon.starts_with("daemon_"),
                "a daemon server escaped its root: {provider_name} -> {daemon}"
            );
            assert_ne!(client, daemon, "two roots must never compose to one name");
        }
    }

    /// A name with nothing a tool name may carry is refused rather than
    /// advertised under its bare root.
    #[test]
    fn a_name_with_nothing_legible_is_refused() {
        assert_eq!(compose_name(ToolLocation::Client, "___"), None);
        assert_eq!(compose_name(ToolLocation::Daemon, "  "), None);
    }

    /// A well-formed provider name is carried through unchanged, so this
    /// composes rather than rewrites.
    #[test]
    fn a_well_formed_provider_name_is_carried_through_unchanged() {
        assert_eq!(
            compose_name(ToolLocation::Daemon, "fileio__read_file").as_deref(),
            Some("daemon_fileio__read_file")
        );
    }

    /// #1216: every advertised name is unique across the whole set. The fixture
    /// carries all three cases - one capability on a client and on the daemon
    /// (two connections, so two names and no fault), a built-in and an external
    /// tool colliding inside one location, and two devices running the same
    /// MCP server.
    #[test]
    fn every_advertised_name_is_unique_across_the_set() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::daemon_builtins(),
            &[def("read_file", "the daemon's own")],
        );
        router.offer(
            &ToolConnection::client_device(),
            &[def("read_file", "the client's own")],
        );
        // A daemon MCP server with no namespace of its own, exposing a name a
        // built-in already holds.
        router.offer(
            &ToolConnection::daemon_server("files"),
            &[def("read_file", "an external tool")],
        );
        // Two devices running the same MCP server, so the same namespace.
        router.offer(
            &ToolConnection::client_device(),
            &[def("fileio__read_file", "device one")],
        );
        router.offer(
            &ToolConnection::client_device(),
            &[def("fileio__read_file", "device two")],
        );

        let names = advertised_names(&router);
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            names.len(),
            unique.len(),
            "duplicate advertised name: {names:?}"
        );
        assert!(names.contains(&"daemon_read_file".to_string()));
        assert!(names.contains(&"client_read_file".to_string()));
    }

    /// A duplicate is a fault, so it is refused and both claimants are named -
    /// a person has to be able to see what to rename.
    #[test]
    fn a_refused_duplicate_names_both_claimants() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::daemon_builtins(),
            &[def("read_file", "the daemon's own")],
        );
        router.offer(
            &ToolConnection::daemon_server("files"),
            &[def("read_file", "an external tool")],
        );
        let duplicates = router.duplicates();
        assert_eq!(duplicates.len(), 1, "the collision is reported once");
        assert_eq!(duplicates[0].name, "daemon_read_file");
        assert_eq!(duplicates[0].held_by, "daemon:built-ins/read_file");
        assert_eq!(duplicates[0].refused, "daemon:files/read_file");
        assert_eq!(
            advertised_names(&router),
            vec!["daemon_read_file".to_string()],
            "the refused claimant is not advertised"
        );
    }

    /// #1216: the daemon addresses a tool as (connection, tool name), and the
    /// composed name the model writes resolves to the same entry. One table,
    /// two ways in.
    #[test]
    fn a_tool_is_addressed_by_connection_and_tool_name() {
        let mut router = ToolRouter::new();
        let laptop = ToolConnection::client_device();
        let fileio = ToolConnection::daemon_server("fileio");
        router.offer(&laptop, &[def("read_file", "on the laptop")]);
        router.offer(&fileio, &[def("fileio__read_file", "on the daemon")]);

        let Route::Found(by_address) = router.resolve_on(&laptop, "read_file") else {
            panic!("a connection's tool must resolve by address");
        };
        let Route::Found(by_name) = router.resolve("client_read_file") else {
            panic!("the composed name must resolve");
        };
        assert_eq!(by_address.advertised_name(), by_name.advertised_name());
        assert_eq!(by_address.definition().description, "on the laptop");

        let Route::Found(daemon_side) = router.resolve_on(&fileio, "fileio__read_file") else {
            panic!("the daemon server's tool must resolve by address");
        };
        assert_eq!(daemon_side.advertised_name(), "daemon_fileio__read_file");
    }

    /// #1216: nothing derives locality by parsing the name. The prefix is a
    /// uniqueness device, and the day metadata replaces it, removing it must
    /// stay a rename rather than a behaviour change.
    ///
    /// Both fixtures would answer the opposite way round if anything read the
    /// name: a client connection whose configured namespace is `daemon`, and a
    /// daemon server whose configured namespace is `client`.
    #[test]
    fn host_is_read_from_the_connection_not_from_the_name() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::client_device(),
            &[def("daemon__read_file", "on the client")],
        );
        router.offer(
            &ToolConnection::daemon_server("client"),
            &[def("client__read_file", "on the daemon")],
        );

        let Route::Found(on_client) = router.resolve("client_daemon__read_file") else {
            panic!("resolves");
        };
        assert!(
            on_client.is_client(),
            "a connection named `daemon` still runs on the client"
        );
        assert_eq!(
            on_client.connection().map(ToolConnection::location),
            Some(ToolLocation::Client)
        );

        let Route::Found(on_daemon) = router.resolve("daemon_client__read_file") else {
            panic!("resolves");
        };
        assert!(
            !on_daemon.is_client(),
            "a server named `client` still runs on the daemon"
        );
    }

    /// #1216: the prefix is a display and routing artifact. What runs, and what
    /// a lesson is keyed on, is the provider's own name.
    #[test]
    fn the_location_prefix_is_stripped_to_the_providers_own_name() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::client_device(),
            &[def("fileio__read_file", "on the laptop")],
        );
        let Route::Found(routed) = router.resolve("client_fileio__read_file") else {
            panic!("resolves");
        };
        assert_eq!(routed.provider_name(), "fileio__read_file");
        assert_eq!(
            strip_location("client_fileio__read_file"),
            "fileio__read_file"
        );
        assert_eq!(
            strip_location("daemon_builtin_tool_search"),
            "builtin_tool_search"
        );
    }

    /// Strip exactly once: a client connection whose provider named a tool
    /// `daemon_read` composes to `client_daemon_read`, and a second strip would
    /// read the provider's own word as a root.
    #[test]
    fn stripping_removes_one_root_and_leaves_the_providers_own_words() {
        assert_eq!(strip_location("client_daemon_read"), "daemon_read");
    }

    /// A name that carries no root is a name from an earlier turn, and is
    /// handed on unchanged.
    #[test]
    fn a_name_with_no_root_is_left_alone() {
        assert_eq!(strip_location("web_read"), "web_read");
    }

    /// One tool is one entry, even when two surfaces offer it: a connection's
    /// tool can be in the deferred fleet and then activated into the block.
    /// That is not two claimants to a name.
    #[test]
    fn one_tool_offered_on_two_surfaces_is_one_entry() {
        let mut router = ToolRouter::new();
        let fileio = ToolConnection::daemon_server("fileio");
        router.offer_deferred(&fileio, &[def("fileio__read_file", "read a file")]);
        router.offer(&fileio, &[def("fileio__read_file", "read a file")]);
        assert!(
            router.duplicates().is_empty(),
            "one tool is not a collision"
        );
        assert_eq!(
            advertised_names(&router),
            vec!["daemon_fileio__read_file".to_string()],
            "once activated, its schema is in the block"
        );
    }

    /// A deferred tool is routable but not advertised: its schema reaches the
    /// model through the provider's own tool search, under its composed name.
    #[test]
    fn a_deferred_tool_is_routable_and_offered_under_its_composed_name() {
        let mut router = ToolRouter::new();
        let fileio = ToolConnection::daemon_server("fileio");
        router.offer_deferred(
            &fileio,
            &[
                def("fileio__read_file", "read a file"),
                def("fileio__write_file", "write a file"),
            ],
        );
        assert!(advertised_names(&router).is_empty());
        assert!(matches!(
            router.resolve("daemon_fileio__read_file"),
            Route::Found(_)
        ));

        let namespaces = vec![ToolNamespace::new(
            "fileio",
            "files",
            vec![
                def("fileio__read_file", "read a file"),
                def("fileio__write_file", "write a file"),
            ],
        )];
        let offered = router.offered_namespaces(&namespaces);
        let names: Vec<&str> = offered[0].tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["daemon_fileio__read_file", "daemon_fileio__write_file"],
            "the model calls what it reads, so a deferred schema carries the composed name"
        );
    }

    /// A tool promoted into the block leaves the deferred namespaces, so the
    /// model is never shown one name twice.
    #[test]
    fn a_tool_promoted_into_the_block_leaves_the_deferred_namespaces() {
        let mut router = ToolRouter::new();
        let fileio = ToolConnection::daemon_server("fileio");
        router.offer_deferred(
            &fileio,
            &[
                def("fileio__read_file", "read a file"),
                def("fileio__write_file", "write a file"),
            ],
        );
        router.offer(&fileio, &[def("fileio__read_file", "read a file")]);
        let namespaces = vec![ToolNamespace::new(
            "fileio",
            "files",
            vec![
                def("fileio__read_file", "read a file"),
                def("fileio__write_file", "write a file"),
            ],
        )];
        let offered = router.offered_namespaces(&namespaces);
        let names: Vec<&str> = offered[0].tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["daemon_fileio__write_file"]);
    }

    /// The turn loop's own control surface has no location, so it takes no
    /// root: it runs in the loop and reaches no executor.
    #[test]
    fn a_core_loop_tool_takes_no_root() {
        let mut router = ToolRouter::new();
        router.offer_core_loop_tool(def("begin_step", "open a step"));
        assert_eq!(advertised_names(&router), vec!["begin_step".to_string()]);
        let Route::Found(routed) = router.resolve("begin_step") else {
            panic!("resolves");
        };
        assert!(routed.is_core_loop());
        assert_eq!(routed.connection(), None);
    }

    /// A caller's allowlist names tools as their providers name them: a
    /// location is this turn's fact about where a tool ran, not part of what
    /// the caller was permitted.
    #[test]
    fn an_allowlist_names_tools_as_their_providers_do() {
        let mut router = ToolRouter::new();
        router.offer(
            &ToolConnection::daemon_builtins(),
            &[
                def("web_read", "read a page"),
                def("run_shell", "run a command"),
            ],
        );
        router.retain(|provider_name| provider_name == "web_read");
        assert_eq!(
            advertised_names(&router),
            vec!["daemon_web_read".to_string()]
        );
    }

    /// A name the turn never offered is not routed here. The turn loop hands it
    /// to the daemon executor, whose table outlives the turn.
    #[test]
    fn a_name_the_turn_never_offered_is_unrouted() {
        let router = ToolRouter::new();
        assert!(matches!(router.resolve("daemon_web_read"), Route::Unrouted));
    }
}
