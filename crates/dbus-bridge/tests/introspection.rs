//! Introspection parity gate (#315) — the acceptance test for the whole
//! dbus-bridge cutover (#281).
//!
//! Every method / signal / property the live in-process `dbus-interface`
//! surface exposes at the well-known name `org.desktopAssistant` must also be
//! present on the bridge with a byte-identical D-Bus signature, so the cutover
//! is invisible to adele-kde (KCM + plasmoid). This test stands each bridge
//! adapter up in-process (no bus required — `Interface::introspect_to_writer`),
//! canonicalizes its introspection XML down to a sorted set of signature lines,
//! and asserts it matches the committed golden for that interface.
//!
//! ## Where the goldens come from (and why that makes this a real gate)
//!
//! The goldens under `tests/goldens/*.canon` are the *canonicalized live
//! surface*, captured from a running daemon — not a snapshot of the bridge
//! against itself. They are therefore authoritative: a divergence is a genuine
//! parity bug in the bridge, not drift in the test. Two documented intentional
//! drops are removed from the captured Settings surface (`Q2_DROPS`): the
//! legacy `Get/SetLlmSettings` (superseded by named connections) and
//! `GenerateWsJwt` (JWT minting is off D-Bus entirely — #281).
//!
//! Re-capture only when the frozen surface legitimately changes (before #319
//! deletes `dbus-interface`):
//!
//! ```text
//! mkdir -p /tmp/live-xml
//! for p in Commands Settings Connections Knowledge Conversations Reload; do
//!   busctl --user introspect org.desktopAssistant /org/desktopAssistant/$p \
//!     --xml-interface > /tmp/live-xml/$p.xml
//! done
//! INTROSPECT_LIVE_DIR=/tmp/live-xml cargo test -p desktop-assistant-dbus-bridge \
//!   --test introspection capture_goldens_from_live -- --ignored --nocapture
//! ```
//!
//! `BackgroundTasks` is intentionally **not** in the parity set: the live
//! in-process daemon does not serve it at all (it has no `/BackgroundTasks`
//! object path); the bridge adds it as a pure superset so D-Bus clients can get
//! `Task*` signals. There is nothing to be "at parity" with, so it is excluded.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::adapter::{
    DbusCommandsAdapter, DbusConnectionsAdapter, DbusConversationsAdapter, DbusKnowledgeAdapter,
    DbusReloadAdapter, DbusSettingsAdapter,
};
use desktop_assistant_dbus_bridge::transport::{BridgeTransport, BridgeTransportError};
use tokio::sync::broadcast;
use zbus::object_server::Interface;

/// Interfaces whose signature must match the live surface, keyed by the golden
/// file label and the D-Bus interface name.
const PARITY: &[(&str, &str)] = &[
    ("Commands", "org.desktopAssistant.Commands"),
    ("Conversations", "org.desktopAssistant.Conversations"),
    ("Settings", "org.desktopAssistant.Settings"),
    ("Connections", "org.desktopAssistant.Connections"),
    ("Knowledge", "org.desktopAssistant.Knowledge"),
    ("Reload", "org.desktopAssistant.Reload"),
];

/// Methods on the live surface that the bridge deliberately does NOT mirror
/// (#314 Q1/Q2): no adele-kde caller, no `api::Command` wire equivalent. These
/// are stripped from the captured golden so the gate treats their absence as
/// expected rather than a regression.
const Q2_DROPS: &[(&str, &str)] = &[
    ("org.desktopAssistant.Settings", "GetLlmSettings"),
    ("org.desktopAssistant.Settings", "SetLlmSettings"),
    ("org.desktopAssistant.Settings", "GenerateWsJwt"),
];

// --- fake transport ---------------------------------------------------------

/// A transport that never connects — introspection never calls `request`, it
/// only needs a constructed adapter. `subscribe_events` returns a live-but-idle
/// receiver to satisfy the trait.
struct NoopTransport {
    events_tx: broadcast::Sender<api::Event>,
}

impl NoopTransport {
    fn arc() -> Arc<Self> {
        let (events_tx, _rx) = broadcast::channel(1);
        Arc::new(Self { events_tx })
    }
}

#[async_trait::async_trait]
impl BridgeTransport for NoopTransport {
    async fn request(
        &self,
        _command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        Err(BridgeTransportError::Daemon("noop transport".into()))
    }

    fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
        self.events_tx.subscribe()
    }
}

/// Introspect one `#[interface]` impl into its raw `<interface>…</interface>`
/// XML fragment, with no bus or object server in the loop.
fn introspect<I: Interface>(iface: &I) -> String {
    let mut out = String::new();
    iface.introspect_to_writer(&mut out, 0);
    out
}

/// The bridge's introspection XML for `iface_name`, built from a fresh adapter.
fn bridge_introspection(iface_name: &str) -> String {
    let t = NoopTransport::arc();
    match iface_name {
        "org.desktopAssistant.Commands" => introspect(&DbusCommandsAdapter::new(Arc::clone(&t))),
        "org.desktopAssistant.Conversations" => {
            introspect(&DbusConversationsAdapter::new(Arc::clone(&t)))
        }
        "org.desktopAssistant.Settings" => introspect(&DbusSettingsAdapter::new(Arc::clone(&t))),
        "org.desktopAssistant.Connections" => {
            introspect(&DbusConnectionsAdapter::new(Arc::clone(&t)))
        }
        "org.desktopAssistant.Knowledge" => introspect(&DbusKnowledgeAdapter::new(Arc::clone(&t))),
        "org.desktopAssistant.Reload" => introspect(&DbusReloadAdapter::new()),
        other => panic!("no bridge adapter wired for {other}"),
    }
}

// --- canonicalization -------------------------------------------------------

/// Read one `key="value"` attribute off an XML element line.
fn attr(line: &str, key: &str) -> Option<String> {
    let pat = format!("{key}=\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Remove XML comments (including multi-line ones — zbus folds `///` docs into
/// `<!-- … -->`, and those differ between the two crates).
fn strip_comments(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + 3..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Reduce an introspection XML blob to the canonical signature of `iface_name`:
/// a sorted set of one-line member descriptors. Member *order* is normalized
/// (sorted) because D-Bus does not order members; argument order *within* a
/// member is preserved because it is the call signature. zbus generates both
/// surfaces, so any codegen quirk (e.g. a signal arg's `direction`) appears
/// identically on both sides and cancels.
fn canonicalize(xml: &str, iface_name: &str) -> String {
    let xml = strip_comments(xml);
    let mut members: Vec<String> = Vec::new();
    let mut in_iface = false;

    enum Cur {
        None,
        Member {
            kind: &'static str,
            name: String,
            args: Vec<String>,
        },
    }
    let mut cur = Cur::None;

    for raw in xml.lines() {
        let line = raw.trim();
        if line.starts_with("<interface ") {
            in_iface = attr(line, "name").as_deref() == Some(iface_name);
            continue;
        }
        if line.starts_with("</interface>") {
            in_iface = false;
            continue;
        }
        if !in_iface {
            continue;
        }

        if line.starts_with("<method ") || line.starts_with("<signal ") {
            let kind = if line.starts_with("<method ") {
                "method"
            } else {
                "signal"
            };
            let name = attr(line, "name").unwrap_or_default();
            if line.ends_with("/>") {
                members.push(format!("{kind} {name}()"));
            } else {
                cur = Cur::Member {
                    kind,
                    name,
                    args: Vec::new(),
                };
            }
        } else if line.starts_with("<arg ") {
            if let Cur::Member { args, .. } = &mut cur {
                let aname = attr(line, "name").unwrap_or_default();
                let atype = attr(line, "type").unwrap_or_default();
                let adir = attr(line, "direction").unwrap_or_default();
                args.push(format!("{adir}:{atype}:{aname}"));
            }
        } else if line.starts_with("</method>") || line.starts_with("</signal>") {
            if let Cur::Member { kind, name, args } = std::mem::replace(&mut cur, Cur::None) {
                members.push(format!("{kind} {name}({})", args.join(", ")));
            }
        } else if line.starts_with("<property ") {
            let name = attr(line, "name").unwrap_or_default();
            let atype = attr(line, "type").unwrap_or_default();
            let access = attr(line, "access").unwrap_or_default();
            members.push(format!("property {name}:{atype}:{access}"));
        }
    }

    members.sort();
    members.join("\n")
}

/// Strip the documented Q2 method drops from a captured canonical surface.
fn drop_q2(iface_name: &str, canon: &str) -> String {
    canon
        .lines()
        .filter(|line| {
            !Q2_DROPS.iter().any(|(iface, method)| {
                *iface == iface_name && line.starts_with(&format!("method {method}("))
            })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn golden_path(label: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{label}.canon"))
}

/// Human-readable symmetric difference of two canonical surfaces.
fn line_diff(want: &str, got: &str) -> String {
    use std::collections::BTreeSet;
    let want_set: BTreeSet<&str> = want.lines().collect();
    let got_set: BTreeSet<&str> = got.lines().collect();
    let mut out = String::new();
    for l in want_set.difference(&got_set) {
        out.push_str(&format!("  - missing on bridge: {l}\n"));
    }
    for l in got_set.difference(&want_set) {
        out.push_str(&format!("  + extra on bridge:   {l}\n"));
    }
    out
}

// --- the gate ---------------------------------------------------------------

#[test]
fn bridge_matches_live_golden() {
    let mut failures = Vec::new();
    for (label, iface) in PARITY {
        let got = canonicalize(&bridge_introspection(iface), iface);
        let path = golden_path(label);
        let want = match std::fs::read_to_string(&path) {
            Ok(s) => s.trim_end().to_string(),
            Err(e) => {
                failures.push(format!(
                    "missing golden for {iface} at {} ({e}); regenerate with capture_goldens_from_live",
                    path.display()
                ));
                continue;
            }
        };
        if got != want {
            failures.push(format!(
                "interface {iface} diverged from the frozen live surface ({}):\n{}",
                path.display(),
                line_diff(&want, &got)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "D-Bus introspection parity gate failed (#315):\n\n{}",
        failures.join("\n\n")
    );
}

/// Regenerate the goldens from a live-daemon capture. Ignored by default;
/// see the module docs for the capture recipe. Writing goldens from the live
/// surface (not the bridge) is what keeps `bridge_matches_live_golden` honest.
#[test]
#[ignore = "regenerates goldens from a live-daemon capture; needs INTROSPECT_LIVE_DIR"]
fn capture_goldens_from_live() {
    let dir = std::env::var("INTROSPECT_LIVE_DIR").expect(
        "set INTROSPECT_LIVE_DIR to a directory of `busctl --xml-interface` captures \
         (one <Label>.xml per interface)",
    );
    std::fs::create_dir_all(golden_path("x").parent().unwrap()).unwrap();
    for (label, iface) in PARITY {
        let xml = std::fs::read_to_string(format!("{dir}/{label}.xml"))
            .unwrap_or_else(|e| panic!("read {dir}/{label}.xml: {e}"));
        let canon = drop_q2(iface, &canonicalize(&xml, iface));
        std::fs::write(golden_path(label), format!("{canon}\n")).unwrap();
        eprintln!("wrote golden {} ({} members)", label, canon.lines().count());
    }
}

#[test]
fn canonicalize_extracts_only_the_named_interface() {
    // Mirrors zbus's real one-element-per-line output. Exercises interface
    // filtering, multi-line comment stripping, an arg-bearing method with an
    // unnamed out arg, a self-closing no-arg method, a signal, and member
    // sorting (Zebra before… no — Alpha sorts first).
    let xml = r#"
<node>
  <interface name="org.desktopAssistant.Other">
    <method name="Ignored">
      <arg name="x" type="s" direction="in"/>
    </method>
  </interface>
  <interface name="org.desktopAssistant.Sample">
    <!-- a doc comment
         spanning two lines, must be stripped -->
    <method name="Zebra">
      <arg name="a" type="s" direction="in"/>
      <arg type="u" direction="out"/>
    </method>
    <method name="Alpha"/>
    <signal name="Ping">
      <arg name="id" type="s"/>
    </signal>
  </interface>
</node>
"#;
    let got = canonicalize(xml, "org.desktopAssistant.Sample");
    assert_eq!(
        got,
        "method Alpha()\nmethod Zebra(in:s:a, out:u:)\nsignal Ping(:s:id)"
    );
}
