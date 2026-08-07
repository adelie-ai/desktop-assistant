# Logging and telemetry

The daemon and the D-Bus bridge produce traces, metrics and logs through
[`adelie-telemetry`](https://github.com/adelie-ai/adelie-telemetry), the crate
every Adelie Rust binary shares. One setup, one set of knobs, so an operator
meets the same behaviour in the daemon, the bridge, the voice service and every
MCP server.

Console output is the default and needs no collector. Export to an
OpenTelemetry collector is *additional*, not a replacement, and is behind an
off-by-default Cargo feature.

## Where it goes

**stderr, always. Never stdout.** stdout carries a process's own output: the
daemon prints an operator-facing result there for its two command-line escape
hatches, an MCP server frames JSON-RPC there, and a client writes the model's
reply there. One log line mixed into that stream corrupts what the reader is
parsing.

Console output is plain text, not JSON, and colour is off. With a collector
doing the real collection, stderr is for a person reading `journalctl` or
`kubectl logs`, and plain text is easier to read there than JSON.

## How much of it

`RUST_LOG` sets the filter, through `EnvFilter`.

```sh
RUST_LOG=info desktop-assistant-daemon
RUST_LOG=info,desktop_assistant_mcp_client=debug desktop-assistant-daemon
```

When `RUST_LOG` is unset or unparseable, each binary falls back to its own
default:

| binary | default filter |
|---|---|
| `desktop-assistant-daemon` | `error` |
| `adelie-dbus-bridge` | `info` |

Every shipped deployment sets `RUST_LOG=info` explicitly - both systemd units,
both container images and the Kubernetes manifests - so the daemon's `error`
default applies only to a bare `cargo run`. One consequence is worth knowing:
the periodic metrics summary is written at INFO, so a run with no `RUST_LOG` at
all does not print it. Ask for `RUST_LOG=info` to see it.

One filter governs the console and the OTLP log exporter together. An operator
who turns the verbosity up expects the same lines wherever they read them.

## What may appear at each level

This contract is a rule, not a preference.

| Level | Carries |
|---|---|
| INFO | ids, counts, byte sizes, durations, model names, tool names, token counts. **Never content.** |
| DEBUG | prompts, the assembled context, tool arguments, tool results, search queries, extracted facts. |

"Content" is anything derived from what a person or a model wrote: a prompt, a
message body, a tool-call argument, a tool result, a search query, an extracted
personal fact. Ids, counts, sizes, durations, and the *names* of tools, models
and providers are not content.

Two things follow.

**`RUST_LOG=debug` means conversation content reaches the journal, and reaches
the collector when one is configured.** That is deliberate, and it is why the
default is `info`. Raise a single target rather than the whole process when you
are debugging one thing:

```sh
RUST_LOG=info,desktop_assistant_mcp_client=debug
```

**A new log line is checked against the contract before it lands.** Named tests
hold the line where content used to leak: `no_content_at_info` and
`content_appears_at_debug` in `crates/core/tests/log_content_contract.rs`,
`no_search_query_at_info` in `crates/mcp-client/src/builtin.rs`, and
`no_extracted_fact_content_at_info` in
`crates/storage/tests/dreaming_db_paths.rs`.

`scripts/tests/systemd-logging.test.sh` holds the shipped systemd units to a
global `RUST_LOG` of `info` or quieter, and forbids `debug` on the three
targets that carry content.

## Span timing

Both binaries turn span-close events on, so a closing span writes a line
carrying how long it was open:

```text
INFO turn{turn_id=4bf92f35...}: close time.busy=208ms time.idle=14.9ms
```

That is what makes turn timing visible to somebody reading a running
container's log, where there is no trace backend to open.

## Metrics

Call sites record through `adelie_telemetry::metrics` and never through an
opentelemetry meter directly. A direct call would make the crate that records
depend on opentelemetry whether or not export is on.

```rust
use adelie_telemetry::metrics::{self, Label};

metrics::increment("llm.requests", &[Label::new("provider", "example")]);
metrics::record_duration("dreaming.scan.duration", elapsed, &[]);
```

**Label values are names, not content.** A prompt or a tool argument used as a
label would be both a disclosure and an unbounded memory leak in a process that
runs for weeks. One metric may carry 64 distinct label sets; past that, further
label sets fold into one series labelled `cardinality=other`.

The registry runs whether or not export is on. It keeps counters and
fixed-bucket histograms in process and writes a summary every 10 minutes, so a
desktop install running a default-feature build gets real numbers in its
journal. Each summary reports the window that just closed beside a running
total, because on a pod that has run for a month a cumulative number is
dominated by history and stops moving.

```text
INFO metrics summary window_seconds=600 uptime_seconds=8400 counters=1 histograms=2
INFO duration metric="dreaming.scan.duration" labels=outcome=ok window_count=1 window_p95_ms=2500 total_count=14 total_p95_ms=5000
```

Durations go into fixed-bucket histograms rather than a count and a sum,
because a mean hides the tail: "the average turn took 3 seconds" and "one turn
in twenty took four minutes" are the same mean, and only the second is the
report a user files.

## Export to a collector

Off by default. With the feature off, no opentelemetry crate is resolved at
all, so `cargo install` costs a desktop user nothing extra - no crates, no
native code, no C toolchain. `scripts/no-opentelemetry.sh` holds that line in
the gate.

```sh
cargo build -p desktop-assistant-daemon --features otel
cargo build -p desktop-assistant-dbus-bridge --features otel
```

With the feature on, the OTLP layers are added *beside* the console layer,
never in place of it, so an exporting build still prints locally. The feature
compiles the TLS backend, which needs a C compiler and an assembler
(`build-essential` on Debian, `gcc` plus `binutils` elsewhere). It does not
need `cmake`.

### Configuration

Everything comes from the standard `OTEL_*` environment variables. There are no
CLI flags and no Adelie-specific variables. Nothing is passed to the exporter
builders in code, so every variable below reaches them.

| Variable | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Endpoint for all three signals. The signal's path (`/v1/traces` and so on) is appended. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Endpoint for traces, used exactly as written, so it must include the path. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | Endpoint for metrics. Same rule. |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Endpoint for log records. Same rule. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc`, `http/protobuf` or `http/json`, for all three. |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces. Per-signal forms exist for metrics and logs too. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers for all three, as `key=value,key=value`. Per-signal forms exist. |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | Export timeout in milliseconds. Per-signal forms exist. |
| `OTEL_EXPORTER_OTLP_COMPRESSION` | `gzip` or `zstd`. Per-signal forms exist. |
| `OTEL_TRACES_EXPORTER` | `none` switches traces off. `OTEL_METRICS_EXPORTER` and `OTEL_LOGS_EXPORTER` do the same for their signal. |
| `OTEL_SDK_DISABLED` | `true` switches all three off. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes, as `key=value,key=value`. |

A per-signal variable beats the generic one.

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.example.com:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  desktop-assistant-daemon
```

Each binary reports its own `service.name`: `adele-daemon` and
`adele-dbus-bridge`.

### Choosing a transport

Both transports are compiled in and `OTEL_EXPORTER_OTLP_PROTOCOL` selects one
at run time.

`http/protobuf` (port 4318) is the safer default. It uses a blocking HTTP
client on the exporter's own thread.

`grpc` (port 4317) needs a running Tokio runtime. Both binaries here are
`#[tokio::main]` and install telemetry inside the runtime, so both transports
work in both.

The two transports trust different certificate stores, and the difference
decides what a container image needs:

| transport | trust anchors | consequence |
|---|---|---|
| `grpc` | compiled-in webpki roots | independent of the image |
| `http/protobuf` | the OS trust store | the image needs `ca-certificates`, or every HTTPS export fails |

### When a pipeline cannot be built

A wrong value in the environment costs the process its export and nothing else.
The console layer is installed either way, the metrics summary keeps running,
and the reason is written at ERROR together with the `OTEL_*` variables that
were set. Header values are never printed, because they routinely carry an API
key. A typo in one variable must not be able to silence a process.

### Shutdown

Telemetry is flushed and shut down when the guard drops at the end of `main`,
in the order traces, metrics, logs, with one final metrics summary. The batch
exporters buffer, and a process that exits without that flush loses whatever
was still buffered - usually the part worth having, because a crash is what was
being investigated. Shutdown is capped at 5 seconds.

**Kubernetes:** set `terminationGracePeriodSeconds` to at least 30 in any
deployment that runs with `otel` on, so the pod has room to flush telemetry and
finish whatever else it was doing before SIGKILL arrives.

## The gate

`just check` covers both configurations:

| step | what it holds |
|---|---|
| `just no-otel` | a default build resolves no opentelemetry crate |
| `just lint-otel` | the exporting path compiles, warnings as errors |
| `just doc-otel` | the exporting path's documentation resolves |
| `just test-otel` | the exporting path's tests run |

The three `-otel` steps exist for the same reason `lint-sqlite` and
`lint-mcp-host` do: the feature is off by default and nothing in a workspace
build turns it on, so the workspace steps compile none of that code. A change
that builds with default features can still fail with `otel` on.
