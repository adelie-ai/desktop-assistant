//! How this binary configures telemetry.
//!
//! The setup itself belongs to `adelie_telemetry`, which every Adelie binary
//! shares so an operator meets the same knobs everywhere. This module holds
//! only the three answers that are the daemon's own: what it calls itself, how
//! loud it is when nobody asked, and whether a closing span writes a line.
//!
//! Where telemetry goes is not decided here. It comes from the standard
//! `OTEL_*` environment variables at run time, so an operator moves it without
//! a rebuild. See the Logging section of `docs/logging.md`.

use adelie_telemetry::Config;

/// The name this binary reports as `service.name`.
///
/// It separates the daemon's telemetry from the bridge's, from voice's, and
/// from every MCP server's, so it has to be the binary's own name and stay
/// stable: a backend query written against it outlives any one release.
pub const SERVICE_NAME: &str = "adele-daemon";

/// The filter that applies when `RUST_LOG` says nothing.
///
/// `error`, which is what the daemon has always done: it used
/// `EnvFilter::from_default_env()`, and that falls back to `ERROR` when the
/// variable is unset. Every shipped deployment sets `RUST_LOG=info` (the
/// systemd units, both container images and the Kubernetes manifests), so this
/// value is what a bare `cargo run` gets and nothing else.
///
/// One consequence is worth knowing: the periodic metrics summary is written
/// at INFO, so a run with no `RUST_LOG` at all does not print it. Ask for
/// `RUST_LOG=info` to see it.
pub const DEFAULT_FILTER: &str = "error";

/// Which periodic maintenance loop a cycle belongs to.
///
/// An enum rather than a string because the value becomes a metric label, and
/// the registry caps a metric at 64 label sets with no eviction. A `&'static
/// str` rendering makes an unbounded value impossible to pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCycle {
    /// Fact extraction over recent conversations.
    Dreaming,
    /// Consolidation of what extraction wrote.
    Consolidation,
}

impl BackendCycle {
    /// The metric this cycle's duration is recorded under.
    fn metric(self) -> &'static str {
        match self {
            Self::Dreaming => "dreaming.scan.duration",
            Self::Consolidation => "consolidation.scan.duration",
        }
    }

    /// The label and log field naming this cycle.
    fn as_label(self) -> &'static str {
        match self {
            Self::Dreaming => "dreaming",
            Self::Consolidation => "consolidation",
        }
    }
}

/// Report one finished maintenance cycle.
///
/// The duration is a **field**, not part of the sentence, so a backend can
/// group and sort by it; it is also a measurement, so it reaches the metrics
/// facade as well. One slow cycle is a log line and a trend is a histogram,
/// and reading a trend out of interpolated text means parsing English.
///
/// `written` is how many facts the cycle wrote, for a cycle that counts them.
pub fn record_scan_cycle(
    cycle: BackendCycle,
    elapsed: std::time::Duration,
    outcome: Result<Option<usize>, String>,
) {
    let label = adelie_telemetry::metrics::Label::new(
        "outcome",
        if outcome.is_ok() { "ok" } else { "error" },
    );
    adelie_telemetry::metrics::record_duration(cycle.metric(), elapsed, &[label]);
    let duration_ms = elapsed.as_millis() as u64;
    match outcome {
        Ok(written) => {
            if let Some(n) = written {
                adelie_telemetry::metrics::add("dreaming.facts.written", n as u64, &[]);
            }
            tracing::info!(
                cycle = cycle.as_label(),
                duration_ms,
                // `-` for a cycle that does not count facts, never `0`: the
                // same rule the token counts follow, for the same reason. A
                // zero here would read as "consolidation wrote nothing".
                facts_written = %match written {
                    Some(n) => n.to_string(),
                    None => "-".to_string(),
                },
                "backend cycle finished"
            );
        }
        Err(error) => tracing::warn!(
            cycle = cycle.as_label(),
            duration_ms,
            error = %error,
            "backend cycle failed"
        ),
    }
}

/// How many distinct label sets one metric may hold before the rest fold into
/// a single `cardinality=other` series.
///
/// The facade's own default is 64, which is right for a small binary and too
/// small here: the daemon fronts a fleet of MCP servers, and
/// `tool.call.duration` carries one label set per (tool, outcome) pair. Thirty
/// or so tools would exhaust the default and collapse the axis that names
/// which tool is slow - permanently, because the budget has no eviction.
///
/// Not a licence to label with anything. Every label the daemon records is
/// still bounded at its call site; this only sizes the ceiling to what a
/// bounded set legitimately needs.
pub const CARDINALITY_CAP: usize = 512;

/// The daemon's telemetry configuration.
pub fn config() -> Config {
    Config::new(SERVICE_NAME)
        .with_default_filter(DEFAULT_FILTER)
        .with_cardinality_cap(CARDINALITY_CAP)
        // A closing span writes how long it was open. That is what makes turn
        // timing readable in `kubectl logs` or `journalctl`, where there is no
        // trace backend to open.
        .with_span_close_events(true)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;
    use std::sync::{Mutex, OnceLock};

    use super::*;

    /// The installed guard, kept for the life of the test binary.
    ///
    /// Telemetry installs once per process. Dropping the guard would stop the
    /// metrics summary thread underneath whichever test runs next, so the
    /// first install is parked here instead.
    static INSTALLED: OnceLock<adelie_telemetry::Guard> = OnceLock::new();

    /// Held while a test redirects the process's stderr. File descriptor 2 is
    /// process-wide, so two tests redirecting it at once would each capture
    /// the other's output.
    static STDERR: Mutex<()> = Mutex::new(());

    /// A buffer a `fmt` layer can write into.
    #[derive(Clone, Default)]
    struct CapturedLog(std::sync::Arc<Mutex<Vec<u8>>>);

    impl CapturedLog {
        fn text(&self) -> String {
            let bytes = self.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
            String::from_utf8(bytes).expect("captured log output is UTF-8")
        }
    }

    impl Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn install() {
        INSTALLED.get_or_init(|| {
            adelie_telemetry::init(config()).expect("install the daemon's telemetry")
        });
    }

    /// Restores the process's stderr when it is dropped.
    ///
    /// A guard rather than a pair of calls around `body`: a panic inside
    /// `body` would skip a plain restore, and the process would then run every
    /// later test with file descriptor 2 pointing at an orphaned temporary
    /// file. That swallows the panic's own message, so the test that broke
    /// would report a failure with nothing to read.
    struct RestoreStderr(libc::c_int);

    impl Drop for RestoreStderr {
        fn drop(&mut self) {
            std::io::stderr().flush().ok();
            // SAFETY: `self.0` is the descriptor `dup` returned in
            // `capture_stderr` and has not been closed. `dup2` puts it back
            // over descriptor 2 and `close` releases the duplicate, which is
            // not used again.
            unsafe {
                libc::dup2(self.0, libc::STDERR_FILENO);
                libc::close(self.0);
            }
        }
    }

    /// Run `body` with the process's stderr redirected into a buffer, and
    /// return what was written.
    ///
    /// The subscriber writes to `std::io::stderr`, which resolves file
    /// descriptor 2 at write time, so redirecting the descriptor captures it.
    /// Nothing short of this observes the real writer: a layer built over a
    /// test buffer would prove only that the test buffer works.
    ///
    /// The capture is not exclusive. Descriptor 2 belongs to the process, and
    /// the mutex below only orders one `capture_stderr` against another - it
    /// cannot stop another test, or a background thread, writing to stderr
    /// while the swap is in place. So assert on what must be present, never on
    /// what must be absent.
    fn capture_stderr<R>(body: impl FnOnce() -> R) -> (R, String) {
        let _serialized = STDERR.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = tempfile::tempfile().expect("a temporary file for stderr");

        // SAFETY: `dup` on the process's own stderr, to keep a handle on it
        // while descriptor 2 points elsewhere. `RestoreStderr` puts it back.
        let saved = unsafe { libc::dup(libc::STDERR_FILENO) };
        assert!(saved >= 0, "duplicate the real stderr");
        let _restore = RestoreStderr(saved);

        assert!(
            // SAFETY: `dup2` of an open descriptor over the process's own stderr.
            // The guard above (`RestoreStderr`) restores it, even on the
            // panicking path. `file` is guaranteed to be a valid open file.
            unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) } >= 0,
            "point stderr at the temporary file"
        );

        let result = body();
        drop(_restore);

        file.seek(SeekFrom::Start(0)).expect("rewind the capture");
        let mut captured = String::new();
        file.read_to_string(&mut captured)
            .expect("read the capture back");
        (result, captured)
    }

    #[test]
    fn telemetry_init_is_idempotent() {
        install();
        // A library hosted in this process, or a second call left behind by a
        // refactor, must not be able to abort the daemon. The old
        // `tracing_subscriber::fmt().init()` panicked here.
        let second = adelie_telemetry::init(config());
        assert!(
            second.is_ok(),
            "a second install must be a no-op, not a failure: {:?}",
            second.err()
        );
    }

    #[test]
    fn span_close_lines_carry_duration() {
        install();
        // An ERROR span so the assertion holds under the default filter as
        // well as under any RUST_LOG a developer has exported.
        let (_, captured) = capture_stderr(|| {
            let span = tracing::error_span!("telemetry_probe");
            span.in_scope(|| {});
            drop(span);
        });

        assert!(
            captured.contains("telemetry_probe"),
            "the closing span must write a line\n--- stderr ---\n{captured}"
        );
        assert!(
            captured.contains("time.busy"),
            "the line must carry how long the span was open\n--- stderr ---\n{captured}"
        );
    }

    #[test]
    fn the_metrics_summary_is_written_without_a_collector() {
        install();
        // The metrics facade is not a no-op with export off. It accumulates in
        // process and writes a summary, so a desktop install running a
        // default-feature build gets real numbers in its journal rather than
        // metrics being the one signal that is simply absent without a
        // collector.
        //
        // The summary is written at INFO. A run with no `RUST_LOG` at all
        // therefore does not show it, because this binary's default filter is
        // `error`; every shipped deployment sets `RUST_LOG=info`, which
        // `scripts/tests/systemd-logging.test.sh` holds the line on. The
        // capture below is at INFO for that reason, and not to make the
        // assertion easier.
        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(captured.clone())
            .with_ansi(false)
            .finish();

        let summary = tracing::subscriber::with_default(subscriber, || {
            adelie_telemetry::metrics::record_duration(
                "dreaming.scan.duration",
                std::time::Duration::from_millis(1_500),
                &[adelie_telemetry::metrics::Label::new("outcome", "ok")],
            );
            adelie_telemetry::metrics::global().dump_now()
        });

        assert!(
            !summary.is_empty(),
            "the facade must record what the daemon measures"
        );
        let text = captured.text();
        assert!(
            text.contains("metrics summary"),
            "the summary must be written\n--- captured at INFO ---\n{text}"
        );
        assert!(
            text.contains("dreaming.scan.duration"),
            "the summary must name the metric that was recorded\n--- captured at INFO ---\n{text}"
        );
    }

    #[test]
    fn dreaming_cycle_logs_duration_as_a_field() {
        // The duration used to be interpolated into the message, which reads
        // fine and cannot be grouped, sorted or aggregated by anything. A
        // field can.
        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(captured.clone())
            .with_ansi(false)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            record_scan_cycle(
                BackendCycle::Dreaming,
                std::time::Duration::from_millis(1_500),
                Ok(Some(7)),
            );
        });

        let text = captured.text();
        assert!(
            text.contains("duration_ms=1500"),
            "the cycle's duration must be a field\n--- captured at INFO ---\n{text}"
        );
        assert!(
            text.contains("facts_written=7"),
            "and so must what the cycle did\n--- captured at INFO ---\n{text}"
        );
        assert!(
            !text.contains("facts_written=0"),
            "a cycle that wrote seven facts must not also report zero\n\
             --- captured at INFO ---\n{text}"
        );
        assert!(
            text.contains("cycle=\"dreaming\""),
            "and which loop it was, so one query covers both\n\
             --- captured at INFO ---\n{text}"
        );
        assert!(
            !text.contains("1.5s") && !text.contains("in 1500ms"),
            "the duration must not also be interpolated into the sentence\n\
             --- captured at INFO ---\n{text}"
        );
    }

    #[test]
    fn a_failed_cycle_reports_the_same_duration_field() {
        // The failure path is a separate log site, so it needs its own case:
        // the arm that is easiest to leave interpolated is the one nobody
        // reads until something is wrong.
        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(captured.clone())
            .with_ansi(false)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            record_scan_cycle(
                BackendCycle::Consolidation,
                std::time::Duration::from_millis(900),
                Err("the backend refused".to_string()),
            );
        });

        let text = captured.text();
        assert!(
            text.contains("duration_ms=900"),
            "--- captured at INFO ---\n{text}"
        );
        assert!(
            text.contains("cycle=\"consolidation\""),
            "--- captured at INFO ---\n{text}"
        );
    }

    #[test]
    fn a_cycle_that_counts_nothing_reports_absence_not_zero() {
        // Consolidation does not count facts. Reporting `0` would read as "it
        // wrote nothing", which is a different and wrong statement.
        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(captured.clone())
            .with_ansi(false)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            record_scan_cycle(
                BackendCycle::Consolidation,
                std::time::Duration::from_millis(10),
                Ok(None),
            );
        });

        let text = captured.text();
        assert!(
            !text.contains("facts_written=0"),
            "--- captured at INFO ---\n{text}"
        );
        assert!(
            text.contains("facts_written=-"),
            "--- captured at INFO ---\n{text}"
        );
    }

    #[test]
    fn the_label_budget_fits_a_tool_fleet() {
        // `tool.call.duration` spends one label set per (tool, outcome) pair,
        // so the facade's default of 64 would collapse the tool axis on a
        // daemon fronting more than about thirty tools. The budget has no
        // eviction, so that collapse is permanent until the process restarts.
        assert!(
            config().cardinality_cap() >= 256,
            "the daemon's label budget must hold a tool fleet; got {}",
            config().cardinality_cap()
        );
    }

    #[test]
    fn the_default_filter_matches_the_previous_subscriber() {
        // `EnvFilter::from_default_env()`, which this replaced, falls back to
        // ERROR when RUST_LOG is unset. Adopting a shared crate must not
        // quietly make a desktop daemon start logging where it used to be
        // silent.
        assert_eq!(config().default_filter(), "error");
    }
}
