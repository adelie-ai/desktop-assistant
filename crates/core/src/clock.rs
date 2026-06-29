//! Canonical "now" snapshot, shared by the `builtin_sys_props` tool and the
//! ambient `[Now]` context block (see [`crate::context`]).
//!
//! Two features need the current wall-clock time and they must never disagree
//! about what "now" looks like — same timezone resolution, same formatting
//! rules. Centralising the capture here means there is exactly one code path
//! that reads the clock and renders it, so the tool's machine-readable property
//! sheet and the human-facing ambient line are always derived the same way.
//!
//! The tool lives in the `mcp-client` crate (which depends on `core`), so the
//! shared logic cannot live there; it lives here, in `core`, where both the
//! tool and the context assembler can reach it.

use chrono::{DateTime, Local, SecondsFormat, Utc};

/// A single captured instant, retained in both UTC and the machine's local
/// zone so callers can render whichever representation they need without
/// re-reading (and thereby risking disagreement with) the clock.
#[derive(Debug, Clone)]
pub struct NowSnapshot {
    utc: DateTime<Utc>,
    local: DateTime<Local>,
}

impl NowSnapshot {
    /// Capture the current instant. The single place the wall clock is read
    /// for both the sys-props tool and the ambient context block.
    pub fn now() -> Self {
        Self {
            utc: Utc::now(),
            local: Local::now(),
        }
    }

    /// Build a snapshot from explicit instants. Exists so tests can pin a
    /// deterministic time; production always goes through [`NowSnapshot::now`].
    pub fn from_parts(utc: DateTime<Utc>, local: DateTime<Local>) -> Self {
        Self { utc, local }
    }

    /// Seconds since the Unix epoch (UTC). Clamped to `0` for pre-epoch
    /// instants so the value is always a non-negative integer (the
    /// `builtin_sys_props` contract exposes `generated_at_epoch` as an
    /// unsigned number).
    pub fn epoch_secs(&self) -> u64 {
        u64::try_from(self.utc.timestamp()).unwrap_or(0)
    }

    /// RFC 3339 in UTC, second precision, `Z` suffix — e.g.
    /// `2026-06-28T18:32:07Z`.
    pub fn utc_rfc3339(&self) -> String {
        self.utc.to_rfc3339_opts(SecondsFormat::Secs, true)
    }

    /// RFC 3339 in the machine's local zone, second precision, with an
    /// explicit numeric offset — e.g. `2026-06-28T14:32:07-04:00`.
    pub fn local_rfc3339(&self) -> String {
        self.local.to_rfc3339_opts(SecondsFormat::Secs, false)
    }

    /// Timezone as numeric offset plus abbreviation — e.g. `-04:00 (EDT)`.
    pub fn timezone(&self) -> String {
        format!(
            "{} ({})",
            self.local.format("%:z"),
            self.local.format("%Z")
        )
    }

    /// Human-facing "now" line for the ambient `[Now]` context block — e.g.
    /// `Sunday, 2026-06-28, 2:32 PM EDT`. Prefers the zone abbreviation when
    /// the platform supplies a name, falling back to the numeric offset when
    /// `%Z` yields only an offset (some tz databases do).
    pub fn ambient_line(&self) -> String {
        let abbr = self.local.format("%Z").to_string();
        // Some platforms / tz databases render `%Z` as a numeric offset
        // (e.g. "+00:00") rather than a name; in that case the explicit
        // offset reads better than a bare number with no sign of being a zone.
        let looks_numeric = abbr.is_empty()
            || abbr
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | ':'));
        let zone = if looks_numeric {
            self.local.format("%:z").to_string()
        } else {
            abbr
        };
        format!(
            "{}, {} {}",
            self.local.format("%A, %Y-%m-%d"),
            self.local.format("%-I:%M %p"),
            zone
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn utc_fields_are_deterministic_from_parts() {
        let utc = Utc.with_ymd_and_hms(2026, 6, 28, 18, 32, 7).unwrap();
        let snap = NowSnapshot::from_parts(utc, utc.with_timezone(&Local));
        assert_eq!(snap.utc_rfc3339(), "2026-06-28T18:32:07Z");
        // 2026-06-28T18:32:07Z is well after the epoch, so the clamp is a no-op.
        assert_eq!(snap.epoch_secs(), u64::try_from(utc.timestamp()).unwrap());
    }

    #[test]
    fn pre_epoch_clamps_to_zero() {
        let utc = Utc.with_ymd_and_hms(1950, 1, 1, 0, 0, 0).unwrap();
        let snap = NowSnapshot::from_parts(utc, utc.with_timezone(&Local));
        assert_eq!(snap.epoch_secs(), 0);
    }

    #[test]
    fn ambient_line_is_well_formed() {
        // Uses the real local zone (can't pin %Z deterministically across
        // machines), so assert structural invariants rather than an exact
        // string: a weekday, an ISO date, a 12-hour meridiem, and — crucially —
        // that no format specifier leaked through unrendered.
        let snap = NowSnapshot::now();
        let line = snap.ambient_line();

        const DAYS: [&str; 7] = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        assert!(
            DAYS.iter().any(|d| line.contains(d)),
            "no weekday in {line:?}"
        );
        assert!(
            line.contains("AM") || line.contains("PM"),
            "no 12-hour meridiem in {line:?}"
        );
        // The ISO date derived from the same snapshot must appear verbatim.
        let date = snap.local.format("%Y-%m-%d").to_string();
        assert!(line.contains(&date), "date {date:?} missing from {line:?}");
        // A leftover '%' means a strftime specifier wasn't understood/rendered.
        assert!(
            !line.contains('%'),
            "unrendered format specifier in {line:?}"
        );
    }
}
