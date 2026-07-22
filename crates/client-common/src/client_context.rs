//! Best-effort resolution of the client's self-reported device/user context
//! (#549) and the gate that decides whether to attach it to the connect
//! handshake.
//!
//! The implementation is added in the accompanying commit; this file starts as
//! the failing spec (TDD).

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_api_model::ClientContext;

    fn full_ctx() -> ClientContext {
        ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada".into()),
            home_dir: Some("/home/ada".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("Ubuntu 24.04".into()),
        }
    }

    #[test]
    fn assemble_drops_absent_and_blank_fields_but_keeps_present_ones() {
        // A `None` field and a present-but-blank field are both treated as
        // absent; a present field is kept verbatim.
        let ctx = assemble_client_context(
            Some("Ada Lovelace".to_string()),
            None,
            Some("   ".to_string()), // blank -> dropped
            Some("analytical-engine".to_string()),
            None,
            Some(String::new()), // empty -> dropped
        );
        assert_eq!(ctx.real_name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(ctx.username, None);
        assert_eq!(ctx.home_dir, None);
        assert_eq!(ctx.hostname.as_deref(), Some("analytical-engine"));
        assert_eq!(ctx.timezone, None);
        assert_eq!(ctx.os, None);
    }

    #[test]
    fn assemble_all_absent_yields_empty_context() {
        let ctx = assemble_client_context(None, None, None, None, None, None);
        assert!(ctx.is_empty());
    }

    #[test]
    fn assemble_trims_surrounding_whitespace() {
        let ctx = assemble_client_context(
            None,
            Some("  ada  ".to_string()),
            None,
            None,
            Some(" Europe/London ".to_string()),
            None,
        );
        assert_eq!(ctx.username.as_deref(), Some("ada"));
        assert_eq!(ctx.timezone.as_deref(), Some("Europe/London"));
    }

    #[test]
    fn context_to_attach_is_none_when_sharing_disabled() {
        // With the setting off the resolver must not even run (no env/syscall
        // reads) and nothing is attached.
        let called = std::cell::Cell::new(false);
        let out = context_to_attach(false, || {
            called.set(true);
            full_ctx()
        });
        assert_eq!(out, None);
        assert!(
            !called.get(),
            "resolver must not run when sharing is disabled"
        );
    }

    #[test]
    fn context_to_attach_returns_resolved_when_enabled() {
        assert_eq!(context_to_attach(true, full_ctx), Some(full_ctx()));
    }

    #[test]
    fn context_to_attach_drops_an_empty_resolved_context() {
        // An all-absent resolved context is equivalent to attaching nothing.
        assert_eq!(context_to_attach(true, ClientContext::default), None);
    }

    #[test]
    fn resolve_client_context_never_panics() {
        // Best-effort contract: whatever the host provides, resolution returns a
        // (possibly empty) context and never panics.
        let _ = resolve_client_context();
    }
}
