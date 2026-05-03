//! PAM-bridge for OS-user password authentication on Linux. Wraps
//! libpam's `pam_start` / `pam_authenticate` / `pam_acct_mgmt` /
//! `pam_end` and provides the conversation callback that supplies the
//! caller-provided password to PAM challenges.
//!
//! Extracted from `config.rs` (#41) — the FFI shape is the most
//! security-sensitive code in the daemon and benefits from living in
//! its own file. Behaviour is unchanged.

#![cfg(target_os = "linux")]

use std::ffi::CString;
use std::ptr;

use anyhow::anyhow;
use libc::{c_char, c_int, c_void};

const PAM_SUCCESS: c_int = 0;
const PAM_PROMPT_ECHO_OFF: c_int = 1;
const PAM_PROMPT_ECHO_ON: c_int = 2;
const PAM_ERROR_MSG: c_int = 3;
const PAM_TEXT_INFO: c_int = 4;
const PAM_CONV_ERR: c_int = 19;

#[repr(C)]
struct PamMessage {
    msg_style: c_int,
    msg: *const c_char,
}

#[repr(C)]
struct PamResponse {
    resp: *mut c_char,
    resp_retcode: c_int,
}

#[repr(C)]
struct PamConv {
    conv: Option<
        extern "C" fn(
            num_msg: c_int,
            msg: *mut *const PamMessage,
            resp: *mut *mut PamResponse,
            appdata_ptr: *mut c_void,
        ) -> c_int,
    >,
    appdata_ptr: *mut c_void,
}

#[repr(C)]
struct PamHandle(c_void);

#[link(name = "pam")]
unsafe extern "C" {
    fn pam_start(
        service_name: *const c_char,
        user: *const c_char,
        pam_conv: *const PamConv,
        pamh: *mut *mut PamHandle,
    ) -> c_int;
    fn pam_end(pamh: *mut PamHandle, pam_status: c_int) -> c_int;
    fn pam_authenticate(pamh: *mut PamHandle, flags: c_int) -> c_int;
    fn pam_acct_mgmt(pamh: *mut PamHandle, flags: c_int) -> c_int;
}

struct ConvData {
    password: *const c_char,
}

/// Free a PAM response array allocated by `libc::calloc` inside the
/// `conversation` callback below.
///
/// # Safety
///
/// The caller must ensure that `responses` is either null or points to
/// a `count`-element array of [`PamResponse`] previously allocated by
/// `libc::calloc` in this module — i.e. only the `conversation` callback
/// (or its error paths) should call this. Each `PamResponse::resp` field,
/// if non-null, must point to a `libc::strdup`'d C string. This module's
/// callbacks uphold both invariants. Calling this with arbitrary
/// pointers is undefined behaviour.
unsafe fn free_responses(responses: *mut PamResponse, count: c_int) {
    // SAFETY: per the function contract, `responses` is either null
    // (early-returned below) or a calloc'd array of `count` PamResponses.
    // The pointer arithmetic stays within bounds and each `.resp` was
    // allocated via `libc::strdup`, so `libc::free` is the matching
    // deallocator.
    if responses.is_null() || count <= 0 {
        return;
    }
    for i in 0..count {
        let entry = unsafe { responses.add(i as usize) };
        if unsafe { !(*entry).resp.is_null() } {
            // Zero the password bytes before handing the buffer
            // back to libc (#38). The buffer was allocated via
            // `libc::strdup` from a NUL-terminated C string, so
            // `strlen` gives the byte count we need to wipe.
            unsafe {
                let len = libc::strlen((*entry).resp);
                if len > 0 {
                    libc::memset((*entry).resp.cast(), 0, len);
                }
                libc::free((*entry).resp.cast());
            }
        }
    }
    unsafe {
        libc::free(responses.cast());
    }
}

extern "C" fn conversation(
    num_msg: c_int,
    msg: *mut *const PamMessage,
    resp: *mut *mut PamResponse,
    appdata_ptr: *mut c_void,
) -> c_int {
    if num_msg <= 0 || msg.is_null() || resp.is_null() || appdata_ptr.is_null() {
        return PAM_CONV_ERR;
    }

    // SAFETY: calloc allocates contiguous zeroed memory for response entries.
    let responses = unsafe {
        libc::calloc(num_msg as usize, std::mem::size_of::<PamResponse>()) as *mut PamResponse
    };
    if responses.is_null() {
        return PAM_CONV_ERR;
    }

    for i in 0..num_msg {
        // SAFETY: msg points to num_msg entries provided by libpam.
        let message_ptr = unsafe { *msg.add(i as usize) };
        if message_ptr.is_null() {
            // SAFETY: responses was allocated above.
            unsafe { free_responses(responses, num_msg) };
            return PAM_CONV_ERR;
        }

        // SAFETY: response slot is within allocated array.
        let response = unsafe { responses.add(i as usize) };
        // SAFETY: appdata_ptr points to ConvData set during pam_start.
        let conv_data = unsafe { &*(appdata_ptr as *const ConvData) };
        // SAFETY: message_ptr is validated above.
        let style = unsafe { (*message_ptr).msg_style };

        match style {
            PAM_PROMPT_ECHO_OFF | PAM_PROMPT_ECHO_ON => {
                // SAFETY: password pointer lives for entire pam_authenticate call.
                let duplicated = unsafe { libc::strdup(conv_data.password) };
                if duplicated.is_null() {
                    // SAFETY: responses was allocated above.
                    unsafe { free_responses(responses, num_msg) };
                    return PAM_CONV_ERR;
                }
                // SAFETY: writing into response slot is valid.
                unsafe {
                    (*response).resp = duplicated;
                    (*response).resp_retcode = 0;
                }
            }
            PAM_ERROR_MSG | PAM_TEXT_INFO => {
                // SAFETY: writing into response slot is valid.
                unsafe {
                    (*response).resp = ptr::null_mut();
                    (*response).resp_retcode = 0;
                }
            }
            _ => {
                // SAFETY: responses was allocated above.
                unsafe { free_responses(responses, num_msg) };
                return PAM_CONV_ERR;
            }
        }
    }

    // SAFETY: resp is valid output pointer from libpam.
    unsafe {
        *resp = responses;
    }
    PAM_SUCCESS
}

pub(super) fn authenticate(username: &str, password: &str) -> anyhow::Result<bool> {
    let service_name = CString::new("login")
        .map_err(|error| anyhow!("invalid PAM service name bytes: {error}"))?;
    let username_c =
        CString::new(username).map_err(|error| anyhow!("invalid username bytes: {error}"))?;
    // Hold the password as a `Zeroizing<Vec<u8>>` so the buffer is
    // wiped before the allocator reclaims it (#38). The PAM
    // conversation callback `strdup`s this into a separate heap
    // buffer, which `free_responses` zeroes separately.
    let password_bytes: zeroize::Zeroizing<Vec<u8>> = zeroize::Zeroizing::new(
        CString::new(password)
            .map_err(|error| anyhow!("invalid password bytes: {error}"))?
            .into_bytes_with_nul(),
    );

    let mut handle: *mut PamHandle = ptr::null_mut();
    // Box the ConvData so its address is heap-stable and cannot be
    // invalidated by stack moves.  The box is kept alive until after
    // pam_end, guaranteeing the pointer remains valid for all callbacks.
    let conv_data = Box::new(ConvData {
        password: password_bytes.as_ptr() as *const c_char,
    });
    let conversation = PamConv {
        conv: Some(conversation),
        appdata_ptr: Box::into_raw(conv_data).cast(),
    };

    // SAFETY: all pointers passed are valid for this call.
    let start = unsafe {
        pam_start(
            service_name.as_ptr(),
            username_c.as_ptr(),
            &conversation,
            &mut handle,
        )
    };
    if start != PAM_SUCCESS {
        return Ok(false);
    }

    // SAFETY: handle is initialized by successful pam_start.
    let mut status = unsafe { pam_authenticate(handle, 0) };
    if status == PAM_SUCCESS {
        // SAFETY: handle remains valid until pam_end.
        status = unsafe { pam_acct_mgmt(handle, 0) };
    }
    // SAFETY: handle came from pam_start and must be terminated once.
    unsafe {
        pam_end(handle, status);
    }
    // SAFETY: reclaim the boxed ConvData that was leaked via Box::into_raw.
    // This must happen after pam_end so the pointer remains valid for all
    // PAM callbacks.
    unsafe {
        drop(Box::from_raw(conversation.appdata_ptr as *mut ConvData));
    }

    Ok(status == PAM_SUCCESS)
}
