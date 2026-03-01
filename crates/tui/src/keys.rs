use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::InputMode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Quit,
    NextConversation,
    PreviousConversation,
    OpenConversation,
    DeleteConversation,
    NewConversation,
    EnterEditMode,
    ExitEditMode,
    SubmitPrompt,
    SubmitTitle,
    InsertNewline,
    ScrollUp,
    ScrollDown,
    ScrollToBottom,
}

/// Handle key events that we intercept before passing to textarea.
/// Returns None for keys that should be forwarded to textarea.input().
pub fn handle_key_event(key: KeyEvent, mode: &InputMode) -> Option<Action> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // Ctrl+u / Ctrl+d / Ctrl+e for scrolling — works in all modes
    if ctrl {
        if matches!(mode, InputMode::Editing) && matches!(key.code, KeyCode::Char('j')) {
            return Some(Action::InsertNewline);
        }
        return match key.code {
            KeyCode::Char('u') => Some(Action::ScrollUp),
            KeyCode::Char('d') => Some(Action::ScrollDown),
            KeyCode::Char('e') => Some(Action::ScrollToBottom),
            _ => None,
        };
    }

    match mode {
        InputMode::Normal => {
            // Ignore Alt/Meta combos in Normal mode
            if alt || key.modifiers.intersects(KeyModifiers::META) {
                return None;
            }
            if key.code == KeyCode::Enter {
                return Some(Action::OpenConversation);
            }
            match key.code {
                KeyCode::Char('q') => Some(Action::Quit),
                KeyCode::Char('j') | KeyCode::Down => Some(Action::NextConversation),
                KeyCode::Char('k') | KeyCode::Up => Some(Action::PreviousConversation),
                KeyCode::Char('d') => Some(Action::DeleteConversation),
                KeyCode::Char('n') => Some(Action::NewConversation),
                KeyCode::Char('i') => Some(Action::EnterEditMode),
                KeyCode::PageUp => Some(Action::ScrollUp),
                KeyCode::PageDown => Some(Action::ScrollDown),
                KeyCode::End => Some(Action::ScrollToBottom),
                _ => None,
            }
        }
        InputMode::Editing => {
            // Shift+Enter inserts a newline while plain Enter submits.
            match key.code {
                KeyCode::Enter => {
                    if key.modifiers.contains(KeyModifiers::SHIFT) {
                        return Some(Action::InsertNewline);
                    }
                    if key.modifiers.is_empty() {
                        return Some(Action::SubmitPrompt);
                    }
                    return None;
                }
                // Preserve terminal-provided newline chars by forwarding them
                // to textarea.input(...), which keeps composer and payload in sync.
                KeyCode::Char('\n') | KeyCode::Char('\r') => Some(Action::InsertNewline),
                KeyCode::Esc => Some(Action::ExitEditMode),
                KeyCode::PageUp => Some(Action::ScrollUp),
                KeyCode::PageDown => Some(Action::ScrollDown),
                KeyCode::End if key.modifiers.contains(KeyModifiers::SHIFT) => {
                    Some(Action::ScrollToBottom)
                }
                // All other keys: return None so they get forwarded to textarea
                _ => None,
            }
        }
        InputMode::CreatingConversation => {
            if key.code == KeyCode::Enter {
                return Some(Action::SubmitTitle);
            }
            match key.code {
                KeyCode::Esc => Some(Action::ExitEditMode),
                // All other keys: return None so they get forwarded to textarea
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn key_with_mod(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    // --- Normal mode tests ---

    #[test]
    fn normal_q_quits() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('q')), &InputMode::Normal),
            Some(Action::Quit)
        );
    }

    #[test]
    fn normal_j_next() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('j')), &InputMode::Normal),
            Some(Action::NextConversation)
        );
    }

    #[test]
    fn normal_down_next() {
        assert_eq!(
            handle_key_event(key(KeyCode::Down), &InputMode::Normal),
            Some(Action::NextConversation)
        );
    }

    #[test]
    fn normal_k_previous() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('k')), &InputMode::Normal),
            Some(Action::PreviousConversation)
        );
    }

    #[test]
    fn normal_up_previous() {
        assert_eq!(
            handle_key_event(key(KeyCode::Up), &InputMode::Normal),
            Some(Action::PreviousConversation)
        );
    }

    #[test]
    fn normal_enter_opens() {
        assert_eq!(
            handle_key_event(key(KeyCode::Enter), &InputMode::Normal),
            Some(Action::OpenConversation)
        );
    }

    #[test]
    fn normal_char_newline_is_ignored() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('\n')), &InputMode::Normal),
            None
        );
    }

    #[test]
    fn normal_d_deletes() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('d')), &InputMode::Normal),
            Some(Action::DeleteConversation)
        );
    }

    #[test]
    fn normal_n_new_conversation() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('n')), &InputMode::Normal),
            Some(Action::NewConversation)
        );
    }

    #[test]
    fn normal_i_enter_edit() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('i')), &InputMode::Normal),
            Some(Action::EnterEditMode)
        );
    }

    #[test]
    fn normal_unknown_key_ignored() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('x')), &InputMode::Normal),
            None
        );
    }

    #[test]
    fn normal_ctrl_modifier_ignored() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('q'), KeyModifiers::CONTROL),
                &InputMode::Normal
            ),
            None
        );
    }

    #[test]
    fn normal_alt_modifier_ignored() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('j'), KeyModifiers::ALT),
                &InputMode::Normal
            ),
            None
        );
    }

    // --- Editing mode tests ---

    #[test]
    fn editing_escape_exits() {
        assert_eq!(
            handle_key_event(key(KeyCode::Esc), &InputMode::Editing),
            Some(Action::ExitEditMode)
        );
    }

    #[test]
    fn editing_enter_submits_prompt() {
        assert_eq!(
            handle_key_event(key(KeyCode::Enter), &InputMode::Editing),
            Some(Action::SubmitPrompt)
        );
    }

    #[test]
    fn editing_shift_enter_inserts_newline() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Enter, KeyModifiers::SHIFT),
                &InputMode::Editing
            ),
            Some(Action::InsertNewline)
        );
    }

    #[test]
    fn editing_newline_char_is_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('\n'), KeyModifiers::NONE),
                &InputMode::Editing
            ),
            Some(Action::InsertNewline)
        );
    }

    #[test]
    fn editing_carriage_return_char_is_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('\r'), KeyModifiers::NONE),
                &InputMode::Editing
            ),
            Some(Action::InsertNewline)
        );
    }

    #[test]
    fn editing_ctrl_j_inserts_newline() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('j'), KeyModifiers::CONTROL),
                &InputMode::Editing
            ),
            Some(Action::InsertNewline)
        );
    }

    #[test]
    fn editing_alt_enter_is_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Enter, KeyModifiers::ALT),
                &InputMode::Editing
            ),
            None
        );
    }

    #[test]
    fn editing_char_forwarded_to_textarea() {
        // Regular chars should return None so they get forwarded to textarea
        assert_eq!(
            handle_key_event(key(KeyCode::Char('a')), &InputMode::Editing),
            None
        );
    }

    #[test]
    fn editing_backspace_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(key(KeyCode::Backspace), &InputMode::Editing),
            None
        );
    }

    #[test]
    fn editing_arrows_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(key(KeyCode::Left), &InputMode::Editing),
            None
        );
        assert_eq!(
            handle_key_event(key(KeyCode::Right), &InputMode::Editing),
            None
        );
        assert_eq!(
            handle_key_event(key(KeyCode::Up), &InputMode::Editing),
            None
        );
        assert_eq!(
            handle_key_event(key(KeyCode::Down), &InputMode::Editing),
            None
        );
    }

    // --- CreatingConversation mode tests ---

    #[test]
    fn creating_escape_exits() {
        assert_eq!(
            handle_key_event(key(KeyCode::Esc), &InputMode::CreatingConversation),
            Some(Action::ExitEditMode)
        );
    }

    #[test]
    fn creating_enter_submits_title() {
        assert_eq!(
            handle_key_event(key(KeyCode::Enter), &InputMode::CreatingConversation),
            Some(Action::SubmitTitle)
        );
    }

    #[test]
    fn creating_char_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('Z')), &InputMode::CreatingConversation),
            None
        );
    }

    #[test]
    fn creating_unknown_key_forwarded_to_textarea() {
        assert_eq!(
            handle_key_event(key(KeyCode::F(1)), &InputMode::CreatingConversation),
            None
        );
    }

    // --- Scroll tests (Ctrl+u/d/e work in all modes) ---

    #[test]
    fn ctrl_u_scrolls_up() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('u'), KeyModifiers::CONTROL),
                &InputMode::Normal
            ),
            Some(Action::ScrollUp)
        );
    }

    #[test]
    fn ctrl_d_scrolls_down() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('d'), KeyModifiers::CONTROL),
                &InputMode::Normal
            ),
            Some(Action::ScrollDown)
        );
    }

    #[test]
    fn ctrl_e_scrolls_to_bottom() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('e'), KeyModifiers::CONTROL),
                &InputMode::Normal
            ),
            Some(Action::ScrollToBottom)
        );
    }

    #[test]
    fn ctrl_u_works_in_editing_mode() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('u'), KeyModifiers::CONTROL),
                &InputMode::Editing
            ),
            Some(Action::ScrollUp)
        );
    }

    #[test]
    fn ctrl_d_works_in_editing_mode() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('d'), KeyModifiers::CONTROL),
                &InputMode::Editing
            ),
            Some(Action::ScrollDown)
        );
    }

    #[test]
    fn ctrl_u_works_in_creating_mode() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('u'), KeyModifiers::CONTROL),
                &InputMode::CreatingConversation
            ),
            Some(Action::ScrollUp)
        );
    }

    #[test]
    fn normal_pageup_scrolls_up() {
        assert_eq!(
            handle_key_event(key(KeyCode::PageUp), &InputMode::Normal),
            Some(Action::ScrollUp)
        );
    }

    #[test]
    fn normal_pagedown_scrolls_down() {
        assert_eq!(
            handle_key_event(key(KeyCode::PageDown), &InputMode::Normal),
            Some(Action::ScrollDown)
        );
    }
}
