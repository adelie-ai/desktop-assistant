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
    InsertChar(char),
    DeleteChar,
    SubmitTitle,
    ScrollUp,
    ScrollDown,
    ScrollToBottom,
}

pub fn handle_key_event(key: KeyEvent, mode: &InputMode) -> Option<Action> {
    // Ignore key events with modifier keys (except shift for typing)
    if key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::META)
    {
        return None;
    }

    match mode {
        InputMode::Normal => match key.code {
            KeyCode::Char('q') => Some(Action::Quit),
            KeyCode::Char('j') | KeyCode::Down => Some(Action::NextConversation),
            KeyCode::Char('k') | KeyCode::Up => Some(Action::PreviousConversation),
            KeyCode::Enter => Some(Action::OpenConversation),
            KeyCode::Char('d') => Some(Action::DeleteConversation),
            KeyCode::Char('n') => Some(Action::NewConversation),
            KeyCode::Char('i') => Some(Action::EnterEditMode),
            KeyCode::PageUp => Some(Action::ScrollUp),
            KeyCode::PageDown => Some(Action::ScrollDown),
            KeyCode::End => Some(Action::ScrollToBottom),
            _ => None,
        },
        InputMode::Editing => match key.code {
            KeyCode::Esc => Some(Action::ExitEditMode),
            KeyCode::Enter => Some(Action::SubmitPrompt),
            KeyCode::Backspace => Some(Action::DeleteChar),
            KeyCode::PageUp => Some(Action::ScrollUp),
            KeyCode::PageDown => Some(Action::ScrollDown),
            KeyCode::End => Some(Action::ScrollToBottom),
            KeyCode::Char(c) => Some(Action::InsertChar(c)),
            _ => None,
        },
        InputMode::CreatingConversation => match key.code {
            KeyCode::Esc => Some(Action::ExitEditMode),
            KeyCode::Enter => Some(Action::SubmitTitle),
            KeyCode::Backspace => Some(Action::DeleteChar),
            KeyCode::Char(c) => Some(Action::InsertChar(c)),
            _ => None,
        },
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
    fn editing_backspace_deletes() {
        assert_eq!(
            handle_key_event(key(KeyCode::Backspace), &InputMode::Editing),
            Some(Action::DeleteChar)
        );
    }

    #[test]
    fn editing_char_inserts() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('a')), &InputMode::Editing),
            Some(Action::InsertChar('a'))
        );
    }

    #[test]
    fn editing_space_inserts() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char(' ')), &InputMode::Editing),
            Some(Action::InsertChar(' '))
        );
    }

    #[test]
    fn editing_unknown_key_ignored() {
        assert_eq!(
            handle_key_event(key(KeyCode::Tab), &InputMode::Editing),
            None
        );
    }

    #[test]
    fn editing_ctrl_modifier_ignored() {
        assert_eq!(
            handle_key_event(
                key_with_mod(KeyCode::Char('c'), KeyModifiers::CONTROL),
                &InputMode::Editing
            ),
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
    fn creating_backspace_deletes() {
        assert_eq!(
            handle_key_event(key(KeyCode::Backspace), &InputMode::CreatingConversation),
            Some(Action::DeleteChar)
        );
    }

    #[test]
    fn creating_char_inserts() {
        assert_eq!(
            handle_key_event(key(KeyCode::Char('Z')), &InputMode::CreatingConversation),
            Some(Action::InsertChar('Z'))
        );
    }

    #[test]
    fn creating_unknown_key_ignored() {
        assert_eq!(
            handle_key_event(key(KeyCode::F(1)), &InputMode::CreatingConversation),
            None
        );
    }

    // --- Scroll tests ---

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

    #[test]
    fn normal_end_scrolls_to_bottom() {
        assert_eq!(
            handle_key_event(key(KeyCode::End), &InputMode::Normal),
            Some(Action::ScrollToBottom)
        );
    }

    #[test]
    fn editing_pageup_scrolls_up() {
        assert_eq!(
            handle_key_event(key(KeyCode::PageUp), &InputMode::Editing),
            Some(Action::ScrollUp)
        );
    }

    #[test]
    fn editing_pagedown_scrolls_down() {
        assert_eq!(
            handle_key_event(key(KeyCode::PageDown), &InputMode::Editing),
            Some(Action::ScrollDown)
        );
    }

    #[test]
    fn editing_end_scrolls_to_bottom() {
        assert_eq!(
            handle_key_event(key(KeyCode::End), &InputMode::Editing),
            Some(Action::ScrollToBottom)
        );
    }
}
