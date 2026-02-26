use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use crate::app::{App, InputMode};

const INPUT_VISIBLE_LINES: u16 = 4;
const INPUT_TOTAL_HEIGHT: u16 = INPUT_VISIBLE_LINES + 2; // +2 for borders
const COLOR_PANEL_BORDER: Color = Color::Rgb(82, 104, 173);
const COLOR_LIST_BORDER: Color = Color::Rgb(62, 125, 146);
const COLOR_INPUT_BORDER_IDLE: Color = Color::Rgb(109, 122, 143);
const COLOR_INPUT_BORDER_EDIT: Color = Color::Rgb(120, 183, 109);
const COLOR_INPUT_BORDER_CREATE: Color = Color::Rgb(203, 152, 95);
const COLOR_LIST_HIGHLIGHT: Color = Color::Rgb(72, 102, 180);
const COLOR_LIST_HIGHLIGHT_FG: Color = Color::Rgb(245, 248, 255);
const COLOR_USER_PREFIX: Color = Color::Rgb(255, 189, 89);
const COLOR_ASSISTANT_PREFIX: Color = Color::Rgb(92, 206, 154);
const COLOR_ASSISTANT_STREAMING: Color = Color::Rgb(132, 218, 193);
const COLOR_STATUS_DIM: Color = Color::Rgb(143, 153, 174);
const COLOR_COUNT_DIM: Color = Color::Rgb(124, 132, 148);

fn mode_chip_style(mode: &InputMode) -> Style {
    match mode {
        InputMode::Normal => Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(122, 163, 255))
            .add_modifier(Modifier::BOLD),
        InputMode::Editing => Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(120, 214, 118))
            .add_modifier(Modifier::BOLD),
        InputMode::CreatingConversation => Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(238, 179, 107))
            .add_modifier(Modifier::BOLD),
    }
}

fn split_display_lines(content: &str) -> Vec<String> {
    content
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .split('\n')
        .map(str::to_string)
        .collect()
}

pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(f.area());

    draw_conversation_list(f, app, chunks[0]);
    draw_chat_panel(f, app, chunks[1]);
}

fn draw_conversation_list(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let items: Vec<ListItem> = app
        .conversations
        .iter()
        .map(|c| {
            ListItem::new(Line::from(vec![
                Span::styled(c.title.as_str(), Style::default().fg(Color::White)),
                Span::styled(
                    format!(" ({})", c.message_count),
                    Style::default().fg(COLOR_COUNT_DIM),
                ),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(COLOR_LIST_BORDER))
                .title(Line::from(Span::styled(
                    "Conversations",
                    Style::default()
                        .fg(Color::Rgb(136, 214, 240))
                        .add_modifier(Modifier::BOLD),
                ))),
        )
        .highlight_style(
            Style::default()
                .bg(COLOR_LIST_HIGHLIGHT)
                .fg(COLOR_LIST_HIGHLIGHT_FG)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    state.select(app.selected_conversation);
    f.render_stateful_widget(list, area, &mut state);
}

fn draw_chat_panel(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(INPUT_TOTAL_HEIGHT),
            Constraint::Length(1),
        ])
        .split(area);

    draw_messages(f, app, chunks[0]);
    draw_input(f, app, chunks[1]);
    draw_status_bar(f, app, chunks[2]);
}

fn draw_messages(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let mut lines: Vec<Line> = Vec::new();

    if let Some(conv) = &app.current_conversation {
        for msg in &conv.messages {
            let (prefix, style) = match msg.role.as_str() {
                "user" => ("You: ", Style::default().fg(COLOR_USER_PREFIX)),
                "assistant" => ("Adele: ", Style::default().fg(COLOR_ASSISTANT_PREFIX)),
                _ => ("", Style::default()),
            };
            // Split content on newlines so ratatui renders them as separate lines
            let mut first = true;
            for text_line in split_display_lines(&msg.content) {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
                        Span::styled(text_line, style),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(Span::styled(text_line, style)));
                }
            }
            lines.push(Line::from("")); // spacing
        }

        // Show streaming buffer as in-progress assistant message
        if !app.streaming_buffer.is_empty() {
            let style = Style::default().fg(COLOR_ASSISTANT_STREAMING);
            let mut first = true;
            for text_line in split_display_lines(&app.streaming_buffer) {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled("Adele: ", style.add_modifier(Modifier::BOLD)),
                        Span::styled(text_line, style),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(Span::styled(text_line, style)));
                }
            }
            // Cursor on last line
            if let Some(last) = lines.last_mut() {
                last.spans.push(Span::styled("▌", style));
            }
        }
    } else {
        lines.push(Line::from("Press 'n' to create a new conversation."));
    }

    let chat_title = app
        .current_conversation
        .as_ref()
        .map(|conv| conv.title.as_str())
        .unwrap_or("Chat");
    let title = if app.scroll_offset > 0 {
        format!("{chat_title} (Ctrl+u/d scroll, Ctrl+e bottom)")
    } else {
        chat_title.to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(COLOR_PANEL_BORDER))
        .title(Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::Rgb(166, 182, 255))
                .add_modifier(Modifier::BOLD),
        )));
    let inner_width = block.inner(area).width;
    let visible_height = block.inner(area).height;

    let messages = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    // Use ratatui's own line_count for accurate wrapped height
    let total_height = messages.line_count(inner_width) as u16;
    let max_scroll = total_height.saturating_sub(visible_height);
    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let messages = messages.scroll((scroll, 0));
    f.render_widget(messages, area);
}

fn draw_input(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let wrap_width = usize::from(area.width.saturating_sub(2)).max(1);
    app.rewrap_textarea_to_width(wrap_width);

    let (title, border_color) = match app.mode {
        InputMode::Normal => ("Input (press 'i' to edit)", COLOR_INPUT_BORDER_IDLE),
        InputMode::Editing => (
            "Input (Esc cancel, Enter send, Shift+Enter newline)",
            COLOR_INPUT_BORDER_EDIT,
        ),
        InputMode::CreatingConversation => (
            "New conversation title (Enter to create, Esc to cancel)",
            COLOR_INPUT_BORDER_CREATE,
        ),
    };

    app.textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Line::from(Span::styled(
                title,
                Style::default().fg(Color::Rgb(216, 223, 236)),
            ))),
    );

    f.render_widget(&app.textarea, area);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let mode_str = match app.mode {
        InputMode::Normal => "NORMAL",
        InputMode::Editing => "EDITING",
        InputMode::CreatingConversation => "CREATING",
    };

    let status = Paragraph::new(Line::from(vec![
        Span::styled(format!(" [{mode_str}] "), mode_chip_style(&app.mode)),
        Span::styled(" • ", Style::default().fg(COLOR_STATUS_DIM)),
        Span::styled(
            app.status_message.as_str(),
            Style::default().fg(Color::White),
        ),
    ]));

    f.render_widget(status, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{ChatMessage, ConversationDetail, ConversationSummary};
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn draw_empty_app_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_with_conversations_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.set_conversations(vec![
            ConversationSummary {
                id: "1".into(),
                title: "Chat 1".into(),
                message_count: 3,
            },
            ConversationSummary {
                id: "2".into(),
                title: "Chat 2".into(),
                message_count: 0,
            },
        ]);
        app.selected_conversation = Some(0);
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_with_messages_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "1".into(),
            title: "Test".into(),
            messages: vec![
                ChatMessage {
                    role: "user".into(),
                    content: "Hello".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "Hi there!".into(),
                },
            ],
        });
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_with_streaming_buffer_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.current_conversation = Some(ConversationDetail {
            id: "1".into(),
            title: "Test".into(),
            messages: vec![],
        });
        app.start_streaming("req1".into());
        app.receive_chunk("req1", "Partial response...");
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_in_editing_mode_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.enter_editing_mode();
        app.textarea.insert_str("typing something");
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_in_creating_mode_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.enter_creating_conversation_mode();
        app.textarea.insert_str("New Chat");
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_with_status_message_does_not_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.status_message = "Error: connection lost".into();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }

    #[test]
    fn draw_small_terminal_does_not_panic() {
        let backend = TestBackend::new(20, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
    }
}
