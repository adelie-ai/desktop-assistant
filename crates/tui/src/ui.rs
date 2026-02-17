use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use crate::app::{App, InputMode};

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
        .map(|c| ListItem::new(Line::from(format!("{} ({})", c.title, c.message_count))))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Conversations"),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let mut state = ListState::default();
    state.select(app.selected_conversation);
    f.render_stateful_widget(list, area, &mut state);
}

fn draw_chat_panel(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    // Dynamic input height: line count + 2 for borders, min 3, max 10
    let line_count = app.textarea.lines().len() as u16;
    let input_height = (line_count + 2).clamp(3, 10);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(input_height),
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
                "user" => ("You: ", Style::default().fg(Color::Cyan)),
                "assistant" => ("AI: ", Style::default().fg(Color::Green)),
                _ => ("", Style::default()),
            };
            // Split content on newlines so ratatui renders them as separate lines
            let mut first = true;
            for text_line in msg.content.split('\n') {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
                        Span::styled(text_line.to_string(), style),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(Span::styled(text_line.to_string(), style)));
                }
            }
            lines.push(Line::from("")); // spacing
        }

        // Show streaming buffer as in-progress assistant message
        if !app.streaming_buffer.is_empty() {
            let style = Style::default().fg(Color::Yellow);
            let mut first = true;
            for text_line in app.streaming_buffer.split('\n') {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled("AI: ", style.add_modifier(Modifier::BOLD)),
                        Span::styled(text_line.to_string(), style),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(Span::styled(text_line.to_string(), style)));
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

    let block = Block::default().borders(Borders::ALL).title(title);
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
    let title = match app.mode {
        InputMode::Normal => "Input (press 'i' to edit)",
        InputMode::Editing => "Input (Esc cancel, Enter send, Alt+Enter newline)",
        InputMode::CreatingConversation => {
            "New conversation title (Enter to create, Esc to cancel)"
        }
    };

    app.textarea
        .set_block(Block::default().borders(Borders::ALL).title(title));

    f.render_widget(&app.textarea, area);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let mode_str = match app.mode {
        InputMode::Normal => "NORMAL",
        InputMode::Editing => "EDITING",
        InputMode::CreatingConversation => "CREATING",
    };

    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(" [{mode_str}] "),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" {}", app.status_message)),
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
