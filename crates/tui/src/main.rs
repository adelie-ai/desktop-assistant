mod app;
mod dbus_client;
mod keys;
mod ui;

use std::io;

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use app::App;
use dbus_client::{DbusClient, SignalEvent};
use keys::{Action, handle_key_event};

#[tokio::main]
async fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::new();

    // Connect to D-Bus
    let (client, mut signal_rx) = match DbusClient::connect().await {
        Ok(c) => {
            match c.list_conversations().await {
                Ok(convs) => app.set_conversations(convs),
                Err(e) => app.status_message = format!("Error loading conversations: {e}"),
            }
            let rx = match c.subscribe_signals().await {
                Ok(rx) => rx,
                Err(e) => {
                    app.status_message = format!("Signal setup error: {e}");
                    mpsc::unbounded_channel().1
                }
            };
            (Some(c), rx)
        }
        Err(e) => {
            app.status_message = format!("D-Bus connection failed: {e}");
            (None, mpsc::unbounded_channel().1)
        }
    };

    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        if app.should_quit {
            break;
        }

        tokio::select! {
            Some(Ok(evt)) = event_stream.next() => {
                if let Event::Key(key) = evt {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    if let Some(action) = handle_key_event(key, &app.mode) {
                        handle_action(&mut app, &client, action).await;
                    }
                }
            }
            Some(signal) = signal_rx.recv() => {
                match signal {
                    SignalEvent::Chunk { request_id, chunk } => {
                        app.receive_chunk(&request_id, &chunk);
                    }
                    SignalEvent::Complete { request_id, full_response } => {
                        app.complete_streaming(&request_id, &full_response);
                    }
                    SignalEvent::Error { request_id, error } => {
                        app.streaming_error(&request_id, &error);
                    }
                }
            }
        }
    }

    Ok(())
}

async fn handle_action(app: &mut App, client: &Option<DbusClient>, action: Action) {
    match action {
        Action::Quit => app.quit(),
        Action::NextConversation => app.next_conversation(),
        Action::PreviousConversation => app.previous_conversation(),
        Action::OpenConversation => {
            if let (Some(client), Some(id)) = (client.as_ref(), app.selected_conversation_id()) {
                let id = id.to_string();
                match client.get_conversation(&id).await {
                    Ok(detail) => {
                        app.load_conversation(detail);
                        app.enter_editing_mode();
                    }
                    Err(e) => app.status_message = format!("Error: {e}"),
                }
            }
        }
        Action::DeleteConversation => {
            if let Some(id) = app.delete_selected_conversation() {
                if let Some(client) = client.as_ref() {
                    if let Err(e) = client.delete_conversation(&id).await {
                        app.status_message = format!("Delete error: {e}");
                    }
                }
            }
        }
        Action::NewConversation => app.enter_creating_conversation_mode(),
        Action::EnterEditMode => {
            if app.current_conversation.is_some() {
                app.enter_editing_mode();
            } else {
                app.status_message = "Open a conversation first (Enter) or create one (n)".into();
            }
        }
        Action::ExitEditMode => app.enter_normal_mode(),
        Action::SubmitPrompt => {
            if let Some((conv_id, prompt)) = app.submit_prompt() {
                if let Some(client) = client.as_ref() {
                    match client.send_prompt(&conv_id, &prompt).await {
                        Ok(request_id) => app.start_streaming(request_id),
                        Err(e) => app.status_message = format!("Send error: {e}"),
                    }
                }
            }
        }
        Action::ScrollUp => app.scroll_up(5),
        Action::ScrollDown => app.scroll_down(5),
        Action::ScrollToBottom => app.scroll_to_bottom(),
        Action::InsertChar(c) => app.insert_char(c),
        Action::DeleteChar => app.delete_char(),
        Action::SubmitTitle => {
            if let Some(title) = app.submit_new_conversation_title() {
                if let Some(client) = client.as_ref() {
                    match client.create_conversation(&title).await {
                        Ok(id) => {
                            // Refresh list and auto-open the new conversation
                            match client.list_conversations().await {
                                Ok(convs) => {
                                    let new_idx = convs.iter().position(|c| c.id == id);
                                    app.set_conversations(convs);
                                    if let Some(idx) = new_idx {
                                        app.selected_conversation = Some(idx);
                                    }
                                }
                                Err(e) => app.status_message = format!("Error refreshing: {e}"),
                            }
                            match client.get_conversation(&id).await {
                                Ok(detail) => {
                                    app.load_conversation(detail);
                                    app.enter_editing_mode();
                                }
                                Err(e) => app.status_message = format!("Error opening: {e}"),
                            }
                        }
                        Err(e) => app.status_message = format!("Create error: {e}"),
                    }
                }
            }
        }
    }
}
