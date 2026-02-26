mod app;
mod dbus_client;
mod keys;
mod ui;
mod ws_client;

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

use app::{App, ConversationDetail, ConversationSummary, InputMode};
use dbus_client::{DbusClient, SignalEvent, generate_ws_jwt};
use keys::{Action, handle_key_event};
use ws_client::WsClient;

const DEFAULT_TRANSPORT: &str = "ws";
const DEFAULT_WS_URL: &str = "ws://127.0.0.1:11339/ws";
const DEFAULT_WS_SUBJECT: &str = "desktop-tui";

enum TransportClient {
    Dbus(DbusClient),
    Ws(WsClient),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransportMode {
    Ws,
    Dbus,
}

fn resolve_transport_mode() -> TransportMode {
    match std::env::var("DESKTOP_ASSISTANT_TUI_TRANSPORT")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_TRANSPORT.to_string())
        .as_str()
    {
        "dbus" => TransportMode::Dbus,
        _ => TransportMode::Ws,
    }
}

fn resolve_ws_url() -> String {
    std::env::var("DESKTOP_ASSISTANT_TUI_WS_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_WS_URL.to_string())
}

fn resolve_ws_subject() -> String {
    std::env::var("DESKTOP_ASSISTANT_TUI_WS_SUBJECT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_WS_SUBJECT.to_string())
}

fn resolve_ws_jwt_from_env() -> Option<String> {
    std::env::var("DESKTOP_ASSISTANT_TUI_WS_JWT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

async fn connect_transport() -> Result<(TransportClient, mpsc::UnboundedReceiver<SignalEvent>)> {
    match resolve_transport_mode() {
        TransportMode::Dbus => {
            let client = DbusClient::connect().await?;
            let signal_rx = client.subscribe_signals().await?;
            Ok((TransportClient::Dbus(client), signal_rx))
        }
        TransportMode::Ws => {
            let ws_url = resolve_ws_url();
            let token = match resolve_ws_jwt_from_env() {
                Some(token) => token,
                None => {
                    let subject = resolve_ws_subject();
                    generate_ws_jwt(&subject).await?
                }
            };
            let (client, signal_rx) = WsClient::connect(&ws_url, &token).await?;
            Ok((TransportClient::Ws(client), signal_rx))
        }
    }
}

async fn client_list_conversations(client: &TransportClient) -> Result<Vec<ConversationSummary>> {
    match client {
        TransportClient::Dbus(client) => client.list_conversations().await,
        TransportClient::Ws(client) => client.list_conversations().await,
    }
}

async fn client_get_conversation(client: &TransportClient, id: &str) -> Result<ConversationDetail> {
    match client {
        TransportClient::Dbus(client) => client.get_conversation(id).await,
        TransportClient::Ws(client) => client.get_conversation(id).await,
    }
}

async fn client_create_conversation(client: &TransportClient, title: &str) -> Result<String> {
    match client {
        TransportClient::Dbus(client) => client.create_conversation(title).await,
        TransportClient::Ws(client) => client.create_conversation(title).await,
    }
}

async fn client_delete_conversation(client: &TransportClient, id: &str) -> Result<()> {
    match client {
        TransportClient::Dbus(client) => client.delete_conversation(id).await,
        TransportClient::Ws(client) => client.delete_conversation(id).await,
    }
}

async fn client_send_prompt(
    client: &TransportClient,
    conversation_id: &str,
    prompt: &str,
) -> Result<String> {
    match client {
        TransportClient::Dbus(client) => client.send_prompt(conversation_id, prompt).await,
        TransportClient::Ws(client) => client.send_prompt(conversation_id, prompt).await,
    }
}

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

    // Connect using configured transport (WS by default, D-Bus optional).
    let (client, mut signal_rx) = match connect_transport().await {
        Ok((transport_client, rx)) => {
            match client_list_conversations(&transport_client).await {
                Ok(convs) => app.set_conversations(convs),
                Err(e) => app.status_message = format!("Error loading conversations: {e}"),
            }
            app.status_message = match resolve_transport_mode() {
                TransportMode::Dbus => "Connected via D-Bus".to_string(),
                TransportMode::Ws => "Connected via WebSocket".to_string(),
            };
            (Some(transport_client), rx)
        }
        Err(e) => {
            app.status_message = format!("Connection failed: {e}");
            (None, mpsc::unbounded_channel().1)
        }
    };

    let mut event_stream = crossterm::event::EventStream::new();

    loop {
        terminal.draw(|f| ui::draw(f, &mut app))?;

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
                    } else if matches!(app.mode, InputMode::Editing | InputMode::CreatingConversation) {
                        // Forward unhandled keys to textarea
                        app.textarea.input(key);
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

async fn handle_action(app: &mut App, client: &Option<TransportClient>, action: Action) {
    match action {
        Action::Quit => app.quit(),
        Action::NextConversation => app.next_conversation(),
        Action::PreviousConversation => app.previous_conversation(),
        Action::OpenConversation => {
            if let (Some(client), Some(id)) = (client.as_ref(), app.selected_conversation_id()) {
                let id = id.to_string();
                match client_get_conversation(client, &id).await {
                    Ok(detail) => {
                        app.load_conversation(detail);
                        app.enter_editing_mode();
                    }
                    Err(e) => app.status_message = format!("Error: {e}"),
                }
            }
        }
        Action::DeleteConversation => {
            if let Some(id) = app.delete_selected_conversation()
                && let Some(client) = client.as_ref()
                && let Err(e) = client_delete_conversation(client, &id).await
            {
                app.status_message = format!("Delete error: {e}");
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
            if let Some((conv_id, prompt)) = app.submit_prompt()
                && let Some(client) = client.as_ref()
            {
                match client_send_prompt(client, &conv_id, &prompt).await {
                    Ok(request_id) if request_id.is_empty() => {
                        app.start_streaming_without_request_id()
                    }
                    Ok(request_id) => app.start_streaming(request_id),
                    Err(e) => app.status_message = format!("Send error: {e}"),
                }
            }
        }
        Action::InsertNewline => {
            app.textarea.insert_newline();
        }
        Action::ScrollUp => app.scroll_up(5),
        Action::ScrollDown => app.scroll_down(5),
        Action::ScrollToBottom => app.scroll_to_bottom(),
        Action::SubmitTitle => {
            if let Some(title) = app.submit_new_conversation_title()
                && let Some(client) = client.as_ref()
            {
                match client_create_conversation(client, &title).await {
                    Ok(id) => {
                        // Refresh list and auto-open the new conversation
                        match client_list_conversations(client).await {
                            Ok(convs) => {
                                let new_idx = convs.iter().position(|c| c.id == id);
                                app.set_conversations(convs);
                                if let Some(idx) = new_idx {
                                    app.selected_conversation = Some(idx);
                                }
                            }
                            Err(e) => app.status_message = format!("Error refreshing: {e}"),
                        }
                        match client_get_conversation(client, &id).await {
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
