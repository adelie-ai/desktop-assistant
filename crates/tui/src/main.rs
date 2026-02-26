mod app;
mod dbus_client;
mod keys;
mod ui;
mod ws_client;

use std::io;

use anyhow::Result;
use clap::Parser;
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

const DEFAULT_WS_URL: &str = "ws://127.0.0.1:11339/ws";
const DEFAULT_WS_SUBJECT: &str = "desktop-tui";

enum TransportClient {
    Dbus(DbusClient),
    Ws(WsClient),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
enum TransportMode {
    Ws,
    Dbus,
}

#[derive(Debug, Parser)]
#[command(name = "desktop-assistant-tui")]
struct CliArgs {
    #[arg(
        long,
        env = "DESKTOP_ASSISTANT_TUI_TRANSPORT",
        value_enum,
        default_value_t = TransportMode::Ws
    )]
    transport: TransportMode,
    #[arg(
        long = "ws-url",
        env = "DESKTOP_ASSISTANT_TUI_WS_URL",
        default_value = DEFAULT_WS_URL
    )]
    ws_url: String,
    #[arg(long = "ws-jwt", env = "DESKTOP_ASSISTANT_TUI_WS_JWT")]
    ws_jwt: Option<String>,
    #[arg(
        long = "ws-subject",
        env = "DESKTOP_ASSISTANT_TUI_WS_SUBJECT",
        default_value = DEFAULT_WS_SUBJECT
    )]
    ws_subject: String,
}

#[derive(Debug, Clone)]
struct RuntimeOptions {
    transport_mode: TransportMode,
    ws_url: String,
    ws_jwt: Option<String>,
    ws_subject: String,
}

impl From<CliArgs> for RuntimeOptions {
    fn from(cli: CliArgs) -> Self {
        let ws_url = {
            let trimmed = cli.ws_url.trim();
            if trimmed.is_empty() {
                DEFAULT_WS_URL.to_string()
            } else {
                trimmed.to_string()
            }
        };

        let ws_jwt = cli
            .ws_jwt
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let ws_subject = {
            let trimmed = cli.ws_subject.trim();
            if trimmed.is_empty() {
                DEFAULT_WS_SUBJECT.to_string()
            } else {
                trimmed.to_string()
            }
        };

        Self {
            transport_mode: cli.transport,
            ws_url,
            ws_jwt,
            ws_subject,
        }
    }
}

fn transport_label(mode: TransportMode) -> &'static str {
    match mode {
        TransportMode::Dbus => "Connected via D-Bus",
        TransportMode::Ws => "Connected via WebSocket",
    }
}

async fn connect_transport(
    options: &RuntimeOptions,
) -> Result<(TransportClient, mpsc::UnboundedReceiver<SignalEvent>)> {
    match options.transport_mode {
        TransportMode::Dbus => {
            let client = DbusClient::connect().await?;
            let signal_rx = client.subscribe_signals().await?;
            Ok((TransportClient::Dbus(client), signal_rx))
        }
        TransportMode::Ws => {
            let token = match options.ws_jwt.clone() {
                Some(token) => token,
                None => generate_ws_jwt(&options.ws_subject).await?,
            };
            let (client, signal_rx) = WsClient::connect(&options.ws_url, &token).await?;
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
    let options = RuntimeOptions::from(CliArgs::parse());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal, &options).await;

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

async fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    options: &RuntimeOptions,
) -> Result<()> {
    let mut app = App::new();

    // Connect using configured transport (WS by default, D-Bus optional).
    let (client, mut signal_rx) = match connect_transport(options).await {
        Ok((transport_client, rx)) => {
            match client_list_conversations(&transport_client).await {
                Ok(convs) => app.set_conversations(convs),
                Err(e) => app.status_message = format!("Error loading conversations: {e}"),
            }
            app.status_message = transport_label(options.transport_mode).to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        let mut out = vec!["desktop-assistant-tui".to_string()];
        out.extend(parts.iter().map(|value| value.to_string()));
        out
    }

    #[test]
    fn clap_parses_transport_flags() {
        let parsed = CliArgs::try_parse_from(args(&[
            "--transport",
            "dbus",
            "--ws-url",
            "wss://example/ws",
            "--ws-jwt",
            "jwt123",
            "--ws-subject",
            "custom-client",
        ]))
        .unwrap();

        assert_eq!(parsed.transport, TransportMode::Dbus);
        assert_eq!(parsed.ws_url, "wss://example/ws");
        assert_eq!(parsed.ws_jwt.as_deref(), Some("jwt123"));
        assert_eq!(parsed.ws_subject, "custom-client");
    }

    #[test]
    fn clap_defaults_map_to_runtime_defaults() {
        let cli = CliArgs::try_parse_from(args(&[])).unwrap();
        let options = RuntimeOptions::from(cli);
        assert_eq!(options.transport_mode, TransportMode::Ws);
        assert_eq!(options.ws_url, DEFAULT_WS_URL);
        assert_eq!(options.ws_subject, DEFAULT_WS_SUBJECT);
        assert_eq!(options.ws_jwt, None);
    }

    #[test]
    fn clap_rejects_invalid_transport_value() {
        let error = CliArgs::try_parse_from(args(&["--transport", "http"]))
            .err()
            .expect("transport should be validated by clap");
        let rendered = error.to_string();
        assert!(rendered.contains("ws"));
        assert!(rendered.contains("dbus"));
    }
}
