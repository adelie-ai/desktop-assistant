mod app;
mod keys;
mod ui;

use std::io;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use desktop_assistant_client_common::{
    AssistantClient, ConnectionConfig, SignalEvent, TransportMode, connect_transport,
    transport::transport_label,
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};

use app::{App, InputMode};
use keys::{Action, handle_key_event};

const DEFAULT_WS_URL: &str = desktop_assistant_client_common::config::DEFAULT_WS_URL;
const DEFAULT_WS_SUBJECT: &str = desktop_assistant_client_common::config::DEFAULT_WS_SUBJECT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
enum CliTransportMode {
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
        default_value_t = CliTransportMode::Ws
    )]
    transport: CliTransportMode,
    #[arg(
        long = "ws-url",
        env = "DESKTOP_ASSISTANT_TUI_WS_URL",
        default_value = DEFAULT_WS_URL
    )]
    ws_url: String,
    #[arg(long = "ws-jwt", env = "DESKTOP_ASSISTANT_TUI_WS_JWT")]
    ws_jwt: Option<String>,
    #[arg(
        long = "ws-login-username",
        env = "DESKTOP_ASSISTANT_TUI_WS_LOGIN_USERNAME"
    )]
    ws_login_username: Option<String>,
    #[arg(
        long = "ws-login-password",
        env = "DESKTOP_ASSISTANT_TUI_WS_LOGIN_PASSWORD"
    )]
    ws_login_password: Option<String>,
    #[arg(
        long = "ws-subject",
        env = "DESKTOP_ASSISTANT_TUI_WS_SUBJECT",
        default_value = DEFAULT_WS_SUBJECT
    )]
    ws_subject: String,
}

impl From<CliArgs> for ConnectionConfig {
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

        let ws_login_username = cli
            .ws_login_username
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let ws_login_password = cli
            .ws_login_password
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

        let transport_mode = match cli.transport {
            CliTransportMode::Ws => TransportMode::Ws,
            CliTransportMode::Dbus => TransportMode::Dbus,
        };

        Self {
            transport_mode,
            ws_url,
            ws_jwt,
            ws_login_username,
            ws_login_password,
            ws_subject,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ConnectionConfig::from(CliArgs::parse());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    );
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal, &config).await;

    // Restore terminal
    disable_raw_mode()?;
    let _ = execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags);
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
    config: &ConnectionConfig,
) -> Result<()> {
    let mut app = App::new();

    // Connect using configured transport (WS by default, D-Bus optional).
    let (client, mut signal_rx) = match connect_transport(config).await {
        Ok((transport_client, rx)) => {
            match transport_client.list_conversations().await {
                Ok(convs) => app.set_conversations(convs),
                Err(e) => app.status_message = format!("Error loading conversations: {e}"),
            }
            app.status_message = transport_label(config.transport_mode).to_string();
            (Some(transport_client), rx)
        }
        Err(e) => {
            app.status_message = format!("Connection failed: {e}");
            (None, tokio::sync::mpsc::unbounded_channel().1)
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
                    if key.kind == KeyEventKind::Release {
                        continue;
                    }
                    if let Some(action) = handle_key_event(key, &app.mode) {
                        handle_action(&mut app, &client, action).await;
                    } else if matches!(app.mode, InputMode::Editing) {
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
                    SignalEvent::Status { request_id: _, message } => {
                        app.status_message = message;
                    }
                    SignalEvent::TitleChanged { conversation_id, title } => {
                        app.update_conversation_title(&conversation_id, &title);
                    }
                    SignalEvent::Disconnected { reason } => {
                        app.status_message = format!("Disconnected: {reason}");
                    }
                }
            }
        }
    }

    Ok(())
}

async fn handle_action(
    app: &mut App,
    client: &Option<desktop_assistant_client_common::TransportClient>,
    action: Action,
) {
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
            if let Some(id) = app.delete_selected_conversation()
                && let Some(client) = client.as_ref()
                && let Err(e) = client.delete_conversation(&id).await
            {
                app.status_message = format!("Delete error: {e}");
            }
        }
        Action::NewConversation => {
            if let Some(client) = client.as_ref() {
                match client.create_conversation("New Conversation").await {
                    Ok(id) => {
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
                match client.send_prompt(&conv_id, &prompt).await {
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
            "--ws-login-username",
            "alice",
            "--ws-login-password",
            "s3cr3t",
            "--ws-subject",
            "custom-client",
        ]))
        .unwrap();

        assert_eq!(parsed.transport, CliTransportMode::Dbus);
        assert_eq!(parsed.ws_url, "wss://example/ws");
        assert_eq!(parsed.ws_jwt.as_deref(), Some("jwt123"));
        assert_eq!(parsed.ws_login_username.as_deref(), Some("alice"));
        assert_eq!(parsed.ws_login_password.as_deref(), Some("s3cr3t"));
        assert_eq!(parsed.ws_subject, "custom-client");
    }

    #[test]
    fn clap_defaults_map_to_runtime_defaults() {
        let cli = CliArgs::try_parse_from(args(&[])).unwrap();
        let config = ConnectionConfig::from(cli);
        assert_eq!(config.transport_mode, TransportMode::Ws);
        assert_eq!(config.ws_url, DEFAULT_WS_URL);
        assert_eq!(config.ws_subject, DEFAULT_WS_SUBJECT);
        assert_eq!(config.ws_jwt, None);
        assert_eq!(config.ws_login_username, None);
        assert_eq!(config.ws_login_password, None);
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
