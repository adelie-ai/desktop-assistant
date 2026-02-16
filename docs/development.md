# Development Guide

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Day-to-day Commands

```bash
# format
cargo fmt

# full test suite
cargo test --workspace

# strict lints
cargo clippy --workspace --all-targets -- -D warnings
```

## Run Components

```bash
# daemon
cargo run -p desktop-assistant-daemon

# tui
cargo run -p desktop-assistant-tui
```

## Environment

Default connector is `ollama` (local, no API key required).

For cloud connectors, set the matching API key:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

Connector key naming convention is generic:
- Secret backend account key defaults to `<connector>_api_key`.
- Environment fallback defaults to `<CONNECTOR>_API_KEY`.
- Connector names are normalized to alphanumeric/underscore (for example, `aws-bedrock` → `aws_bedrock_api_key` and `AWS_BEDROCK_API_KEY`).

Optional:

```bash
export OPENAI_MODEL=gpt-4o
export OPENAI_BASE_URL=https://api.openai.com/v1
export RUST_LOG=info
```

## Testing MCP Integration

- E2E tests may require external binaries (`fileio-mcp`, `python3`)
- Tests skip gracefully if optional tools are unavailable

Useful targeted runs:

```bash
cargo test -p desktop-assistant-mcp-client
cargo test -p desktop-assistant-mcp-client --test e2e_fileio
cargo test -p desktop-assistant-mcp-client --test e2e_dynamic_list_changed
```

## Project Conventions

- Keep core logic in `crates/core` independent of adapters
- Prefer trait-based boundaries over direct dependency coupling
- Keep docs and tests updated when interfaces change
