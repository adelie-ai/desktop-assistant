FROM rust:1-bookworm AS builder

WORKDIR /workspace
COPY . .
RUN cargo build --release --locked -p desktop-assistant-daemon

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 assistant
WORKDIR /home/assistant

COPY --from=builder /workspace/target/release/desktop-assistant-daemon /usr/local/bin/desktop-assistant-daemon

ENV RUST_LOG=info
ENV DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339
ENV DESKTOP_ASSISTANT_DBUS_REQUIRED=false
ENV XDG_CONFIG_HOME=/home/assistant/.config
ENV XDG_DATA_HOME=/home/assistant/.local/share
ENV XDG_STATE_HOME=/home/assistant/.local/state
ENV XDG_CACHE_HOME=/home/assistant/.cache

EXPOSE 11339
USER assistant

ENTRYPOINT ["desktop-assistant-daemon"]
