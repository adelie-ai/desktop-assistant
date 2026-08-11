FROM rust:1.97-bookworm AS builder

WORKDIR /workspace

# The daemon links libpam (WS local-system-auth); the base rust image lacks the
# dev headers, so the final link fails with `-lpam not found` without this.
#
# Nothing else is needed for `OTEL=1`. Its TLS backend compiles native code
# through aws-lc-rs, which wants a C compiler and an assembler and not cmake -
# and this image already compiles aws-lc-sys on every build, because
# jsonwebtoken pulls it in as a normal dependency of the daemon.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpam0g-dev \
    && rm -rf /var/lib/apt/lists/*

# Whether this image can export telemetry. Off by default, so an unmodified
# build produces the image it has always produced. `--build-arg OTEL=1`
# compiles the OTLP exporter in; see scripts/cargo-otel.sh and the Telemetry
# section of deploy/k8s/README.md.
ARG OTEL=0

# Copied on its own, before the source, so the build reaches cargo through the
# same wrapper the fleet image uses.
COPY scripts/cargo-otel.sh /usr/local/bin/cargo-otel

COPY . .
RUN cargo-otel build --release --locked -p desktop-assistant-daemon

FROM debian:bookworm-slim

# ca-certificates is load-bearing once telemetry is exported over https. The two
# OTLP transports read the operating system trust store, so an image without a
# CA bundle fails every https export. Removing this package, or moving to a
# distroless or scratch base without adding a bundle, breaks that silently.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libpam0g \
    && rm -rf /var/lib/apt/lists/*

# Carried through from the builder stage so the finished image says whether it
# can export. `1` means the OTLP exporter is compiled in.
ARG OTEL=0
LABEL ai.adelie.otel="${OTEL}"

RUN useradd --create-home --uid 10001 assistant
WORKDIR /home/assistant

COPY --from=builder /workspace/target/release/desktop-assistant-daemon /usr/local/bin/desktop-assistant-daemon

ENV RUST_LOG=info
ENV DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339
ENV XDG_CONFIG_HOME=/home/assistant/.config
ENV XDG_DATA_HOME=/home/assistant/.local/share
ENV XDG_STATE_HOME=/home/assistant/.local/state
ENV XDG_CACHE_HOME=/home/assistant/.cache

EXPOSE 11339
USER assistant

ENTRYPOINT ["desktop-assistant-daemon"]
