FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG CODEX_VERSION
RUN test -n "$CODEX_VERSION" && npm install -g "@openai/codex@$CODEX_VERSION"
