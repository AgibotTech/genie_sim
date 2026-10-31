# syntax=docker/dockerfile:1
# Agent2Act inference server — motion mode, real LLM backend.
# Build from source/agent2act: docker build -t <tag> .
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
COPY scripts ./scripts
COPY examples ./examples
COPY LICENSES ./LICENSES
COPY LICENSE THIRD_PARTY_NOTICES.md ./
RUN uv sync --locked --no-dev

# The image already contains the locked environment. Avoid re-resolving the
# editable project when a pod starts without internet access.
ENV UV_NO_SYNC=1

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--locked", "--no-dev", "python", "scripts/serve_llm.py", \
    "--config", "examples/g2_llm_recipe.json", "--depth", \
    "--host", "0.0.0.0", "--port", "8000"]
