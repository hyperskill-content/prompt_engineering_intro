FROM python:3.14-slim

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/local

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY notebooks/ /workspace/notebooks/

EXPOSE 8888

RUN useradd -m -u 1000 marimo && \
    chown -R marimo:marimo /workspace

USER marimo

CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "8888", "--no-token", "notebooks/"]