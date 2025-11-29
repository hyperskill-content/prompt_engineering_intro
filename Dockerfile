FROM python:3.14-slim

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

COPY requirements.txt ./
RUN uv pip install -r requirements.txt

COPY notebooks/ /workspace/notebooks/

EXPOSE 8888

RUN useradd -m -u 1000 jupyter && \
    chown -R jupyter:jupyter /workspace

USER jupyter

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]