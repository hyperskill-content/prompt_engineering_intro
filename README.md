# Intro to LLMs and pipelines

This repo contains support notebooks for topics used or covered in the first module. Notebooks in this repo are [marimo](https://marimo.io) notebooks: plain, git-diffable `.py` files with reactive cells, run either through the marimo UI or locally via `uv`.

## Notebooks

- `litellm_intro.py`: one function call, run against several providers through the course's litellm proxy.
- `model_zoo.py`: the proxy's full model catalog, plus a live test drive of a subset of it.
- `raw_api_layer.py`: Chat Completions vs the Responses API through the same proxy, and an intro on how tool calling works.

## Usage

### Option 1: Docker

Create the `.env` file from a template:
```shell
cp .env.template .env
```

Build and start the container:
```shell
docker compose up -d
```
Access the marimo editor at: http://localhost:8888

Stop the container:
```shell
docker compose down
```

The `notebooks/` folder is mounted as a volume, so any changes you make in the marimo UI are
reflected locally and vice versa.

If you want to add packages, edit `pyproject.toml` (or run `uv add <package>`), then rebuild:
```shell
docker compose up -d --build
```

### Option 2: Local with uv

```shell
cp .env.template .env
uv sync
uv run marimo edit notebooks/
```
