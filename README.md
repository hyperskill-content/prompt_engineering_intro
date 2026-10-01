# Intro to LLMs and pipelines

This repo contains support notebooks for topics used or covered in the first module. Notebooks in this repo are [marimo](https://marimo.io) notebooks: plain, git-diffable `.py` files with reactive cells, run either through the marimo UI or locally via `uv`.

## Notebooks

- `prompting.py`: prompting basics through one running example, turning requests into GitHub search queries: message roles, temperature, few-shot, chain-of-thought, structured output and validation, and handling ambiguous requests.
- `litellm_intro.py`: one function call, run against several providers through the course's LiteLLM proxy.
- `model_zoo.py`: the proxy's full model catalog, plus a live test drive of a subset of it.
- `raw_api_layer.py`: Chat Completions vs the Responses API through the same proxy, and an intro on how tool calling works.

## Access

The notebooks call models through a LiteLLM proxy that we host for the course. Participants get a proxy URL and a personal API key when they enroll, and those go into `.env` as `OPENAI_BASE_URL` and `OPENAI_API_KEY`. 
Without them, the notebooks won't run as-is.

If you're not enrolled, you can run your own [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)
with your own provider keys. For the notebooks to work unchanged, the proxy needs:

- An OpenAI pass-through route, with `OPENAI_BASE_URL` pointing at it and ending in `/openai`. The notebooks strip that suffix to reach the proxy's main endpoint.
- The model names the notebooks use, configured under the same names. Otherwise, change the model names in the notebooks to whatever your proxy serves.


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
