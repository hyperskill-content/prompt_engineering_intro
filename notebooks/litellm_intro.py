import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # litellm: one call, many providers

    Each provider ships its own SDK and its own request and response
    format. OpenAI's Chat Completions API doesn't look like Anthropic's
    Messages API, which doesn't look like Google's Gemini API.

    ```python
    # OpenAI Chat Completions
    {"model": "gpt-4o-mini",
     "messages": [{"role": "system", "content": "Be brief."},
                  {"role": "user", "content": "Hi"}]}
    # reply text at: choices[0].message.content

    # Anthropic Messages
    {"model": "claude-sonnet-4-5",
     "max_tokens": 1024,
     "system": "Be brief.",
     "messages": [{"role": "user", "content": "Hi"}]}
    # reply text at: content[0].text

    # Google Gemini (the model name goes in the URL, not the body)
    {"systemInstruction": {"parts": [{"text": "Be brief."}]},
     "contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
    # reply text at: candidates[0].content.parts[0].text
    ```
    Switch providers and you normally have to rewrite how you build requests and parse responses.

    `litellm` uses OpenAI's Chat Completions shape as the interface for every provider: the same
    `messages` list going in, the same `choices[0].message.content` coming out, whichever provider answers. What changes is who translates that shape into the provider's real API.

    Point `litellm.completion()` at Anthropic with your Anthropic key, and litellm does the translation locally before sending the request. Point it at this proxy instead, as we do below, and litellm just sends an OpenAI-shaped request over the wire. The proxy, which is also litellm running server-side, translates it into the target
    provider's format on the other end. That's also why the key you were issued works for every provider: it's a key to the proxy, not to any one of them.

    The rest of this notebook covers four things a shared format allows: swapping providers with a one-line change, running several at once, streaming, and handling errors without a branch per provider.
    """)
    return


@app.cell
def _():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return (os,)


@app.cell
def _(os):
    import litellm
    import openai

    # Quiets litellm's internal retry/error logging so the demos below stay readable;
    # the exceptions themselves are still raised normally.
    litellm.suppress_debug_info = True

    api_key = os.environ["OPENAI_API_KEY"]
    # The root proxy (not the /openai pass-through) is where model names get
    # routed to their real provider. We tell litellm to speak the
    # OpenAI-compatible dialect to it via the "openai/" prefix; the actual
    # provider routing happens server-side in the proxy.
    api_base = os.environ["OPENAI_BASE_URL"].removesuffix("/openai") + "/v1"
    return api_base, api_key, litellm, openai


@app.cell
def _(mo):
    mo.md(r"""
    Below, the same call runs against whichever provider you pick. Only
    the model name changes.
    """)
    return


@app.cell
def _(mo):
    model_options = {
        "gpt-4o-mini (provider: openai)": "gpt-4o-mini",
        "gpt-4.1-mini (provider: openai)": "gpt-4.1-mini",
        "mistral-large-latest (provider: mistral)": "mistral-large-latest",
        "claude-sonnet-4-5 (provider: anthropic)": "claude-sonnet-4-5",
    }
    model_picker = mo.ui.dropdown(
        options=model_options,
        value="gpt-4o-mini (provider: openai)",
        label="Model",
    )
    prompt_box = mo.ui.text_area(value="Say hi in exactly five words.", label="Prompt")
    ask_button = mo.ui.run_button(label="Ask")
    mo.vstack([model_picker, prompt_box, ask_button])
    return ask_button, model_picker, prompt_box


@app.cell
def _(api_base, api_key, ask_button, litellm, mo, model_picker, prompt_box):
    mo.stop(
        not ask_button.value,
        mo.md("*Pick a model, write a prompt, and click **Ask**.*"),
    )

    response = litellm.completion(
        model="openai/" + model_picker.value,
        messages=[{"role": "user", "content": prompt_box.value}],
        api_base=api_base,
        api_key=api_key,
    )
    return (response,)


@app.cell
def _(mo, model_picker, response):
    mo.md(f"""
    **{model_picker.value}** replied:\n\n> {response.choices[0].message.content}
    """)
    return


@app.cell
def _(litellm, mo, response):
    try:
        cost_line = f"${litellm.completion_cost(response):.6f}"
    except Exception:
        # Expected: custom/proxy-only model names aren't in litellm's
        # built-in pricing table, so cost lookup can't resolve them.
        cost_line = "n/a (model not in litellm's pricing table)"

    usage = response.usage
    mo.md(
        f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out "
        f"&nbsp;·&nbsp; cost: {cost_line}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## One prompt, several providers at once

    `litellm.acompletion()` is the async version of `completion()` and takes the same arguments. Fire it at several providers concurrently with `asyncio.gather` and the total wait is roughly the slowest single call, not the sum of all of them. That makes it a cheap way to compare how different models answer the same prompt side by side.
    """)
    return


@app.cell
def _(mo):
    race_prompt = mo.ui.text_area(
        value="What's a good name for a cat that only sleeps?",
        label="Prompt",
    )
    race_models = mo.ui.multiselect(
        options=["gpt-4o-mini", "gpt-4.1-mini", "mistral-large-latest", "claude-sonnet-4-5"],
        value=["gpt-4o-mini", "mistral-large-latest", "claude-sonnet-4-5"],
        label="Providers to race",
    )
    race_button = mo.ui.run_button(label="Race")
    mo.vstack([race_prompt, race_models, race_button])
    return race_button, race_models, race_prompt


@app.cell
async def _(
    api_base,
    api_key,
    litellm,
    mo,
    race_button,
    race_models,
    race_prompt,
):
    mo.stop(
        not race_button.value,
        mo.md("*Pick a prompt and at least one provider, then click **Race**.*"),
    )

    import asyncio
    import time

    async def ask_one(model: str) -> dict:
        start = time.time()
        try:
            resp = await litellm.acompletion(
                model="openai/" + model,
                messages=[{"role": "user", "content": race_prompt.value}],
                api_base=api_base,
                api_key=api_key,
            )
            try:
                cost = f"${litellm.completion_cost(resp):.6f}"
            except Exception:
                cost = "n/a"
            return {
                "model": model,
                "seconds": round(time.time() - start, 2),
                "cost": cost,
                "reply": resp.choices[0].message.content,
            }
        except Exception as e:
            return {
                "model": model,
                "seconds": round(time.time() - start, 2),
                "cost": "n/a",
                "reply": f"error: {type(e).__name__}",
            }

    race_results = await asyncio.gather(*(ask_one(m) for m in race_models.value))
    return (race_results,)


@app.cell
def _(mo, race_results):
    mo.ui.table(race_results, label="Same prompt, every provider, one round trip")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Streaming

    A normal call makes you wait until the whole reply is finished. With streaming, you get the text as it's generated, so the first words show up almost immediately. The total time is about the same, but it feels much faster, which is why every chat interface does it, and it matters most for long replies.

    Pass `stream=True`, the same flag OpenAI's SDK uses, and `litellm.completion()` returns an iterator of chunks instead of one finished response. litellm converts each provider's streaming format into the OpenAI chunk shape, so the reply text is always at `chunk.choices[0].delta.content`. The loop below doesn't change if you swap the model picked in the first section.
    """)
    return


@app.cell
def _(mo):
    stream_prompt = mo.ui.text_area(
        value="Count from 1 to 10, one number per line.", label="Prompt"
    )
    stream_button = mo.ui.run_button(label="Stream")
    mo.vstack([stream_prompt, stream_button])
    return stream_button, stream_prompt


@app.cell
def _(
    api_base,
    api_key,
    litellm,
    mo,
    model_picker,
    stream_button,
    stream_prompt,
):
    mo.stop(not stream_button.value, mo.md("*Write a prompt, click **Stream**.*"))

    streamed_text = ""
    for _chunk in litellm.completion(
        model="openai/" + model_picker.value,
        messages=[{"role": "user", "content": stream_prompt.value}],
        api_base=api_base,
        api_key=api_key,
        stream=True,
    ):
        _delta = _chunk.choices[0].delta.content
        if _delta:
            streamed_text += _delta
            mo.output.replace(mo.md(f"**{model_picker.value}**, streaming:\n\n{streamed_text}"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Same errors, whichever provider breaks

    Every provider fails differently: different status codes, different error bodies, different field names for what went wrong. litellm maps all of that onto OpenAI's exception classes, like `openai.NotFoundError` and `openai.RateLimitError`, which all inherit from `openai.APIError`. So error handling doesn't need a branch per provider either.

    The first model name below, `llama-3.1-8b-instant`, doesn't exist on the proxy, so that call is guaranteed to fail. A real outage or a retired model would look the same to your code. Catching `openai.APIError` and moving on to the next candidate makes a complete fallback chain in a handful of lines, and it works the same no matter which providers are in the list or how each one fails:
    """)
    return


@app.cell
def _(mo):
    fallback_prompt = mo.ui.text_area(value="Say hi in exactly five words.", label="Prompt")
    fallback_button = mo.ui.run_button(label="Try it")
    mo.vstack([fallback_prompt, fallback_button])
    return fallback_button, fallback_prompt


@app.cell
def _(
    api_base,
    api_key,
    fallback_button,
    fallback_prompt,
    litellm,
    mo,
    openai,
):
    mo.stop(not fallback_button.value, mo.md("*Click **Try it**.*"))

    _candidates = ["llama-3.1-8b-instant", "gpt-4o-mini"]
    _log = []
    fallback_used, fallback_reply = None, None
    for _candidate in _candidates:
        try:
            _resp = litellm.completion(
                model="openai/" + _candidate,
                messages=[{"role": "user", "content": fallback_prompt.value}],
                api_base=api_base,
                api_key=api_key,
            )
            _log.append(f"`{_candidate}`: succeeded")
            fallback_used, fallback_reply = _candidate, _resp.choices[0].message.content
            break
        except openai.APIError as e:
            _log.append(f"`{_candidate}`: {type(e).__name__}, falling back")

    mo.md("\n\n".join(_log) + f"\n\n**{fallback_used}** replied:\n\n> {fallback_reply}")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    One interface allows to swap providers by changing a string, run several at once, stream from any of them, and catch every provider's failures the same way. All four work because litellm translates each provider's format into one shape. To see how much that saves you, try the same request against a provider's direct API with litellm stripped away.
    """)
    return


if __name__ == "__main__":
    app.run()
