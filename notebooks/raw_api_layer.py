import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The raw API layer

    The course proxy speaks several APIs. In this notebook we use two of them, both OpenAI-style, through the plain `openai` SDK.

    Chat Completions takes `messages` in and returns `choices`. It's the long-standing format, and the closest thing to a portable standard. Most other providers and inference servers (Groq, Together, vLLM, Ollama) offer an OpenAI-compatible endpoint, and libraries like LangChain and LlamaIndex use the same shape whenever you point them at one, which is why pointing the `openai` client at a different `base_url` is common. The proxy accepts Chat Completions for every provider it routes to and handles the translation on the other end. Our Chat Completions calls go through its unified endpoint and can reach any of them.

    Responses takes `input` in and returns `output` out, but it's more than a different shape. It's OpenAI's own product, and much of it runs on their servers: conversation state, built-in tools, remote MCP integration. Other providers and gateways can accept the request shape, but they don't ship the same feature set, and things like storage end up configured somewhere else. So it isn't a portable protocol the way Chat Completions is, at least for now. Our Responses calls go through the proxy's OpenAI-only pass-through, which gets us the actual product.

    Then we look at tool calling: how the model signals that it wants to call a tool, and how the harness (your code, not the model) turns that signal into a function call and a follow-up turn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a tool call actually is

    A model has no clock, no database connection, no way to run code. Whatever it knows was baked in at training time. Tool calling allows to describe capabilities in the model's context, and let it ask for a tool instead of answering from it's internal knowledge.

    There's no separate tool-calling mode the model switches into. It's predicting the next token, as always. During training it was taught that certain requests call for a specific pattern instead of a normal-language reply: a name from the tool list, plus a JSON object matching that tool's schema. The exact recipe (fine-tuning, reinforcement learning, whatever mix a given lab uses) doesn't matter here. What matters is that the model isn't aware it triggered anything. It generated tokens in a shape your code knows how to recognize.

    For your own functions, acting on that shape is the harness's job. Recognizing it usually falls to the API, which looks at what came out and wraps it in a discrete field so your code doesn't have to guess. Some tools are the exception: the built-in ones in the Responses API run on OpenAI's servers, so your code never executes them. This notebook focuses on custom functions, covering the shape of the output, how you know it's a tool call, and the function call your code makes to get a result back to the model.
    """)
    return


@app.cell
def _():
    import json
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return json, os


@app.cell
def _(os):
    from openai import OpenAI

    api_key = os.environ["OPENAI_API_KEY"]
    # OPENAI_BASE_URL is the OpenAI-only pass-through (".../openai"): real OpenAI behavior,
    # used below for the Responses calls. Strip that suffix and add "/v1" to reach the proxy's
    # unified endpoint instead, which routes a Chat Completions request to whatever provider
    # actually serves the model name you give it.
    responses_base_url = os.environ["OPENAI_BASE_URL"]
    cc_base_url = responses_base_url.removesuffix("/openai") + "/v1"

    cc_client = OpenAI(api_key=api_key, base_url=cc_base_url)
    responses_client = OpenAI(api_key=api_key, base_url=responses_base_url)

    # Responses is OpenAI's own product. No picker; it only ever means OpenAI.
    RESPONSES_MODEL = "gpt-4o-mini"
    return RESPONSES_MODEL, cc_client, responses_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chat Completions doesn't depend on the provider as much.
    """)
    return


@app.cell
def _(mo):
    cc_model_picker = mo.ui.dropdown(
        options={
            "claude-sonnet-4-5 (provider: anthropic)": "claude-sonnet-4-5",
            "mistral-large-latest (provider: mistral)": "mistral-large-latest",
            "gpt-4o-mini (provider: openai)": "gpt-4o-mini",
        },
        value="claude-sonnet-4-5 (provider: anthropic)",
        label="Chat Completions model",
    )
    cc_model_picker
    return (cc_model_picker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same request, two shapes

    Chat Completions takes a list of role-tagged messages and returns a list of `choices`, plural because you can ask for more than one candidate completion. In practice you almost always read the first one: `choices[0].message.content`.

    Responses takes `input`, either a string or a list of items, and returns a flat `output` list of typed items. A plain text reply is a single `message` item with the text nested inside it, and the SDK's `response.output_text` shortcut digs it out for you.

    The cell below sends the same prompt to both.
    """)
    return


@app.cell
def _(mo):
    shape_prompt = mo.ui.text_area(
        value="Explain what a race condition is in one sentence.",
        label="Prompt",
        full_width=True,
    )
    shape_button = mo.ui.run_button(label="Run")
    mo.vstack([shape_prompt, shape_button])
    return shape_button, shape_prompt


@app.cell
def _(
    RESPONSES_MODEL,
    cc_client,
    cc_model_picker,
    mo,
    responses_client,
    shape_button,
    shape_prompt,
):
    mo.stop(not shape_button.value, mo.md("*Write a prompt, click Run.*"))

    shape_cc_response = cc_client.chat.completions.create(
        model=cc_model_picker.value,
        messages=[{"role": "user", "content": shape_prompt.value}],
    )
    shape_responses_response = responses_client.responses.create(
        model=RESPONSES_MODEL,
        input=shape_prompt.value,
    )
    return shape_cc_response, shape_responses_response


@app.cell
def _(cc_model_picker, json, mo, shape_cc_response, shape_responses_response):
    _cc_choice = shape_cc_response.choices[0]
    _cc_summary = {
        "model": cc_model_picker.value,
        "response.id": shape_cc_response.id,
        "choices[0].finish_reason": _cc_choice.finish_reason,
        "choices[0].message.role": _cc_choice.message.role,
        "choices[0].message.content": _cc_choice.message.content,
    }
    _responses_summary = {
        "response.id": shape_responses_response.id,
        "output_text": shape_responses_response.output_text,
        "output": [item.type for item in shape_responses_response.output],
    }
    mo.hstack(
        [
            mo.vstack(
                [mo.md("**Chat Completions**"), mo.md(f"```json\n{json.dumps(_cc_summary, indent=2)}\n```")]
            ),
            mo.vstack(
                [
                    mo.md("**Responses**"),
                    mo.md(f"```json\n{json.dumps(_responses_summary, indent=2)}\n```"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Chat Completions replies from OpenAI, Anthropic, and Mistral all look alike: the text at `choices[0].message.content`, a `finish_reason` on the choice, the role on the message. Only the values differ, like the model name, the id, and the wording. That's the proxy's translation. Anthropic's native Messages format doesn't look this way.

    The format only changes when we switch APIs. Responses puts the text in `output_text` and describes everything the model produced as typed items in `output`, so the type (`message`, here) tells you what you got. Chat Completions has no equivalent, and you infer the same thing from `finish_reason` and which fields on the message are filled in. That second difference matters once we bring tools into it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Giving the model something to call

    A tool is a JSON Schema plus a name and description. Both APIs want the same schema, but they
    wrap it differently. Chat Completions nests it under `function`; Responses flattens it:
    """)
    return


@app.cell
def _():
    WEATHER_SCHEMA = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Lisbon'"},
        },
        "required": ["location"],
        "additionalProperties": False,
    }

    CURRENCY_SCHEMA = {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "3-letter currency code"},
            "to_currency": {"type": "string", "description": "3-letter currency code"},
        },
        "required": ["amount", "from_currency", "to_currency"],
        "additionalProperties": False,
    }

    # `strict: True` is what actually engages constrained decoding. More on that below.
    cc_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": WEATHER_SCHEMA,
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount from one currency to another.",
                "parameters": CURRENCY_SCHEMA,
                "strict": True,
            },
        },
    ]

    responses_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": WEATHER_SCHEMA,
            "strict": True,
        },
        {
            "type": "function",
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another.",
            "parameters": CURRENCY_SCHEMA,
            "strict": True,
        },
    ]
    return cc_tools, responses_tools


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both APIs get the same `WEATHER_SCHEMA` and `CURRENCY_SCHEMA`, only the formatting around them changes. The tools are mocks, deterministic with no network call, so the notebook stays self-contained. They can be swapped for live API calls and nothing else here changes, as long as the function keeps the same signature and return shape:
    """)
    return


@app.cell
def _():
    def get_weather(location: str) -> dict:
        return {"location": location, "temperature_c": 18, "condition": "cloudy"}

    def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
        rates = {
            ("USD", "EUR"): 0.92,
            ("EUR", "USD"): 1.09,
            ("USD", "GBP"): 0.79,
            ("GBP", "USD"): 1.27,
        }
        rate = rates.get((from_currency.upper(), to_currency.upper()))
        if rate is None:
            return {"error": f"no rate for {from_currency} -> {to_currency}"}
        return {
            "amount": amount,
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "result": round(amount * rate, 2),
        }

    # The dispatch table: every harness has some version of this, mapping the name the
    # model output to the callable that actually does something.
    TOOLS = {"get_weather": get_weather, "convert_currency": convert_currency}
    return (TOOLS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Structured output enforcement

    Getting valid JSON out of `arguments` can be enforced. Many serving stacks use constrained decoding: at each generation step, the sampler masks out any token that would make the output violate the tool's JSON Schema (wrong type, invalid key, malformed syntax). The mask comes from a grammar or finite-state machine derived from the schema. The model is still just predicting tokens, but the decoder mechanically prevents invalid JSON from being sampled.

    `strict: True` on the tool definitions above is the setting for this. OpenAI documents it as turning on constrained decoding. Other providers accept the same flag through the proxy and returned schema-valid `arguments` in our testing, but they don't necessarily document how they guarantee it. It could be constrained decoding, or something else.

    Not every system enforces this. Some rely on fine-tuning alone, plus a retry loop for when the JSON comes back malformed. Still, strict mode is common, because it eliminates a whole class of failures for schemas like the ones above.

    Note: Strict mode guarantees the shape of the arguments, not that they're sensible value-wise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. The API signals a tool call

    When the model emits a tool call instead of an answer, the response doesn't leave your code to guess from the text. It comes back with a discrete signal your code can branch on, and each API spells it differently. The cell below asks something that needs the weather tool and prints the raw signal from both:
    """)
    return


@app.cell
def _(mo):
    signal_prompt = mo.ui.text_area(
        value="What's the weather in Lisbon?", label="Prompt", full_width=True
    )
    signal_button = mo.ui.run_button(label="Run")
    mo.vstack([signal_prompt, signal_button])
    return signal_button, signal_prompt


@app.cell
def _(
    RESPONSES_MODEL,
    cc_client,
    cc_model_picker,
    cc_tools,
    mo,
    responses_client,
    responses_tools,
    signal_button,
    signal_prompt,
):
    mo.stop(not signal_button.value, mo.md("*Write a prompt, click Run.*"))

    signal_cc_response = cc_client.chat.completions.create(
        model=cc_model_picker.value,
        messages=[{"role": "user", "content": signal_prompt.value}],
        tools=cc_tools,
    )
    signal_responses_response = responses_client.responses.create(
        model=RESPONSES_MODEL,
        input=signal_prompt.value,
        tools=responses_tools,
    )
    return signal_cc_response, signal_responses_response


@app.cell
def _(
    cc_model_picker,
    json,
    mo,
    signal_cc_response,
    signal_responses_response,
):
    _cc_choice = signal_cc_response.choices[0]
    _cc_call = _cc_choice.message.tool_calls[0] if _cc_choice.message.tool_calls else None
    _cc_summary = {
        "model": cc_model_picker.value,
        "choices[0].finish_reason": _cc_choice.finish_reason,
        "choices[0].message.tool_calls[0]": _cc_call.model_dump() if _cc_call else None,
    }

    _responses_call = next(
        (item for item in signal_responses_response.output if item.type == "function_call"), None
    )
    _responses_summary = {
        "output item types": [item.type for item in signal_responses_response.output],
        "function_call item": _responses_call.model_dump() if _responses_call else None,
    }

    mo.hstack(
        [
            mo.vstack(
                [mo.md("**Chat Completions**"), mo.md(f"```json\n{json.dumps(_cc_summary, indent=2)}\n```")]
            ),
            mo.vstack(
                [
                    mo.md("**Responses**"),
                    mo.md(f"```json\n{json.dumps(_responses_summary, indent=2)}\n```"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chat Completions sets `finish_reason` to `"tool_calls"`, while Responses puts a `function_call` item in `output`. The shapes differ, but the message is that this isn't a final answer, so go run the tool.

    One thing does match across both, and it's easy to get wrong: `arguments` is a JSON string, not a parsed object (note the escaped quotes in both outputs above). The harness still has to `json.loads()` it. Strict mode makes a parse failure rare but not impossible, since a reply cut off at `max_tokens` can leave the JSON incomplete, so the harness should handle that case too.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Parsing and dispatch

    The harness reads the call, parses `arguments`, looks `name` up in the dispatch table from
    earlier, and invokes it with the parsed arguments as kwargs:
    """)
    return


@app.cell
def _(TOOLS, json):
    def dispatch(name: str, arguments_json: str) -> dict:
        arguments = json.loads(arguments_json)
        tool = TOOLS[name]
        return tool(**arguments)

    return (dispatch,)


@app.cell
def _(dispatch, json, mo, signal_cc_response):
    mo.stop(
        not signal_cc_response.choices[0].message.tool_calls,
        mo.md("*Run a prompt above that triggers a tool call first.*"),
    )
    _call = signal_cc_response.choices[0].message.tool_calls[0]
    _result = dispatch(_call.function.name, _call.function.arguments)
    mo.md(f"`dispatch({_call.function.name!r}, ...)` returns:\n\n```json\n{json.dumps(_result, indent=2)}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The full loop

    That's one turn, and harness loops: call the model, check the signal, run the tool if it's a tool call, feed the result back, and call again, until the signal says this is a final answer or you hit a turn limit. The limit is there because nothing in the model knows when to stop. It can keep asking for tools indefinitely, so limiting the loop is the harness's job. This is the loop underneath most tool-calling agent frameworks, plus more scaffolding.

    Feeding the result back is where Chat Completions and Responses diverge again.

    Chat Completions has no server-side memory, so every call resends the entire conversation. You append the assistant's tool-call message, then a `role: "tool"` message carrying the result and the matching `tool_call_id`, and send the whole list again.

    Responses can be stateful. Pass `previous_response_id` and send only the new `function_call_output` item, carrying the `call_id` from the call you're answering. The server reattaches it to the conversation it already has. That works because OpenAI stores the response on their side, which is the "runs on their servers" point from the intro, and it's why these calls go through the OpenAI-only pass-through. It's also optional. You can use Responses statelessly by resending the full input, including the `function_call` item next to its output, the same way you would with Chat Completions. The trade-off is that a stateful conversation lives on OpenAI's servers, and a stateless one lives in your code.

    A single response can contain more than one tool call. Ask for the weather in Lisbon and the EUR to USD rate, and the model may request both at once. Each call needs its own result: one `tool` message per `tool_call_id` in Chat Completions, one `function_call_output` per `call_id` in Responses.
    """)
    return


@app.cell
def _(cc_client, cc_tools, dispatch, json):
    def run_cc_loop(model: str, user_message: str, max_turns: int = 4) -> str:
        messages = [{"role": "user", "content": user_message}]
        for _ in range(max_turns):
            response = cc_client.chat.completions.create(model=model, messages=messages, tools=cc_tools)
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                return choice.message.content
            messages.append(choice.message.model_dump(exclude_none=True))
            for call in choice.message.tool_calls:
                result = dispatch(call.function.name, call.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
        return "(gave up after max_turns)"

    return (run_cc_loop,)


@app.cell
def _(RESPONSES_MODEL, dispatch, json, responses_client, responses_tools):
    def run_responses_loop(user_message: str, max_turns: int = 4) -> str:
        response = responses_client.responses.create(
            model=RESPONSES_MODEL, input=user_message, tools=responses_tools
        )
        for _ in range(max_turns):
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text
            outputs = [
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(dispatch(call.name, call.arguments)),
                }
                for call in calls
            ]
            response = responses_client.responses.create(
                model=RESPONSES_MODEL,
                input=outputs,
                tools=responses_tools,
                previous_response_id=response.id,
            )
        return "(gave up after max_turns)"

    return (run_responses_loop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run a prompt through both full loops:
    """)
    return


@app.cell
def _(mo):
    loop_prompt = mo.ui.text_area(
        value="What's the weather in Lisbon, and what's 50 USD in EUR?",
        label="Prompt",
        full_width=True,
    )
    loop_button = mo.ui.run_button(label="Run")
    mo.vstack([loop_prompt, loop_button])
    return loop_button, loop_prompt


@app.cell
def _(
    cc_model_picker,
    loop_button,
    loop_prompt,
    mo,
    run_cc_loop,
    run_responses_loop,
):
    mo.stop(not loop_button.value, mo.md("*Write a prompt, click Run.*"))

    loop_cc_answer = run_cc_loop(cc_model_picker.value, loop_prompt.value)
    loop_responses_answer = run_responses_loop(loop_prompt.value)
    mo.vstack(
        [
            mo.md(f"**Chat Completions loop ({cc_model_picker.value}):**\n\n{loop_cc_answer}"),
            mo.md(f"**Responses loop (gpt-4o-mini):**\n\n{loop_responses_answer}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    The field names differ, but the loop is identical in both APIs: the model asks for a tool, the harness runs it, and the model writes its answer from the result. Agent frameworks are mostly this loop with more scaffolding around it.
    """)
    return


if __name__ == "__main__":
    app.run()
