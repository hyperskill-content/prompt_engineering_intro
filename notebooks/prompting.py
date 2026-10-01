import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Introduction to prompting")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Introduction to prompting

    Users ask questions like "show me popular Python repos from last month". The GitHub API needs `language:python created:>2024-10-29 sort:stars`.
    We'll work with the GitHub API as our example, translating requests like "show me popular Python repos from last month" into formatted query strings with the right parameters, filters, and syntax.
    This notebook shows how to write prompts to support this translation. We'll cover parameter extraction, handling ambiguous requests, and preventing common failures.

    Let's load environment variables and initialize the client, pointed at the proxy. The `call_llm()` function sends prompts to the model with optional system instructions and returns the response text.

    Most cells use `gpt-5.6-luna`, a current reasoning model. A few also use `gpt-4o-mini`, a non-reasoning model, in the places where the difference between the two is important. `gpt-5.6-luna` is slower and bills for hidden reasoning tokens, so a call costs more than its price per token suggests.
    """)
    return


@app.cell
def _():
    import json
    import os
    from datetime import date
    from typing import Literal

    from dotenv import load_dotenv
    from openai import OpenAI
    from pydantic import BaseModel, ValidationError, field_validator

    load_dotenv(override=True)

    # OPENAI_BASE_URL is the proxy's OpenAI-only pass-through (".../openai"). Strip the
    # suffix and add "/v1" to reach the unified endpoint, which routes any model name to
    # its provider, so swapping a model below is a one-line change.
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"].removesuffix("/openai") + "/v1",
    )

    REASONING_MODEL = "gpt-5.6-luna"
    NON_REASONING_MODEL = "gpt-4o-mini"
    return (
        BaseModel,
        Literal,
        NON_REASONING_MODEL,
        REASONING_MODEL,
        ValidationError,
        client,
        date,
        field_validator,
        json,
    )


@app.cell
def _(REASONING_MODEL, client):
    def call_llm(
        prompt: str,
        system_prompt: str | None = None,
        model: str = REASONING_MODEL,
        **kwargs,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content

    return (call_llm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Message roles

    `call_llm()` sends the model a list of messages, each with a role: `system` (optional
    instructions for how to behave), `user` (the request), or `assistant` (the model's own prior
    replies, used in multi-turn conversations).

    Edit the system and user prompts below and click Run to compare the response with and without
    the system prompt.
    """)
    return


@app.cell
def _():
    roles_system_input = mo.ui.text_area(
        value="You are a terse senior engineer. Answer in a single sentence, no caveats.",
        label="System prompt",
        full_width=True,
    )
    roles_user_input = mo.ui.text_area(
        value="What's the difference between a GET and a POST request?",
        label="User prompt",
        full_width=True,
    )
    roles_run_button = mo.ui.run_button(label="Run")
    mo.vstack([roles_system_input, roles_user_input, roles_run_button])
    return roles_run_button, roles_system_input, roles_user_input


@app.cell
def _(call_llm, roles_run_button, roles_system_input, roles_user_input):
    mo.stop(not roles_run_button.value, mo.md("_Click **Run** to compare the two responses._"))

    _without = call_llm(roles_user_input.value)
    _with = call_llm(roles_user_input.value, system_prompt=roles_system_input.value)
    mo.vstack(
        [
            mo.md(f"**Without system prompt:**\n\n{_without}"),
            mo.md(f"**With your system prompt:**\n\n{_with}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling parameters: temperature

    `temperature` controls how much randomness goes into picking the next token. Low temperature makes the model consistently pick its highest-probability continuation; high temperature makes it more likely to pick lower-probability tokens.

    This demo uses `gpt-4o-mini`. `gpt-5.6-luna` only accepts the default temperature of 1 and returns an error for any other value, which is common for OpenAI's reasoning models specifically. Set a temperature below and run the same prompt three times.
    """)
    return


@app.cell
def _():
    temperature_slider = mo.ui.slider(
        start=0.0, stop=1.5, step=0.1, value=0.0, label="temperature", show_value=True
    )
    temperature_run_button = mo.ui.run_button(label="Run 3x")
    mo.vstack([temperature_slider, temperature_run_button])
    return temperature_run_button, temperature_slider


@app.cell
def _(
    NON_REASONING_MODEL,
    call_llm,
    temperature_run_button,
    temperature_slider,
):
    mo.stop(
        not temperature_run_button.value,
        mo.md("_Click **Run 3x** to sample the model three times at this temperature._"),
    )

    _prompt = "Suggest one fun, short name for a Python repo about weather forecasting. Respond with just the name."
    _samples = [
        call_llm(_prompt, model=NON_REASONING_MODEL, temperature=temperature_slider.value)
        for _ in range(3)
    ]
    mo.md("\n".join(f"{i}. {s}" for i, s in enumerate(_samples, 1)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At `temperature=0` the three samples should come back identical or nearly so. Turn the slider
    up and re-run to see them diverge. Use low temperature for tasks with one right answer
    (extraction, classification, code) and higher temperature for tasks that benefit from variety.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Zero-shot and few-shot prompting
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We'll start with zero-shot prompting — giving the model a task with no examples. Then we'll add examples (few-shot) to improve consistency.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Ask the model to convert the request directly, specifying only the output format. The cell below sends the same prompt to both models:
    """)
    return


@app.cell
def _():
    zero_shot_button = mo.ui.run_button(label="Run on both models (2 calls)")
    zero_shot_button
    return (zero_shot_button,)


@app.cell
def _(NON_REASONING_MODEL, REASONING_MODEL, call_llm, zero_shot_button):
    mo.stop(not zero_shot_button.value, mo.md("_Click above to run the zero-shot prompt._"))

    _user_request = "Find Python repositories created in 2024 with over 1000 stars"
    _zero_shot_prompt = f"""Convert this request into GitHub API search parameters:
    "{_user_request}"

    Return as JSON with these fields:
    - q: the search query string
    - sort: sort field (stars, forks, updated)
    - order: asc or desc
    """

    for _model in (NON_REASONING_MODEL, REASONING_MODEL):
        print(f"--- {_model} ---")
        print(call_llm(_zero_shot_prompt, model=_model))
        print()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model correctly extracts the key components: language filter, star threshold, and date range. It also makes reasonable assumptions — sorting by stars in descending order — since the request implied we want the most popular repos. In our runs, `gpt-4o-mini` wrapped the JSON in an explanation and markdown code fences, which will break if you try to parse it directly with `json.loads()`, while `gpt-5.6-luna` returned the bare JSON. Either way, asking for JSON in the prompt doesn't guarantee you get it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we provide examples showing input-output pairs. This teaches the model the expected format and demonstrates how to handle different query types. The cell below sends the same request to `gpt-4o-mini` with and without the examples, since that's the model where the difference shows:
    """)
    return


@app.cell
def _():
    few_shot_button = mo.ui.run_button(label="Run zero-shot and few-shot (2 calls)")
    few_shot_button
    return (few_shot_button,)


@app.cell
def _(NON_REASONING_MODEL, call_llm, few_shot_button):
    mo.stop(not few_shot_button.value, mo.md("_Click above to compare zero-shot and few-shot._"))

    _user_request = "Find TypeScript repositories from 2024 with more than 500 stars and active issues"
    _zero_shot_prompt = f"""Convert this request into GitHub API search parameters:
    "{_user_request}"

    Return as JSON with these fields:
    - q: the search query string
    - sort: sort field (stars, forks, help-wanted-issues, updated)
    - order: asc or desc
    """
    _few_shot_prompt = f"""Convert natural language requests into GitHub API search parameters.

    Examples:

    Request: "Find popular JavaScript repos"
    Output: {{
      "q": "language:javascript",
      "sort": "stars",
      "order": "desc"
    }}

    Request: "Show me repos about machine learning, most recently updated first"
    Output: {{
      "q": "topic:machine-learning",
      "sort": "updated",
      "order": "desc"
    }}

    Request: "Python projects with good first issues"
    Output: {{
      "q": "language:python good-first-issues:>0",
      "sort": "stars",
      "order": "desc"
    }}


    Now convert this request: {_user_request}


    Output:"""

    for _label, _prompt in (("zero-shot", _zero_shot_prompt), ("few-shot", _few_shot_prompt)):
        print(f"--- {_label} ({NON_REASONING_MODEL}) ---")
        print(call_llm(_prompt, model=NON_REASONING_MODEL))
        print()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In our runs, the examples got `gpt-4o-mini` to drop the explanation and copy the exact field
    layout. It still wrapped the JSON in code fences, though, and "active issues" still turned into
    a qualifier repository search doesn't support: `is:open` or `is:issue` in some runs, an
    invented one like `issues:>0` in others. None of the examples covered that case. Examples are
    good at shaping format and weaker at teaching the model what the API accepts. Both problems
    come back later in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Chain-of-thought
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Chain-of-thought makes the model show its reasoning before answering. Ask the model to reason
    through ambiguous terms before generating the query. "Trending" could mean recent stars,
    activity, or creation date. "Beginner-friendly" might map to labels like `good-first-issue` or
    documentation tags. Explicit reasoning reduces inconsistent interpretations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Asking for step-by-step reasoning matters on a non-reasoning model like `gpt-4o-mini`.
    Reasoning models like `gpt-5.6-luna` already reason before answering, whether you ask or not,
    and bill for it as hidden reasoning tokens. The cell below runs the step-by-step prompt on
    `gpt-4o-mini` and the plain request on `gpt-5.6-luna`, and shows how many hidden reasoning
    tokens each one spent.
    """)
    return


@app.cell
def _():
    cot_button = mo.ui.run_button(label="Run both (2 calls)")
    cot_button
    return (cot_button,)


@app.cell
def _(NON_REASONING_MODEL, REASONING_MODEL, client, cot_button):
    mo.stop(not cot_button.value, mo.md("_Click above to compare the two approaches._"))

    _user_request = "Find trending AI repositories that are beginner-friendly"
    _cot_prompt = f"""Convert this request into GitHub API search parameters:
    {_user_request}

    Think step-by-step:
    1. What does "trending" mean? (recent activity, stars, etc.)
    2. How do we identify "AI" repositories?
    3. What makes a repo "beginner-friendly"?
    4. What filters should we combine?

    Then provide the final JSON output."""
    _plain_prompt = f"""Convert this request into GitHub API search parameters:
    {_user_request}

    Provide the final JSON output."""

    for _model, _label, _prompt in (
        (NON_REASONING_MODEL, "step-by-step prompt", _cot_prompt),
        (REASONING_MODEL, "plain prompt", _plain_prompt),
    ):
        _response = client.chat.completions.create(
            model=_model, messages=[{"role": "user", "content": _prompt}]
        )
        _details = _response.usage.completion_tokens_details
        _reasoning_tokens = (_details.reasoning_tokens if _details else 0) or 0
        print(f"--- {_model}, {_label} ---")
        print(f"hidden reasoning tokens: {_reasoning_tokens}")
        print(_response.choices[0].message.content)
        print()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `gpt-4o-mini` writes its reasoning into the visible answer and spends no hidden tokens.
    `gpt-5.6-luna` writes a short answer after hundreds of hidden ones. Notice also that the
    step-by-step answer mixes reasoning text with the JSON, so it can't be parsed directly. The
    structured-output schema below gives reasoning its own field instead.

    Visible reasoning (which is present in open-source reasoning models) helps with debugging, but it isn't a reliable check on correctness.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Structured output and validation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Asking for JSON in the prompt doesn't guarantee you'll get parseable, schema-compliant output.
    The model might add markdown, rearrange fields, or return malformed JSON.
    """)
    return


@app.cell
def _():
    prompt_only_button = mo.ui.run_button(label="Run (1 call)")
    prompt_only_button
    return (prompt_only_button,)


@app.cell
def _(call_llm, prompt_only_button):
    mo.stop(not prompt_only_button.value, mo.md("_Click above to run the prompt-only version._"))

    _user_request = "Show me the best React libraries from last year"
    _prompt_only = f"""Convert this request into GitHub API search parameters:
    {_user_request}

    Return ONLY valid JSON with this structure:
    {{
      "query_params": {{...}},
      "reasoning": "...",
      "confidence": "high|medium|low"
    }}
    """

    _response = call_llm(_prompt_only)
    # No guarantee this is valid JSON or follows the schema
    print(_response)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    OpenAI's structured output mode guarantees the response matches your schema. Define the structure with Pydantic, and the API will only return valid objects (not semantically correct, but structurally valid objects)
    """)
    return


@app.cell
def _(BaseModel, Literal):
    class QueryParams(BaseModel):
        q: str
        sort: Literal["stars", "forks", "help-wanted-issues", "updated"]
        order: Literal["asc", "desc"]
        per_page: int = 30

    class GitHubQueryResponse(BaseModel):
        query_params: QueryParams
        reasoning: str
        potential_issues: list[str]
        confidence: Literal["high", "medium", "low"]

    return (GitHubQueryResponse,)


@app.cell
def _(BaseModel, REASONING_MODEL, client):
    def call_llm_structured(
        prompt: str,
        response_format: type[BaseModel],
        system_prompt: str | None = None,
        model: str = REASONING_MODEL,
    ) -> BaseModel:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.parsed

    return (call_llm_structured,)


@app.cell
def _():
    structured_button = mo.ui.run_button(label="Run the structured examples (3 calls)")
    structured_button
    return (structured_button,)


@app.cell
def _(GitHubQueryResponse, call_llm_structured, json, structured_button):
    mo.stop(not structured_button.value, mo.md("_Click above to run the structured examples._"))

    _user_request = "Show me the best React libraries from last year"
    _structured_prompt = f"""Convert this request into GitHub API search parameters:
    "{_user_request}"

    Consider:
    - What does "best" mean?
    - How to filter for "last year"?
    - Flag any ambiguities or assumptions"""

    _result = call_llm_structured(_structured_prompt, GitHubQueryResponse)

    # result is now a GitHubQueryResponse object, guaranteed to match schema
    print(f"Query: {_result.query_params.q}")
    print(f"Sort: {_result.query_params.sort}")
    print(f"Confidence: {_result.confidence}")
    print(f"Issues: {_result.potential_issues}")

    # Can also convert to dict
    print(json.dumps(_result.model_dump(), indent=2))
    return


@app.cell
def _(
    GitHubQueryResponse,
    call_llm,
    call_llm_structured,
    json,
    structured_button,
):
    mo.stop(not structured_button.value)

    # Test with an ambiguous request
    _ambiguous_request = "Find good repos for learning web development"

    print("=" * 60)
    print("Without structured output:")
    print("=" * 60)
    _prompt = f"""Convert to GitHub API params: "{_ambiguous_request}"
    Return JSON with: query_params, reasoning, potential_issues, confidence"""
    _unstructured_response = call_llm(_prompt)
    print(_unstructured_response)
    print("\nNotice: May have inconsistent format, extra text, invalid JSON\n")

    print("=" * 60)
    print("With structured output:")
    print("=" * 60)
    _structured_response = call_llm_structured(
        f"""Convert this request into GitHub API search parameters: "{_ambiguous_request}"

    Think about:
    - What makes a repo "good for learning"?
    - What does "web development" encompass?
    - What filters capture this?""",
        GitHubQueryResponse,
    )
    print(json.dumps(_structured_response.model_dump(), indent=2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Structured output guarantees the shape of the response: `q` will be a string, and `sort` one of
    the allowed values. It can't guarantee the content is valid for the API. Both models in this
    notebook will put `label:` into a repository search if the request mentions labels, and GitHub
    only supports that qualifier in issue search.

    A Pydantic validator can catch that on your side, after the response comes back. Nothing
    retries automatically: `parse()` raises a `ValidationError`, and your code decides what to do
    next.
    """)
    return


@app.cell
def _():
    validator_button = mo.ui.run_button(label="Run (1 call)")
    validator_button
    return (validator_button,)


@app.cell
def _(
    BaseModel,
    Literal,
    ValidationError,
    call_llm_structured,
    field_validator,
    validator_button,
):
    mo.stop(not validator_button.value, mo.md("_Click above to run a request the validator rejects._"))

    class StrictQueryParams(BaseModel):
        q: str
        sort: Literal["stars", "forks", "help-wanted-issues", "updated"]
        order: Literal["asc", "desc"]
        per_page: int = 30

        @field_validator("q")
        @classmethod
        def no_issue_qualifiers(cls, v: str) -> str:
            issue_only = ("label:", "state:", "is:issue", "is:pr", "is:open", "is:closed")
            found = [qualifier for qualifier in issue_only if qualifier in v]
            if found:
                raise ValueError(f"issue-search qualifiers in a repository search: {found}")
            return v

    _request = "Python repos with issues labeled good first issue"
    try:
        _result = call_llm_structured(
            f"Convert to GitHub repository search parameters: {_request}",
            response_format=StrictQueryParams,
        )
        _output = mo.md(f"Valid: `{_result.model_dump()}`")
    except ValidationError as e:
        _error = e.errors()[0]
        _output = mo.md(f"Rejected `q = {_error['input']!r}`\n\n> {_error['msg']}")
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    From here your code can retry with the error message added to the prompt, or fall back to a
    simpler query. The better fix is usually upstream: tell the model what repository search
    supports, which is what the prompt at the end of this notebook does.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Handling edge cases and ambiguity

    Users type "find some good python stuff" or "show me trending repos". The prompts need to
    surface ambiguity and document assumptions.
    """)
    return


@app.cell
def _():
    edge_cases_button = mo.ui.run_button(label="Run examples 1 to 3 (4 calls)")
    edge_cases_button
    return (edge_cases_button,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Example 1: Vague and ambiguous queries
    """)
    return


@app.cell
def _(GitHubQueryResponse, call_llm_structured, edge_cases_button, json):
    mo.stop(not edge_cases_button.value, mo.md("_Click **Run examples 1 to 3** above._"))

    _user_query = "find some good python stuff"
    _prompt = f"""Convert this user request to GitHub API parameters:
    {_user_query}

    This query is vague. In your response:
    1. Identify ALL ambiguous terms (e.g., "good", "stuff")
    2. Document your interpretation of each term
    3. List alternative interpretations the user might have meant
    4. Set confidence level appropriately

    Make reasonable assumptions but be transparent about them."""

    print("Prompt that surfaces ambiguity:")
    print("=" * 60)
    _response = call_llm_structured(_prompt, GitHubQueryResponse)
    print(json.dumps(_response.model_dump(), indent=2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `confidence` field is the model's own label for how sure it is, not a measured
    probability. It's useful for routing, for example sending low-confidence queries to a person,
    but it isn't a reliable estimate of accuracy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Example 2: Conflicting requirements

    User asks for "new projects with lots of commit history" - these might conflict. The following
    prompt explicitly asks the LLM to detect conflicts, explain prioritization decisions, and
    document trade-offs:
    """)
    return


@app.cell
def _(GitHubQueryResponse, call_llm_structured, edge_cases_button, json):
    mo.stop(not edge_cases_button.value)

    _conflicting_request = "Find new projects with lots of commit history"

    _conflict_aware_prompt = f"""Convert this user request to GitHub API parameters:
    "{_conflicting_request}"

    IMPORTANT: Check for logical conflicts between requirements.
    If you find conflicts:
    - Clearly identify them in potential_issues
    - Explain which requirement you prioritized and why
    - Suggest what the user might actually want

    Be explicit about trade-offs made."""

    print("Handling conflicting requirements:")
    print("=" * 60)
    _result = call_llm_structured(_conflict_aware_prompt, GitHubQueryResponse)
    print(json.dumps(_result.model_dump(), indent=2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The prompt forced the LLM to:

    1. Recognize the conflict
    2. Make a defensible choice
    3. Document it for the user
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Example 3: Impossible or invalid constraints

    User asks for something the GitHub API doesn't support (e.g., "repos with most downloads").
    The cell runs it twice: once as a plain request, and once with the API's supported filters
    spelled out.
    """)
    return


@app.cell
def _(
    GitHubQueryResponse,
    call_llm,
    call_llm_structured,
    edge_cases_button,
    json,
):
    mo.stop(not edge_cases_button.value)

    _invalid_request = "Find Python repos with the most downloads this week"

    print("Plain request, no constraints:")
    print("=" * 60)
    print(
        call_llm(
            f"""Convert this user request to GitHub API search parameters, as JSON with q, sort, order:
    "{_invalid_request}\""""
        )
    )
    print()

    # Prompt with API constraint awareness
    _api_aware_prompt = f"""Convert this user request to GitHub API parameters:
    "{_invalid_request}"

    CRITICAL: GitHub API search supports these filters:
    - language, stars, forks, size, created, pushed, topics, license
    - Does NOT support: downloads, npm/pip installs, weekly metrics

    If the request asks for unsupported features:
    1. Flag this clearly in potential_issues
    2. Suggest the closest supported alternative
    3. Explain why it's only an approximation
    4. Set confidence to 'low' if using poor proxy metrics"""

    print("Handling impossible constraints:")
    print("=" * 60)
    _result = call_llm_structured(_api_aware_prompt, GitHubQueryResponse)
    print(json.dumps(_result.model_dump(), indent=2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Without this, the LLM might:

    - Silently use `stars` as a proxy for `downloads`
    - Not warn the user about the limitation
    - Return results that don't match user intent

    Providing an explicit list of supported/unsupported features lets the LLM flag impossible
    requests and suggest alternatives instead of silently guessing.

    In our testing, the plain run did the first two: it sorted by stars and said nothing about
    downloads.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Example 4: Relative dates

    Requests like "last month" or "from last year" depend on today's date, which the model doesn't
    necessarily have. The cell below sends the same request to `gpt-4o-mini` three ways: with no
    date, with today's date as a bare statement in the system prompt, and with the date placed
    right next to the request.
    """)
    return


@app.cell
def _():
    date_button = mo.ui.run_button(label="Run all three (3 calls)")
    date_button
    return (date_button,)


@app.cell
def _(NON_REASONING_MODEL, call_llm, date, date_button):
    mo.stop(not date_button.value, mo.md("_Click above to run the request three ways._"))

    _today = date.today().isoformat()
    _request = (
        "Convert to a GitHub repo search query string, reply with just the query: "
        '"popular Python repos created last month"'
    )
    _no_date = call_llm(_request, model=NON_REASONING_MODEL, temperature=0)
    _system_date = call_llm(
        _request,
        system_prompt=f"Today's date is {_today}.",
        model=NON_REASONING_MODEL,
        temperature=0,
    )
    _inline_date = call_llm(
        f"Today's date is {_today}.\n\n{_request}", model=NON_REASONING_MODEL, temperature=0
    )
    mo.md(
        f"Today is {_today}.\n\n"
        f"**No date:** `{_no_date.strip('`')}`\n\n"
        f"**Date in the system prompt:** `{_system_date.strip('`')}`\n\n"
        f"**Date next to the request:** `{_inline_date.strip('`')}`"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Without the date, `gpt-4o-mini` has to guess, and it guesses a date from its training period.
    The middle result is the surprising one: in our testing, a bare "Today's date is ..." in the
    system prompt was ignored every time, and the model still used its training-era date. Placing
    the date in the user message, right next to the request, fixed it every time. So did keeping
    it in the system prompt with an explicit instruction to resolve relative dates from it.
    Including the date isn't always enough on its own; the model has to connect it to the task.

    `gpt-5.6-luna` got the date right even with no date in the prompt, which means something on the
    serving side supplies it. That isn't guaranteed across providers or models, so pass the date
    explicitly whenever a request depends on it.

    The same goes for anything else the model can't know on its own, like the user's time zone or
    which API version you're targeting. Injecting it into the prompt is more reliable than giving
    the model a tool to look it up, because a model that doesn't realize it's missing something
    won't call the tool.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Best practices for prompts

    Key techniques demonstrated above:
    """)
    return


@app.cell
def _(date):
    def build_prompt(user_request: str) -> str:
        return f"""Convert this user request to GitHub API search parameters:
    "{user_request}"

    CONTEXT - GitHub API Capabilities:
    - Today's date: {date.today().isoformat()}
    - Searchable: language, stars, forks, size, created/pushed dates, topics, license, good-first-issues, help-wanted-issues
    - NOT searchable: downloads, page views, active users, external metrics
    - Issue-search qualifiers (label:, state:, is:issue, is:open) do NOT work in repository search
    - Sort options: stars, forks, help-wanted-issues, updated
    - Time filters: Use created:>YYYY-MM-DD or pushed:>YYYY-MM-DD format

    YOUR TASK:
    1. Parse the user's intent from potentially vague language
    2. Map requirements to available GitHub filters
    3. Identify ambiguous terms (e.g., "good", "popular", "recent")
    4. Flag conflicts (e.g., "new" + "mature", "simple" + "feature-rich")
    5. Note impossible constraints and suggest alternatives
    6. Document ALL assumptions in the reasoning field
    7. List issues in potential_issues array
    8. Set confidence based on ambiguity level:
       - high: Clear, supported requirements
       - medium: Some ambiguity but reasonable interpretation
       - low: Heavy assumptions or unsupported features

    Make the best query possible, but be transparent about limitations."""

    return (build_prompt,)


@app.cell
def _():
    best_practices_button = mo.ui.run_button(label="Run the test cases (4 calls)")
    best_practices_button
    return (best_practices_button,)


@app.cell
def _(
    GitHubQueryResponse,
    best_practices_button,
    build_prompt,
    call_llm_structured,
):
    mo.stop(not best_practices_button.value, mo.md("_Click above to run `build_prompt` on four requests._"))

    # Test with multiple edge cases
    _test_cases = [
        "find good python stuff",
        "trending ML repos for beginners",
        "repos with most npm downloads",
        "new projects with stable APIs",
    ]

    print("Prompt in action:")
    print("=" * 60)
    for _test_query in _test_cases:
        print(f'\nQuery: "{_test_query}"')
        print("-" * 60)
        _result = call_llm_structured(build_prompt(_test_query), GitHubQueryResponse)
        print(f"Search: {_result.query_params.q}")
        print(f"Confidence: {_result.confidence}")
        if _result.potential_issues:
            print(f"Issues: {', '.join(_result.potential_issues[:2])}...")  # show first 2
        print()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Try it yourself

    Type your own request and run it through `build_prompt`:
    """)
    return


@app.cell
def _():
    playground_input = mo.ui.text_area(
        value="show me abandoned but popular JS projects",
        label="User request",
        full_width=True,
    )
    playground_run_button = mo.ui.run_button(label="Run")
    mo.vstack([playground_input, playground_run_button])
    return playground_input, playground_run_button


@app.cell
def _(
    GitHubQueryResponse,
    build_prompt,
    call_llm_structured,
    json,
    playground_input,
    playground_run_button,
):
    mo.stop(not playground_run_button.value, mo.md("_Click **Run** to convert your request._"))

    _result = call_llm_structured(build_prompt(playground_input.value), GitHubQueryResponse)
    mo.md(f"```json\n{json.dumps(_result.model_dump(), indent=2)}\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    Most of what made the GitHub queries reliable came from telling the model things it couldn't
    work out alone, like which filters the API supports, which it doesn't, and what today's date is.
    Asking it to flag ambiguity, conflicts and unsupported requests turned guesses into more explicit assumptions. The `confidence` label it returns is useful for routing, as long as you don't read it as a probability.

    Checking the output happens at two levels. Structured outputs guarantee the shape of the
    response, and validators in your own code catch content that has the right shape but is wrong
    for the API, like `label:` in a repository search. Which model you use shifts where the effort
    goes: a reasoning model already thinks before answering and needed fewer formatting hints here,
    while a non-reasoning one benefited from explicit steps and examples.
    """)
    return


if __name__ == "__main__":
    app.run()
