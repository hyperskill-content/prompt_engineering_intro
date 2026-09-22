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
    # The model zoo

    There are a lot of language models out there, from a lot of different providers. This notebook is a quick look at the landscape.

    From there we'll look at benchmarks. They're mostly for telling you roughly what tier a model sits in, not for predicting how it will do on your task. We'll finish with a repeatable way to decide when a cheaper model is fine for part of a pipeline.
    """)
    return


@app.cell
def _():
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    import requests
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI, ThreadPoolExecutor, as_completed, os, pd, requests, time


@app.cell
def _(OpenAI, os):
    api_key = os.getenv("OPENAI_API_KEY")
    # OPENAI_BASE_URL points at the OpenAI-only pass-through (".../openai").
    # The root proxy also exposes a unified endpoint that routes each model
    # name to its real provider; that's what /model/info describes.
    base_url = os.environ["OPENAI_BASE_URL"].removesuffix("/openai")
    client = OpenAI(api_key=api_key, base_url=f"{base_url}/v1")
    return api_key, base_url, client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model landscape

    The providers can be divided into roughly three groups.

    The big closed-weight labs (OpenAI, Anthropic, Google, Amazon) sell access through an API and keep their flagship weights private. These are the names everyone already knows. The open-weight labs (Alibaba's Qwen, DeepSeek, Zhipu's GLM, Moonshot's Kimi, and a growing list of others) publish weights you can download, run yourself, or rent from any of a dozen inference providers. Then there are thousands of task-specific fine-tunes on Hugging Face, mostly built by small teams or individuals for one narrow job: a [classifier](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending), a [translator](https://huggingface.co/models?pipeline_tag=translation&sort=trending), a [summarizer](https://huggingface.co/models?pipeline_tag=summarization&sort=trending), etc.

    New releases happen almost daily, and which model counts as the best shifts on a similar timescale. What we have access to, through the course's litellm proxy, is a curated list of the first two groups. The cell below fetches the proxy's catalog, showing every configured model, its provider, and its cost.
    """)
    return


@app.cell
def _(api_key, base_url, requests):
    def fetch_catalog():
        resp = requests.get(
            f"{base_url}/model/info",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    catalog = fetch_catalog()
    return (catalog,)


@app.cell
def _(catalog, pd):
    def to_row(entry):
        info = entry.get("model_info") or {}
        input_cost = info.get("input_cost_per_token")
        output_cost = info.get("output_cost_per_token")
        return {
            "model": entry.get("model_name"),
            "provider": info.get("litellm_provider") or "unknown",
            "mode": info.get("mode") or "unknown",
            "context_window": info.get("max_input_tokens"),
            "$ / 1M input tok": round(input_cost * 1_000_000, 3) if input_cost else None,
            "$ / 1M output tok": round(output_cost * 1_000_000, 3) if output_cost else None,
        }

    catalog_df = (
        pd.DataFrame(to_row(e) for e in catalog)
        .sort_values(["provider", "model"])
        .reset_index(drop=True)
    )
    return (catalog_df,)


@app.cell
def _(catalog_df, mo):
    provider_counts = catalog_df["provider"].value_counts()
    mo.md(
        f"""
        **{len(catalog_df)} model names** configured, across
        **{catalog_df['provider'].nunique()} providers**:
        {", ".join(f"{v} × `{k}`" for k, v in provider_counts.items())}.
        """
    )
    return


@app.cell
def _(catalog_df, mo):
    mo.ui.table(catalog_df, label="Configured models (sortable, filterable)")
    return


@app.cell
def _(mo):
    mo.md("""
    Same data, grouped down to just the providers:
    """)
    return


@app.cell
def _(catalog_df, mo):
    providers_df = (
        catalog_df["provider"]
        .value_counts()
        .rename_axis("provider")
        .reset_index(name="model count")
    )
    mo.ui.table(providers_df, label="Providers configured on the proxy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What's in the instance?

    Some models are deprecated aliases or point at an expired key. Pick a few model names (substrings are fine) and make requests.
    """)
    return


@app.cell
def _(mo):
    models_to_test = mo.ui.text(
        value="gpt-4o-mini, gpt-4.1-mini, claude-3-5-sonnet-20241022, "
        "codestral-latest, llama-3.1-8b-instant, gemma2-9b-it, "
        "text-embedding-3-small",
        label="Models to test (comma-separated substrings)",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Run test drive")
    mo.vstack([models_to_test, run_button])
    return models_to_test, run_button


@app.cell
def _(catalog_df, models_to_test):
    filters = [f.strip().lower() for f in models_to_test.value.split(",") if f.strip()]
    matched_models = sorted(
        {m for m in catalog_df["model"] for f in filters if f in m.lower()}
    )
    return (matched_models,)


@app.cell
def _(
    ThreadPoolExecutor,
    as_completed,
    catalog,
    client,
    matched_models,
    mo,
    run_button,
    time,
):
    mo.stop(
        not run_button.value,
        mo.md(f"*Matched {len(matched_models)} models. Click **Run test drive** above to test them.*"),
    )

    mode_by_model = {
        e["model_name"]: (e.get("model_info") or {}).get("mode") or "chat" for e in catalog
    }

    def test_one(model_name):
        mode = mode_by_model.get(model_name, "chat")
        start = time.time()
        try:
            if mode == "embedding":
                resp = client.embeddings.create(model=model_name, input="hello world", timeout=20)
                detail = f"embedding dims={len(resp.data[0].embedding)}"
            elif mode in ("chat", "completion"):
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
                    max_tokens=10,
                    timeout=20,
                )
                detail = (resp.choices[0].message.content or "").strip()
            else:
                return {
                    "model": model_name, "mode": mode, "status": "SKIPPED",
                    "seconds": 0, "detail": f"no generic test for mode={mode}",
                }
            return {
                "model": model_name, "mode": mode, "status": "OK",
                "seconds": round(time.time() - start, 2), "detail": detail,
            }
        except Exception as e:
            return {
                "model": model_name, "mode": mode, "status": "FAIL",
                "seconds": round(time.time() - start, 2), "detail": str(e)[:200],
            }

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(test_one, m) for m in matched_models]
        test_results = [f.result() for f in as_completed(futures)]
    return (test_results,)


@app.cell
def _(pd, test_results):
    results_df = pd.DataFrame(test_results).sort_values(["status", "model"]).reset_index(drop=True)
    return (results_df,)


@app.cell
def _(mo, results_df):
    counts = results_df["status"].value_counts()
    mo.md(
        f"**{counts.get('OK', 0)} OK · {counts.get('FAIL', 0)} FAIL · "
        f"{counts.get('SKIPPED', 0)} SKIPPED** out of {len(results_df)} tested."
    )
    return


@app.cell
def _(mo, results_df):
    mo.ui.table(results_df, label="Test drive results")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmarks: what to trust, what not to

    Every model release comes with a chart. [MMLU](https://artificialanalysis.ai/evaluations/mmlu-pro), [GPQA](https://artificialanalysis.ai/evaluations/gpqa-diamond), [SWE-bench](https://www.swebench.com/), whatever's fashionable this quarter, usually with the new model's bar just ahead of everyone else's. That number is a hint about roughly what tier a model is in, and not much more. A few things are going on underneath it.

    Questions leak into training data, deliberately or not (contamination), so a high score can mean the model has seen these questions before rather than that it reasons well. Labs also choose which benchmarks to headline, and can pick prompting or scaffolding that flatters their own model. And older benchmarks like MMLU and HellaSwag are close to maxed out across frontier models (saturation), so a percentage-point difference there tells you almost nothing.

    Benchmarks are still useful for telling you a model exists and roughly what class it's in. They're not a substitute for testing on your own task. To make that concrete, here are two models, roughly comparable in tier and price, given the same slightly awkward instruction:
    """)
    return


@app.cell
def _(mo):
    bench_button = mo.ui.run_button(label="Run the comparison (2 calls)")
    bench_button
    return (bench_button,)


@app.cell
def _(bench_button, client, mo):
    mo.stop(not bench_button.value, mo.md("*Click above to run both models on the same prompt.*"))

    bench_prompt = 'Write a sentence that ends in the word "the".'
    bench_models = ["gpt-4o-mini", "mistral-small-latest"]

    bench_replies = {
        model: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": bench_prompt}],
            max_tokens=150,
        )
        .choices[0]
        .message.content
        for model in bench_models
    }

    mo.vstack(
        [mo.md(f"**{model}**:\n\n> {reply}") for model, reply in bench_replies.items()]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The models are roughly same tier with the same prompt: write a sentence that ends in the word "the". Both models technically comply, but neither writes a real sentence. GPT-4o-mini trails off mid-thought, and Mistral hands back a fragment wrapped in commentary. Nothing in the benchmark scores would have shown this, and the only way to find out is to run it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing models for a pipeline

    The rest of this notebook works through an example: a support-ticket triage pipeline.
    Given an incoming ticket, it needs to categorize it, pull out structured fields, score how
    urgent it is, and draft a reply.

    Five steps, in order:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: prototype with the most capable model

    Before deciding anything is worth optimizing, check that the task is solvable. Run the whole pipeline, end to end, on the most capable model available. This establishes a ceiling.

    If the frontier model can't do the task, the problem is probably the task, prompt, or data design, and that's what needs fixing, not the choice of model. If it succeeds, a working design exists, and it's safe to start decomposing and optimizing the components.
    """)
    return


@app.cell
def _():
    sample_ticket = (
        "Hi, I'm Sarah Chen (order #48291). My wireless headphones (SoundWave Pro) stopped "
        "charging after two weeks, and I have a flight tomorrow morning I was hoping to use "
        "them for. Can someone help ASAP?"
    )
    return (sample_ticket,)


@app.cell
def _(mo):
    proto_button = mo.ui.run_button(label="Run the frontier model on the full ticket (1 call)")
    proto_button
    return (proto_button,)


@app.cell
def _(client, mo, proto_button, sample_ticket):
    mo.stop(not proto_button.value, mo.md("*Click above to run the full pipeline in one call.*"))

    proto_prompt = f"""Read this support ticket and handle it completely:

    "{sample_ticket}"

    Return JSON with these fields:
    - category: one of "billing", "technical", "account", "other"
    - extracted_fields: object with any of customer_name, order_number, product, deadline you can find
    - urgency: one of "Low", "Medium", "High"
    - draft_reply: a short, empathetic reply to send the customer

    Return only the JSON, no other text."""

    proto_response = (
        client.chat.completions.create(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": proto_prompt}],
        )
        .choices[0]
        .message.content
    )
    return (proto_response,)


@app.cell
def _(mo, proto_response):
    mo.md(f"""
    ```json\n{proto_response}\n```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A single call to a frontier model handles all four subtasks correctly. So the question from here is whether every piece of the final task needs to run on the most expensive model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: isolate subtasks

    Break the end-to-end flow into pieces and ask the same four questions of each one:

    - Is the output space closed (a fixed set of labels) or open-ended (free text)?
    - Is this a well-established task type, or something novel the model has to reason through
      fresh?
    - Does it need broad context or world knowledge, or is it self-contained given just the ticket?
    - What does a subtle failure cost?

    | Subtask | Output space | Established? | Self-contained? | Cost of a subtle failure |
    |---|---|---|---|---|
    | Categorize | closed (4 labels) | yes, classification | yes | low: misrouted ticket, recoverable |
    | Extract fields | closed-ish (structured) | yes, extraction | yes | low-medium: a missed field is usually noticed downstream |
    | Score urgency | closed (3 labels) | yes, classification | yes | medium: a silently under-scored urgent ticket sits in the wrong queue |
    | Draft reply | open-ended | somewhat, needs judgment | no, needs policy/tone context | high: a bad reply reaches the customer directly |

    Three of these look like reasonable candidates for a cheaper model: the output is closed, the task shape is established, the ticket is self-contained, and a failure is either cheap or likely to get caught downstream. Urgency is the borderline one, since a wrong score fails silently, but it still qualifies. Drafting the reply doesn't: it's open-ended, needs context this prompt doesn't carry (policy, brand voice), and that one needs a human reviewing replies before they
    go out, whichever model drafts them, so it sits outside this cost question entirely.

    We'll follow the scoring urgency. It's the borderline case, which makes it the most interesting one to test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: find a cheaper candidate

    This proxy doesn't have a model fine-tuned for something like urgency scoring. The catalog above is general-purpose foundation models, not the long tail of task-specific fine-tunes from the landscape section. What it does have is a range of sizes and prices. Sorted by cost per output token, the cheapest chat models are the small, fast tier of each provider's lineup:
    """)
    return


@app.cell
def _(catalog_df, mo):
    cheap_candidates = (
        catalog_df[catalog_df["mode"] == "chat"]
        .dropna(subset=["$ / 1M output tok"])
        .sort_values("$ / 1M output tok")
        .head(10)
    )
    mo.ui.table(cheap_candidates, label="Cheapest chat models on the proxy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A specialized model would be a fine-tune built for exactly that kind of classification, the sort you'd find by searching Hugging Face by task tag rather than in a general chat provider's catalog. Two examples give a sense of the range: [a sentiment classifier tagged for customer feedback](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) that's been downloaded widely, and [a customer support ticket classifier](https://huggingface.co/interneuronai/customer_support_ticket_classification_pegasus) that has barely been touched. Downloads aren't proof of quality, just a weaker version of the benchmark problem. At best they tell you people have used a model, not that it works on your task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: build a small task-specific eval set

    Public benchmarks won't tell you whether `ministral-3b-latest` can score urgency on *your* tickets. The only way to find out is to build a small set of examples with expected answers and run both models against it. The set below has 15. A production pipeline would want more, but the point holds.
    """)
    return


@app.cell
def _():
    eval_set = [
        ("Just wanted to say the support team was great last time, thanks!", "Low"),
        ("Quick question, does the premium plan include API access?", "Low"),
        ("Love the new dashboard redesign, much easier to use now.", "Low"),
        ("Is there a dark mode option somewhere in settings?", "Low"),
        ("Do you offer discounts for students?", "Low"),
        ("My package arrived damaged, the box was crushed and the item inside is broken. "
         "I'd like a replacement.", "Medium"),
        ("The app crashes every time I open the settings page. Not a huge deal but wanted "
         "to flag it.", "Medium"),
        ("Can you update the shipping address on my last order? It hasn't shipped yet.", "Medium"),
        ("The color of the product I received doesn't quite match the photo, but it's still "
         "usable. Not asking for a return, just flagging feedback.", "Medium"),
        ("My data export has been \"processing\" for two hours. Not urgent yet, just checking in.",
         "Medium"),
        ("URGENT: I was charged twice for my subscription this month and my card is now over "
         "its limit, I need this fixed today.", "High"),
        ("My account was locked after I tried logging in from a new device, and I can't access "
         "any of my saved data before an important presentation in an hour.", "High"),
        ("I've emailed three times about my refund and haven't heard back in two weeks. This "
         "is getting ridiculous.", "High"),
        ("I think there might be a security issue: I'm seeing login attempts from a country "
         "I've never visited in my account activity log.", "High"),
        ("Our team's workspace has been down for 40 minutes and we have a client demo starting "
         "in 15 minutes. Please help immediately.", "High"),
    ]
    return (eval_set,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5: swap and measure

    Run both models against every example, compare each answer to the expected label, and look at
    the actual diffs, not just an aggregate score. The design was already validated in step 1;
    this is only deciding whether this one subtask can move to the cheaper model.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Run both models against all 15 examples (30 calls)")
    eval_button
    return (eval_button,)


@app.cell
def _(client, eval_button, eval_set, mo):
    mo.stop(
        not eval_button.value,
        mo.md("*Click above to run `claude-sonnet-4-5` and `ministral-3b-latest` against the eval set.*"),
    )

    def score_urgency(model, ticket):
        prompt = (
            "Classify the urgency of this customer support ticket as exactly one word: "
            f'Low, Medium, or High.\n\nTicket: "{ticket}"'
        )
        reply = (
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            .choices[0]
            .message.content
        )
        return reply.strip().strip(".*").strip()

    eval_rows = []
    for ticket, expected in eval_set:
        frontier_answer = score_urgency("claude-sonnet-4-5", ticket)
        candidate_answer = score_urgency("ministral-3b-latest", ticket)
        eval_rows.append(
            {
                "ticket": ticket[:60] + ("..." if len(ticket) > 60 else ""),
                "expected": expected,
                "claude-sonnet-4-5": frontier_answer,
                "frontier correct": frontier_answer == expected,
                "ministral-3b-latest": candidate_answer,
                "candidate correct": candidate_answer == expected,
            }
        )
    return (eval_rows,)


@app.cell
def _(eval_rows, mo, pd):
    eval_df = pd.DataFrame(eval_rows)
    frontier_acc = eval_df["frontier correct"].mean()
    candidate_acc = eval_df["candidate correct"].mean()
    mo.md(
        f"**claude-sonnet-4-5**: {frontier_acc:.0%} correct &nbsp;·&nbsp; "
        f"**ministral-3b-latest**: {candidate_acc:.0%} correct, out of {len(eval_df)} examples."
    )
    return (eval_df,)


@app.cell
def _(eval_df, mo):
    mo.ui.table(eval_df, label="Per-example results")
    return


@app.cell
def _(eval_df, mo):
    mo.md(
        "Rows worth reading closely: where the two models disagree, or either one missed the "
        "expected label. An aggregate accuracy number hides this."
        if (~eval_df["frontier correct"] | ~eval_df["candidate correct"]).any()
        else "Both models matched the expected label on every example here."
    )
    return


@app.cell
def _(eval_df, mo):
    disagreements = eval_df[
        eval_df["claude-sonnet-4-5"] != eval_df["ministral-3b-latest"]
    ]
    mo.ui.table(disagreements, label="Where the two models disagreed") if len(
        disagreements
    ) else mo.md("*The two models agreed on every example.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Whether the candidate's accuracy is good enough isn't something this notebook can answer. It depends on what a wrong urgency score costs, and that's specific to a pipeline. This process gives you evidence to make a call with.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Naming the pattern

    Routing simple requests to cheap models and hard ones to expensive models is known as **LLM routing**, or sometimes **model routing**. The simplest version is what steps 1 through 5 walked through by hand: decide per subtask, once, which tier each piece of the pipeline needs. A more dynamic version puts a small, cheap classifier in front of the whole pipeline that decides, per request, which tier to send it to. A close relative is **model cascading**, where every request goes to the cheap model first and only escalates to the expensive one when the answer looks unreliable. All of these are worth knowing by name so you can look up how they are built.

    For a broader look at what's available and roughly how models compare on cost and capability across providers, [Artificial Analysis](https://artificialanalysis.ai/) is a useful independent reference. [LMArena](https://lmarena.ai/) is built on people voting on which of two anonymous responses they prefer, so it rewards answers that look good at a glance, which doesn't always correlate with correctness.
    """)
    return


if __name__ == "__main__":
    app.run()
