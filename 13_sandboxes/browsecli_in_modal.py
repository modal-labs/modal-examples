# ---
# cmd: ["modal", "run", "browsecli_in_modal.py"]
# ---

# # Run a browser agent inside a Modal Function
#
# A Modal Function is a great place to run an **agent loop** — but a Firecracker
# sandbox is a poor place to run a *browser*. It has a datacenter IP (often
# blocked by bot-detection), no fingerprint hardening, and no way to solve a
# CAPTCHA. Bundling Playwright + Chromium into the image doesn't help: it still
# browses *from the datacenter IP*, and it bloats the image and slows cold starts.
#
# This example keeps the browser **out** of the sandbox. The Function runs a small
# Anthropic agent loop whose only tool is the
# [`browse`](https://github.com/browserbase/stagehand/tree/main/packages/cli) CLI.
# Each tool call shells out to `browse`, which drives a **remote** browser on
# [Browserbase](https://www.browserbase.com) over CDP. The heavy browser lives on
# Browserbase; the Function just runs the model and the CLI.
#
# ```
# ┌──────────────────────────────┐    CDP over wss    ┌────────────────────────┐
# │  Modal Function              │ ─────────────────▶ │  Browserbase browser   │
# │  Claude agent loop           │                    │  (runs the real Chrome,│
# │  tool: `browse ... --remote` │ ◀───────────────── │   returns page data)   │
# └──────────────────────────────┘    page data       └────────────────────────┘
# ```
#
# The agent's task (overridable): pull each company's most recent 10-Q from SEC
# EDGAR and return a sourced comparison of revenue, growth, RPO, and top risk.
#
# To run it:
#
# ```bash
# export ANTHROPIC_API_KEY=sk-ant-...
# export BROWSERBASE_API_KEY=bb_live_...
# modal run browsecli_in_modal.py
# ```

import os
import shlex
import subprocess

import modal

# ## Build the image
#
# Modal's `debian_slim` has no Node, so we start from the official `node:20-slim`
# image (Node is what runs `browse`), add a Python interpreter so Modal can run
# the agent inside, install the `browse` CLI globally, and `pip install anthropic`
# for the agent loop. We also `apt_install("ca-certificates")` so Python's HTTPS
# stack can verify TLS to the Anthropic API (the slim base ships without the CA
# bundle Python needs). **No Chrome/Chromium is installed** — the browser lives
# on Browserbase and is reached over CDP at run time.

image = (
    modal.Image.from_registry("node:20-slim", add_python="3.12")
    .apt_install("ca-certificates")
    .run_commands("npm install -g browse@latest", "browse --version")
    .pip_install("anthropic")
)

app = modal.App("browsecli-in-modal", image=image)

# ## Credentials
#
# We inject both keys with `Secret.from_dict`, reading them from the **local**
# environment at launch — so no pre-created Modal Secret is required to try this.
# (For production, store them once as a named Secret — `modal secret create
# browser-agent ANTHROPIC_API_KEY=... BROWSERBASE_API_KEY=...` — and swap the
# `Secret.from_dict(...)` below for `modal.Secret.from_name("browser-agent")`.)

agent_secret = modal.Secret.from_dict(
    {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "BROWSERBASE_API_KEY": os.environ.get("BROWSERBASE_API_KEY", ""),
    }
)

# ## The agent
#
# The model's only tool is `browse`. Its handler shells out to the CLI, scoping
# every call to one shared remote session (`--session agent`) so the agent keeps
# the same browser tab across calls. We slice the output to keep tool results
# inside the model's context budget.
#
# The system prompt is deliberately generic: it describes the `browse` commands and
# good research habits (use multiple sources, cross-check, don't retry dead pages),
# then lets the model plan its own steps for whatever task it's given.

SESSION = "agent"
DEFAULT_TASK = (
    "For Snowflake, Datadog, and MongoDB, find each company's most recent 10-Q "
    "filing on SEC EDGAR (start at "
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany). Open the actual "
    "primary filing document — not the filing index, cover page, or an exhibit — "
    "and extract quarterly revenue, year-over-year revenue growth, remaining "
    "performance obligations (RPO), and the single most significant risk factor. "
    "Return a comparison table across all three companies and cite each filing's URL."
)
MODEL = "claude-sonnet-4-5"
MAX_STEPS = 40

SYSTEM_PROMPT = f"""You are an autonomous deep-research agent. You answer questions by investigating the live web with a real browser that runs remotely on Browserbase. Each tool call runs: browse <your args> (every call is automatically scoped to one shared session "{SESSION}").
Useful commands:
  open <url> --remote   # navigate (ALWAYS include --remote so it uses the cloud browser)
  get markdown body     # read the current page as markdown (keeps links/URLs)
  get text body         # read the current page as plain text
Use "--help" to discover more commands.

Plan your own research: break the question into sub-questions, find and open relevant sources, follow links, and read pages to gather evidence. Use several independent sources and cross-check key facts. If a page returns ERROR or looks empty, try a different source instead of retrying it unchanged. When you can answer thoroughly, stop browsing and return a concise, well-sourced synthesis that cites the URLs you used."""

BROWSE_TOOL = {
    "name": "browse",
    "description": (
        'Run a browse CLI command (omit the leading "browse"). '
        "e.g. open https://example.com --remote ; "
        "get markdown body ; get text body"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "args": {
                "type": "string",
                "description": "Arguments passed to the browse CLI, without the leading 'browse'.",
            }
        },
        "required": ["args"],
    },
}


def run_browse(args: str) -> str:
    """Execute one `browse` command against the shared remote session."""
    print(f"-> browse {args}")
    try:
        # Tokenize the model's free-form arg string and re-quote each piece, so a
        # URL with shell metacharacters like `&` (e.g. SEC EDGAR query strings)
        # isn't split by the shell into broken commands.
        cmd = "browse " + " ".join(shlex.quote(a) for a in shlex.split(args))
        result = subprocess.run(
            ["bash", "-lc", f"{cmd} --session {SESSION}"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        out = result.stdout or result.stderr or ""
    except subprocess.TimeoutExpired:
        out = "ERROR: command timed out"
    except Exception as e:  # surface, don't crash the loop
        out = f"ERROR: {e}"
    is_err = out.startswith("ERROR")
    print(f"   <- {len(out)} chars{' [ERR]' if is_err else ''}")
    return out[:40000]


@app.function(secrets=[agent_secret])
def run_agent(task: str = DEFAULT_TASK) -> str:
    # CI guard. Modal runs every gallery example live on each push, where no keys
    # exist. In that case `from_dict` injected empty strings — print a clear
    # "skipping" message and return cleanly instead of failing CI. With keys
    # present, the live run is cheap — one short remote session.
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get(
        "BROWSERBASE_API_KEY"
    ):
        print(
            "[browser-agent] skipping live run (missing ANTHROPIC_API_KEY or "
            "BROWSERBASE_API_KEY). Set both in your env before `modal run`."
        )
        return ""

    import anthropic

    client = anthropic.Anthropic()

    # The agent loop: call the model, run any tool_use blocks it returns, feed the
    # results back as tool_result blocks, and repeat until it stops calling tools
    # (or we hit the step cap). This is the Python twin of an AI SDK tool loop.
    messages = [{"role": "user", "content": task}]
    final_text = ""

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[BROWSE_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # Collect this turn's text and any tool calls.
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")
        if text:
            final_text = text

        if response.stop_reason != "tool_use" or not tool_uses:
            break

        # Execute each tool call and return all results in ONE user message.
        tool_results = []
        for block in tool_uses:
            output = run_browse(block.input["args"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Close the remote session so it doesn't linger.
    run_browse("stop")

    answer = final_text or "(no answer)"
    print("\n===== FINAL ANSWER =====\n" + answer)
    return answer


# ## Local entrypoint
#
# `modal run browsecli_in_modal.py` triggers this, which runs the Function in the
# cloud. Pass a different goal with `--task "..."`.


@app.local_entrypoint()
def main(task: str = DEFAULT_TASK):
    run_agent.remote(task)


# > **Note on protected sites.** This example uses a plain `--remote` browser,
# > which works on **any** Browserbase plan. To reach sites behind aggressive
# > bot-detection, Browserbase also offers Verified browsers (residential IP +
# > automatic CAPTCHA solving via `--proxies --verified --solve-captchas`), which
# > require a Scale plan — see https://www.browserbase.com/pricing.
