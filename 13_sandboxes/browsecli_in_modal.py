# ---
# cmd: ["modal", "run", "13_sandboxes/browsecli_in_modal.py"]
# pytest: false
# ---

# # Run a browser agent in a Modal Sandbox

# A Modal [Sandbox](https://modal.com/docs/guide/sandbox) is a great place to run an
# **agent's tools** — but a Firecracker microVM is a poor place to run a *browser*.
# It has a datacenter IP (often blocked by bot-detection), no fingerprint hardening,
# and no way to solve a CAPTCHA. Bundling Playwright + Chromium into the image doesn't
# help: it still browses *from the datacenter IP*, and it bloats the image and slows
# cold starts.

# This example keeps the browser **out** of the Sandbox. The Sandbox holds a single
# tool — the [`browse`](https://github.com/browserbase/stagehand/tree/main/packages/cli)
# CLI — and an Anthropic agent loop drives it via Claude's native **bash tool**. Each
# bash command runs inside the Sandbox with `sandbox.exec`, and `browse` connects out
# over CDP to a **remote** browser on [Browserbase](https://www.browserbase.com). The
# heavy browser lives on Browserbase; the Sandbox just runs the CLI.

# ```
# ┌─────────────────────┐   bash tool    ┌──────────────────┐  CDP over wss  ┌────────────────────────┐
# │  Modal Function     │ ─────────────▶ │  Modal Sandbox   │ ─────────────▶ │  Browserbase browser   │
# │  Claude agent loop  │  sandbox.exec  │  `browse` CLI    │ ◀───────────── │  (the real Chrome)     │
# └─────────────────────┘ ◀───────────── └──────────────────┘   page data    └────────────────────────┘
# ```
# The agent's task (overridable): research each company's recent SEC EDGAR filing
# activity and return a sourced comparison of their most recent 10-Q and 10-K.

# To run it:

# ```bash
# export ANTHROPIC_API_KEY=sk-ant-...
# export BROWSERBASE_API_KEY=bb_live_...
# modal run 13_sandboxes/browsecli_in_modal.py
# ```

import os

import modal

# ## Build the Sandbox image
#
# Modal's `debian_slim` has no Node, so we start from the official `node:20-slim`
# image (Node is what runs `browse`) and install the `browse` CLI globally. **No
# Chrome/Chromium is installed** — the browser lives on Browserbase and is reached
# over CDP at run time. The agent loop itself runs in a Modal Function and only needs
# `anthropic`, so we add that to the Function's image separately below.

sandbox_image = modal.Image.from_registry("node:20-slim", add_python="3.12").run_commands(
    "npm install -g browse@latest", "browse --version"
)

app = modal.App("example-browsecli-in-modal")

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
# The model's only tool is Claude's built-in **bash tool**
# (`{"type": "bash_20250124", "name": "bash"}`) — a schema-less tool whose commands we
# execute ourselves. We run each command inside the Sandbox with `sandbox.exec`, so the
# model gets a real shell whose one useful program is `browse`.
#
# We don't hand-roll a custom `browse` tool or inject flags into the model's commands.
# Instead the Sandbox's environment does the steering: `BROWSERBASE_API_KEY` makes
# `browse` default to a **remote** Browserbase browser, and `BROWSE_SESSION=agent`
# scopes every call to one shared session, so the agent keeps the same browser tab
# across commands. The model just runs `browse open <url>`, `browse get markdown body`,
# and so on.
#
# The system prompt is deliberately generic: it describes the `browse` commands and
# good research habits, then lets the model plan its own steps for whatever task it's
# given.

SESSION = "agent"
DEFAULT_TASK = (
    "For Snowflake, Datadog, and MongoDB, find each company's single most recent 10-Q "
    "filing on SEC EDGAR (start at "
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany). For each company report "
    "the filing date, the fiscal period it covers, and the direct URL of the primary "
    "filing document (the company's own 10-Q .htm document, not the filing index, a cover "
    "page, an exhibit, or a viewer/preview page). Also report the date of each company's "
    "most recent 10-K. Return a comparison table across all three companies and cite each "
    "URL."
)
MODEL = "claude-sonnet-5"
MAX_STEPS = 40
MAX_OUTPUT_CHARS = 40_000  # cap each tool result so it fits the context budget

SYSTEM_PROMPT = f"""You are an autonomous deep-research agent. You answer questions by investigating the live web with a real browser that runs remotely on Browserbase, which you drive through the `browse` CLI in a bash shell. Every browser command is automatically scoped to one shared remote session, so you keep the same browser tab across commands.
Useful commands:
  browse open <url>          # navigate (uses the shared remote browser)
  browse get markdown body   # read the current page as markdown (keeps links/URLs)
  browse get text body       # read the current page as plain text
Run `browse --help` to discover more commands.

Plan your own research: break the question into sub-questions, find and open relevant sources, follow links, and read pages to gather evidence. Use several independent sources and cross-check key facts. If a page returns an error or looks empty, try a different source instead of retrying it unchanged.

To stay effective:
- Pages are fully rendered (JavaScript runs) before you read them — the text/markdown you get back IS the real content. Read it carefully and extract what you need; don't assume a page "needs JavaScript" or abandon a source that already has the answer.
- Read each page once. Don't fetch the same page twice or as both markdown and text (for long pages "get text body" is best), and don't chase detours when a page you already have answers the question.
- Your steps are limited: once you have what you need for one item, move on, and leave yourself a step to write the final answer.

Reporting rules that apply to every task:
- When you report the URL of a document, give the direct link to the document itself, not a viewer, preview, or print-friendly wrapper around it.
- When a question asks about "the most recent" item, identify the single most recent one before reporting, rather than the first plausible match you find.

When you can answer thoroughly, stop browsing and return a concise, well-sourced synthesis that cites the URLs you used."""

# Claude's native bash tool. The schema is built into the model — we declare it by
# type and name only, and our handler executes the `command` Claude sends.
BASH_TOOL = {"type": "bash_20250124", "name": "bash"}


@app.function(image=modal.Image.debian_slim().pip_install("anthropic"), secrets=[agent_secret])
def run_agent(task: str = DEFAULT_TASK) -> str:
    # CI guard. Modal runs every gallery example live on each push, where no keys
    # exist. In that case `from_dict` injected empty strings — print a clear
    # "skipping" message and return cleanly instead of failing CI. With keys
    # present, the live run is cheap — one short remote session.
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("BROWSERBASE_API_KEY"):
        print(
            "[browser-agent] skipping live run (missing ANTHROPIC_API_KEY or "
            "BROWSERBASE_API_KEY). Set both in your env before `modal run`."
        )
        return ""

    import anthropic

    # Boot the Sandbox that holds the `browse` CLI. The environment steers every
    # command to a shared *remote* browser, so the model never has to pass flags.
    sandbox = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        secrets=[agent_secret],
        env={"BROWSE_SESSION": SESSION},
        timeout=15 * 60,
    )
    print(f"Sandbox ID: {sandbox.object_id}")

    def run_bash(command: str) -> str:
        """Execute one bash command inside the Sandbox and return its output."""
        print(f"-> {command}")
        proc = sandbox.exec("bash", "-c", command, timeout=60)
        proc.wait()
        out = (proc.stdout.read() or "") + (proc.stderr.read() or "")
        print(f"   <- {len(out)} chars")
        return out[:MAX_OUTPUT_CHARS] or "(no output)"

    client = anthropic.Anthropic()

    # The agent loop: call the model, run any bash commands it requests, feed the
    # results back as tool_result blocks, and repeat until it stops calling tools
    # (or we hit the step cap).
    messages = [{"role": "user", "content": task}]
    final_text = ""

    try:
        for _ in range(MAX_STEPS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=[BASH_TOOL],
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text = "".join(b.text for b in response.content if b.type == "text")
            if text:
                final_text = text

            if response.stop_reason != "tool_use" or not tool_uses:
                break

            # Run every requested command, returning all results in ONE user message.
            tool_results = []
            for block in tool_uses:
                if block.input.get("restart"):
                    output = "(bash session restart is a no-op; each command runs fresh)"
                else:
                    output = run_bash(block.input["command"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
    finally:
        # Close the remote browser session, then terminate the Sandbox.
        run_bash("browse stop")
        sandbox.terminate()

    answer = final_text or "(no answer)"
    print("\n===== FINAL ANSWER =====\n" + answer)
    return answer


# ## Local entrypoint
#
# `modal run 13_sandboxes/browsecli_in_modal.py` triggers this, which runs the agent
# Function in the cloud. Pass a different goal with `--task "..."`.


@app.local_entrypoint()
def main(task: str = DEFAULT_TASK):
    run_agent.remote(task)


# > **Note on protected sites.** This example uses a plain remote browser, which works
# > on **any** Browserbase plan. To reach sites behind aggressive bot-detection,
# > Browserbase also offers Verified browsers (residential IP + automatic CAPTCHA
# > solving), which require a Scale plan — see https://www.browserbase.com/pricing.
