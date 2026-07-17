# ---
# lambda-test: false  # needs Kernel + Anthropic credentials and provisions a cloud browser
# ---

# # QA a preview deployment with a Kernel headful browser + Claude computer use
#
# On every pull request you want to know one thing: does the feature still work in a real
# browser? This example wires up an agent that does exactly that. On each run it:
#
# 1. Serves a tiny demo web app on Modal (a feedback form) in two variants - a **working**
#    one and a **broken** one (a deliberately injected regression). Pretend the broken one
#    is a bad PR preview.
# 2. Spins up a Kernel **headful** browser and points a **Claude computer-use** agent at the
#    page. The agent drives the browser the way a person would - from screenshots and pixel
#    coordinates - filling the form, clicking Submit, and judging whether it worked.
# 3. **Records the whole session** with Kernel's replay and saves the mp4 to a **Modal Volume**,
#    so the trace outlives the run.
# 4. Returns a pass/fail verdict. It **passes** the working variant and **catches the
#    regression** in the broken one.
#
# Like the sibling [web scraper example](https://modal.com/docs/examples/kernel_webscraper), the
# browser runs on Kernel, so **there is no browser binary in the Modal image** - Modal just
# talks to it over the API. The scraper uses a fast model for one-shot extraction; this
# example uses a stronger model because agentic, vision-driven navigation needs it. The
# same Kernel `computer` API works with other computer-use models like Gemini and OpenAI;
# see Kernel's [Computer-Use Overview](https://www.kernel.sh/docs/integrations/computer-use/overview).
#
# A `modal deploy` path at the bottom turns this into a bot that QAs every PR against its
# preview and comments the verdict back; `modal run` needs none of it.
#
# ## What you'll need
#
# A [Kernel](https://www.kernel.sh) API key, an [Anthropic](https://console.anthropic.com)
# API key, and a [Modal](https://modal.com) account. Store the keys as Modal Secrets - there's
# no preset for Kernel, so create a custom secret named `kernel`:
#
# ```
# modal secret create kernel KERNEL_API_KEY=...
# modal secret create anthropic-secret ANTHROPIC_API_KEY=...
# ```

import modal

MINUTES = 60  # seconds, for readable timeouts
VIEWPORT = {
    "width": 1280,
    "height": 800,
}  # must equal the computer-use tool's display_*_px
MODEL = "claude-sonnet-4-6"  # Sonnet is the sweet spot for computer use; swap to claude-opus-4-8 in one line for harder UIs
MAX_ITERS = 22  # a two-field form needs only a few steps; this bounds cost and runtime
TYPING_DELAY_MS = (
    12  # per-character typing delay (matches Kernel's public computer-use template)
)
SCREENSHOT_SETTLE_S = 2.0  # let the page react before each capture
REPLAY_GRACE_S = 1.5  # let the final banner land in the recording before we stop it

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "kernel==0.74.0",  # <1.0: pin exact patch per modal-examples policy
    "anthropic==0.116.0",  # <1.0: pin exact patch
    "fastapi==0.139.0",  # <1.0: pin exact patch
)
app = modal.App("example-kernel-pr-qa-agent", image=image)

KERNEL_SECRET = modal.Secret.from_name("kernel", required_keys=["KERNEL_API_KEY"])
ANTHROPIC_SECRET = modal.Secret.from_name(
    "anthropic-secret", required_keys=["ANTHROPIC_API_KEY"]
)

# A Modal Volume to persist each session recording (mp4) so the traces outlive the run.
TRACES = modal.Volume.from_name("kernel-pr-qa-traces", create_if_missing=True)


# ## The app under test
#
# A tiny self-contained feedback form, served at two routes off one Modal web app. The only
# difference is the submit handler: `/working` shows a green success banner; `/broken` shows
# a red error - a believable "someone refactored the form and the submit broke" regression.
# We make the failure a *visible* red state (not just a missing banner) so the vision model
# can reliably tell pass from fail.


def _feedback_html(broken: bool) -> str:
    if broken:
        on_submit = (
            "r.textContent = '✗ Something went wrong — please try again.';"
            "r.className = 'result error'; r.style.display = 'block';"
        )
    else:
        on_submit = (
            "r.textContent = '✓ Thanks, your feedback was submitted.';"
            "r.className = 'result success'; r.style.display = 'block';"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="data:,"><title>Send us feedback</title><style>
  body {{ font-family: system-ui, sans-serif; background: #f4f4f5; display: flex;
          justify-content: center; padding: 60px; }}
  .card {{ background: #fff; padding: 40px; border-radius: 12px; width: 440px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  h1 {{ font-size: 24px; margin: 0 0 8px; }}
  p.sub {{ color: #6b7280; margin: 0 0 20px; }}
  label {{ display: block; font-size: 14px; font-weight: 600; margin: 16px 0 6px; }}
  input, textarea {{ width: 100%; font-size: 16px; padding: 10px; box-sizing: border-box;
                     border: 1px solid #ccc; border-radius: 8px; }}
  button {{ margin-top: 22px; width: 100%; font-size: 16px; padding: 12px; border: 0;
            border-radius: 8px; background: #4f46e5; color: #fff; cursor: pointer; }}
  .result {{ margin-top: 20px; padding: 14px; border-radius: 8px; font-size: 16px; display: none; }}
  .success {{ background: #dcfce7; color: #166534; }}
  .error {{ background: #fee2e2; color: #991b1b; }}
</style></head><body>
  <div class="card">
    <h1>Send us feedback</h1>
    <p class="sub">We read everything.</p>
    <form id="f">
      <label for="name">Name</label>
      <input id="name" name="name" type="text" />
      <label for="message">Message</label>
      <textarea id="message" name="message" rows="4"></textarea>
      <button type="submit">Submit</button>
    </form>
    <div id="result" class="result"></div>
  </div>
  <script>
    document.getElementById('f').addEventListener('submit', function (e) {{
      e.preventDefault();
      var r = document.getElementById('result');
      {on_submit}
    }});
  </script>
</body></html>"""


@app.function()
@modal.asgi_app()
def demo_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    @web.get("/working")
    def working():
        return HTMLResponse(_feedback_html(broken=False))

    @web.get("/broken")
    def broken():
        return HTMLResponse(_feedback_html(broken=True))

    return web


# ## The computer-use loop
#
# This is the hero. A Claude computer-use agent drives the Kernel browser: we send it a
# screenshot, it replies with an action at pixel coordinates (click here, type this), we
# execute that action through Kernel's `computer` API, take a fresh screenshot, and repeat
# until it returns a verdict. The action translation below follows Kernel's public
# computer-use template (https://github.com/kernel/cli).

SYSTEM_PROMPT = (
    "You are a meticulous QA engineer testing a web app in a browser. The browser runs in "
    "kiosk mode, so each screenshot is entirely page content, with no toolbar, address bar, or "
    "tabs to account for. Study every screenshot before you act. After submitting a form, pause "
    "briefly and take a fresh screenshot before you judge the outcome, so any success or error "
    "state has time to appear. If you cannot find something, try scrolling before concluding it "
    "is absent."
)

TOOL = {
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": VIEWPORT["width"],
    "display_height_px": VIEWPORT["height"],
    "display_number": 1,
}
# A strict tool for the final result, so the verdict is structured data (an enum + a reason)
# instead of a sentinel string we parse out of the model's prose.
SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Report the final QA result once you have verified the change end to end. "
    "Call this exactly once, when you are done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": "pass if the change works as described, fail otherwise",
            },
            "reason": {"type": "string", "description": "one-line justification"},
        },
        "required": ["verdict", "reason"],
    },
}
BETAS = ["computer-use-2025-11-24", "prompt-caching-2024-07-31"]

# Map common key names to the xdotool keysyms Kernel's `computer` API expects. Adapted from
# Kernel's public computer-use template (https://github.com/kernel/cli).
_KEY_MAP = {
    "return": "Return",
    "enter": "Return",
    "space": "space",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "home": "Home",
    "end": "End",
    "pageup": "Page_Up",
    "page_up": "Page_Up",
    "pagedown": "Page_Down",
    "page_down": "Page_Down",
    "delete": "Delete",
    "backspace": "BackSpace",
    "tab": "Tab",
    "esc": "Escape",
    "escape": "Escape",
    "insert": "Insert",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    "minus": "minus",
    "equal": "equal",
    "plus": "plus",
}
_MODIFIER_MAP = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "cmd": "super",
    "command": "super",
    "win": "super",
    "meta": "super",
    "shift": "shift",
}


def _to_xdotool(key: str) -> str:
    """Resolve a key name (or '+'-joined combo) to xdotool keysyms, mirroring the lookup in
    Kernel's public computer-use template: normalize, then check the modifier and key maps."""
    k = key.lower().strip()
    if k in _MODIFIER_MAP:
        return _MODIFIER_MAP[k]
    if k in _KEY_MAP:
        return _KEY_MAP[k]
    if "+" in key:
        parts = []
        for part in key.split("+"):
            p = part.strip().lower()
            parts.append(_MODIFIER_MAP.get(p) or _KEY_MAP.get(p) or p)
        return "+".join(parts)
    return key


def _clamp(value, hi: int) -> int:
    return max(0, min(hi, int(value)))


def _screenshot(client, sid: str) -> str:
    """Capture the browser as a base64 PNG. The screenshot endpoint returns a binary
    response, so read() it; we let the page settle for SCREENSHOT_SETTLE_S first so the
    capture reflects the last action."""
    import base64
    import time

    time.sleep(SCREENSHOT_SETTLE_S)
    resp = client.browsers.computer.capture_screenshot(sid)
    raw = resp if isinstance(resp, (bytes, bytearray)) else resp.read()
    return base64.b64encode(raw).decode()


def _execute(client, sid: str, inp: dict) -> None:
    """Translate one Anthropic computer action into Kernel `computer` calls. There are no
    per-action sleeps here; the settle delay in _screenshot gives the page time to react
    before the next capture."""
    import time

    computer = client.browsers.computer
    action = inp.get("action")
    coord = inp.get("coordinate")
    if coord and len(coord) >= 2:
        coord = [
            _clamp(coord[0], VIEWPORT["width"]),
            _clamp(coord[1], VIEWPORT["height"]),
        ]
    else:
        coord = None
    text = inp.get("text")

    if action in ("left_click", "click") and coord:
        button = inp.get("button")
        if button in ("right", "middle"):
            computer.click_mouse(sid, x=coord[0], y=coord[1], button=button)
        else:
            computer.click_mouse(sid, x=coord[0], y=coord[1])
    elif action == "right_click" and coord:
        computer.click_mouse(sid, x=coord[0], y=coord[1], button="right")
    elif action == "double_click" and coord:
        computer.click_mouse(sid, x=coord[0], y=coord[1], num_clicks=2)
    elif action == "triple_click" and coord:
        computer.click_mouse(sid, x=coord[0], y=coord[1], num_clicks=3)
    elif action == "type" and text:
        computer.type_text(sid, text=text, delay=TYPING_DELAY_MS)
    elif action in ("key", "keypress"):
        if text:
            computer.press_key(sid, keys=[_to_xdotool(text)])
        else:
            for key in inp.get("keys", []):
                computer.press_key(sid, keys=[_to_xdotool(key)])
    elif action == "scroll":
        if coord:
            cx, cy = coord[0], coord[1]
        else:
            cx, cy = VIEWPORT["width"] // 2, VIEWPORT["height"] // 2
        notches = max(
            inp.get("scroll_amount") or 1, 1
        )  # wheel units, as the Kernel SDK expects
        direction = inp.get("scroll_direction") or "down"
        dx = notches if direction == "right" else -notches if direction == "left" else 0
        dy = notches if direction == "down" else -notches if direction == "up" else 0
        computer.scroll(sid, x=cx, y=cy, delta_x=dx, delta_y=dy)
    elif action in ("mouse_move", "move") and coord:
        computer.move_mouse(sid, x=coord[0], y=coord[1])
    elif action == "wait":
        time.sleep(min(float(inp.get("duration") or 1.0), 5.0))
    # "screenshot" and anything unrecognized: fall through; a fresh screenshot is taken below.


def _inject_prompt_caching(messages: list, breakpoints: int = 3) -> None:
    """Mark the newest user turns as prompt-cache breakpoints so each request re-reads the
    prior conversation (mostly screenshots) from cache instead of resending it. The breakpoints
    slide forward as the conversation grows; 3 here plus the system prompt is Anthropic's max
    of 4."""
    remaining = breakpoints
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], list):
            if remaining:
                remaining -= 1
                message["content"][-1]["cache_control"] = {"type": "ephemeral"}
            else:
                message["content"][-1].pop("cache_control", None)
                break


def _run_cua_loop(client, anthropic_client, sid: str, goal: str):
    messages: list = [{"role": "user", "content": goal}]
    reprompted = False

    for i in range(MAX_ITERS):
        # Cache the conversation prefix so each turn re-reads the screenshots cheaply.
        _inject_prompt_caching(messages)
        resp = anthropic_client.beta.messages.create(
            model=MODEL,
            max_tokens=8192,  # headroom for thinking on more complex UIs
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
            tools=[TOOL, SUBMIT_VERDICT_TOOL],
            betas=BETAS,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
        )
        # Append the whole response (text + thinking + tool_use) so thinking signatures survive.
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        # The agent ends by calling submit_verdict; read the structured result straight from the
        # tool input, no prose to parse. That's unambiguous, so we act on it immediately.
        verdict_call = next(
            (tu for tu in tool_uses if tu.name == "submit_verdict"), None
        )
        if verdict_call:
            inp = dict(verdict_call.input)
            verdict = "pass" if inp.get("verdict") == "pass" else "fail"
            print(f"  turn {i + 1}/{MAX_ITERS}: submit_verdict({verdict})")
            return verdict, str(inp.get("reason") or "").strip(), i + 1

        actions = (
            ", ".join(str(tu.input.get("action")) for tu in tool_uses) or "no action"
        )
        print(f"  turn {i + 1}/{MAX_ITERS}: {actions}")

        if not tool_uses:
            # No action and no verdict - nudge once to submit, then give up.
            if not reprompted:
                reprompted = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Call the submit_verdict tool with your pass/fail result and a one-line reason.",
                    }
                )
                continue
            return "inconclusive", "model ended without submitting a verdict", i + 1

        tool_results = []
        for tu in tool_uses:
            try:
                _execute(client, sid, dict(tu.input))
            except Exception as exc:
                # A transient Kernel API error or a bad coordinate shouldn't kill a paid,
                # non-idempotent run: note it and let the model re-plan from a fresh screenshot.
                print(f"  action failed, model will re-plan: {exc}")
            try:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _screenshot(client, sid),
                                },
                            }
                        ],
                    }
                )
            except Exception as exc:
                # A failed capture shouldn't abort the run either; tell the model to retry so the
                # loop still reaches a verdict and saves the recording.
                print(f"  screenshot failed, model will retry: {exc}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": "screenshot failed; take another screenshot and continue",
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return "inconclusive", f"hit iteration cap ({MAX_ITERS}) with no verdict", MAX_ITERS


def _goal(change: str) -> str:
    """Turn a plain-English description of the change under review into the agent's task."""
    return (
        "You are reviewing a web app to confirm a specific change works. The change under test:\n"
        f"{change}\n\n"
        "Interact with the page like a user to verify it end to end. When you are done, call the "
        "submit_verdict tool with 'pass' if the change works as described or 'fail' if it does "
        "not, plus a one-line reason."
    )


def _save_trace(client, sid: str, replay_id: str, label: str):
    """Download the finished replay (an mp4) and persist it to the Modal Volume so it outlives
    the run. The recording can take a moment to finalize after stop(), so we retry briefly."""
    import time

    raw = b""
    for _ in range(10):
        try:
            raw = client.browsers.replays.download(replay_id, id=sid).read()
            if raw:
                break
        except Exception:
            pass
        time.sleep(2)
    if not raw:
        print("recording not ready to download; skipping trace save")
        return None
    path = f"/traces/{label}.mp4"
    with open(path, "wb") as f:
        f.write(raw)
    TRACES.commit()
    print(f"saved recording to Modal Volume: {path} ({len(raw)} bytes)")
    return path


# ## verify_pr: the QA function
#
# Point a Kernel headful browser at a preview URL, run the computer-use agent to verify the
# described `change`, record the session, save the mp4 to a Modal Volume, and always delete the
# browser in a `finally` so we never leak a paid session. `retries=0` because the loop is
# non-idempotent and paid.


@app.function(
    secrets=[KERNEL_SECRET, ANTHROPIC_SECRET],
    volumes={"/traces": TRACES},
    timeout=15 * MINUTES,
    retries=0,
)
def verify_pr(
    preview_url: str, change: str, label: str = "run", watch: bool = False
) -> dict:
    import time

    from anthropic import Anthropic
    from kernel import Kernel

    client = Kernel()  # reads KERNEL_API_KEY
    anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY

    # kiosk_mode hides the address bar/tabs so the whole screenshot is page content, matching
    # the system prompt and keeping pixel coordinates aligned with the content.
    kb = client.browsers.create(
        headless=False,
        stealth=True,
        start_url=preview_url,
        kiosk_mode=True,
        viewport=VIEWPORT,
        timeout_seconds=10 * MINUTES,
    )
    sid = kb.session_id
    replay = None
    replay_id = None
    replay_view_url = None
    trace_path = None
    verdict, reason, iterations = "inconclusive", "did not run", 0
    try:
        replay = client.browsers.replays.start(
            sid, framerate=10, max_duration_in_seconds=10 * MINUTES
        )
        replay_id = replay.replay_id
        replay_view_url = getattr(replay, "replay_view_url", None)  # from start()
        # This URL carries a session JWT, so we only print it for an interactive run (watch=True).
        # verify_pr also runs as the deploy webhook, whose logs are retained - never log it there.
        if watch:
            print(f"recording: {replay_view_url}")
        verdict, reason, iterations = _run_cua_loop(
            client, anthropic_client, sid, _goal(change)
        )
        # Let the final state land in the recording, then stop it and save the mp4 to the Volume.
        time.sleep(REPLAY_GRACE_S)
        client.browsers.replays.stop(replay_id, id=sid)
        replay = None  # stopped; don't stop again in the finally
        trace_path = _save_trace(client, sid, replay_id, label)
    finally:
        if replay is not None:
            try:
                client.browsers.replays.stop(replay_id, id=sid)
            except Exception:
                pass
        try:
            client.browsers.delete_by_id(sid)
        except Exception as exc:
            # Don't let a teardown blip discard a verdict that already cost a paid browser.
            print(f"  browser delete failed: {exc}")

    return {
        "preview_url": preview_url,
        "verdict": verdict,
        "reason": reason,
        "trace_path": trace_path,
        "replay_view_url": replay_view_url,
        "iterations": iterations,
    }


# The change we ask the agent to verify against the demo app. The working variant satisfies it
# (PASS); the broken variant regressed it (FAIL).
DEMO_CHANGE = (
    "The feedback form accepts a name and a message, and after clicking Submit it shows a "
    "success confirmation."
)


# ## Try it
#
# ```
# modal run 10_integrations/kernel_pr_qa_agent.py
# ```
#
# This serves the demo app, QAs the working variant (expect PASS) and the broken variant
# (expect FAIL), and prints each verdict plus a link to the recording.
#
# Rough cost/runtime: each run provisions two Kernel browsers (one per variant) for a minute
# or two each, plus up to `MAX_ITERS` computer-use turns per variant (usually only a
# handful). `MAX_ITERS` and the model are one-line knobs if you want to bound or raise that.


def _wait_until_up(url: str, attempts: int = 30, delay: float = 2.0) -> None:
    import time
    import urllib.request

    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"demo app never became reachable at {url}")


@app.local_entrypoint()
def main():
    base = demo_app.get_web_url()
    assert base, "demo app has no web URL"

    results = {}
    for variant in ("working", "broken"):
        url = f"{base}/{variant}"
        _wait_until_up(url)  # Modal web containers boot on first request
        print(f"\nVerifying the {variant} variant: {url}")
        r = verify_pr.remote(url, DEMO_CHANGE, label=variant, watch=True)
        results[variant] = r
        print(f"  -> {r['verdict']}: {r['reason']}")
        if r.get("trace_path"):
            print(f"  recording saved to Modal Volume: {r['trace_path']}")

    print(
        f"\nworking -> {results['working']['verdict']}   broken -> {results['broken']['verdict']}"
    )
    print("(expected: working -> pass, broken -> fail)")


# ## Deploy: QA every pull request automatically
#
# `modal run` above is the demo. In production you'd trigger this on every PR: a GitHub webhook
# hits a deployed endpoint that checks the payload signature, kicks off `verify_pr` against the
# PR's preview URL, and posts the verdict back as a PR comment.
#
# It ships commented so the demo stays zero-setup. Modal resolves every function's secrets when
# the app starts, so a live webhook that needs a GitHub token would force even a plain
# `modal run` to have those secrets. To turn the bot on:
#
# ```
# modal secret create github-webhook GITHUB_WEBHOOK_SECRET=... ALLOWED_PREVIEW_HOST=preview.example.com  # webhook secret + the only host the bot will drive
# modal secret create github-token GITHUB_TOKEN=...  # a token allowed to comment on the repo
# # uncomment the block below, then:
# modal deploy 10_integrations/kernel_pr_qa_agent.py
# # register the printed github_webhook URL under the repo's Settings -> Webhooks
# # (content type application/json, the same secret, sending the deployment_status event).
# ```
#
# The webhook takes a raw `Request`, so add `from fastapi import Request` to the module-level
# imports when you enable this block.
#
# ````python
# @app.function(
#     secrets=[modal.Secret.from_name("github-token", required_keys=["GITHUB_TOKEN"])],
#     timeout=20 * MINUTES,
# )
# def qa_and_comment(repo: str, pr_number: int, preview_url: str, change: str):
#     """QA the PR's preview with verify_pr, then post the verdict back as a PR comment."""
#     import json
#     import os
#     import urllib.request
#
#     result = verify_pr.remote(preview_url, change, label=f"pr-{pr_number}")
#     mark = "✅" if result["verdict"] == "pass" else "❌"
#     # `reason` is written by the agent from the (untrusted) preview page. Strip backticks so it
#     # can't break out of the code fence, and treat the verdict as a QA signal, not an approval.
#     # Don't include result['replay_view_url'] here - it carries a session JWT.
#     comment = f"{mark} **Kernel PR-QA: {result['verdict'].upper()}**"
#     if result["reason"]:
#         safe_reason = result["reason"].replace("`", "'")
#         comment += f"\n\n```\n{safe_reason}\n```"
#     req = urllib.request.Request(
#         f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
#         data=json.dumps({"body": comment}).encode(),
#         headers={
#             "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
#             "Accept": "application/vnd.github+json",
#             "Content-Type": "application/json",
#         },
#         method="POST",
#     )
#     with urllib.request.urlopen(req, timeout=30) as resp:
#         print(f"commented on {repo}#{pr_number}: HTTP {resp.status}")
#
#
# @app.function(
#     secrets=[
#         modal.Secret.from_name(
#             "github-webhook",
#             required_keys=["GITHUB_WEBHOOK_SECRET", "ALLOWED_PREVIEW_HOST"],
#         )
#     ]
# )
# @modal.fastapi_endpoint(method="POST")
# async def github_webhook(request: Request):
#     import hashlib
#     import hmac
#     import json
#     import os
#     import urllib.parse
#
#     from fastapi import HTTPException
#
#     body = await request.body()
#     digest = hmac.new(
#         os.environ["GITHUB_WEBHOOK_SECRET"].encode(), body, hashlib.sha256
#     ).hexdigest()
#     if not hmac.compare_digest(
#         "sha256=" + digest, request.headers.get("x-hub-signature-256", "")
#     ):
#         raise HTTPException(status_code=401, detail="bad signature")
#
#     event = await request.json()
#     # Where the PR number, preview URL, and change come from depends on your CI. This reads a
#     # deployment_status event whose target_url is the preview and whose deployment payload
#     # carries the PR number and change; adapt it to however your pipeline exposes them.
#     repo = event.get("repository", {}).get("full_name", "")
#     preview_url = event.get("deployment_status", {}).get("target_url", "")
#     payload = event.get("deployment", {}).get("payload") or {}
#     if isinstance(payload, str):  # GitHub's deployment payload can be an object or a string
#         try:
#             payload = json.loads(payload)
#         except ValueError:
#             payload = {}
#     pr_number = payload.get("pr_number")
#     change = payload.get("change", "the change in this PR")
#     if not (repo and preview_url and pr_number):
#         raise HTTPException(status_code=400, detail="missing repo/preview_url/pr_number")
#
#     # The signature proves the request came from GitHub, not that preview_url is safe to
#     # drive. Only drive URLs on your own preview host - set ALLOWED_PREVIEW_HOST to yours.
#     parsed = urllib.parse.urlparse(preview_url)
#     allowed = os.environ.get("ALLOWED_PREVIEW_HOST", "")  # e.g. "preview.example.com"
#     host = (parsed.hostname or "").lower()
#     if parsed.scheme != "https" or not allowed or not (
#         host == allowed or host.endswith("." + allowed)
#     ):
#         raise HTTPException(status_code=400, detail="preview_url is not on the allowed host")
#     qa_and_comment.spawn(repo, int(pr_number), preview_url, change)
#     return {"status": "queued"}
# ````
