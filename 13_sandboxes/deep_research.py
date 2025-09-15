""" """

from __future__ import annotations

import json
import os
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import modal

# # Homemade Deep Research Agent using nothing but Modal, vLLM, and PyTorch

# ## Constants
MEM_MNT = Path("/mem")
APP_NAME = "example-deep-research"
AGENT_MODEL_ID = os.environ.get("JEN_NANO_MODEL", "Menlo/Jan-Nano")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"

# ## Images
# ### Principal Research Agent
# We wil have a principal research agent, Jen Nano, running on a high-throughput
# vLLM server.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # Model serving
        "vllm==0.5.4",
        # Simple token counting & utils (optional)
        "tiktoken==0.7.0",
    )
    .env({"HF_HOME": (MEM_MNT / "/hf_cache").as_posix()})
)
with vllm_image.imports():
    from vllm import LLM, SamplingParams

# ### SubAgents
# We will have a separate framework for subagents. They will run in
# Modal sandboxes with a lighter weight compute requisition.
researcher_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "httpx==0.27.2",
    "duckduckgo_search==6.2.4",
    "readability-lxml==0.8.1",
    "beautifulsoup4==4.12.3",
    "lxml==5.3.0",
    "trafilatura==1.8.1",
    "rapidfuzz==3.9.7",
)
with researcher_image.imports():
    import httpx
    import trafilatura
    from duckduckgo_search import DDGS

# vision_subagent_image = () # TODO:


# ## Storage
# ### KV Cache
context_dict = modal.Dict.from_name("example-dprsrch-context", create_if_missing=True)
# ### Persistent Memory
mem_vol = modal.Volume.from_name("example-dprsrch-memory", create_if_missing=True)

# ## Helper Functions


def _extract_json(text: str) -> str:
    """Extract first JSON object or array from a string."""
    import re

    matches = re.findall(r"(\{.*\}|\[.*\])", text, flags=re.S)
    return matches[0] if matches else "{}"


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ## App
app = modal.App(APP_NAME, volumes={MEM_MNT: mem_vol})

# # LLM Gateway: Jen Nano :: vLLM


@app.cls(gpu="H100", image=vllm_image, volumes={MEM_MNT: mem_vol}, timeout=60 * 60)
class NanoGateway:
    @modal.enter()  # TODO: try GPU snapshot, TODO: upgrade to AsyncLLM
    def load_model(self):
        # Reuse HF cache inside the shared Volume to avoid re-downloading weights

        self.engine = LLM(
            model=AGENT_MODEL_ID,
            dtype="auto",
            max_model_len=32768,  # adjust for your Jen Nano variant
            gpu_memory_utilization=0.95,
        )
        self.default_params = SamplingParams(
            temperature=0.2, top_p=0.95, max_tokens=1024, presence_penalty=0.0
        )

    @modal.method()
    def generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2
    ) -> str:
        """Simple text-in, text-out. Keep it dead simple for the demo."""

        params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        outputs = self.engine.generate([prompt], params)
        # vLLM returns a list of RequestOutputs. Take the first, first output piece.
        return outputs[0].outputs[0].text


# # Sub Agents


# ## Web Searcher
@app.function(image=researcher_image, timeout=60)
@modal.concurrent(max_inputs=256)
def search_web(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Fast web search (no API key) using DuckDuckGo."""
    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            # r contains title, href, body
            results.append(
                {
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                }
            )
    return results


# ## Data Extractor
@app.function(image=researcher_image, timeout=90)
@modal.concurrent(max_inputs=256)
def fetch_and_read(url: str) -> Dict[str, Any]:
    """Fetch a URL and extract main text using trafilatura/readability."""

    try:
        resp = httpx.get(
            url, headers={"User-Agent": UA}, follow_redirects=True, timeout=25
        )
        html = resp.text
        text = trafilatura.extract(html, include_comments=False) or ""
        return {
            "url": url,
            "status": resp.status_code,
            "text": text[:200_000],
        }  # guard size
    except Exception as e:
        return {"url": url, "error": str(e), "text": ""}


# ## Snippet Reranker
@app.function(image=researcher_image, timeout=90, concurrency_limit=256)
def rerank_with_llm(
    run_id: str,
    query: str,
    passages: List[Dict[str, Any]],
    max_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    CPU worker that *calls the shared GPU LLM* to rerank snippets.
    This demonstrates many CPU sandboxes leaning on one hot model.
    """
    # Build a tiny prompt; keep costs small.
    prompt = textwrap.dedent(f"""
    You are ranking passages for the query: {query!r}.
    Return the top 10 with a short reason. Output as JSON list:
    [{{"idx": <index in input>, "score": <0..1>, "reason": "..."}}]
    Passages:
    {json.dumps(passages[:30], ensure_ascii=False)[:12000]}
    """)
    # Call the shared gateway (RPC, not HTTP).
    llm = modal.Cls.from_name(APP_NAME, "LLMGateway")()
    text = llm.generate.remote(prompt, max_tokens=max_tokens)

    # Best-effort JSON parse
    try:
        arr = json.loads(
            text.strip().split("{", 1)[0]
        )  # this line is too hacky; safer parser below
    except Exception:
        try:
            arr = json.loads(_extract_json(text))
        except Exception:
            arr = []  # fall back
    # Attach scores back to passages
    by_idx = {e.get("idx"): e for e in arr if isinstance(e, dict)}
    out = []
    for i, p in enumerate(passages):
        score = (by_idx.get(i) or {}).get("score", 0.0)
        out.append({**p, "score": float(score)})
    # Sort by score desc
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    _write_json(f"/mem/runs/{run_id}/rerank.json", out)
    return out[:10]


# # Planner

SYSTEM_PLAN = """You are a research lead. Produce a JSON object with:
{
  "clarifying_questions": [ ... ],
  "tasks": [
    {"tool": "search_web", "args": {"query": "<string>"}},
    {"tool": "fetch_and_read", "args": {"url": "<string>"}}
  ],
  "next_step_policy": "Describe when to iterate vs. stop.",
  "budget_suggestion": {"max_llm_calls": 50}
}
Keep it compact and executable. Only output JSON, nothing else.
"""


def plan_prompt(user_prompt: str, prior_context: str) -> str:
    return textwrap.dedent(f"""
    {SYSTEM_PLAN}

    Context summary:
    {prior_context}

    User prompt:
    {user_prompt}
    """)


@app.cls(image=researcher_image, timeout=60 * 30)
class DeepResearchAgent:
    @modal.enter()
    def initialize(self):
        # Initialize a reference to LLM app
        self.agent = modal.Cls.from_name(APP_NAME, "LLMGateway")

    @modal.method()
    def research(
        self, prompt: str, user_id: str, max_rounds: int = 2
    ) -> Dict[str, Any]:
        """
        Synchronous "one-shot" endpoint for now.
        - Plans with the LLM (gateway).
        - Fans out search/scrape workers.
        - Reranks with LLM (via rerank_with_llm worker that calls the gateway).
        - Compiles a draft answer with citations.
        """
        run_id = str(uuid.uuid4())
        run_dir = MEM_MNT / f"runs/{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        prior = context_dict.get(user_id) or ""
        plan_text = self.agent.generate.remote(
            plan_prompt(prompt, prior), max_tokens=800
        )
        try:
            plan = json.loads(_extract_json(plan_text))
        except Exception:
            plan = {"clarifying_questions": [], "tasks": []}

        _write_json(f"{run_dir}/plan_round0.json", plan)

        # (Optionally) you'd ask the user clarifying questions here via a web UI.
        # For the skeleton, we proceed with tasks as-is.

        # ----- Round loop -----
        all_evidence: List[Dict[str, Any]] = []
        for round_idx in range(max_rounds):
            tasks: List[Dict[str, Any]] = plan.get("tasks", [])
            if not tasks:
                break

            # 1) Group by tool and fan out each group
            group: Dict[str, List[Dict[str, Any]]] = {}
            for t in tasks:
                name = t.get("tool")
                group.setdefault(name, []).append(t.get("args", {}))

            round_results: Dict[str, List[Any]] = {}

            # Fan-out: SEARCH
            if "search_web" in group:
                q_items = [
                    a.get("query") for a in group["search_web"] if a.get("query")
                ]
                if q_items:
                    # Burst to 100+ workers instantly
                    search_batches = list(search_web.map(q_items))
                    # Flatten and tag the origin query
                    found = []
                    for q, batch in zip(q_items, search_batches):
                        for item in batch:
                            item["query"] = q
                            found.append(item)
                    round_results["search_web"] = found

            # Fan-out: FETCH
            if "fetch_and_read" in group:
                urls = [a.get("url") for a in group["fetch_and_read"] if a.get("url")]
                if urls:
                    fetched = list(fetch_and_read.map(urls))
                    round_results["fetch_and_read"] = fetched

            _write_json(f"{run_dir}/round{round_idx}_raw.json", round_results)

            # 2) Aggregate evidence and (optionally) rerank with the LLM gateway
            passages: List[Dict[str, Any]] = []

            for item in round_results.get("search_web", []):
                passages.append(
                    {
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "url": item.get("url"),
                    }
                )

            for doc in round_results.get("fetch_and_read", []):
                if doc.get("text"):
                    passages.append(
                        {
                            "title": doc.get("url"),
                            "snippet": doc["text"][:800],
                            "url": doc.get("url"),
                        }
                    )

            if not passages:
                break

            # Rerank with many CPU workers, each calling the *shared* LLM
            # (Here we just do one call for simplicity; you can sharded-map if thousands of passages.)
            top = rerank_with_llm.call(
                run_id, prompt, passages
            )  # .call is blocking; .remote if you want a Future
            all_evidence.extend(top)

            # 3) Decide whether to iterate again (ask gateway for next plan)
            decision_prompt = textwrap.dedent(f"""
            You have current evidence (top passages with URLs):
            {json.dumps(top, ensure_ascii=False)[:14000]}

            Given the user goal: {prompt!r}
            Decide whether to continue another round of tasks or finalize an answer.
            Output JSON: {{"continue": true|false, "tasks": [... if continue], "draft": "..." }}
            """)
            decision_text = self.agent.generate.remote(decision_prompt, max_tokens=800)
            try:
                decision = json.loads(_extract_json(decision_text))
            except Exception:
                decision = {"continue": False, "draft": ""}

            _write_json(f"{run_dir}/decision_round{round_idx}.json", decision)

            if not decision.get("continue"):
                # 4) Finalize report
                final_prompt = textwrap.dedent(f"""
                Compile a concise, well-cited answer. Use inline references like [^1], [^2] matching the bibliography JSON.
                Evidence:
                {json.dumps(all_evidence, ensure_ascii=False)[:15000]}
                Output JSON:
                {{
                  "answer_markdown": "<markdown>",
                  "bibliography": [{{"ref": 1, "title": "...", "url": "..."}}]
                }}
                """)
                final_text = self.agent.generate.remote(final_prompt, max_tokens=1200)
                try:
                    final = json.loads(_extract_json(final_text))
                except Exception:
                    final = {
                        "answer_markdown": decision.get("draft", ""),
                        "bibliography": [],
                    }
                _write_json(f"{run_dir}/final.json", final)
                # Update a tiny rolling context summary for the user
                context_dict[user_id] = (context_dict.get(user_id) or "")[:4000]
                return {"run_id": run_id, "final": final, "rounds": round_idx + 1}

            # else continue
            plan = {"tasks": decision.get("tasks", [])}

        # If loop ends without finalize:
        return {
            "run_id": run_id,
            "final": {"answer_markdown": "_No result_", "bibliography": []},
            "rounds": 0,
        }


# -----------------------------------------------------------------------------
# Web endpoints (simple)
# -----------------------------------------------------------------------------


@app.web_endpoint(method="POST")
def research_endpoint(request):
    """
    Minimal HTTP entrypoint.
    Body JSON: {"prompt": "...", "user_id": "u123"}
    """
    payload = request.json
    prompt = payload.get("prompt") or ""
    user_id = payload.get("user_id") or "anon"
    res = DeepResearchAgent().research.call(prompt, user_id, max_rounds=2)
    return res


class SandboxWrapper:
    def __init__(self):
        self.box = modal.Sandbox.create(
            image=agent_image,
            timeout=60 * 20,
            app=app,
            # Modal sandboxes support GPUs!
            gpu="T4",
            # you can also pass secrets here -- note that the main app's secrets are not shared
        )

        # TODO: Initialize Jen Nano server

    def inference(self, prompt):
        # TODO: pass `prompt` onto Jen Nano server
        pass

    def exec(self, code: str) -> tuple[str, str]:
        exc = self.box.exec("python", "-c", code)
        exc.wait()

        stdout = exc.stdout.read()
        stderr = exc.stderr.read()

        if exc.returncode != 0:
            print(f"📦: Failed with exitcode {self.box.returncode}")

        return stdout, stderr
