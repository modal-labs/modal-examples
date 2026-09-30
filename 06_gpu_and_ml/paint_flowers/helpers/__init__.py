import base64
import io
import json
import math
import os
import pathlib
import random
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import modal
from PIL import Image

SYSTEM_PROMPT = """\
You paint watercolours by writing a p5.js 2.x sketch that uses the p5.brush library, available as the global `brush`.

Reply with one ```javascript fence containing a complete sketch, and nothing else.

Use exactly this skeleton:

  async function setup() {
    createCanvas(600, 600, WEBGL);
    brush.scaleBrushes(3);
    angleMode(DEGREES);
    noLoop();
  }

  function draw() {
    translate(-width / 2, -height / 2);
    background(hex);
    // painting goes here
  }

So: angles are in degrees, and after the translate the canvas runs from 0 to 600 on both axes with the origin at the top LEFT, not the centre. The centre of the canvas is (300, 300). Every coordinate you paint at must be between 0 and 600: a negative coordinate is off the canvas and paints nothing at all. Colours are hex strings like "#e08a72".

Follow the user brief for species and palette.

Paint one flower, centred, filling most of the frame, with a stem and leaves. Do not fill the canvas edge to edge.

Three numbers matter more than any adjective:
- Five petals. Paint each petal two or three times over, not once: a first pass at full size, then a smaller and more opaque pass inside it, and a small dark one near the centre. That layering is where a watercolour gets its depth, and it puts the whole painting at fifteen to thirty filled shapes. Keep the petals as five broad lobes, not fifteen separate little marks.
- Petals reaching 200 to 240 units from the centre, so the flower occupies the frame.
- Opacity never below 150, and 180 to 230 on the petals, with brush.fillBleed between 0.2 and 0.3. A dilute wash with a wide bleed disperses until no pigment reaches density and the flower comes out invisible.

These ten brush methods exist. Nothing else on `brush` exists, and there is no way to
name or select a brush: everything is painted as a filled shape. Do not call any other brush
method, and never use the bare p5 drawing functions such as ellipse, rect, vertex or beginShape:
  brush.scaleBrushes(factor)
  brush.noStroke()
  brush.fill(colorHex, opacity)
  brush.noFill()
  brush.fillBleed(amount)
  brush.fillTexture(amount, borderIntensity)
  brush.beginShape(curvature)
  brush.vertex(x, y)
  brush.endShape(true)
  brush.circle(x, y, radius, scribble)

colorHex is a string like "#e08a72". opacity runs 0 to 255. amount, curvature,
borderIntensity and scribble run 0 to 1.

Every mark is a filled shape. Build petals and leaves with brush.beginShape, a run of at least
three brush.vertex calls, then brush.endShape(true), and call brush.fill before each one. A stem
is a long narrow filled shape, not a line. brush.circle fills a disc.
"""

_JS_FENCE = re.compile(r"```(?:javascript|js)\s*\n(.*?)```", re.DOTALL)
_BANNED = re.compile(
    r"\b(loadImage|loadBytes|loadJSON|fetch|XMLHttpRequest|WebSocket|document\.|window\.|eval|import|require)\b"
)
_ENTRY = re.compile(
    r"\b(?:function\s+(?:setup|draw)\s*\(|(?:setup|draw)\s*=\s*(?:function|\())"
)
_BRUSH_CALL = re.compile(r"\bbrush\s*\.\s*([A-Za-z_$][\w$]*)")
_TEXT_CALL = re.compile(r"(^|[^\w.])text\s*\(")
_BARE_P5 = re.compile(
    r"(^|[^.\w])(arc|beginShape|bezier|box|circle|cone|curve|cylinder|ellipse|"
    r"line|plane|point|quad|rect|sphere|square|torus|triangle|vertex)\s*\(",
    re.MULTILINE,
)
BRUSH_METHODS = (
    "scaleBrushes",
    "noStroke",
    "fill",
    "noFill",
    "fillBleed",
    "fillTexture",
    "beginShape",
    "vertex",
    "endShape",
    "circle",
)
_PAINTING = frozenset({"circle", "endShape"})
_KNOWN = frozenset(BRUSH_METHODS) | {"load"}
MIN_PAINT_FRACTION = 0.005


def extract_sketch(response: str, parse_response) -> str | None:
    parsed = parse_response(response)
    m = _JS_FENCE.search(parsed.content or "")
    if not m:
        return None
    code = m.group(1).strip()
    if not code or not _ENTRY.search(code) or len(code) > 24000:
        return None
    if _BANNED.search(code) or re.search(r"data\s*:\s*image/", code, re.I):
        return None
    return code


RENDER_JS = r"""
const fs = require('fs');
const puppeteer = require('puppeteer-core');

function readFirst(candidates) {
  for (const p of candidates) {
    if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8');
  }
  throw new Error('missing ' + candidates[0]);
}

(async () => {
  const sketch = fs.readFileSync(process.argv[2], 'utf8');
  const p5js = readFirst([
    '/render/node_modules/p5/lib/p5.min.js',
    '/render/node_modules/p5/dist/p5.min.js',
  ]);
  const brushjs = fs.readFileSync('/render/node_modules/p5.brush/dist/p5.brush.js', 'utf8');
  const buildHtml = (src) => `<!DOCTYPE html><html><head><meta charset="utf-8">
<script>
(function () {
  let s = 1;
  Math.random = function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
})();
</script>
<script>${p5js}</script>
<script>${brushjs}</script>
<style>html,body{margin:0}</style>
</head><body>
<script>window.__err=null;window.__setupDone=false;window.onerror=(m)=>{window.__err=String(m)};window.onunhandledrejection=(e)=>{window.__err=String(e.reason)};</script>
<script>try{(0,eval)(${JSON.stringify(src).replace(/</g, '\\u003c')} + ";if(typeof setup==='function')window.setup=setup;if(typeof draw==='function')window.draw=draw")}catch(e){window.__err=String(e)}</script>
<script>
if (typeof window.setup === "function") {
  const original = window.setup;
  window.setup = function () {
    try { return original.apply(this, arguments); }
    finally { window.__setupDone = true; }
  };
} else {
  window.__setupDone = true;
}
</script>
</body></html>`;
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium',
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
           '--use-gl=angle', '--use-angle=swiftshader',
           '--enable-unsafe-swiftshader', '--disable-gpu-sandbox'],
  });
  const attempt = async (src) => {
    const page = await browser.newPage();
    await page.setViewport({ width: 700, height: 700 });
    try {
      await page.setContent(buildHtml(src), { waitUntil: 'load', timeout: 20000 });
      await page.waitForFunction(
        'window.__err !== null || (window.__setupDone && document.querySelector("canvas"))',
        { timeout: 15000 }).catch(() => {});
      let dataUrl = null;
      let prev = '';
      for (let i = 0; i < 12; i++) {
        await new Promise(r => setTimeout(r, 1000));
        dataUrl = await page.evaluate(
          "document.querySelector('canvas') ? document.querySelector('canvas').toDataURL('image/png') : null"
        );
        if (!dataUrl) break;
        if (dataUrl === prev) break;
        prev = dataUrl;
      }
      const err = await page.evaluate('window.__err');
      if (!dataUrl) return { err: err || 'no canvas', buf: null };
      const buf = Buffer.from(dataUrl.split(',')[1], 'base64');
      return { err, buf };
    } finally {
      await page.close();
    }
  };

  try {
    const res = await attempt(sketch);
    if (!res.buf) {
      console.error('SKETCH_ERROR: ' + (res.err || 'no canvas'));
      process.exit(2);
    }
    process.stdout.write('PNGB64:' + Buffer.from(res.buf).toString('base64'));
    process.exit(0);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error('RENDER_ERROR: ' + e); process.exit(3); });
"""


def renderer_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("chromium", "nodejs", "npm", "fonts-liberation")
        .run_commands(
            "mkdir -p /render",
            "cd /render && npm install --no-audit --no-fund"
            " p5@2.3.2 p5.brush@2.2.1 puppeteer-core@23.11.1",
        )
    )


REMOTE_ASSETS_DIR = "/root/flower_assets"
REFERENCE_POOL = "HuggingEnvs/watercolour-reference-pool"
JUDGE_REFS = 4

HPSV3_APP = "dojo-flower-hpsv3"
HPSV3_PROMPT = "a loose watercolour flower"

_CACHE: dict[str, object] = {}
_LOCKS: dict[str, threading.Lock] = {}

hpsv3_app = modal.App(HPSV3_APP)
HPS_IMAGE = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "hpsv3==1.0.0",
    "matplotlib",
    "tensorboard",
)


@hpsv3_app.function(image=HPS_IMAGE, gpu="H100", timeout=600, scaledown_window=3600)
def score_image(png: bytes) -> float:
    from hpsv3 import HPSv3RewardInferencer

    inferencer = _CACHE.get("hpsv3_model")
    if inferencer is None:
        inferencer = HPSv3RewardInferencer(device="cuda")
        _CACHE["hpsv3_model"] = inferencer
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(png)
        tmp.flush()
        rewards = inferencer.reward(prompts=[HPSV3_PROMPT], image_paths=[tmp.name])
    return float(rewards[0][0].item())


def launch_hpsv3() -> None:
    with modal.enable_output():
        hpsv3_app.deploy()


def bake_reference_pool() -> None:
    os.environ["HF_HOME"] = "/tmp/hf-bake"
    from datasets import load_dataset

    dest = pathlib.Path(REMOTE_ASSETS_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(REFERENCE_POOL, split="train")
    seen: set[str] = set()
    for row in rows:
        tier = str(row["tier"])
        if tier not in ("love", "okay"):
            continue
        seen.add(tier)
        stem = pathlib.Path(str(row["source_file"])).stem
        row["image"].convert("RGB").save(
            dest / f"{stem}.webp", format="WEBP", quality=80, method=6
        )
    if seen != {"love", "okay"}:
        raise RuntimeError("reference pool is missing love or okay images")


def overlay_flower_image(image: modal.Image) -> modal.Image:
    return (
        image.uv_pip_install(
            "modal~=1.5.5", "httpx~=0.28.1", "pillow~=11.1", "datasets"
        )
        .add_local_file(__file__, remote_path="/root/helpers.py", copy=True)
        .run_function(bake_reference_pool)
        .run_commands("rm -rf /tmp/hf-bake /root/.cache/huggingface")
    )


def assets_dir() -> str:
    return REMOTE_ASSETS_DIR


def _load_tier(root: str, tier: str) -> list[bytes]:
    refs = []
    for name in sorted(os.listdir(root)):
        if not (name.startswith(f"{tier}_") and name.endswith(".webp")):
            continue
        buf = io.BytesIO()
        Image.open(os.path.join(root, name)).convert("RGB").save(buf, format="PNG")
        refs.append(buf.getvalue())
    return refs


def reference_pools() -> dict[str, list[bytes]]:
    with _LOCKS.setdefault("refs", threading.Lock()):
        if "refs" not in _CACHE:
            root = assets_dir()
            _CACHE["refs"] = {
                "love": _load_tier(root, "love"),
                "okay": _load_tier(root, "okay"),
            }
    return _CACHE["refs"]


def pick_references(k: int = JUDGE_REFS) -> list[bytes]:
    pools = reference_pools()
    n_love = k // 2
    n_okay = k - n_love
    if len(pools["love"]) < n_love or len(pools["okay"]) < n_okay:
        raise RuntimeError("reference pool is missing love or okay images")
    return random.sample(pools["love"], n_love) + random.sample(pools["okay"], n_okay)


def probe_score(png: bytes) -> float:
    with _LOCKS.setdefault("hpsv3", threading.Lock()):
        if "hpsv3" not in _CACHE:
            _CACHE["hpsv3"] = modal.Function.from_name(HPSV3_APP, "score_image")
    mu = _CACHE["hpsv3"].remote(png)
    return 1.0 / (1.0 + math.exp(-mu / 4.0))


def paint_fraction(png: bytes) -> float:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    if w == 0 or h == 0:
        return 0.0
    corners = [
        im.getpixel((0, 0)),
        im.getpixel((w - 1, 0)),
        im.getpixel((0, h - 1)),
        im.getpixel((w - 1, h - 1)),
    ]
    bg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))
    painted = 0
    n = 0
    for p in im.getdata():
        n += 1
        if abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > 30:
            painted += 1
    return painted / n


def gate_sketch(code: str, png: bytes) -> tuple[bool, dict]:
    body = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    body = re.sub(r"//[^\n]*", " ", body)
    used = set(_BRUSH_CALL.findall(body))
    reasons = []
    if "WEBGL" not in code:
        reasons.append("not_webgl")
    if _TEXT_CALL.search(body):
        reasons.append("text")
    if _BARE_P5.search(body):
        reasons.append("bare_p5")
    if used - _KNOWN:
        reasons.append("unknown_brush")
    if not (used & _PAINTING):
        reasons.append("no_brush_paint")
    frac = paint_fraction(png)
    if frac < MIN_PAINT_FRACTION:
        reasons.append("pigment")
    return (not reasons), {
        "gate": ",".join(reasons) or "ok",
        "paint": round(frac, 4),
        "brush_calls": sorted(used),
    }


PAIRWISE_PROMPT = """You are judging two generative watercolour paintings, image A first and image B second.
Which one is the better hibiscus watercolour? Prefer a recognisable flower with petals around a centre, a stem and leaves, over a washed disc or formless coloured mass. Then prefer soft pigment bleeds, layered translucent washes that show through each other, varied edge softness where edges dissolve rather than stop, and deliberate composition. Muddy opaque blobs, uniform scribbles, hard flat edges and near-empty canvases are worse.

Answer with strict JSON only: {"winner": "A" or "B", "why": "<8 words>"}"""


def judge_pair(candidate: bytes, reference: bytes, flip: bool, judge):
    def uri(png: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png).decode()

    a, b = (reference, candidate) if flip else (candidate, reference)
    try:
        with _LOCKS.setdefault("judge", threading.Lock()):
            judge.wait_until_ready()
        msg = judge.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PAIRWISE_PROMPT},
                        {"type": "text", "text": "Image A:"},
                        {"type": "image_url", "image_url": {"url": uri(a)}},
                        {"type": "text", "text": "Image B:"},
                        {"type": "image_url", "image_url": {"url": uri(b)}},
                    ],
                }
            ],
            timeout=180,
            max_tokens=96,
            temperature=0.3,
            chat_template_kwargs={"enable_thinking": False},
        )
        text = msg.get("content") or ""
    except Exception as e:
        print(f"judge_pair failed: {type(e).__name__}: {e}", flush=True)
        return None
    winner = None
    try:
        winner = json.loads(text[text.index("{") : text.rindex("}") + 1]).get("winner")
    except Exception:
        for token in ('"A"', '"B"'):
            if token in text:
                winner = token.strip('"')
                break
    if winner not in ("A", "B"):
        return None
    return float((winner == "A") != flip)


def judge_win_rate(png: bytes, judge) -> tuple[float | None, dict]:
    refs = pick_references(JUDGE_REFS)
    jobs = [(ref, flip) for ref in refs for flip in (False, True)]
    with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        raw = list(
            pool.map(
                lambda rf: judge_pair(png, rf[0], flip=rf[1], judge=judge),
                jobs,
            )
        )
    got = []
    for i in range(0, len(raw), 2):
        a, b = raw[i], raw[i + 1]
        if a is None or b is None:
            continue
        got.append((a + b) / 2)
    if not got:
        print("judge_win_rate: no usable votes", flush=True)
        return None, {
            "judge_votes": 0,
            "judge_wins": 0.0,
            "judge_refs": len(refs),
        }
    return sum(got) / len(got), {
        "judge_votes": len(got),
        "judge_wins": round(sum(got), 2),
        "judge_refs": len(refs),
    }


GATE_WEIGHT = 0.05
LENGTH_WEIGHT = 0.05
PROBE_WEIGHT = 0.30
JUDGE_WEIGHT = 0.60
MIN_LENGTH_TOKENS = 150
TARGET_LENGTH_TOKENS = 3000
RUNAWAY_LENGTH_TOKENS = 6000


def length_score(source: str) -> float:
    tokens = len(source) / 4
    if tokens < MIN_LENGTH_TOKENS or tokens > RUNAWAY_LENGTH_TOKENS:
        return 0.0
    if tokens >= TARGET_LENGTH_TOKENS:
        return 1.0
    return (tokens - MIN_LENGTH_TOKENS) / (TARGET_LENGTH_TOKENS - MIN_LENGTH_TOKENS)


def score_png(
    png: bytes | None, code: str, judge, render_meta: dict
) -> tuple[float | None, dict, bytes | None]:
    if png is None:
        if render_meta.get("render") == "unavailable":
            return None, {**render_meta, "infra": "render"}, None
        return 0.0, render_meta, None
    meta = dict(render_meta)
    ok, gate_meta = gate_sketch(code, png)
    meta.update(gate_meta)
    if not ok:
        return 0.0, meta, png
    length = length_score(code)
    meta["length"] = round(length, 3)
    reward = GATE_WEIGHT
    reward += LENGTH_WEIGHT * length
    try:
        probe = probe_score(png)
    except Exception as e:
        print(f"probe_score failed: {type(e).__name__}: {e}", flush=True)
        return None, {**meta, "infra": "hpsv3"}, png
    wins, judge_meta = judge_win_rate(png, judge)
    meta.update(probe=round(probe, 3), **judge_meta)
    if wins is None:
        return None, {**meta, "infra": "judge"}, png
    meta["wins"] = round(wins, 3)
    reward += PROBE_WEIGHT * probe + JUDGE_WEIGHT * wins
    return round(reward, 4), meta, png


def skip_infra_rewards(args, samples):
    raw = []
    kept = []
    for sample in samples:
        value = sample.reward
        if value is None or getattr(sample, "remove_sample", False):
            raw.append(0.0)
            kept.append(False)
        else:
            raw.append(float(value))
            kept.append(True)
    n = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    out = list(raw)
    for start in range(0, len(raw), n):
        idx = range(start, min(start + n, len(raw)))
        vals = [raw[i] for i in idx if kept[i]]
        if not vals:
            for i in idx:
                out[i] = 0.0
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            scale = var**0.5 + 1e-6
        else:
            scale = 1e-6
        for i in idx:
            out[i] = 0.0 if not kept[i] else (raw[i] - mean) / scale
    return raw, out
