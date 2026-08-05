# Internal Development Guide for Modal Examples

This guide is for fixing bugs in the Modal examples repository. These bugs are
typically triggered by synthetic monitoring (synmon) alerts.

## Prime Directive

**The code must run.** If example code does not run, we have failed the user.
Examples are tested regularly because code rots — external dependencies change,
APIs evolve, and artifacts move. Your job is to make the code run again.

## Scope

- Fix bugs in numbered example directories (`01_getting_started/` through
  `14_clusters/`)
- Ignore the `misc/` folder entirely — it is not continuously tested
- Do not change anything in the `internal/` folder — it contains CI
  infrastructure, not examples. Use its contents, including this file, to guide
  your investigation and testing

## Common Bug Types

Most bugs fall into one of two categories:

1. **Dependency changes** — upstream packages release breaking changes or
   deprecate APIs; other artifacts or APIs change in breaking ways
2. **Modal SDK changes** — the Modal API evolves and examples need updates

If the root cause is outside this repository (e.g., a bug in the Modal SDK
itself), report it rather than implementing a workaround. Include the error,
affected example, and any relevant context.

## Testing Examples

Run examples using the internal test runner:

```bash
MODAL_ENVIRONMENT=examples python -m internal.run_example <example_stem>
```

Where `<example_stem>` is the filename without `.py` (e.g., `hello_world` for
`01_getting_started/hello_world.py`).

This uses the `cmd` and `args` from the example's frontmatter, or defaults to
`modal run <path>`. The test environment automatically sets
`MODAL_SERVE_TIMEOUT=5.0`.

## Updating Dependencies

When fixing dependency-related bugs:

1. **Check PyPI for the latest version** of the problematic package
2. **Update to the latest working version** — not just any version that fixes
   the immediate bug
3. **Confirm the new dependencies work** by running the example
4. **Fix warnings, not just errors** — examples should run cleanly without
   deprecation warnings or other noise

### Pinning Requirements

Follow these pinning rules for container image dependencies:

- Pin to SemVer minor version: `~=x.y.z` or `==x.y`
- For packages with `version < 1.0`: pin to patch version `==0.y.z`
- Pin container base images to stable tags (e.g., `v1`), never `latest`
- Always specify `python_version` when using base images that support it

Example:

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
)
```

## Updating Prose

Examples use literate programming style — prose is written as markdown in
comments on their own lines:

```python
# # This is a Title
#
# This is a paragraph of prose that becomes documentation.
# It continues on multiple lines.

import modal  # This comment stays with the code

# ## This is a Section Header
#
# More prose explaining the next code block.

app = modal.App("example-name")
```

**Critical rules:**

- **Always update prose when code changes affect what it describes** — the prose
  is the documentation and is equally important as the code
- **Do not rename files** — file paths are referenced elsewhere and renaming
  breaks links

## Code Style

### Time constants

Define `MINUTES = 60` (seconds) at module level for readable timeouts:

```python
MINUTES = 60  # seconds

@app.function(timeout=10 * MINUTES)
```

### Pin model revisions

Always pin model revisions to avoid surprises when upstream repos update:

```python
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf"  # pin to avoid surprises!
```

### Hugging Face downloads

Use the high-performance download flag and pin `huggingface-hub`:

```python
.uv_pip_install("huggingface-hub==0.36.0")
.env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster downloads
```

### Volume conventions

- Use descriptive kebab-case names: `"huggingface-cache"`, `"vllm-cache"`
- Always use `create_if_missing=True` for first-run convenience

```python
cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
```

### App naming

Use `example-` prefix with kebab-case:

```python
app = modal.App("example-vllm-inference")
```

### Path references

Use `Path(__file__).parent` for paths relative to the script:

```python
here = Path(__file__).parent
input_path = here / "data" / "config.yaml"
```

### Clean up warnings and logs

Examples should run cleanly. Fix deprecation warnings and suppress noisy logs —
unhandled warnings are amateurish and make debugging harder.

- Run `.entrypoint([])` on NVIDIA container images to disable license logging:
  ```python
  modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
      .entrypoint([])  # disable noisy NVIDIA license logging
  ```

### Use fully-qualified Modal names

Always use `modal.X` instead of importing names directly:

```python
# ✅ Good
import modal

image = modal.Image.debian_slim()
vol = modal.Volume.from_name("my-vol", create_if_missing=True)

# ❌ Bad
from modal import Image, Volume

image = Image.debian_slim()
```

### No local dependencies except `modal` and `fastapi`

Examples must not require local Python dependencies beyond `modal` (and
`fastapi` if needed). Use the Python standard library for HTTP requests and
other utilities. Dependencies inside Modal Functions are fine and encouraged.

### Each line should spark joy

Prefer clarity and economy. Use `pathlib.Path` over `os.path`. Use meaningful
variable names. Remove code that foregrounds machine concerns over reader
concerns.

## Prose Style

### Capitalize Modal product features

Modal's features are proper nouns: Image, Volume, Function, Cls, App, Secret.
Use `monospace` only when referring to the actual Python object.

```python
# ✅ "Modal Volumes provide distributed storage."
# ✅ "`modal.Volume` has a `from_name` method."
# ❌ "Modal volumes provide distributed storage."
```

### Use active, descriptive headers

Headers should express purpose, not just content:

```python
# ✅ ## Cache model weights
# ❌ ## Modal Volume
# ❌ ## Model weights loading
```

### Use blank lines for visual separation

Add a blank line before code blocks to match the visual separation in rendered
docs:

```python
# ✅ Good

# We define an image with our dependencies.

image = modal.Image.debian_slim().pip_install("torch")

# ❌ Bad

# We define an image with our dependencies.
image = modal.Image.debian_slim().pip_install("torch")
```

### Defer detailed explanations

Don't duplicate Guide content. Keep explanations brief and link to docs:

```python
# ✅ Good
# [Modal Volumes](https://modal.com/docs/guide/volumes) add distributed storage.
# Here, we use one to cache compiler artifacts.

# ❌ Bad (too much detail)
# Modal Volumes provide a high-performance distributed file system...
# (paragraph of text duplicating the Guide)
```

### Link to Modal docs

Link to relevant Modal documentation using full URLs:

```python
# We wrap inference in a Modal [Cls](https://modal.com/docs/guide/lifecycle-functions)
# that ensures models are loaded once when a new container starts.
```

Common links:

- Images: `https://modal.com/docs/guide/images`
- Volumes: `https://modal.com/docs/guide/volumes`
- GPUs: `https://modal.com/docs/guide/gpu`
- Model weights: `https://modal.com/docs/guide/model-weights`
- Lifecycle: `https://modal.com/docs/guide/lifecycle-functions`

## Linting

Before committing changes:

```bash
source venv/bin/activate
ruff check --fix <modified_files>
ruff format <modified_files>
```

See the repository root `CLAUDE.md` for full Python development rules.

## Pull Request Guidelines

See `.github/pull_request_template.md` for the full checklist. Key points:

- Example must be testable with `modal run` or have a custom `cmd` in
  frontmatter
- Example must run with no arguments or have `args` defined in frontmatter
- All container dependencies must be pinned
- No local third-party dependencies required (except `fastapi`)
