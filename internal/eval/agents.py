"""Agent backends for code generation.

Supports Claude (Anthropic) and Codex (OpenAI) as configurable agents.
Each agent receives a task description and documentation content,
and generates Modal Python code.
"""

import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = """\
You are an expert Python developer. Your task is to write Python code \
using the Modal cloud platform based ONLY on the documentation provided below.

IMPORTANT RULES:
1. Use ONLY the provided documentation as your reference for Modal APIs.
2. Do NOT rely on any prior knowledge of Modal that contradicts the docs.
3. Write complete, runnable Python code.
4. Include all necessary imports.
5. Follow Modal best practices as described in the documentation.

## Documentation

{docs_content}
"""

USER_PROMPT = """\
Write complete, runnable Modal Python code for the following task:

{task_description}

Requirements:
- The code must be a complete Python script that can be run with `modal run`
- Include all necessary imports
- Follow the patterns shown in the documentation
- Output ONLY the Python code inside a single ```python code block
"""


def extract_code(text: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code in ```python blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find code in generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks found, return the full text (might be raw code)
    return text.strip()


@dataclass
class AgentConfig:
    """Configuration for an agent run."""

    agent_type: str  # "claude" or "codex"
    model: str | None = None  # Model override (uses default if None)
    temperature: float = 0.0
    max_tokens: int = 8192
    extra_params: dict = field(default_factory=dict)

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "codex": "o3-mini",
        }
        return defaults[self.agent_type]


def generate_with_claude(
    task_description: str,
    docs_content: str,
    config: AgentConfig,
) -> str:
    """Generate code using Anthropic's Claude API."""
    import anthropic

    client = anthropic.Anthropic()

    system = SYSTEM_PROMPT.format(docs_content=docs_content)
    user_msg = USER_PROMPT.format(task_description=task_description)

    response = client.messages.create(
        model=config.resolved_model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw_text = response.content[0].text
    return extract_code(raw_text)


def generate_with_codex(
    task_description: str,
    docs_content: str,
    config: AgentConfig,
) -> str:
    """Generate code using OpenAI's API (Codex-style)."""
    import openai

    client = openai.OpenAI()

    system = SYSTEM_PROMPT.format(docs_content=docs_content)
    user_msg = USER_PROMPT.format(task_description=task_description)

    response = client.chat.completions.create(
        model=config.resolved_model,
        temperature=config.temperature,
        max_completion_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    raw_text = response.choices[0].message.content
    return extract_code(raw_text)


def generate_with_codex_responses(
    task_description: str,
    docs_content: str,
    config: AgentConfig,
) -> str:
    """Generate code using OpenAI's Responses API (for Codex agent)."""
    import openai

    client = openai.OpenAI()

    system = SYSTEM_PROMPT.format(docs_content=docs_content)
    user_msg = USER_PROMPT.format(task_description=task_description)

    response = client.responses.create(
        model=config.resolved_model,
        instructions=system,
        input=user_msg,
    )

    raw_text = response.output_text
    return extract_code(raw_text)


AGENT_BACKENDS = {
    "claude": generate_with_claude,
    "codex": generate_with_codex,
    "codex_responses": generate_with_codex_responses,
}


def generate_code(
    task_description: str,
    docs_content: str,
    config: AgentConfig,
) -> str:
    """Generate code using the configured agent backend."""
    backend = AGENT_BACKENDS.get(config.agent_type)
    if backend is None:
        raise ValueError(
            f"Unknown agent type: {config.agent_type}. "
            f"Available: {list(AGENT_BACKENDS.keys())}"
        )
    return backend(task_description, docs_content, config)
