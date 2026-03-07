"""Code evaluation logic for generated Modal code.

Evaluates generated code on multiple dimensions:
1. Syntax validity (AST parsing)
2. Modal API pattern usage (structural checks)
3. Semantic similarity to reference (LLM-as-judge)
"""

import ast
import re
from dataclasses import dataclass


@dataclass
class SyntaxResult:
    valid: bool
    error: str | None = None


@dataclass
class PatternScores:
    """Scores for Modal API pattern usage (each 0 or 1)."""

    has_modal_import: bool = False
    has_app_creation: bool = False
    has_function_decorator: bool = False
    has_entrypoint: bool = False
    has_image_definition: bool = False
    uses_remote_or_map: bool = False
    has_web_endpoint: bool = False
    has_gpu_config: bool = False
    has_volume_or_mount: bool = False
    has_secret_usage: bool = False
    has_sandbox_usage: bool = False
    has_schedule: bool = False

    @property
    def matched_count(self) -> int:
        return sum(
            1
            for v in [
                self.has_modal_import,
                self.has_app_creation,
                self.has_function_decorator,
                self.has_entrypoint,
            ]
            if v
        )

    @property
    def core_score(self) -> float:
        """Score for core Modal patterns (0-1). These are almost always needed."""
        core = [
            self.has_modal_import,
            self.has_app_creation,
            self.has_function_decorator,
        ]
        return sum(1 for v in core if v) / len(core)


@dataclass
class JudgeScores:
    """Scores from LLM-as-judge evaluation."""

    api_correctness: int = 0  # 0-5: Correct use of Modal APIs
    functional_match: int = 0  # 0-5: Functionally equivalent to reference
    code_quality: int = 0  # 0-5: General code quality
    reasoning: str = ""  # Judge's reasoning

    @property
    def normalized_score(self) -> float:
        """Normalized score (0-1)."""
        return (self.api_correctness + self.functional_match + self.code_quality) / 15


@dataclass
class EvalScores:
    """Combined evaluation scores."""

    syntax: SyntaxResult
    patterns: PatternScores
    judge: JudgeScores | None = None

    @property
    def overall_score(self) -> float:
        """Weighted overall score (0-1)."""
        if not self.syntax.valid:
            return 0.0

        pattern_weight = 0.3
        judge_weight = 0.7

        score = self.patterns.core_score * pattern_weight
        if self.judge:
            score += self.judge.normalized_score * judge_weight
        else:
            # If no judge, patterns get full weight
            score = self.patterns.core_score

        return score

    def to_dict(self) -> dict:
        result = {
            "syntax_valid": self.syntax.valid,
            "syntax_error": self.syntax.error,
            "patterns": {
                "has_modal_import": self.patterns.has_modal_import,
                "has_app_creation": self.patterns.has_app_creation,
                "has_function_decorator": self.patterns.has_function_decorator,
                "has_entrypoint": self.patterns.has_entrypoint,
                "has_image_definition": self.patterns.has_image_definition,
                "uses_remote_or_map": self.patterns.uses_remote_or_map,
                "has_web_endpoint": self.patterns.has_web_endpoint,
                "has_gpu_config": self.patterns.has_gpu_config,
                "core_score": self.patterns.core_score,
            },
            "overall_score": self.overall_score,
        }
        if self.judge:
            result["judge"] = {
                "api_correctness": self.judge.api_correctness,
                "functional_match": self.judge.functional_match,
                "code_quality": self.judge.code_quality,
                "reasoning": self.judge.reasoning,
                "normalized_score": self.judge.normalized_score,
            }
        return result


def check_syntax(code: str) -> SyntaxResult:
    """Check if the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return SyntaxResult(valid=True)
    except SyntaxError as e:
        return SyntaxResult(valid=False, error=str(e))


def check_patterns(code: str) -> PatternScores:
    """Check for Modal API patterns in the generated code."""
    scores = PatternScores()

    # Check imports
    scores.has_modal_import = bool(re.search(r"import\s+modal", code))

    # Check App creation
    scores.has_app_creation = bool(
        re.search(r"modal\.App\s*\(", code) or re.search(r"App\s*\(", code)
    )

    # Check function decorator
    scores.has_function_decorator = bool(
        re.search(r"@app\.function", code) or re.search(r"@app\.cls", code)
    )

    # Check entrypoint
    scores.has_entrypoint = bool(
        re.search(r"@app\.local_entrypoint", code)
        or re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', code)
    )

    # Check image definition
    scores.has_image_definition = bool(
        re.search(r"modal\.Image", code) or re.search(r"Image\.", code)
    )

    # Check remote/map usage
    scores.uses_remote_or_map = bool(
        re.search(r"\.remote\s*\(", code)
        or re.search(r"\.map\s*\(", code)
        or re.search(r"\.starmap\s*\(", code)
    )

    # Check web endpoint decorators
    scores.has_web_endpoint = bool(
        re.search(r"@modal\.(fastapi_endpoint|asgi_app|wsgi_app|web_server)", code)
        or re.search(r"@(modal\.)?fastapi_endpoint", code)
    )

    # Check GPU configuration
    scores.has_gpu_config = bool(re.search(r"gpu\s*=", code))

    # Check volume/mount usage
    scores.has_volume_or_mount = bool(
        re.search(r"modal\.Volume", code)
        or re.search(r"modal\.Mount", code)
        or re.search(r"Volume\.", code)
    )

    # Check secret usage
    scores.has_secret_usage = bool(
        re.search(r"modal\.Secret", code) or re.search(r"Secret\.", code)
    )

    # Check sandbox usage
    scores.has_sandbox_usage = bool(
        re.search(r"modal\.Sandbox", code) or re.search(r"Sandbox\.", code)
    )

    # Check scheduling
    scores.has_schedule = bool(
        re.search(r"modal\.Period", code)
        or re.search(r"modal\.Cron", code)
        or re.search(r"schedule\s*=", code)
    )

    return scores


JUDGE_PROMPT = """\
You are an expert code reviewer evaluating generated Modal Python code.

Compare the GENERATED code against the REFERENCE code and rate on these dimensions:

1. **API Correctness** (0-5): Does the generated code use Modal APIs correctly?
   - Correct imports, decorators, function signatures
   - Proper use of App, Function, Image, etc.

2. **Functional Match** (0-5): Is the generated code functionally equivalent to the reference?
   - Does it accomplish the same task?
   - Does it use similar patterns and approaches?

3. **Code Quality** (0-5): General code quality
   - Clean, readable, well-structured
   - Proper error handling where appropriate
   - Good naming conventions

## REFERENCE CODE:
```python
{reference_code}
```

## GENERATED CODE:
```python
{generated_code}
```

Respond in this EXACT JSON format (no other text):
{{
  "api_correctness": <0-5>,
  "functional_match": <0-5>,
  "code_quality": <0-5>,
  "reasoning": "<brief explanation>"
}}
"""


def judge_with_llm(
    generated_code: str,
    reference_code: str,
    judge_agent: str = "claude",
    judge_model: str | None = None,
) -> JudgeScores:
    """Use an LLM to judge the quality of generated code."""
    import json

    prompt = JUDGE_PROMPT.format(
        reference_code=reference_code,
        generated_code=generated_code,
    )

    if judge_agent == "claude":
        import anthropic

        client = anthropic.Anthropic()
        model = judge_model or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
    elif judge_agent == "codex":
        import openai

        client = openai.OpenAI()
        model = judge_model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
    else:
        raise ValueError(f"Unknown judge agent: {judge_agent}")

    # Parse JSON response
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(raw)

        return JudgeScores(
            api_correctness=int(data.get("api_correctness", 0)),
            functional_match=int(data.get("functional_match", 0)),
            code_quality=int(data.get("code_quality", 0)),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError) as e:
        return JudgeScores(reasoning=f"Failed to parse judge response: {e}\n{raw}")


def evaluate_code(
    generated_code: str,
    reference_code: str,
    use_judge: bool = True,
    judge_agent: str = "claude",
    judge_model: str | None = None,
) -> EvalScores:
    """Run full evaluation on generated code."""
    syntax = check_syntax(generated_code)
    patterns = check_patterns(generated_code)

    judge = None
    if use_judge and syntax.valid:
        judge = judge_with_llm(
            generated_code,
            reference_code,
            judge_agent=judge_agent,
            judge_model=judge_model,
        )

    return EvalScores(syntax=syntax, patterns=patterns, judge=judge)
