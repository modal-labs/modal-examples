# Docs eval framework for A/B testing documentation variants.
#
# Runs coding agents (Claude Code CLI or Codex CLI) inside Modal Sandboxes
# with documentation mounted as files for controlled access.
#
# Usage:
#   # Generate eval tasks from examples
#   modal run internal/eval/generate_tasks.py
#
#   # Run evaluation with Claude Code agent
#   modal run internal/eval/runner.py --docs-variant llms_txt --agent claude
#
#   # Run evaluation with Codex agent
#   modal run internal/eval/runner.py --docs-variant llms_txt --agent codex
#
#   # Compare two doc variants
#   modal run internal/eval/runner.py --docs-variant llms_txt --docs-variant llms_txt_v2
