# Docs eval framework for A/B testing documentation variants.
#
# Usage:
#   # Generate eval tasks from examples
#   modal run internal/eval/generate_tasks.py
#
#   # Run evaluation
#   modal run internal/eval/runner.py --docs-variant llms_txt --agent claude
#
#   # Compare two doc variants
#   modal run internal/eval/runner.py --docs-variant llms_txt --docs-variant llms_txt_v2 --agent claude
