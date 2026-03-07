# Documentation Variants

This directory contains different documentation variants for A/B testing.

## Structure

Each subdirectory or file represents a "docs variant" that can be tested.
The runner will load all `.txt` and `.md` files from a variant directory
and concatenate them as the documentation context for the LLM agent.

## Adding a new variant

1. Create a new directory (e.g., `my_variant/`)
2. Add documentation files (`.txt` or `.md`)
3. Run the eval with `--docs-variant my_variant`

## Default behavior

If no variant is specified, the runner fetches the current
`https://modal.com/llms.txt` as the documentation source.

## Example variants to create

- `llms_txt_current/` - Snapshot of the current llms.txt
- `llms_txt_minimal/` - A stripped-down version with only core API docs
- `llms_txt_with_examples/` - llms.txt plus inline code examples
- `full_reference/` - Complete API reference documentation
- `guide_only/` - Only the guide documentation, no API reference

## Fetching docs

You can fetch and save the current llms.txt as a variant:

```bash
mkdir -p internal/eval/docs/llms_txt_current
curl -s https://modal.com/llms.txt > internal/eval/docs/llms_txt_current/llms.txt
```
