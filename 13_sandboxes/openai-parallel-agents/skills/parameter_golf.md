# Parameter Golf

## Premise

OpenAI's **Parameter Golf** challenge (March 18 – April 30) asks participants to train the smallest, most efficient language model possible within strict constraints. The goal is L(N) optimization: achieve the lowest loss given a fixed parameter budget, with no constraints on architecture, training steps, or tokenizer choice.

## Rules

- **Size**: The submission artifact (code + compressed weights) must fit within **16 MB** (16,000,000 bytes).
- **Compute**: Training must complete in under **10 minutes on 8× H100 GPUs**. Evaluation is also capped at 10 minutes.
- **Self-contained**: No network calls or external downloads during evaluation.
- **Data**: Train on the **FineWeb** dataset (provided).

## Scoring

Performance is measured in **bits per byte (BPB)** on the FineWeb validation set, evaluated tokenizer-agnostically. Lower is better. New SOTA records must beat the current best by ≥ 0.005 nats with statistical significance (p < 0.01); systems-only optimizations are exempt from this threshold.

## Leaderboard tracks

- **Standard**: ≤10 min on 8× H100 - the main competition.
- **Unlimited compute**: Experimental submissions, not bound by the time limit.

## What's encouraged

- Novel architectures: depth recurrence, parameter tying, test-time compute
- Compression: quantization, low-precision training, custom tokenizers
- SOTA as of writing: ~1.0810 BPB (SP8192 vocab, 3-layer recurrence, parallel residuals, QK-Gain, test-time training)

## Incentives

- $1,000,000 in compute credits from OpenAI
- Potential hiring for early-career researchers and Olympiad medalists

## Repo

https://github.com/openai/parameter-golf - includes local training scripts (MLX for Apple Silicon), RunPod cloud templates, and submission tooling.

## Recommended Skills
Load these
- parallel_research