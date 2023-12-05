# gpt-fast on Modal

This is a demo of https://github.com/pytorch-labs/gpt-fast running on
[Modal](https://modal.com). It demonstrates how to use speculative sampling,
quantized models, and pytorch compilation to achieve upwards of 125 tokens/s
with batch sizes of 1 (i.e. no vLLM-style continuous batching), on 7B models
running on individual A100 80GB GPUs. It's a multi-file Modal app that
integrates into an existing codebase (files other than `modal.py` were mostly
taken as-is from `pytorch-labs/gpt-fast`), makes of container-lifecyle
primitives, streams responses, and is also able to invoke already-deployed
functions.

TODO:
- [ ] Doc-ify modal.py, publish to website.
- [ ] Make use of draft models for speculative sampling. Takes >10m to compile,
      which runs into runner timeouts.
- [ ] Make use of tensor parallelism.
- [ ] Make use of GPU checkpointing to avoid long cold starts (?)

To run one-off inference:
```
    ۩ modal run gpt-fast.modal::main --prompt "Implement fibonacci in python" 
        \ --no-compile-model
    ...
    Loading model weights ...
    Using int8 weight-only quantization!
    Loading model weights took 11.08 seconds
    Starting inference for prompt = 'Implement fibonacci in python'
     with memoization.

    The time complexity should be O(n)
    The space complexity should be O(n)
    """

    def fibonacci(n, mem=dict()):
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n in mem:
            return mem[n]
    Time for inference 1: 13.24 sec total, 7.55 tokens/sec
    Bandwidth achieved: 51.91 GB/s
    ...
```

Compile the model for faster inference, at the cost of much longer cold-starts:
```
    ۩ modal run gpt-fast.modal::main --prompt "Implement fibonacci in python" \
        --compile-model
    ...
    Running warmup inference ...
    Model compilation time: 298.49 seconds
    Starting inference for prompt = 'Implement fibonacci in python'
    ...
    Time for inference 1: 0.81 sec total, 123.54 tokens/sec
    Bandwidth achieved: 856.83 GB/s
```

Deploy the model and run inference against a container that's already compiled
the pytorch model:
```
    ۩ modal deploy gpt-fast.modal

    # Should happen instantaneously once deployed model is fully compiled, at
    # upwards of 125 tokens/sec.
    ۩ modal run gpt-fast.modal::main --lookup-existing \
        --prompt "Add two numbers in python" --num-samples 10
```

Run a web-version of the app using:
```
    ۩ modal serve gpt-fast.modal::app
```
