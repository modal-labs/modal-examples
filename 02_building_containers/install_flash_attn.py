import modal

app = modal.App("example-install-flash-attn")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint(
        []  # removes chatty prints on entry
    )
    .pip_install("ninja", "packaging", "wheel", "torch")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
)


@app.function(gpu="L40S", image=image)
def run_flash_attn():
    import torch
    from flash_attn import flash_attn_func

    batch_size, seqlen, nheads, headdim, nheads_k = 2, 4, 3, 16, 3

    q = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=torch.float16
    ).to("cuda")
    k = torch.randn(
        batch_size, seqlen, nheads_k, headdim, dtype=torch.float16
    ).to("cuda")
    v = torch.randn(
        batch_size, seqlen, nheads_k, headdim, dtype=torch.float16
    ).to("cuda")

    out = flash_attn_func(q, k, v)
    assert out.shape == (batch_size, seqlen, nheads, headdim)


@app.local_entrypoint()
def main():
    run_flash_attn.remote()
