import modal

app = modal.App("example-install-flash-attn")

# Find releases at https://github.com/Dao-AILab/flash-attention/releases
flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl"
)

image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "torch==2.6.0", "numpy==2.2.4", flash_attn_release
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
