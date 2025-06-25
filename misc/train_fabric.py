"""
# PyTorch Lightning with a single node and 2 GPUs

```bash
modal run train_fabric.py
```
"""
from pathlib import Path
import modal

app = modal.App("fabric-lightning")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.7.1",
    "click==8.1.8",
    "lightning==2.5.1.post0",
    "requests==2.32.3",
)
wiki_volume = modal.Volume.from_name("fabric-wiki", create_if_missing=True)
data_path = Path("/wiki")


@app.function(
    image=image,
    volumes={data_path.as_posix(): wiki_volume},
    cpu=12,
    memory=30 * 1024,
    gpu="A100:2",
)
def train_modal():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "lightning.fabric.cli", "--devices", "2", __file__]
    )


if __name__ == "__main__":
    from pathlib import Path
    import lightning as L
    import torch
    import torch.nn.functional as F
    from lightning.pytorch.demos import Transformer, WikiText2
    from torch.utils.data import DataLoader

    L.seed_everything(42)

    fabric = L.Fabric()

    data_path = Path("/wiki")
    torch.set_float32_matmul_precision("high")
    with fabric.rank_zero_first():
        dataset = WikiText2(download=True, data_dir=data_path)

    model = Transformer(vocab_size=dataset.vocab_size)
    train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    max_steps = len(train_dataloader)
    for batch_idx, batch in enumerate(train_dataloader):
        input, target = batch
        output = model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            fabric.print(f"iteration: {batch_idx}/{max_steps} - loss {loss.item():.4f}")
