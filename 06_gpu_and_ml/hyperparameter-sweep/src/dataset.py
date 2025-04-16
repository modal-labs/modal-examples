# ---
# pytest: false
# ---

import torch


class Dataset:
    """Manage text dataset and batching."""

    def __init__(
        self,
        encoded_text,
        train_percent,
        batch_size,
        context_size,
        device,
    ):
        self.device = device
        self.batch_size = batch_size
        self.context_size = context_size
        assert (train_percent > 0.0) and (train_percent < 1.0), (
            "train_percent must be in (0,1)"
        )

        # Train/Validation split.
        data = torch.tensor(encoded_text, dtype=torch.long)
        n = len(data)
        self.train_data = data[: int(train_percent * n)]
        self.val_data = data[int(train_percent * n) :]

    def get_batch(self, split):
        """Get a batch of train or validation data."""
        data = self.train_data if split == "train" else self.val_data

        starts = torch.randint(len(data) - self.context_size, (self.batch_size,))

        x = torch.stack([data[start : start + self.context_size] for start in starts])

        # +1 because we want to predict the next token.
        y = torch.stack(
            [data[start + 1 : start + self.context_size + 1] for start in starts]
        )
        return x.to(self.device), y.to(self.device)
