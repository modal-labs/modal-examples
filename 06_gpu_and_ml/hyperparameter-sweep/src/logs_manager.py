# ---
# pytest: false
# ---

from torch.utils.tensorboard import SummaryWriter


class LogsManager:
    def __init__(self, experiment_name, hparams, num_parameters, tb_log_path):
        self.model_name = (
            f"{experiment_name}"
            f"_context_size={hparams.context_size}_n_heads={hparams.n_heads}"
            f"_dropout={hparams.dropout}"
        )

        model_log_dir = tb_log_path / f"{experiment_name}/{self.model_name}"
        model_log_dir.mkdir(parents=True, exist_ok=True)
        self.train_writer = SummaryWriter(log_dir=f"{model_log_dir}/train")
        self.val_writer = SummaryWriter(log_dir=f"{model_log_dir}/val")

        # save hyperparameters to TensorBoard for easy reference
        pretty_hparams_str = "\n".join(f"{k}: {v}" for k, v in hparams.__dict__.items())
        pretty_hparams_str += f"\nNum parameters: {num_parameters}"
        self.train_writer.add_text("Hyperparameters", pretty_hparams_str)

    def add_train_scalar(self, name, value, step):
        self.train_writer.add_scalar(name, value, step)

    def add_val_scalar(self, name, value, step):
        self.val_writer.add_scalar(name, value, step)

    def add_val_text(self, name, text, step):
        self.val_writer.add_text(name, text, step)

    def flush(self):
        self.train_writer.flush()
        self.val_writer.flush()
