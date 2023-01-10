"""
Defines minimal data structures and command-line interface (CLI) commands for a model registry.
The CLI commands are operationally useful, used to inspect prior trained models and promote the
most promising models to production serving.
"""
import dataclasses
import json
import sys
from typing import NamedTuple, Optional

from . import config
from .app import stub, volume


class TrainMetrics(NamedTuple):
    # human-readable identifier for the dataset used in training.
    dataset_id: str
    # How many examples in the evaluation subset.
    eval_set_size: int
    # (TP + TN) / (TP + TN + FP + FN)
    accuracy: Optional[float] = None
    # TP / (TP + FP)
    precision: Optional[float] = None
    # TP / (TP + FN)
    recall: Optional[float] = None


class ModelMetadata(NamedTuple):
    impl_name: str
    save_date: str  # UTC+ISO8601 formatted.
    git_commit_hash: str
    metrics: Optional[TrainMetrics] = None

    def serialize(self) -> dict:
        d = self._asdict()
        if d["metrics"]:
            d["metrics"] = d["metrics"]._asdict()
        return d

    @classmethod
    def from_dict(cls, m: dict) -> "ModelMetadata":
        if "metrics" not in m or m["metrics"] is None:
            metrics = None
        else:
            metrics = TrainMetrics(
                dataset_id=m["metrics"]["dataset_id"],
                eval_set_size=m["metrics"]["eval_set_size"],
                accuracy=m["metrics"]["accuracy"],
                precision=m["metrics"]["precision"],
                recall=m["metrics"]["recall"],
            )
        return cls(
            impl_name=m["impl_name"],
            save_date=m["save_date"],
            git_commit_hash=m["git_commit_hash"],
            metrics=metrics,
        )


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def _list_models() -> dict[str, ModelMetadata]:
    registry_filepath = config.MODEL_STORE_DIR / config.MODEL_REGISTRY_FILENAME
    with open(registry_filepath, "r") as f:
        registry_data = json.load(f)
    return {m_id: ModelMetadata.from_dict(m) for m_id, m in registry_data.items()}


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def delete_model(
    # sha256 hashtag of model. eg 'sha256.1234567890abcd'
    model_id: str,
    # Don't actually delete, just show deletion plan.
    dry_run: bool = True,
) -> None:
    """Remove a model from registry and storage."""
    pass


@stub.local_entrypoint
def list_models() -> None:
    """Show all models in registry."""
    with stub.run():
        models = _list_models.call()
    newest_to_oldest = sorted(
        [(key, value) for key, value in models.items()], key=lambda item: item[1].save_date, reverse=True
    )
    for model_id, metadata in newest_to_oldest:
        print(f"\033[96m {model_id} \033[0m{metadata.impl_name}\033[93m {metadata.save_date} \033[0m")


if __name__ == "__main__":
    print("USAGE: modal run spam_detect.model_registry [FUNCTION]")
    raise SystemExit(1)
