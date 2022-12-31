import argparse
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


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def _list_models() -> dict[str, ModelMetadata]:
    registry_filepath = config.MODEL_STORE_DIR / config.MODEL_REGISTRY_FILENAME
    with open(registry_filepath, "r") as f:
        registry_data = json.load(f)
    return {
        m_id: ModelMetadata(
            impl_name=m["impl_name"],
            save_date=m["save_date"],
            git_commit_hash=m["git_commit_hash"],
            metrics=TrainMetrics(
                dataset_id=m["metrics"]["dataset_id"],
                eval_set_size=m["metrics"]["eval_set_size"],
                accuracy=m["metrics"]["accuracy"],
                precision=m["metrics"]["precision"],
            ),
        )
        for m_id, m in registry_data.items()
    }


@stub.function(shared_volumes={config.VOLUME_DIR: volume})
def _delete_model(model_id: str) -> None:
    pass


def run_list() -> None:
    with stub.run():
        models = _list_models()
    newest_to_oldest = sorted(
        [(key, value) for key, value in models.items()], key=lambda item: item[1].save_date, reverse=True
    )
    for model_id, metadata in newest_to_oldest:
        print(f"\033[96m {model_id} \033[0m{metadata.impl_name}\033[93m {metadata.save_date} \033[0m")


def run_delete_model() -> None:
    pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="model-registry")
    parser.add_argument("--foo", action="store_true", help="foo is great option")
    sub_parsers = parser.add_subparsers(dest="subcommand")

    # create the parser for the "list" sub-command
    parser_list = sub_parsers.add_parser("list", help="Show all models in registry.")
    parser_list.add_argument("--json", type=int, help="Output in JSON format instead of a plaintext table.")

    # create the parser for the "delete-model" sub-command
    parser_delete = sub_parsers.add_parser("delete-model", help="Remove a model from registry and storage.")
    parser_delete.add_argument(
        "--dry-run", action="store_true", default=False, help="Don't actually delete, just show deletion plan."
    )
    parser_delete.add_argument("--model-id", action="store", help="sha256 hashtag of model. eg 'sha256.1234567890abcd'")

    args = parser.parse_args()
    if args.subcommand == "list":
        run_list()
    elif args.subcommand == "delete-model":
        run_delete_model()
    elif args.subcommand is None:
        parser.print_help(sys.stderr)
    else:
        raise AssertionError(f"Unimplemented subcommand '{args.subcommand}' was invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
