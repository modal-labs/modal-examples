import argparse
import dataclasses
from typing import Optional


from .app import stub, volume


@dataclasses.dataclass
class ModelMetadata:
    impl_name: str
    save_date: str  # UTC+ISO8601 formatted.
    git_commit_hash: str


def list_models() -> list[ModelMetadata]:
    pass


def delete_model(model_id: str) -> None:
    pass


def main(argv: Optional[list[str]] = None) -> int:
    # create the top-level parser
    parser = argparse.ArgumentParser(prog="model-registry")
    parser.add_argument("--foo", action="store_true", help="foo is great option")

    # create sub-parser
    sub_parsers = parser.add_subparsers(help="sub-command help")

    # create the parser for the "ahoy" sub-command
    parser_list = sub_parsers.add_parser("list", help="Show all models in registry.")
    parser_list.add_argument("--json", type=int, help="Output in JSON format instead of a plaintext table.")

    # create the parser for the "booo" sub-command
    parser_delete = sub_parsers.add_parser("delete-model", help="Remove a model from registry and storage.")
    parser_delete.add_argument(
        "--dry-run", action="store_true", default=False, help="Don't actually delete, just show deletion plan."
    )
    parser_delete.add_argument("--model-id", action="store", help="sha256 hashtag of model. eg 'sha256.1234567890abcd'")

    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    raise SystemExit(main())
