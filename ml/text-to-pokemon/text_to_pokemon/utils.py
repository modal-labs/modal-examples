import argparse
import io
import json
import sys
import urllib.request

from . import config
from .main import stub, volume
from .pokemon_naming import (
    fetch_pokemon_names,
    generate_names,
    rnn_image,
    train_rnn,
)


@stub.function(shared_volumes={config.CACHE_DIR: volume})
def reset_diskcache(dry_run=True) -> None:
    """
    Delete all Pokémon character samples and cards from disk cache.
    Useful when a changes are made to character or card generation process
    and you want create cache misses so the changes so up for users.
    """
    if config.POKEMON_IMGS.exists():
        files = [f for f in config.POKEMON_IMGS.glob("**/*") if f.is_file()]
        i = 0
        for i, filepath in enumerate(files):
            if not dry_run:
                filepath.unlink()
        if files and dry_run:
            print(f"dry-run: would have deleted {i+1} Pokémon character samples")
        elif files:
            print(f"deleted {i+1} Pokémon character samples")
        else:
            print("No Pokémon character samples to delete")

        if not dry_run:
            dirs = [f for f in config.POKEMON_IMGS.glob("**/*") if f.is_dir()]
            for d in dirs:
                d.rmdir()

    if config.FINAL_IMGS.exists():
        files = [f for f in config.FINAL_IMGS.glob("**/*") if f.is_file()]
        i = 0
        for i, filepath in enumerate(files):
            if not dry_run:
                filepath.unlink()

        if files and dry_run:
            print(f"dry-run: would have deleted {i+1} Pokémon card images")
        elif files:
            print(f"deleted {i+1} Pokémon card images")
        else:
            print("No Pokémon character card images to delete")

        if not dry_run:
            dirs = [f for f in config.FINAL_IMGS.glob("**/*") if f.is_dir()]
            for d in dirs:
                d.rmdir()


@stub.function
def extract_colors(num=3) -> None:
    """
    Extracts the colors for all Pokémon cards contained in `config` module
    and updates the card config with color data.

    Copy-paste this function's output back into the `config` module.
    """
    import colorgram

    for card in config.POKEMON_CARDS:
        print(f"Processing {card['name']}")
        req = urllib.request.Request(
            card["images"]["large"],
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
            },
        )
        image_bytes = urllib.request.urlopen(req).read()
        colors = colorgram.extract(io.BytesIO(image_bytes), num)
        card["colors"] = [list(color.rgb) for color in colors]

    print(json.dumps(config.POKEMON_CARDS, indent=4))


@stub.function(
    image=rnn_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=15 * 60,
)
def generate_pokemon_names():
    """
    Use a text-generation ML model to create new Pokémon names
    and persist them in a volume for later use in the card creation
    process.
    """
    desired_generations = 100
    poke_names = fetch_pokemon_names()
    # Hyphenated Pokémon names, eg. Hakamo-o, don't play well with RNN model.
    training_names = [n for n in poke_names if "-" not in n]
    max_sequence_len = max([len(name) for name in training_names])
    model = train_rnn(
        training_names=training_names,
        max_sequence_len=max_sequence_len,
    )
    new_names = generate_names(
        model=model,
        training_names=training_names,
        num=desired_generations,
        max_sequence_len=max_sequence_len,
    )

    print(f"Storing {desired_generations} generated names. eg. '{new_names[0]}'")
    output_path = config.POKEMON_NAMES / "rnn.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(new_names))


def main() -> int:
    parser = argparse.ArgumentParser(prog="text-to-pokemon-utils")
    sub_parsers = parser.add_subparsers(dest="subcommand")
    sub_parsers.add_parser("extract-colors", help="Extract colors for all Pokémon base cards.")
    sub_parsers.add_parser("gen-pokemon-names", help="Generate new Pokémon names.")
    parser_reset_diskcache = sub_parsers.add_parser(
        "reset-diskcache", help="Delete all cached Pokémon card parts from volume."
    )
    parser_reset_diskcache.add_argument(
        "--nodry-run", action="store_true", default=False, help="Actually delete files from volume."
    )

    args = parser.parse_args()
    if args.subcommand == "gen-pokemon-names":
        with stub.run():
            generate_pokemon_names()
    elif args.subcommand == "extract-colors":
        with stub.run():
            extract_colors()
    elif args.subcommand == "reset-diskcache":
        with stub.run():
            reset_diskcache(dry_run=not args.nodry_run)
    elif args.subcommand is None:
        parser.print_help(sys.stderr)
    else:
        raise AssertionError(f"Unimplemented subcommand '{args.subcommand}' was invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
