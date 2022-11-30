import io
import json
import urllib.request
import shutil

from . import config
from .main import stub, volume


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
            # Set a user agent to avoid 403 response from some podcast audio servers.
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
            },
        )
        image_bytes = urllib.request.urlopen(req).read()
        colors = colorgram.extract(io.BytesIO(image_bytes), num)
        card["colors"] = [list(color.rgb) for color in colors]

    print(json.dumps(config.POKEMON_CARDS, indent=4))


if __name__ == "__main__":
    with stub.run():
        reset_diskcache(dry_run=False)
