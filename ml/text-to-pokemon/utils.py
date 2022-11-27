import io
import json
import urllib.request

from . import config
from .main import stub


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
