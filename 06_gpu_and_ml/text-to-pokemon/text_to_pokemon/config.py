import pathlib
import time

import modal

CACHE_DIR = "/cache"
MODEL_CACHE = pathlib.Path("/models")
# Where generated Pokémon images are stored, by hash of prompt.
POKEMON_IMGS = pathlib.Path(CACHE_DIR, "generated_samples")
# Where human-generated and ML-generated new Pokémon names are stored.
POKEMON_NAMES = pathlib.Path(CACHE_DIR, "pokemon_names")
# Where fully compose Pokémon card output images are stored, by hash of prompt.
FINAL_IMGS = pathlib.Path(CACHE_DIR, "final_cards")
# Location of web frontend assets.
ASSETS_PATH = pathlib.Path(__file__).parent / "frontend" / "dist"
# Card composite component images
CARD_PART_IMGS = pathlib.Path(CACHE_DIR, "card_parts")
# Sometimes the NSFW checker is confused by the Pokémon images.
# You can disable it at your own risk.
DISABLE_SAFETY = True


POKEMON_CARDS = [
    {
        "id": "swshp-SWSH039",
        "name": "Pikachu",
        "supertype": "Pokémon",
        "subtypes": ["Basic"],
        "number": "SWSH039",
        "rarity": "Rare Holo Galaxy",
        "images": {
            "small": "https://images.pokemontcg.io/swshp/SWSH039.png",
            "large": "https://images.pokemontcg.io/swshp/SWSH039_hires.png",
        },
        "colors": [[246, 207, 87], [242, 186, 14], [210, 180, 140]],
    },
    {
        "id": "sm35-1",
        "name": "Bulbasaur",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "1",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm35/1.png",
            "large": "https://images.pokemontcg.io/sm35/1_hires.png",
        },
        "colors": [[221, 221, 64], [164, 199, 63], [131, 184, 156]],
    },
    {
        "id": "sm10-33",
        "name": "Squirtle",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "33",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm10/33.png",
            "large": "https://images.pokemontcg.io/sm10/33_hires.png",
        },
        "colors": [[87, 186, 227], [253, 224, 100], [191, 225, 240]],
    },
    {
        "id": "sm115-7",
        "name": "Charmander",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "7",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/sm115/7.png",
            "large": "https://images.pokemontcg.io/sm115/7_hires.png",
        },
        "colors": [[235, 131, 68], [235, 88, 52], [252, 222, 98]],
    },
    {
        "id": "swsh45-35",
        "name": "Morpeko",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "35",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/swsh45/35.png",
            "large": "https://images.pokemontcg.io/swsh45/35_hires.png",
        },
        "colors": [[252, 220, 55], [202, 167, 99], [238, 236, 194]],
    },
    {
        "id": "swsh9-120",
        "name": "Bidoof",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "120",
        "rarity": "Common",
        "images": {
            "small": "https://images.pokemontcg.io/swsh9/120.png",
            "large": "https://images.pokemontcg.io/swsh9/120_hires.png",
        },
        "colors": [[236, 231, 223], [249, 224, 101], [190, 154, 113]],
    },
    {
        "id": "sm8-142",
        "name": "Dedenne",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "142",
        "rarity": "Uncommon",
        "images": {
            "small": "https://images.pokemontcg.io/sm8/142.png",
            "large": "https://images.pokemontcg.io/sm8/142_hires.png",
        },
        "colors": [[216, 77, 140], [226, 118, 169], [252, 223, 100]],
    },
    {
        "id": "pgo-24",
        "name": "Articuno",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "24",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/24.png",
            "large": "https://images.pokemontcg.io/pgo/24_hires.png",
        },
        "colors": [[90, 184, 225], [254, 225, 99], [186, 220, 237]],
    },
    {
        "id": "pgo-29",
        "name": "Zapdos",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "29",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/29.png",
            "large": "https://images.pokemontcg.io/pgo/29_hires.png",
        },
        "colors": [[253, 220, 56], [121, 173, 202], [224, 175, 69]],
    },
    {
        "id": "pgo-12",
        "name": "Moltres",
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"],
        "number": "12",
        "rarity": "Rare Holo",
        "images": {
            "small": "https://images.pokemontcg.io/pgo/12.png",
            "large": "https://images.pokemontcg.io/pgo/12_hires.png",
        },
        "colors": [[238, 131, 72], [236, 89, 59], [253, 222, 98]],
    },
]


def load_stable_diffusion_pokemon_model():
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "lambdalabs/sd-pokemon-diffusers"
    cache_dir = MODEL_CACHE / model_id
    if cache_dir.exists():
        print(f"Using diskcached model for '{model_id}'")
        local_files_only = True
        load_action = "loading"
    else:
        print(f"No diskcached model found for '{model_id}'")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_files_only = False
        load_action = "downloading"
    load_start_time = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    print(
        f"finished {load_action} model, took {time.time() - load_start_time:.3f}s."
    )

    if DISABLE_SAFETY:

        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety
    return pipe


volume = modal.Volume.from_name(
    "txt-to-pokemon-cache-vol", create_if_missing=True
)
image = (
    modal.Image.debian_slim()
    .pip_install(
        "accelerate",
        "colorgram.py",
        "diffusers~=0.11.1",
        "ftfy",
        "huggingface_hub~=0.21.1",
        "torch",
        "transformers",
        "scipy",
        "fastapi[standard]",
    )
    .run_function(load_stable_diffusion_pokemon_model)
    .add_local_dir(local_path=ASSETS_PATH, remote_path="/assets")
)
app = modal.App(name="example-text-to-pokemon", image=image)
