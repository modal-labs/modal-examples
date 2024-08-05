"""
Our AI-generated Pokémon characters need their own names!
"""

import dataclasses
import json
import time
import urllib.request
from typing import Any

import modal

from . import config

rnn_image = modal.Image.debian_slim().pip_install(
    "keras",
    "pandas",
    "numpy<2",
    "tensorflow",
)

# Longer names don't fit on Pokémon card
MAX_NAME_LEN = 14
# Discard names too short to make sense
MIN_NAME_LEN = 4

rnn_names_output_path = config.POKEMON_NAMES / "rnn.txt"


@dataclasses.dataclass
class TrainingDataset:
    X: Any  # numpy arr
    Y: Any  # numpy arr
    num_unique_chars: int


def load_names(
    include_model_generated: bool,
    include_human_generated: bool,
) -> set[str]:
    names = set()
    if include_model_generated:
        if rnn_names_output_path.exists():
            model_names = set(rnn_names_output_path.read_text().split("\n"))
            names.update(model_names)
        else:
            print(
                f"Model generated names at `{rnn_names_output_path}` are not ready, skipping"
            )
    if include_human_generated:
        names.update(FANDOM_NAMES)
        names.update(PREFILL_PROMPT_NAMES)
    return names


def prompt_2_name(prompt: str, candidates: set[str]) -> str:
    if not prompt:
        raise ValueError("`prompt` argument cannot be empty")
    return max(
        (cand for cand in candidates),
        key=lambda c: len(lcs(prompt, c)),
    )


def lcs(one: str, two: str) -> str:
    matrix = [["" for x in range(len(two))] for x in range(len(one))]
    for i in range(len(one)):
        for j in range(len(two)):
            if one[i] == two[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = one[i]
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + one[i]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)

    longest = matrix[-1][-1]
    return longest


def generate_names(
    model,
    training_names: set[str],
    num: int,
    max_sequence_len: int,
):
    """Accepts training dataset and trained model, and generates `num` new Pokémon names."""
    import numpy as np

    concat_names = "\n".join(training_names).lower()
    # Start sequence generation from end of the input sequence
    sequence = concat_names[-(max_sequence_len - 1) :] + "\n"

    new_names: set[str] = set()
    chars = sorted(list(set(concat_names)))
    num_chars = len(chars)

    # Build translation dictionaries
    char2idx = {c: i for i, c in enumerate(chars)}  # a -> 0
    idx2char = {i: c for i, c in enumerate(chars)}  # 0 -> a

    while len(new_names) < num:
        # Vectorize sequence for prediction
        x = np.zeros((1, max_sequence_len, num_chars))
        for i, char in enumerate(sequence):
            x[0, i, char2idx[char]] = 1

        # Sample next char from predicted probabilities
        probs = model.predict(x, verbose=0)[0]
        probs /= probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx2char[next_idx]
        sequence = sequence[1:] + next_char

        # Newline means we have a new name
        if next_char == "\n":
            gen_name = [name for name in sequence.split("\n")][1]

            # Never start name with two identical chars
            if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
                gen_name = gen_name[1:]

            if len(gen_name) > MAX_NAME_LEN:
                continue
            elif len(gen_name) >= MIN_NAME_LEN:
                # Only allow new and unique names
                if gen_name not in training_names and gen_name not in new_names:
                    new_names.add(gen_name)

            if len(new_names) % 10 == 0:
                print("generated {} new names".format(len(new_names)))
    return list(new_names)


def prep_dataset(
    training_names: list[str], max_sequence_len: int
) -> TrainingDataset:
    import numpy as np

    step_length = (
        1  # The step length we take to get our samples from our corpus
    )
    # Make it all to a long string
    concat_names = "\n".join(training_names).lower()

    chars = sorted(list(set(concat_names)))
    num_chars = len(chars)

    # Build translation dictionary, 'a' -> 0
    char2idx = dict((c, i) for i, c in enumerate(chars))

    # Use longest name length as our sequence window
    max_sequence_len = max([len(name) for name in training_names])

    print(f"Total chars: {num_chars}")
    print("Corpus length:", len(concat_names))
    print("Number of names: ", len(training_names))
    print("Longest name: ", max_sequence_len)

    sequences = []
    next_chars = []

    # Loop over our data and extract pairs of sequances and next chars
    for i in range(0, len(concat_names) - max_sequence_len, step_length):
        sequences.append(concat_names[i : i + max_sequence_len])
        next_chars.append(concat_names[i + max_sequence_len])

    num_sequences = len(sequences)

    print("Number of sequences:", num_sequences)
    print("First 10 sequences and next chars:")
    for i in range(10):
        print(
            "X=[{}]   y=[{}]".replace("\n", " ")
            .format(sequences[i], next_chars[i])
            .replace("\n", " ")
        )

    X = np.zeros((num_sequences, max_sequence_len, num_chars), dtype=bool)
    Y = np.zeros((num_sequences, num_chars), dtype=bool)

    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            X[i, j, char2idx[char]] = 1
        Y[i, char2idx[next_chars[i]]] = 1

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return TrainingDataset(
        X=X,
        Y=Y,
        num_unique_chars=num_chars,
    )


def train_rnn(
    training_names: list[str],
    max_sequence_len: int,
):
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from keras.optimizers import RMSprop

    epochs = 100  # Number of times we train on our full data
    batch_size = 32  # Data samples in each training step
    latent_dim = 64  # Size of our LSTM
    dropout_rate = 0.2  # Regularization with dropout
    verbosity = 1  # Print result for each epoch

    dataset = prep_dataset(training_names, max_sequence_len)

    input_shape = (
        max_sequence_len,
        dataset.num_unique_chars,
    )
    model = Sequential()
    model.add(
        LSTM(
            latent_dim, input_shape=input_shape, recurrent_dropout=dropout_rate
        )
    )
    model.add(Dense(units=dataset.num_unique_chars, activation="softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    model.summary()

    start = time.time()
    print("Training for {} epochs".format(epochs))
    model.fit(
        dataset.X,
        dataset.Y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbosity,
    )
    print(f"Finished training - time elapsed: {(time.time() - start)} seconds")
    return model


def fetch_pokemon_names() -> list[str]:
    """
    Source training data by getting all Pokémon names from the pokeapi.co API.
    There are 1008 Pokémon as of early December 2022.
    """
    get_all_url = "https://pokeapi.co/api/v2/pokemon?limit=1500"  # Set limit > than total number of Pokémon.
    req = urllib.request.Request(
        get_all_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/35.0.1916.47 Safari/537.36"
            )
        },
    )
    response = urllib.request.urlopen(req)
    data = json.load(response)

    pokemon_names = [item["name"] for item in data["results"]]
    print(f"Fetched {len(pokemon_names)} Pokémon names")
    return pokemon_names


# Hand-writing good Pokémon names for the prefill prompts defined in the frontend.
PREFILL_PROMPT_NAMES: set[str] = {
    "abrahamad",  # Abraham Linclon
    "jordasaur",  # Air Jordans
    "rattlebub",  # A Happy Baby With A Rattle
    "bananapeel",  # Banana in Pajamas
    "cheeseclaw",  # Crab Made of Cheese
    "Trumpistan",  # Donald Trump
    "duckhoof",  # Duck sized horse
    "elephhix",  # Elephant With Six Legs
    "frodomon",  # Frodo Baggins
    "goldsealy",  # Golden Seal
    "homerimpson",  # Homer Simpson
    "hoofduck",  # Horse sized duck
    "iphoneuous",  # IPhone 7 Device
    "jokerclown",  # Joker Evil
    "kingkongmon",  # King Kong
    "popandafu",  # Kung Fu Panda
    "limamonk",  # Lima Monkey
    "marvin",  # Marvin The Paranoid Robot
    "nocturas",  # Nocturnal Animal
    "buddhismo",  # Old Buddhist Monk in Orange Robes
    "pdp-11",  # PDP-11 Computer
    "coupleous",  # Power Couple
    "questsight",  # Question Mark With Eyes
    "roomba",  # Roomba
    "ragesound",  # Rage Against The Machine
    "metalflight",  # Snake With Metal Wings
    "armorgator",  # Suit of Armor Alligator
    "stevejobs",  # Steve Jobs
    "devilmon",  # The Devil
    "fearmon",  # The Fear
    "uranus",  # Uranus The Planet
    "vladmarx",  # Vladimir Lenin
    "willycat",  # Willy Wonka Cat
    "xenomorphmon",  # Xenomorph Alien
    "yoyoma",  # Yoyo Toy
    "zoroblade",  # Zoro The Masked Bandit
}

FANDOM_NAMES: set[str] = {
    "azelfuel",
    "billiaze",
    "bronzera",
    "camyke",
    "cocodunt",
    "cocomut",
    "colirus",
    "cysting",
    "eleafant",
    "elephfern",
    "eleplant",
    "eloha",
    "elopun",
    "gladiatron",
    "golerno",
    "ivoany",
    "oliosa",
    "pachygerm",
    "palmtrunk",
    "pinealf",
    "rute",
    "scorbit",
    "scrash",
    "sproutrunk",
    "stampyro",
    "taphromet",
    "tephracorna",
    "troot",
    "tropiphant",
    "truncoco",
    "trute",
    "vectol",
    "virachnid",
    "virachnus",
}
