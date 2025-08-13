# # Enforcing JSON outputs on LLMs

# [Outlines](https://github.com/outlines-dev/outlines) is a tool that lets you control the generation of language models to make their output more predictable.

# This includes things like:

# - Reducing the completion to a choice between multiple possibilities
# - Type constraints
# - Efficient regex-structured generation
# - Efficient JSON generation following a Pydantic model
# - Efficient JSON generation following a JSON schema

# Outlines is considered an alternative to tools like [JSONFormer](https://github.com/1rgs/jsonformer), and can be used on top of a variety of LLMs, including:

# - OpenAI models
# - LLaMA
# - Mamba

# In this guide, we will show how you can use Outlines to enforce a JSON schema on the output of Mistral-7B.

# ## Build image

#  First, you'll want to build an image and install the relevant Python dependencies:
# `outlines` and a Hugging Face inference stack.

import modal

app = modal.App(name="example-outlines-generate")

outlines_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines==1.2.3",
    "transformers==4.41.2",
    "sentencepiece==0.2.0",
    "datasets==2.18.0",
    "accelerate==0.27.2",
    "numpy<2",
    "pydantic==2.11.7",
)

# ## Download the model

# Next, we download the Mistral 7B model from Hugging Face.
# We do this as part of the definition of our Modal Image so that
# we don't need to download it every time our inference function is run.

MODEL_NAME = "mistral-community/Mistral-7B-v0.2"


def import_model(model_name):
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )


outlines_image = outlines_image.run_function(
    import_model, kwargs={"model_name": MODEL_NAME}
)


# ## Define the function

# Next, we define the generation function.
# We use the `@app.function` decorator to tell Modal to run this function on the app we defined above.
# Note that we import `outlines` from inside the Modal function. This is because the `outlines` package exists in the container, but not necessarily locally.

# We specify that we want to use the Mistral-7B model, and then ask for a character, and we'll receive structured data with the right schema.

# We also define the schema that we want to enforce on the output of Mistral-7B. This schema is for a character description, and includes a name, age, armor, weapon, and strength.


@app.function(image=outlines_image, gpu="A100-40GB")
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    from enum import Enum

    import outlines
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"

    class Weapon(str, Enum):
        sword = "sword"
        axe = "axe"
        mace = "mace"
        spear = "spear"
        bow = "bow"
        crossbow = "crossbow"

    class Character(BaseModel):
        name: str = Field(..., max_length=10, title="Name")
        age: int = Field(..., title="Age")
        armor: Armor
        weapon: Weapon
        strength: int = Field(..., title="Strength")

    character = model(
        f"Give me a character description. Describe {prompt}.",
        Character,
        max_new_tokens=256,
    )

    return character


# ## Define the entrypoint

# Finally, we define the entrypoint that will connect our local computer
# to the functions above, that run on Modal, and we are done!
#
# When you run this script with `modal run`, you should see something like this printed out:
#
#  `{'name': 'Amiri', 'age': 53, 'armor': 'leather', 'weapon': 'sword', 'strength': 10}`


@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    print(generate.remote(prompt))
