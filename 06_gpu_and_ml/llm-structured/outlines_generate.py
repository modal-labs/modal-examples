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

app = modal.App(name="outlines-app")

outlines_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.44",
    "transformers==4.41.2",
    "sentencepiece==0.2.0",
    "datasets==2.18.0",
    "accelerate==0.27.2",
    "numpy<2",
)

# ## Download the model

# Next, we download the Mistral 7B model from Hugging Face.
# We do this as part of the definition of our Modal Image so that
# we don't need to download it every time our inference function is run.

MODEL_NAME = "mistral-community/Mistral-7B-v0.2"


def import_model(model_name):
    import outlines

    outlines.models.transformers(model_name)


outlines_image = outlines_image.run_function(
    import_model, kwargs={"model_name": MODEL_NAME}
)


# ## Define the schema

# Next, we define the schema that we want to enforce on the output of Mistral-7B. This schema is for a character description, and includes a name, age, armor, weapon, and strength.

schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""

# ## Define the function

# Next, we define the generation function.
# We use the `@app.function` decorator to tell Modal to run this function on the app we defined above.
# Note that we import `outlines` from inside the Modal function. This is because the `outlines` package exists in the container, but not necessarily locally.

# We specify that we want to use the Mistral-7B model, and then ask for a character, and we'll receive structured data with the right schema.


@app.function(image=outlines_image, gpu=modal.gpu.A100(size="40GB"))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines

    model = outlines.models.transformers(MODEL_NAME, device="cuda")

    generator = outlines.generate.json(model, schema)
    character = generator(
        f"Give me a character description. Describe {prompt}."
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
