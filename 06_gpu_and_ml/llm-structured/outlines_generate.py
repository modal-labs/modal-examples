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
# - Transformers models
# - Llama
# - Mamba

# In this guide, we will show how you can use Outlines to enforce a JSON schema on the output of Mistral-7B.

# ## Build image
#
#  First, you'll want to build an image and install the relevant Python dependencies:
# `outlines` and a Hugging Face inference stack.

from modal import App, Image, gpu

app = App(name="outlines-app")  # Note: prior to April 2024, "app" was called "stub"

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.34",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)

# ## Download the model
#
# Next, we download the Mistral-7B model from Hugging Face.
# We do this as part of the definition of our Modal image so that
# we don't need to download it every time our inference function is run.


def import_model():
    import outlines

    outlines.models.transformers("mistralai/Mistral-7B-v0.1")


outlines_image = outlines_image.run_function(import_model)


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


@app.function(image=outlines_image, gpu=gpu.A100(memory=80))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines

    model = outlines.models.transformers(
        "mistralai/Mistral-7B-v0.1", device="cuda"
    )

    generator = outlines.generate.json(model, schema)
    character = generator(
        f"Give me a character description. Describe {prompt}."
    )

    print(character)


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
    generate.remote(prompt)
