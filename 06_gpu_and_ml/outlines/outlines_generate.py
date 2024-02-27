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
#  First, you'll want to build an image, and install the relevant Python dependency, which is just `outlines`.

from modal import Image, Stub

stub = Stub(name="outlines-app")

outlines_image = Image.debian_slim().pip_install("outlines")

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
# We use the `@stub.function` decorator to tell Modal to run this function on the stub we defined above.
# Note that we import `outlines` from inside the Modal function. This is because the `outlines` package exists in the container, but not necessarily locally.

# We specify that we want to use the Mistral-7B model, and then ask for a character in the correct schema.


@stub.function(image=outlines_image, timeout=60 * 20)
def generate():
    import outlines

    model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

    generator = outlines.generate.json(model, schema)
    character = generator(
        "Give me a character description for Alice, a 53-year-old woman with a sword, shield, and a strength of 10."
    )

    print(character)


# ## Define the entrypoint

# Finally, we define the entrypoint that will call the function above, and we are done!


@stub.local_entrypoint()
def main():
    generate.remote()


# When you run this script with `modal run`, you should see something like this printed out:

#  `{'name': 'Alice', 'age': 53, 'armor': 'leather', 'weapon': 'sword', 'strength': 10}`
