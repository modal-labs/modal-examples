# # Dynamic batching for ASCII and character conversion

# This example demonstrates how to dynamically batch a simple
# application that converts ASCII codes to characters and vice versa.

# For more details about using dynamic batching and optimizing
# the batching configurations for your application, see
# the [dynamic batching guide](https://modal.com/docs/guide/dynamic-batching).

# ## Setup

# Let's start by defining the image for the application.

import modal

app = modal.App(
    "example-dynamic-batching",
    image=modal.Image.debian_slim(python_version="3.11"),
)


# ## Defining a Batched Function

# Now, let's define a function that converts ASCII codes to characters. This
# async Batched Function allows us to convert up to four ASCII codes at once.


@app.function()
@modal.batched(max_batch_size=4, wait_ms=1000)
async def asciis_to_chars(asciis: list[int]) -> list[str]:
    return [chr(ascii) for ascii in asciis]


# If there are fewer than four ASCII codes in the batch, the Function will wait
# for one second, as specified by `wait_ms`, to allow more inputs to arrive before
# returning the result.

# The input `asciis` to the Function is a list of integers, and the
# output is a list of strings. To allow batching, the input list `asciis`
# and the output list must have the same length.

# You must invoke the Function with an individual ASCII input, and a single
# character will be returned in response.

# ## Defining a class with a Batched Method

# Next, let's define a class that converts characters to ASCII codes. This
# class has an async Batched Method `chars_to_asciis` that converts characters
# to ASCII codes.

# Note that if a class has a Batched Method, it cannot have other Batched Methods
# or Methods.


@app.cls()
class AsciiConverter:
    @modal.batched(max_batch_size=4, wait_ms=1000)
    async def chars_to_asciis(self, chars: list[str]) -> list[int]:
        asciis = [ord(char) for char in chars]
        return asciis


# ## ASCII and character conversion

# Finally, let's define the `local_entrypoint` that uses the Batched Function
# and Class Method to convert ASCII codes to characters and
# vice versa.

# We use [`map.aio`](https://modal.com/docs/reference/modal.Function#map) to asynchronously map
# over the ASCII codes and characters. This allows us to invoke the Batched
# Function and the Batched Method over a range of ASCII codes and characters
# in parallel.
#
# Run this script to see which characters correspond to ASCII codes 33 through 38!


@app.local_entrypoint()
async def main():
    ascii_converter = AsciiConverter()
    chars = []
    async for char in asciis_to_chars.map.aio(range(33, 39)):
        chars.append(char)

    print("Characters:", chars)

    asciis = []
    async for ascii in ascii_converter.chars_to_asciis.map.aio(chars):
        asciis.append(ascii)

    print("ASCII codes:", asciis)
