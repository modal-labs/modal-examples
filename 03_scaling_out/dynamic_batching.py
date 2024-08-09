# # Dynamic batching for ASCII and character conversion
#
# This example demonstrates how dynamic batching can be used in a simple
# application for converting ASCII codes to characters and vice versa.
#
# For more details about using dynamic batching and tuning its configuration, see
# the [Dynamic Batching](/docs/guide/dynamic-batching) guide.
#
# ## Setup
#
# First, let's define the image for the application.

import modal

app = modal.App(
    "example-dynamic-batching-ascii-conversion",
    image=modal.Image.debian_slim()
)

# ## The batched function
#
# Now, let's define a Function that converts ASCII codes to characters. This
# async batched Function allows us to convert at most four ASCII codes at once.
# If there are fewer than four ASCII codes in the batch, the function will wait
# for one second to allow more inputs to arrive before returning the result.
#
# In the function signature, the input `asciis` is a list of integers, and the
# output is a list of strings. This is because the `asciis` input is batched,
# and the output should be a list of the same length as the input.
#
# When the function is invoked, however, the input will be a single integer,
# and the return value to the invocation will be a single string.

@app.function()
@modal.batched(max_batch_size=4, wait_ms=1000)
async def asciis_to_chars(asciis: list[int]) -> list[str]:
    return [chr(ascii) for ascii in asciis]


# ## The Class with a batched method
#
# Next, let's define a Class that converts characters to ASCII codes. This
# Class has an async batched method `chars_to_asiics` that converts characters
# to ASCII codes and has the same configuration as the batched Function above.
#
# Note that if a Class has a batched method, the Class cannot implement other
# batched methods or `@modal.method`s.


@app.cls()
class AsciiConverter:
    @modal.batched(max_batch_size=4, wait_ms=1000)
    async def chars_to_asciis(self, chars: list[str]) -> list[int]:
        asciis = [ord(char) for char in chars]
        return asciis


# ## ASCII and character conversion
#
# Finally, let's define the `local_entrypoint` that uses the batched Function
# and the Class with a batched method to convert ASCII codes to characters and
# vice versa.
#
# We use [`map.aio`](/docs/reference/modal.Function#map) to asynchronously map
# over the ASCII codes and characters. This allows us to invoke the batched
# Function and the batched method over a range of ASCII codes and characters
# in parallel.
#
# Run this script to see what ASCII codes from 33 to 38 correspond to in characters!


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
