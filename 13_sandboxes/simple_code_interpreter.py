# ---
# cmd: ["python", "13_sandboxes/simple_code_interpreter.py"]
# pytest: false
# ---

# # Build a stateful, sandboxed code interpreter

# This example demonstrates how to build a stateful code interpreter using a Modal
# [Sandbox](https://modal.com/docs/guide/sandbox).

# We'll create a Modal Sandbox that listens for code to execute and then
# executes the code in a Python interpreter. Because we're running in a sandboxed
# environment, we can safely use the "unsafe" `exec()` to execute the code.

# ## Setting up a code interpreter in a Modal Sandbox

# Our code interpreter uses a Python "driver program" to listen for code
# sent in JSON format to its standard input (`stdin`), execute the code,
# and then return the results in JSON format on standard output (`stdout`).

import inspect
import json
from typing import Any

import modal


def driver_program():
    import json
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    # When you `exec` code in Python, you can pass in a dictionary
    # that defines the global variables the code has access to.

    # We'll use that to store state.

    globals: dict[str, Any] = {}
    while True:
        command = json.loads(input())  # read a line of JSON from stdin
        if (code := command.get("code")) is None:
            print(json.dumps({"error": "No code to execute"}))
            continue

        # Capture the executed code's outputs
        stdout_io, stderr_io = StringIO(), StringIO()
        with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
            try:
                exec(code, globals)
            except Exception as e:
                print(f"Execution Error: {e}", file=sys.stderr)

        print(
            json.dumps(
                {
                    "stdout": stdout_io.getvalue(),
                    "stderr": stderr_io.getvalue(),
                }
            ),
            flush=True,
        )


# Now that we have the driver program, we can write a function to take a
# `ContainerProcess` that is running the driver program and execute code in it.


def run_code(p: modal.container_process.ContainerProcess, code: str):
    p.stdin.write(json.dumps({"code": code}))
    p.stdin.write("\n")
    p.stdin.drain()
    next_line = next(iter(p.stdout))
    result = json.loads(next_line)
    print(result["stdout"], end="")
    print("\033[91m" + result["stderr"] + "\033[0m", end="")


# We've got our driver program and our code runner. Now we can create a Sandbox
# and run the driver program in it.

# We have to convert the driver program to a string to pass it to the Sandbox.
# Here we use `inspect.getsource` to get the source code as a string,
# but you could also keep the driver program in a separate file and read it in.

driver_program_text = inspect.getsource(driver_program)
driver_program_command = f"""{driver_program_text}\n\ndriver_program()"""

app = modal.App.lookup("code-interpreter", create_if_missing=True)
sb = modal.Sandbox.create(app=app)
p = sb.exec("python", "-c", driver_program_command)

# ## Running code in a Modal Sandbox

# Now we can execute some code in the Sandbox!

run_code(p, "print('hello, world!')")  # hello, world!

# The Sandbox and our code interpreter are stateful,
# so we can define variables and use them in subsequent code.

run_code(p, "x = 10")
run_code(p, "y = 5")
run_code(p, "result = x + y")
run_code(p, "print(f'The result is: {result}')")  # The result is: 15

# We can also see errors when code fails.

run_code(p, "print('Attempting to divide by zero...')")
run_code(p, "1 / 0")  # Execution Error: division by zero

# Finally, let's clean up after ourselves and terminate the Sandbox.

sb.terminate()
