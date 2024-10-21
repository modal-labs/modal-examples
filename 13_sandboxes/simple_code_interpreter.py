# ---
# cmd: ["python", "13_sandboxes/simple_code_interpreter.py"]
# tags: ["use-case-sandboxed-code-execution"]
# pytest: false
# ---

# # Building a stateful sandboxed code interpreter
#
# This example demonstrates how to build a stateful code interpreter using a Modal
# [Sandbox](/docs/guide/sandbox).

import inspect
import json
from typing import Any

import modal

# In this example, we'll create a Sandbox that listens for code to execute, and then
# executes the code in a Python interpreter. Because we're running in a sandboxed
# environment, we can "unsafely" use `exec()` to execute the code.
#
# This example uses a simple JSON protocol over the Sandbox's stdin/stdout to send
# code to execute and get the results. First, we define the driver program that'll
# run in the Sandbox.


def driver_program():
    import json
    import sys
    from io import StringIO

    # We'll use a global dictionary to store the state of the interpreter. For each
    # command, we'll update the global state, execute the code, and then return the
    # result.

    globals: dict[str, Any] = {}
    while True:
        command = json.loads(input())
        code = command.get("code", None)
        if code is None:
            print(json.dumps({"error": "No code to execute"}))
            continue

        # Redirect stdout and stderr to capture the output of the code.
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            exec(code, globals)
        except Exception as e:
            print(f"Execution Error: {e}", file=sys.stderr)

        # Return the stdout and stderr of the code execution.
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        print(json.dumps({"stdout": stdout, "stderr": stderr}), flush=True)


# We want to execute this code in a Sandbox, so we'll need to convert the driver
# function to a string and invoke it.

driver_program_text = inspect.getsource(driver_program)
driver_program_command = f"""{driver_program_text}\n\ndriver_program()"""

# Now that we have the driver program, we can write a function to take a running
# instance of the program and execute code in it. This function will print out stdout
# in white and stderr in red.


def run_code(p: modal.Sandbox, code: str):
    p.stdin.write(json.dumps({"code": code}))
    p.stdin.write("\n")
    p.stdin.drain()
    next_line = next(iter(p.stdout))
    result = json.loads(next_line)
    print(result["stdout"], end="")
    print("\033[91m" + result["stderr"] + "\033[0m", end="")


# We've got our driver program and our code runner. Now we can create a Sandbox
# and run the driver program in it.

app = modal.App.lookup("code-interpreter", create_if_missing=True)
sb = modal.Sandbox.create("python", "-c", driver_program_command, app=app)

# Now we can execute some code in the Sandbox!

run_code(sb, "print('hello, world!')")

# ```
# hello, world!
# ```

run_code(sb, "x = 10")
run_code(sb, "y = 5")
run_code(sb, "result = x + y")
run_code(sb, "print(f'The result is: {result}')")

# ```
# The result is: 15
# ```
#
# We can also cause stderr output by causing an error.

run_code(sb, "print('Attempting to divide by zero...')")
run_code(sb, "1 / 0")

# ```
# Attempting to divide by zero...
# <span style="color:red">Execution Error: division by zero</span>
# ```
#
# Finally, we'll terminate the Sandbox.

sb.terminate()
