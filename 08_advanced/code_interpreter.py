# ---
# cmd: ["python", "08_advanced/code_interpreter.py"]
# ---
#
# This script demonstrates a simple code interpeter for LLM apps on top of
# Modal's [Sandbox](https://modal.com/docs/guide/sandbox) functionality.
#
# Modal's Sandboxes are a secure, trustless compute environment with access to all of Modal's features:
# custom container images, remote and local filesystem storage, cloud bucket mounts, GPUs, and more.
import uuid

import modal

# This unique string is exchanged between the Sandbox and the code interpreter wrapper
# to indicate that some code ran but returned no output.
NULL_MARKER = str(uuid.uuid4())

# Our code interpreter will use a Debian base with Python 3.11 and install the `IPython`
# package for its cleaner REPL interface.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "IPython==8.26.0"
)

# This is the Python program which runs inside the modal.Sandbox container process and receives
# code to execute from the code interpreter client.
#
# Its implementation is kept rudimentary for brevity. A production driver program should
# have improved error handling at the least.
DRIVER_PROGRAM = f"""import sys
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()

print(
    "IPython kernel is running. Enter Python code to execute (Ctrl+D to exit):",
    file=sys.stderr,
)

while True:
    try:
        code = input(">>> ") # Read input from stdin
        result = shell.run_cell(code)
        if result.result is not None:
            print(result.result, flush=True)
        else:
            print("{NULL_MARKER}", flush=True)
    except EOFError:
        break
    except Exception as e:
        # Print any exceptions that occur
        print(f"Error: {{e}}")
"""

# `ExecutionResult` is currently a very simple wrapper around the Sandbox's
# text output, but we could extend it further by returning execution metadata
# from the Sandbox driver program and attaching it to this execution result
# object.


class ExecutionResult:
    def __init__(self, text: str):
        self.text = text


# Our code interpreter could take many forms (it could even not run Python!)
# but for this example we implement a Jupyter Notebook-like interface.


class Notebook:
    def __init__(self, sb=None):
        self.environment = {}
        self.sb = sb

    def exec_cell(self, code: str) -> ExecutionResult:
        if not self.sb:
            return self._exec_cell_local(code)
        else:
            return self._exec_cell_remote(code)

    def _exec_cell_remote(self, code: str) -> ExecutionResult:
        self.sb.stdin.write(code.encode("utf-8"))
        self.sb.stdin.write(b"\n")
        self.sb.stdin.drain()
        message = next(iter(self.sb.stdout))
        if message.strip() == NULL_MARKER:
            return ExecutionResult(None)
        return ExecutionResult(message)

    def _exec_cell_local(self, code: str) -> ExecutionResult:
        try:
            exec(code, self.environment)
            # Retrieve the last expression's value
            result = eval(code.split(";")[-1], self.environment)
            return ExecutionResult(str(result))
        except Exception as e:
            return ExecutionResult(f"Error: {str(e)}")


# The `CodeInterpreter` is a context manager class which manages
# the lifecycle of the underlying `modal.Sandbox`.


class CodeInterpreter:
    def __init__(self, timeout: int = 600, debug: bool = False):
        self.sb = modal.Sandbox.create(
            "python",
            "-c",
            DRIVER_PROGRAM,
            timeout=timeout,
            image=image,
            app=modal.App.lookup(
                "example-code-interpreter", create_if_missing=True
            ),
        )
        self.notebook = Notebook(self.sb)
        self.debug = debug

    def __enter__(self) -> "CodeInterpreter":
        if self.debug:
            print("Entering Code interpreter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            print("Exiting Code interpreter")
        self.sb.stdin.write_eof()
        self.sb.stdin.drain()
        self.sb.wait()
        if self.sb.returncode != 0:
            print(f"Sandbox execution failed with code {self.sb.returncode}")
            print(self.sb.stderr.read())
        if self.debug:
            print(self.sb.stdout.read())
        return False  # don't suppress any Exception

    def exec(self, *cmds: str) -> str:
        process = self.sb.exec(*cmds)
        process.wait()
        if process.returncode != 0:
            c = process.returncode
            raise RuntimeError(
                f"command exec failed exit {c}: {process.stderr.read()}"
            )
        return process.stdout.read()


# Finally, we demonstrate the basics of the code interpreter's functionality.
# We can modify the interpreter's state by setting variables and then mutating
# those variables. We can modify the sandbox's filesystem and then inspect that
# filesystem modification by exec'ing arbitrary Linux shell commands in the sandbox
# container.


def main():
    with CodeInterpreter() as sandbox:
        sandbox.notebook.exec_cell("x = 1")
        execution = sandbox.notebook.exec_cell("x+=10; x")
        print(execution.text)  # outputs 11

        # Demo filesystem mutation and querying
        execution = sandbox.notebook.exec_cell(
            "import pathlib; pathlib.Path('/tmp/foo').write_text('hello!')"
        )
        stdout = sandbox.exec("cat", "/tmp/foo")  # outputs 'hello!'
        print(stdout)


if __name__ == "__main__":
    main()
