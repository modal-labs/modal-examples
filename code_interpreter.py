import uuid

import modal

NULL_MARKER = str(uuid.uuid4())

image = modal.Image.debian_slim().pip_install("IPython")

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


class ExecutionResult:
    def __init__(self, text: str):
        self.text = text


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
        for message in self.sb.stdout:
            if message.strip() == NULL_MARKER:
                return ExecutionResult(None)
            return ExecutionResult(message)

    def _exec_cell_local(self, code: str) -> ExecutionResult:
        try:
            # Execute the code in the environment
            exec(code, self.environment)
            # Retrieve the last expression's value
            result = eval(code.split(";")[-1], self.environment)
            return ExecutionResult(str(result))
        except Exception as e:
            return ExecutionResult(f"Error: {str(e)}")


class CodeInterpreter:
    def __init__(self, timeout: int = 600, debug: bool = False):
        self.sb = modal.Sandbox.create(
            "python",
            "-c",
            DRIVER_PROGRAM,
            timeout=timeout,
            image=image,
        )
        self.notebook = Notebook(self.sb)
        self.debug = debug

    def __enter__(self) -> "CodeInterpreter":
        if self.debug:
            print("Entering Code interpreter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
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
            raise RuntimeError(f"command exec failed exit {c}: {process.stderr.read()}")
        return process.stdout.read()


def main():
    with CodeInterpreter() as sandbox:
        sandbox.notebook.exec_cell("x = 1")
        execution = sandbox.notebook.exec_cell("x+=1; x")
        print(execution.text)  # outputs 2

        # Demo filesystem mutation and querying
        execution = sandbox.notebook.exec_cell("import pathlib; pathlib.Path('/tmp/foo').write_text('hello!')")
        stdout = sandbox.exec("ls", "/tmp/")  # outputs 'foo\n'
        print(stdout)


if __name__ == "__main__":
    main()

