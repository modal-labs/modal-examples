from textwrap import dedent
import json
import pickle
import base64
from typing import Any
import inspect
import modal


def driver_program():
    import json
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    globals: dict[str, Any] = {}
    while True:
        command = json.loads(input())  # read a line of JSON from stdin
        code = command.get("code")
        if code is None:
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


def run_code(p: modal.container_process.ContainerProcess, code: str):
    p.stdin.write(json.dumps({"code": code}))
    p.stdin.write("\n")
    p.stdin.drain()
    next_line = next(iter(p.stdout))
    result = json.loads(next_line)
    return result["stdout"], result["stderr"]


def get_executor_cls():
    from smolagents.remote_executors import RemotePythonExecutor
    from smolagents.monitoring import LogLevel
    from smolagents.utils import AgentError
    from smolagents.tools import get_tools_definition_code

    class ModalExecutor(RemotePythonExecutor):
        def __init__(self, additional_imports: list[str], logger, app, **kwargs):
            super().__init__(additional_imports, logger)

            sandbox_image = (
                modal.Image.debian_slim().pip_install(*additional_imports).add_local_python_source("sql_engine")
            )
            self.additional_imports = additional_imports

            self.sandbox = modal.Sandbox.create(app=app, image=sandbox_image)

            self.logger.log("ModalExecutor is running", level=LogLevel.INFO)

            driver_program_text = inspect.getsource(driver_program)
            driver_program_command = f"""{driver_program_text}\n\ndriver_program()"""
            self.p = self.sandbox.exec("python", "-c", driver_program_command)

        def send_variables(self, variables: dict):
            if not variables:
                return
            return super().send_variables(variables)

        def send_tools(self, tools):
            # Install tool packages
            print("tools", tools)
            packages_to_install = {
                pkg
                for tool in tools.values()
                for pkg in tool.to_dict()["requirements"]
                if pkg not in self.installed_packages + ["smolagents"]
            }
            if packages_to_install:
                self.installed_packages += self.install_packages(list(packages_to_install))
            # Get tool definitions
            code = get_tools_definition_code(tools)
            if code:
                execution = self.run_code_raise_errors(code)
                self.logger.log(execution[1])

        def run_code_raise_errors(self, code_action, return_final_answer=False):
            PICKLE_PREFIX = "RESULT_PICKLE:"
            wrapped_code = code_action
            if return_final_answer:
                match = self.final_answer_pattern.search(code_action)
                if match:
                    pre_final_answer_code = self.final_answer_pattern.sub("", code_action)
                    result_expr = match.group(1)
                    wrapped_code = pre_final_answer_code + dedent(f"""
                        import pickle, base64
                        _result = {result_expr}
                        print("{PICKLE_PREFIX}" + base64.b64encode(pickle.dumps(_result)).decode())
                        """)

            stdout, stderr = run_code(self.p, wrapped_code)
            result = None

            if stderr:
                logs = f"Executing code yielded an error: {stderr}"
                raise AgentError(logs, self.logger)

            if return_final_answer:
                if not stdout:
                    raise AgentError("No result returned by executor!", self.logger)

                if stdout.startswith("RESULT_PICKLE:"):
                    pickle_data = stdout[len(PICKLE_PREFIX) :].strip()
                    result = pickle.loads(base64.b64decode(pickle_data))

            return result, stdout

        def install_packages(self, additional_imports):
            if additional_imports:
                out = self.sandbox.exec("python", "-m", "pip", "install", *additional_imports)
                self.logger.log(out.stdout.read())
            return additional_imports

        def cleanup(self):
            self.sandbox.terminate()

    return ModalExecutor
