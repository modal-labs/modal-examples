import asyncio
from typing import Any

from agents import AgentHooks, ModelSettings, function_tool


SUBAGENT_DEVELOPER_INSTRUCTIONS = """\
You are a relentless executor. Your job is to complete the task fully - not to report that it's hard, \
not to ask for permission, not to stop because something wasn't pre-installed. Get it done.

## Default assumptions
- Writing and running code is expected. Do it.
- Installing dependencies, cloning repos, downloading data - this is normal setup work, not a blocker. Just do it.
- If something is unclear, make a reasonable assumption and proceed. Don't ask.
- If an approach fails, try another. Be creative. Persist.

## Status updates
Before every tool call, call set_status with a one-sentence description of what you are about to do, \
unless it is unchanged from the previous call.

## Execution
- Use the shell tool to write, run, and iterate on code.
- Prefer selectively viewing files (with `head` or `tail`) vs `cat` to save time and bandwidth.
- Prefer redirecting output to files or piping between commands, rather than viewing the output.
- Split complex commands into multiple tool calls, and use intermediate files to pass data between steps.
- Apply reasonable timeouts to shell commands to avoid hanging.
- Use the apply_patch tool to create or edit files. Paths must be relative to the workspace root \
- If the same error recurs three times with no variation in approach, stop and report - don't loop blindly.

## Reporting
Your output goes directly into the orchestrator's context. Keep it tight:
- Report results only: file paths, metrics, outputs, errors.
- No process narration, no "I started by...".
- If genuinely blocked after exhausting options, state the blocker in one sentence.\
"""
from agents.extensions.sandbox.modal import ModalSandboxClient, ModalSandboxSession
from agents.memory import SQLiteSession
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox import SandboxAgent
from agents.sandbox.capabilities import Capability
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.sandbox.apply_patch import WorkspaceEditor
from agents.tool import (
    ApplyPatchTool,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
    Tool,
)
from agents.run import RunConfig, Runner
from agents.sandbox import SandboxRunConfig


class SubAgentCapability(Capability):
    """Custom capability (tools) used to execute shell commands in the sandbox. Bound to a sandbox session instance."""

    def __init__(self, on_exec_command: Any = None) -> None:
        super().__init__(type="workspace_shell")
        self._session: ModalSandboxSession | None = None
        self._on_exec_command = on_exec_command

    def bind(self, session: ModalSandboxSession) -> None:
        # Bind is auto-called by the Agent SDK when the capability is added to the agent
        self._session = session

    def tools(self) -> list[Tool]:
        return [  # type: ignore
            ShellTool(executor=self._execute_shell),
            ApplyPatchTool(editor=self),
        ]

    # Shell tool executor, to implement the ShellTool
    async def _execute_shell(self, request: ShellCommandRequest) -> ShellResult:
        if self._session is None:
            raise RuntimeError("Workspace shell is not bound to a sandbox session.")

        timeout_s = (
            request.data.action.timeout_ms / 1000
            if request.data.action.timeout_ms is not None
            else None
        )
        outputs: list[ShellCommandOutput] = []
        for command in request.data.action.commands:
            if self._on_exec_command is not None:
                self._on_exec_command(command)

            result = await self._session.exec(command, timeout=timeout_s, shell=True)
            outputs.append(
                ShellCommandOutput(
                    command=command,
                    stdout=result.stdout.decode("utf-8", errors="replace"),
                    stderr=result.stderr.decode("utf-8", errors="replace"),
                    outcome=ShellCallOutcome(type="exit", exit_code=result.exit_code),
                )
            )
        return ShellResult(output=outputs)

    # ApplyPatchEditor methods, to implement the ApplyPatchTool
    async def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        if self._session is None:
            raise RuntimeError("Apply patch editor is not bound to a sandbox session.")
        return await WorkspaceEditor(self._session).apply_operation(operation)  # type: ignore[arg-type]

    async def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        if self._session is None:
            raise RuntimeError("Apply patch editor is not bound to a sandbox session.")
        return await WorkspaceEditor(self._session).apply_operation(operation)  # type: ignore[arg-type]

    async def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        if self._session is None:
            raise RuntimeError("Apply patch editor is not bound to a sandbox session.")
        return await WorkspaceEditor(self._session).apply_operation(operation)  # type: ignore[arg-type]



class SubAgentHooks(AgentHooks):  # type: ignore[type-arg]
    """Custom lifecycle hooks used to track exec progress in the SubAgent struct"""
    def __init__(self) -> None:
        self._agent: "SubAgent | None" = None

    def bind(self, agent: "SubAgent") -> None:
        self._agent = agent

    def on_exec_command(self, command: str) -> None:
        if self._agent is not None:
            self._agent.last_command = " ".join(command.splitlines())

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: Any, input_items: Any) -> None:
        # Standard lifecycle hook override used by the Agent SDK
        if self._agent is not None:
            self._agent.turns += 1


class SubAgent:
    def __init__(self, id: str, name: str, model: str, objective: str, sandbox_client: ModalSandboxClient, sandbox_session: SandboxSession, gpu: str | None = None) -> None:
        self.id = id # unique identifier for the subagent
        self.name = name # human-readable label for the subagent
        self.gpu = gpu # GPU type allocated to this subagent (e.g. "A10", "A100", "H100:8", or None)

        self.sandbox_client = sandbox_client
        self.sandbox_session = sandbox_session # handle to the live sandbox session (instrumented wrapper)
        self.conv_session = SQLiteSession(session_id=f"{id}_conv") # session to track conversation history

        # Handle to an async run task
        self.run_task: asyncio.Task[Any] | None = None

        # Metrics tracking, updated by hooks and used by display / LLM summaries
        self.turns = 0
        self.last_command: str | None = None
        self.status: str | None = None

        hooks = SubAgentHooks()
        capability = SubAgentCapability(on_exec_command=hooks.on_exec_command)

        def set_status(status: str) -> str:
            """Set a one-sentence status summary shown in the live display.
            Call this before every tool invocation unless the status is unchanged."""
            self.status = status
            return "Status updated."

        self.agent = SandboxAgent(
            name=name,
            model=model,
            instructions=objective,
            developer_instructions=SUBAGENT_DEVELOPER_INSTRUCTIONS,
            capabilities=[capability],
            tools=[function_tool(set_status)],
            model_settings=ModelSettings(tool_choice="required"),
            hooks=hooks,
        )

        hooks.bind(self)

    @property
    def state(self) -> str:
        if self.run_task is None:
            return "idle"
        if not self.run_task.done():
            return "running"
        if self.run_task.exception() is not None:
            return "error"
        return "done"

    def text_summary(self) -> str:
        """Summary for LLM tool output and human inspection."""
        lines = [
            f"ID: {self.id}",
            f"Name: {self.name}",
            f"State: {self.state}",
            f"GPU: {self.gpu or 'none'}",
            f"Turns: {self.turns}",
        ]
        if self.status:
            lines.append(f"Status: {self.status}")
        if self.last_command:
            lines.append(f"Last command: {self.last_command}")
        return "  ".join(lines)

    @property
    def modal_session(self) -> ModalSandboxSession:
        """Unwrap the instrumented SandboxSession to access the underlying ModalSandboxSession."""
        inner = self.sandbox_session._inner
        assert isinstance(inner, ModalSandboxSession)
        return inner

    async def run(self, prompt: str, trace_id: str) -> None:
        """Create a background task to run the subagent."""
        self.run_task = asyncio.create_task(Runner.run(
            self.agent,
            prompt,
            max_turns=200,
            session=self.conv_session,
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=self.sandbox_client,
                    session=self.sandbox_session,
                ),
                trace_id=trace_id,
            ),
        ))