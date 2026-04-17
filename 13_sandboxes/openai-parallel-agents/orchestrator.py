import asyncio
from typing import Any

from typing import Literal

from agents import AgentHooks, ModelSettings, StopAtTools, function_tool
from agents.extensions.sandbox.modal import ModalSandboxClient, ModalSandboxClientOptions
from agents.memory import SQLiteSession
from agents.run import Agent, Runner, RunConfig
from openai.types.shared import Reasoning, ReasoningEffort

from display import LiveDisplay
from subagent_pool import SubAgentPool

from pathlib import Path

ORCHESTRATOR_INSTRUCTIONS = """
You are a relentless execution agent. Your only job is to get the user's task fully done - not \
partially done, not blocked on questions, fully done. Do not stop. Do not ask for clarification \
unless you have exhausted every reasonable interpretation and attempt.

## Default assumptions
- The user wants working code. Write it, run it, fix it until it works.
- The user expects sandbox environments to be set up. Installing dependencies, cloning repos, \
downloading data - this is all expected work, not a blocker. Just do it.
- If something is unclear, make a reasonable assumption, proceed, and note the assumption in your \
finish() summary.
- Creativity and persistence are the job. If one approach fails, try another. Adapt.

## Subagent instructions
Write your subagent objectives and task messages with the same mindset: assume the work is expected, \
assume setup is needed, and instruct them to persist. Never let a subagent stop just because something \
wasn't pre-installed or pre-configured. Subagents should be told to figure it out. \
Turn on high reasoning if subagents are particularly stuck.

## Status updates
Before every tool call, call set_status with a one-sentence description of what you are about to do, \
unless status is unchanged. One sentence, no filler.

## Skills
Start each session by calling list_skills and loading any that are relevant.
Skills contain detailed operational guidance - follow them closely.
"""
class OrchestratorHooks(AgentHooks):  # type: ignore[type-arg]
    """Custom lifecycle hooks used to track orchestrator state and reset reasoning effort."""
    def bind(self, orchestrator: "OrchestratorAgent") -> None:
        self._orchestrator = orchestrator

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: Any, input_items: Any) -> None:
        self._orchestrator.turns += 1
        self._orchestrator.is_thinking = True

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self._orchestrator.is_thinking = False
        if self._orchestrator._reasoning_effort is not None:
            self._orchestrator._reasoning_effort = None

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self._orchestrator.current_tool = tool.name

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self._orchestrator.current_tool = None


class OrchestratorAgent:
    """Orchestrator agent that manages a pool of sandbox agents to complete user tasks."""
    def __init__(
        self,
        model: str,
        sandbox_client: ModalSandboxClient,
        sandbox_options: ModalSandboxClientOptions,
        trace_id: str,
    ) -> None:
        self._sandbox_client = sandbox_client
        self._trace_id = trace_id

        # Prepare an pool of sandbox subagents
        self._pool = SubAgentPool(
            client=sandbox_client,
            options=sandbox_options,
            model=model,
        )

        self._reasoning_effort: ReasoningEffort | None = None
        self.turns: int = 0
        self.current_tool: str | None = None
        self.status: str | None = None
        self.is_thinking: bool = False

        # Registry mapping label → (image_id, description) for images captured via snapshot_image.
        self._image_registry: dict[str, tuple[str, str]] = {}
        self.loaded_skills: list[str] = []

        orchestrator_hooks = OrchestratorHooks()
        orchestrator_hooks.bind(self)

        self._agent = Agent(
            name="Orchestrator Agent",
            model=model,
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model_settings=ModelSettings(tool_choice="auto"),
            tool_use_behavior=StopAtTools(stop_at_tool_names=["finish"]),
            hooks=orchestrator_hooks,
            tools=[
                function_tool(self.set_status),
                function_tool(self.set_reasoning_effort),
                function_tool(self.list_skills),
                function_tool(self.load_skill),
                function_tool(self.spawn_subagent),
                function_tool(self.list_subagents),
                function_tool(self.gpu_capacity),
                function_tool(self.snapshot_image),
                function_tool(self.list_images),
                function_tool(self.invoke_subagent),
                function_tool(self.wait_for_subagent_result),
                function_tool(self.wait_for_first_subagent_result),
                function_tool(self.delete_subagent),
                function_tool(self.delete_all),
                function_tool(self.finish),
            ],
        )
        self._session = SQLiteSession(session_id="orchestrator")

    async def run(self, prompt: str) -> str:
        """Run a single turn with the orchestrator."""
        display = LiveDisplay(self, self._pool)
        display.start()
        result = await Runner.run(
            self._agent,
            prompt,
            max_turns=200,
            session=self._session,
            run_config=RunConfig(
                trace_id=self._trace_id,
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort=self._reasoning_effort),
                ),
            ),
        )
        display.stop()
        return result.final_output or ""

    ## -----
    ## Tools

    def set_status(self, status: str) -> str:
        """Set a one-sentence status summary shown in the live display.
        Call this before every tool invocation unless the status is unchanged."""
        self.status = status
        return "Status updated."

    def set_reasoning_effort(self, effort: ReasoningEffort) -> str:
        """Set the reasoning effort for the subsequent LLM call. Resets to None (disabled) automatically."""
        self._reasoning_effort = effort
        return f"Reasoning effort set to {effort}. Proceed to plan and execute the next steps."

    _SKILLS_DIR = Path(__file__).parent / "skills"

    def list_skills(self) -> str:
        """List all skills available to the orchestrator."""
        return "\n".join(f"- {skill.stem}" for skill in self._SKILLS_DIR.glob("*.md"))

    def load_skill(self, skill_name: str) -> str:
        """Load a skill's instructions into context."""
        skill_path = self._SKILLS_DIR / f"{skill_name}.md"
        if not skill_path.exists():
            return f"Skill '{skill_name}' not found."
        if skill_name not in self.loaded_skills:
            self.loaded_skills.append(skill_name)
        return skill_path.read_text()

    def list_subagents(self) -> str:
        """Check active subagents and their status. Returns IDs, names, task status, and turn counts."""
        entries = self._pool.list_entries()
        if not entries:
            return "No active subagents."
        return "\n".join(e.text_summary() for e in entries)

    def gpu_capacity(self) -> str:
        """Check available GPU capacity across all types in the pool.
        Returns limit, running count, and available slots per GPU type."""
        capacity = self._pool.gpu_capacity()
        lines = []
        for gpu_type, info in capacity.items():
            lines.append(f"{gpu_type}: {info['running']}/{info['limit']} running")
        return "\n".join(lines)

    async def spawn_subagent(
        self,
        agent_name: str,
        objective: str = "You are a general purpose coding agent with access to a shell.",
        gpu: Literal["H100", "H100:8"] | None = None,
        image_id: str | None = None,
    ) -> str:
        """Spawn a new sandbox subagent with the given name and objective.
        Returns the subagent's unique ID, which must be used for subsequent invoke_subagent calls.

        The objective becomes the subagent's instructions (system prompt) - its persistent role
        throughout its lifetime. Write it as an identity statement describing who the agent is
        and what it owns, not as a task. Tasks are sent separately via invoke_subagent.

        Arguments:
            agent_name: A short human-readable label for this subagent.
            objective: The subagent's persistent role and scope (becomes its instructions/system prompt).
                       Example: "You are a benchmarking agent responsible for measuring GPU throughput."
            gpu: Optional GPU type to attach to this subagent's sandbox.
            image_id: Optional Modal image ID to boot from (e.g. from snapshot_image or list_images).
                      The sandbox filesystem will be restored from that snapshot.
        """
        try:
            entry = await self._pool.create(agent_name, objective, gpu=gpu, image_id=image_id)
        except ValueError as exc:
            gpu_type = str(exc)
            return (
                f"No more {gpu_type}s can be spawned, limit hit. "
                f"Consider a different gpu type. "
                f"If {gpu_type} is needed, consider deleting idle subagents."
            )
        return f"Subagent '{agent_name}' spawned with id={entry.id}"

    async def snapshot_image(self, agent_id: str, label: str, description: str) -> str:
        """Freeze a subagent's sandbox filesystem as a Modal image and store it in the image registry.

        The image can later be used to boot a new subagent from the same filesystem state by
        passing the returned image ID (or the label) to spawn_subagent.

        Arguments:
            agent_id: ID of the subagent whose filesystem to snapshot.
            label: A short human-readable label to identify this image in the registry.
            description: A two-sentence description of what this image contains and what it was built for.
        """
        entry = self._pool.get(agent_id)
        if entry is None:
            return f"Subagent '{agent_id}' not found."
        try:
            image_id = await entry.modal_session.snapshot_filesystem()
        except Exception as exc:
            return f"Snapshot failed: {exc}"
        self._image_registry[label] = (image_id, description)
        return f"Snapshot complete. label={label!r} image_id={image_id}"

    def list_images(self) -> str:
        """List all images captured via snapshot_image, with their labels, Modal image IDs, and descriptions."""
        if not self._image_registry:
            return "No images captured yet."
        return "\n".join(
            f"{label}: {image_id}\n  {description}"
            for label, (image_id, description) in self._image_registry.items()
        )

    async def delete_subagent(self, agent_id: str) -> str:
        """Terminate a subagent and free its resources when done."""
        entry = self._pool.get(agent_id)
        if entry is None:
            return f"Subagent '{agent_id}' not found."
        await self._pool.delete(agent_id)
        return f"Subagent '{agent_id}' terminated."

    async def delete_all(self) -> None:
        """Terminate all sandbox sessions."""
        await self._pool.delete_all()

    async def invoke_subagent(self, agent_id: str, message: str) -> str:
        """Start a task on a subagent and return immediately (non-blocking).
        Use wait_for_subagent_result or wait_for_first_subagent_result to get the result."""
        entry = self._pool.get(agent_id)
        if entry is None:
            return f"Subagent '{agent_id}' not found. Use spawn_subagent first."
        if entry.run_task is not None and not entry.run_task.done():
            return f"Subagent '{agent_id}' is still running. Use wait_for_subagent_result tool to await its result."

        await entry.run(message, trace_id=self._trace_id)
        return f"Subagent '{agent_id}' task started."

    async def wait_for_subagent_result(self, agent_id: str) -> str:
        """Block until a specific subagent finishes and return its result."""
        entry = self._pool.get(agent_id)
        if entry is None:
            return f"Subagent '{agent_id}' not found."
        if entry.run_task is None:
            return f"Subagent '{agent_id}' has no active task."

        result = await entry.run_task
        entry.run_task = None # mark as done
        return result.final_output or ""

    async def wait_for_first_subagent_result(self) -> str:
        """Block until any running subagent finishes and return its ID and result."""
        running = {e.run_task: e for e in self._pool._entries.values() if e.run_task and not e.run_task.done()}
        if not running:
            return "No running subagents."
        done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
        entry = running[next(iter(done))]
        result = await self.wait_for_subagent_result(entry.id)
        return f"Subagent '{entry.id}' ({entry.name}) completed: {result}"


    def finish(self, summary: str) -> str:
        """Signal completion with a concise summary of what was accomplished,
        including key outputs, file paths, or results produced.
        Call this only after collecting all results you need from subagents.

        Arguments:
            summary: A concise summary of what was accomplished.
        """
        return summary

