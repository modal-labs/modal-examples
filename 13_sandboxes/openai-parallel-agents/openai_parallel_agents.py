# ---
# cmd: ["uv", "run", "--directory", "13_sandboxes/openai-parallel-agents", "openai_parallel_agents.py", "--prompt", "Implement and train a model on the MNIST dataset using three separate approaches: pytorch, tensorflow, and jax"]
# pytest: false
# mypy: ignore-errors
# # Building Massively Parallel Agents with OpenAI Agent SDK and Modal

# The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) is a powerful new tool
# for building agent harnesses for coding, deep research, and more. It plugs neatly into Modal via a
# sandbox extension, giving agents a home computer to work on, with the right tools to truly take
# advantage of the scale Modal is known for.

# In this example, we'll show you how to build a custom agent harness, from scratch, on top of the OpenAI Agents SDK.
# This harness will be built with parallelism in mind, with the ability to spawn and manage multiple coding subagents in parallel.
# 
# We'll integrate Modal [Sandboxes](https://modal.com/docs/guide/sandbox) to give those agents computers
# (even GPUs) to run in, and use it to massively parallelize
# [Parameter Golf](https://openai.com/index/parameter-golf/) experiments across multiple GPUs simultaneously.

# ## Memory with Sessions

# By default, `Agents` are stateless. `Sessions` are our solution: objects you can pass across agent
# runs to accumulate the context window, across multiple user prompts and even across agent instances:
#
# ```python
# session = SQLiteSession(session_id="my-session")
#
# while True:
#     user_input = input("> ")
#     result = await Runner.run(agent, user_input, session=session)
#     print(result.final_output)
# ```
#
# With memory, we now have a new challenge: how to intelligently manage context windows to avoid large bills, slow responses, and [context rot](https://www.trychroma.com/research/context-rot).

# ## Introducing subagents for efficient long-horizon work

# Coding agents are incredibly token-heavy as they explore codebases and ingest stdout/stderr. The
# effective lifetime of one of these agents is short. To allow for long-horizon work, we split into two
# agents: an [orchestrator](https://developers.openai.com/api/docs/guides/agents/orchestration) and a
# subagent.
#
# The orchestrator is our main chat agent, accumulating memory for the entire task. It has a tool,
# `invoke_subagent`, which is itself an agent with a completely fresh context window and `Session`.
# The task description goes in, a summary of work done comes out, keeping orchestrator context tight.
#
# ```python
# class OrchestratorAgent:
#     def __init__(self, model, sandbox_client, sandbox_options):
#         self._pool = SubAgentPool(client=sandbox_client, options=sandbox_options, model=model)
#
#         self._agent = Agent(
#             name="Orchestrator",
#             instructions="Delegate coding tasks to subagents. spawn → invoke → collect → finish.",
#             tool_use_behavior=StopAtTools(stop_at_tool_names=["finish"]),
#             tools=[function_tool(self.spawn_subagent), function_tool(self.invoke_subagent), ...],
#         )
#         self._session = SQLiteSession(session_id="orchestrator")
#
#     async def run(self, prompt: str) -> str:
#         result = await Runner.run(self._agent, prompt, session=self._session)
#         return result.final_output
# ```

# Rather than blocking the orchestrator, `invoke_subagent` stores an `asyncio.Task` so the orchestrator
# can manage multiple parallel subagents and wait for them selectively:
#
# ```python
# async def invoke_subagent(self, agent_id: str, prompt: str) -> str:
#     """Start a task on a subagent and return immediately (non-blocking)."""
#     entry = self._pool.get(agent_id)
#
#     # .task is now a Future handle to the async result
#     entry.task = asyncio.create_task(Runner.run(entry.agent, prompt, session=entry.conv_session, ...))
#     return f"Subagent '{agent_id}' task started."
#
# async def wait_for_subagent_result(self, agent_id: str) -> str:
#     """Block until a specific subagent finishes and return its result."""
#     entry = self._pool.get(agent_id)
#     result = await entry.task
#     return result.final_output
#
# async def wait_for_first_subagent_result(self) -> str:
#     """Block until any running subagent finishes and return its ID and result."""
#     running = {e.task: e for e in self._pool.entries.values() if e.task and not e.task.done()}
#     done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
#     entry = running[next(iter(done))]
#     result = await entry.task
#     return f"Subagent '{entry.id}' ({entry.name}) completed: {result.final_output}"
# ```

# ## Agent oversight via Hooks

# Since subagents run in the background, the orchestrator needs another way to see what they're doing.
# We use SDK lifecycle `Hooks` to track turn counts and the current shell command, and expose a
# `set_status` tool so subagents can report progress without exiting back to the orchestrator.
#
# ```python
# class SubAgentHooks(AgentHooks):
#     def bind(self, agent: SubAgent) -> None:
#         self._agent = agent
#
#     def on_exec_command(self, command: str) -> None:
#         # Called by our Capability whenever a shell command is dispatched
#         self._agent.last_command = command
#
#     async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
#         # Built-in SDK lifecycle hook, called at the start of each LLM call
#         self._agent.turns += 1
#
# hooks = SubAgentHooks()
#
# def set_status(status: str) -> str:
#     """Set a one-sentence status shown in the live display."""
#     self.status = status
#     return "Status updated."
#
# self.agent = SandboxAgent(
#     ...
#     tools=[function_tool(set_status)],
#     hooks=hooks,
# )
# hooks.bind(self)
# ```

# These fields are surfaced to the orchestrator via a `list_subagents` tool, giving it a live status
# view without pulling subagents out of their tasks.

# ## GPU quotas

# With async tasks, we're handing LLMs the potentially expensive ability to spin up unbounded GPU
# subagents. `SubAgentPool` enforces hard limits:
#
# ```python
# GPU_LIMITS: dict[str, int] = {
#     "H100": 8,
#     "H100:8": 2,
# }
#
# async def create(self, agent_name, objective, gpu=None, image_id=None) -> SubAgent:
#     if gpu is not None:
#         limit = GPU_LIMITS.get(gpu)
#         if limit is not None:
#             in_use = sum(1 for e in self._entries.values() if e.gpu == gpu)
#             if in_use >= limit:
#                 raise ValueError(gpu)  # caller converts to a friendly tool response
#     # ...
#
# async def spawn_subagent(self, agent_name, objective, gpu=None, image_id=None) -> str:
#     try:
#         entry = await self._pool.create(agent_name, objective, gpu=gpu, image_id=image_id)
#     except ValueError as exc:
#         gpu_type = str(exc)
#         return f"No more {gpu_type}s can be spawned, limit hit."
#     return f"Subagent '{agent_name}' spawned with id={entry.id}"
# ```

# ## Filesystem Snapshots

# With subagents all starting from base sandboxes, they each waste precious GPU time on the same setup
# work: pulling repos and installing dependencies. Filesystem Snapshots freeze a live sandbox into an
# image ID that future subagents can boot from, branching from known checkpoints.
#
# ```python
# # registry mapping label -> Modal image ID
# self._image_registry: dict[str, str] = {}
#
# async def snapshot_image(self, agent_id: str, label: str) -> str:
#     """Freeze a subagent's sandbox filesystem as a Modal image."""
#     entry = self._pool.get(agent_id)
#     image_id = await entry.modal_session.snapshot_filesystem()
#     self._image_registry[label] = image_id
#     return f"Snapshot complete. label={label!r} image_id={image_id}"
#
# def list_images(self) -> str:
#     """List all captured snapshots."""
#     return "\n".join(f"{label}: {image_id}" for label, image_id in self._image_registry.items())
#
# # spawn_subagent already accepts image_id; the pool boots the sandbox from that snapshot
# async def spawn_subagent(self, agent_name, objective, gpu=None, image_id=None) -> str:
#     entry = await self._pool.create(agent_name, objective, gpu=gpu, image_id=image_id)
#     return f"Subagent '{agent_name}' spawned with id={entry.id}"
#
# # In SubAgentPool.create(), image_id scopes the ModalSandboxClient to that image:
# if image_id is not None:
#     client = ModalSandboxClient(image=ModalImageSelector.from_id(image_id))
# ```

# In addition to the time savings, the added benefit of filesystem snapshots is context management.
# Filesystems can be used to offload memory!

# Just like we keep context hot in `Session` state, the filesystem of a live sandbox also acts as an
# implicit form of memory when viewed via shell tools. This on-disk memory can be implemented
# explicitly, via writing skills/memory files, or implicitly by the nature of the codebase already
# having been put in a working state.

# Now, the orchestrator can progress threads of work, snapshot that filesystem, and then drop fresh
# subagents into that snapshot with basic follow-up instructions. The subagent will quickly get its
# bearings and resume work, without the context bloat of everything that led to this point.

# ## Skills subsystem

# Our final step for running Parameter Golf efficiently is prompting. Rather than baking task-specific
# context into the system prompt, we give the orchestrator tools to opt into it on demand via a
# `skills/` directory of markdown files.
#
# ```python
# _SKILLS_DIR = Path("skills/")
#
# def list_skills(self) -> str:
#     """List all skills available to the orchestrator."""
#     return "\n".join(f"- {skill.stem}" for skill in self._SKILLS_DIR.glob("*.md"))
#
# def load_skill(self, skill_name: str) -> str:
#     """Load a skill's instructions into context."""
#     return (self._SKILLS_DIR / f"{skill_name}.md").read_text()
# ```

# ## Running the harness

# Our harness is now complete. You can kick off a Parameter Golf parallel experiment with:
#
# ```
# uv run openai_parallel_agents.py --prompt "Prepare a Parameter Golf experiment environment, generate a plan for five parallel experiments to run, and run those experiments. Report back your findings, including final training loss and score."
# ```

# The complete implementation lives in the supporting modules in this directory:
# `orchestrator.py`, `subagent.py`, `subagent_pool.py`, and `display.py`.
#
# You'll need a Modal account and an `OPENAI_API_KEY` set in your environment.

import asyncio
import os
import uuid

from agents.extensions.sandbox.modal import ModalSandboxClient, ModalSandboxClientOptions  # type: ignore[import]

from orchestrator import OrchestratorAgent


async def main(prompt: str | None, interactive: bool = False) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment before running.")
    trace_id = "trace_" + str(uuid.uuid4())
    model = "gpt-5.1"

    modal_sandbox_client = ModalSandboxClient()
    modal_sandbox_client_options = ModalSandboxClientOptions(
        app_name="example-parallel-agents",
        workspace_persistence="snapshot_filesystem",
    )

    orchestrator = OrchestratorAgent(
        model=model,
        sandbox_client=modal_sandbox_client,
        sandbox_options=modal_sandbox_client_options,
        trace_id=trace_id,
    )

    magenta = "\033[35m"
    reset = "\033[0m"
    print(
        f"\nView trace on OpenAI:\n{magenta}https://platform.openai.com/logs/trace?trace_id={trace_id}{reset}\n"
    )

    try:
        if prompt is not None:
            output = await orchestrator.run(prompt)
            print(output)
            return

        if interactive:
            while True:
                user_input = input("> ")
                if user_input.lower() == "exit":
                    return
                if not user_input.strip():
                    continue
                output = await orchestrator.run(user_input)
                print(output)

        print("No prompt provided. Use --prompt \"<prompt>\" or --interactive to run.")
    finally:
        await orchestrator.delete_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", default=False)
    args = parser.parse_args()
    asyncio.run(main(args.prompt, args.interactive))
