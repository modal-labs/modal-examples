# ---
# lambda-test: false  # auxiliary-file
# ---
import uuid

from agents.extensions.sandbox.modal import (
    ModalImageSelector,
    ModalSandboxClient,
    ModalSandboxClientOptions,
)
from subagent import SubAgent

# Maximum number of concurrently active subagents per GPU type.
GPU_LIMITS: dict[str, int] = {
    # "A10": 10,
    # "A100": 5,
    "H100": 8,
    "H100:8": 2,
}

SANDBOX_TIMEOUT_S: int = 1 * 60 * 60  # 1 hour


class SubAgentPool:
    """Pool of sandbox agents that can be spawned, invoked, and managed by an orchestrator."""

    def __init__(
        self,
        client: ModalSandboxClient,
        options: ModalSandboxClientOptions,
        model: str,
    ) -> None:
        self._client = client
        self._options = options
        self._model = model
        self._entries: dict[str, SubAgent] = {}

    async def create(
        self,
        agent_name: str,
        objective: str,
        gpu: str | None = None,
        image_id: str | None = None,
    ) -> SubAgent:
        """Create a new SubAgent with its own sandbox session.

        Raises ValueError if the GPU limit for the requested type is already reached.

        Args:
            agent_name: Human-readable label for the subagent.
            objective: Persistent role/instructions for the subagent.
            gpu: Optional GPU type (e.g. "A10", "A100", "H100:8").
            image_id: Optional Modal image ID to boot from. When provided the sandbox
                filesystem is restored from that snapshot instead of starting fresh.
        """
        if gpu is not None:
            limit = GPU_LIMITS.get(gpu)
            if limit is None:
                raise ValueError(f"Invalid GPU type: {gpu}")
            if limit is not None:
                in_use = sum(1 for e in self._entries.values() if e.gpu == gpu)
                if in_use >= limit:
                    raise ValueError(gpu)

        per_sandbox_options = ModalSandboxClientOptions(
            app_name=self._options.app_name,
            sandbox_create_timeout_s=self._options.sandbox_create_timeout_s,
            workspace_persistence=self._options.workspace_persistence,
            snapshot_filesystem_timeout_s=self._options.snapshot_filesystem_timeout_s,
            snapshot_filesystem_restore_timeout_s=self._options.snapshot_filesystem_restore_timeout_s,
            exposed_ports=self._options.exposed_ports,
            gpu=gpu,
            timeout=SANDBOX_TIMEOUT_S,
        )

        # If an image_id is given, create a short-lived client scoped to that image so the
        # sandbox boots from the snapshotted filesystem.
        # TODO: I don't like this, it feels sloppy.
        if image_id is not None:
            client = ModalSandboxClient(image=ModalImageSelector.from_id(image_id))
        else:
            client = self._client

        agent_id = "sub_" + str(uuid.uuid4()).replace("-", "")
        sandbox_session = await client.create(options=per_sandbox_options)
        subagent = SubAgent(
            agent_id,
            agent_name,
            self._model,
            objective,
            self._client,
            sandbox_session,
            gpu=gpu,
        )
        self._entries[agent_id] = subagent
        return subagent

    def get(self, agent_id: str) -> SubAgent | None:
        """Get a subagent by its ID."""
        return self._entries.get(agent_id)

    async def delete(self, agent_id: str) -> None:
        """Delete a subagent by its ID. Deletes the underlying sandbox session and conversation session."""
        subagent = self._entries.pop(agent_id, None)
        if subagent is not None:
            if subagent.run_task is not None and not subagent.run_task.done():
                subagent.run_task.cancel()
            await self._client.delete(subagent.sandbox_session)
            subagent.conv_session.close()

    async def delete_all(self) -> None:
        """Delete all subagents. Deletes the underlying sandbox sessions and conversation sessions."""
        for subagent in self._entries.values():
            if subagent.run_task is not None and not subagent.run_task.done():
                subagent.run_task.cancel()
            await self._client.delete(subagent.sandbox_session)
            subagent.conv_session.close()
        self._entries.clear()

    def list_entries(self) -> list[SubAgent]:
        return list(self._entries.values())

    def gpu_capacity(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for gpu_type, limit in GPU_LIMITS.items():
            running = sum(1 for e in self._entries.values() if e.gpu == gpu_type)
            result[gpu_type] = {"limit": limit, "running": running}
        return result
