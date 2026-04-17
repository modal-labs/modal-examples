from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from subagent_pool import SubAgentPool

if TYPE_CHECKING:
    from orchestrator import OrchestratorAgent

_SPINNER_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]


class LiveDisplay:
    """Live updating display showing orchestrator state and the subagent pool."""

    def __init__(self, orchestrator: OrchestratorAgent, pool: SubAgentPool) -> None:
        self._orchestrator = orchestrator
        self._pool = pool
        self._frame = 0
        self._live = Live(auto_refresh=False, transient=False)
        self._task: asyncio.Task[None] | None = None

    def _render_orchestrator(self) -> Panel:
        orch = self._orchestrator
        reasoning = orch._reasoning_effort or "-"
        tool = orch.current_tool or "-"
        turns = orch.turns

        spinner = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)] if orch.is_thinking else " "

        stats = Text()
        stats.append(f"{spinner} ", style="yellow" if orch.is_thinking else "dim")
        stats.append("Reasoning: ", style="dim")
        stats.append(str(reasoning))
        stats.append("    Turns: ", style="dim")
        stats.append(str(turns))
        stats.append("    Tool: ", style="dim")
        stats.append(tool, style="yellow" if orch.current_tool else "")

        lines: list[Text] = [stats]

        if orch.loaded_skills:
            skills_line = Text("\n")
            skills_line.append("Skills: ", style="dim")
            skills_line.append(", ".join(orch.loaded_skills), style="cyan")
            lines.append(skills_line)

        if orch.status:
            status_line = Text("\n")
            status_line.append("Status: ", style="dim")
            status_line.append(orch.status, style="italic")
            lines.append(status_line)

        content: Text | object = Text.assemble(*lines)

        return Panel(content, title="Orchestrator", border_style="dim")

    def _render_subagents(self) -> Panel:
        entries = self._pool.list_entries()
        table = Table(box=None, show_header=True, padding=(0, 2), header_style="dim", expand=True)
        table.add_column("", width=2)
        table.add_column("Agent")
        table.add_column("State", width=9)
        table.add_column("GPU", width=6)
        table.add_column("Turns", width=6)
        table.add_column("Status", ratio=1, no_wrap=True, overflow="ellipsis")

        state_styles: dict[str, tuple[str, str]] = {
            "running": ("yellow", _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]),
            "done":    ("green",  "✓"),
            "error":   ("red",    "✗"),
            "idle":    ("dim",    "·"),
        }
        for agent in entries:
            style, icon = state_styles.get(agent.state, ("", "?"))
            display_status = agent.status or agent.last_command or ""
            table.add_row(
                Text(icon, style=style),
                Text(agent.name),
                Text(f"[{agent.state}]", style=style),
                Text(agent.gpu or "", style="dim") if agent.gpu else Text(""),
                Text(str(agent.turns), style="dim") if agent.turns else Text(""),
                Text(display_status, style="italic" if agent.status else "dim italic"),
            )

        count = len(entries)
        title = f"Subagents - {count} active" if count else "Subagents - idle"
        return Panel(table, title=title, border_style="dim")

    def _render(self) -> Group:
        return Group(self._render_orchestrator(), self._render_subagents())

    async def _loop(self) -> None:
        while True:
            self._live.update(self._render(), refresh=True)
            self._frame += 1
            await asyncio.sleep(0.25)

    def start(self) -> None:
        print()
        self._live.start()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._live.stop()
        print()
