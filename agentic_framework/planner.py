"""Simple planner implementation for the scaffold."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config


class Planner:
    """Planner that materialises a LangGraph-style plan from configuration."""

    def __init__(self, config: Config):
        self._config = config

    def generate_plan(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a list of tool invocations.

        If the configuration provides an explicit plan the planner returns a
        deep copy of those steps. Otherwise it falls back to a simple heuristic
        that invokes each configured tool once with empty arguments.
        """

        configured_plan = self._config.plan()
        if configured_plan:
            return [dict(step) for step in configured_plan]

        return [{"tool": tool_name, "args": {}} for tool_name in self._config.agent.tools]
