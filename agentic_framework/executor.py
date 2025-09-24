"""Executor responsible for running tools defined in the plan."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List

from .config import Config
from .tools import ToolRegistry

LOGGER = logging.getLogger(__name__)


class Executor:
    """Execute a plan by resolving tools from the registry."""

    def __init__(self, config: Config):
        self._config = config
        self._registry = ToolRegistry(config)

    def execute(self, plan: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute every step in the plan and return structured results."""

        results: List[Dict[str, Any]] = []
        for index, step in enumerate(plan):
            tool_name = step.get("tool")
            if not tool_name:
                raise ValueError(f"Step {index} is missing the 'tool' field: {step}")

            args = step.get("args", {})
            LOGGER.debug("Executing step %s with tool '%s'", index, tool_name)
            tool = self._registry.get(tool_name)
            started = time.perf_counter()
            output = tool.execute(**args)
            duration = time.perf_counter() - started
            LOGGER.debug("Tool '%s' finished in %.4fs", tool_name, duration)
            results.append(
                {
                    "tool": tool_name,
                    "args": dict(args),
                    "output": output,
                    "duration": duration,
                }
            )
        return results
