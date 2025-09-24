"""Reflection utilities to analyse execution traces."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .config import Config


class Reflector:
    """Derive lightweight insights from executed steps."""

    def __init__(self, config: Config):
        self._config = config

    def reflect_on_execution(
        self, plan: Iterable[Dict[str, Any]], results: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarise execution metadata for downstream evaluation."""

        plan_steps = list(plan)
        executed_steps = list(results)
        total_duration = sum(step.get("duration", 0.0) for step in executed_steps)
        tools_used = [step.get("tool") for step in executed_steps]
        summary = {
            "total_steps": len(executed_steps),
            "tools_used": tools_used,
            "total_duration": total_duration,
            "successful": len(plan_steps) == len(executed_steps),
        }
        if tools_used:
            summary["summary"] = (
                f"Executed {len(tools_used)} steps using {', '.join(tools_used)} "
                f"in {total_duration:.4f} seconds."
            )
        else:
            summary["summary"] = "No steps executed."
        return summary
