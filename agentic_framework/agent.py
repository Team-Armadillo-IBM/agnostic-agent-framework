"""Agent orchestrator tying together planning, execution, and reflection."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import Config
from .executor import Executor
from .planner import Planner
from .reflect import Reflector


class Agent:
    """High-level façade for the agentic workflow."""

    def __init__(self, config_file: Any):
        self.config = Config(config_file)
        self.planner = Planner(self.config)
        self.executor = Executor(self.config)
        self.reflector = Reflector(self.config)

    def run(self, goal: Optional[str] = None) -> Dict[str, Any]:
        """Run the planner → executor → reflector loop.

        Parameters
        ----------
        goal:
            Optional textual goal that can be used by the planner. The default
            planner in this scaffold does not use it directly but passing it
            through demonstrates the API surface for custom planners.
        """

        plan = self.planner.generate_plan(goal=goal)
        results = self.executor.execute(plan)
        reflection = self.reflector.reflect_on_execution(plan, results)
        return {"plan": plan, "results": results, "reflection": reflection}
