import unittest
from pathlib import Path

from agentic_framework.config import Config
from agentic_framework.executor import Executor
from agentic_framework.planner import Planner
from agentic_framework.reflect import Reflector


class ReflectorTests(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
        self.config = Config(config_path)
        self.planner = Planner(self.config)
        self.executor = Executor(self.config)
        self.reflector = Reflector(self.config)

    def test_reflection_summary(self):
        plan = self.planner.generate_plan()
        results = self.executor.execute(plan)
        summary = self.reflector.reflect_on_execution(plan, results)
        self.assertEqual(summary["total_steps"], len(plan))
        self.assertTrue(summary["successful"])
        self.assertIn("Executed", summary["summary"])


if __name__ == "__main__":
    unittest.main()
