import unittest
from pathlib import Path

from agentic_framework.config import Config
from agentic_framework.planner import Planner


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
        self.config = Config(config_path)
        self.planner = Planner(self.config)

    def test_generate_plan_from_config(self):
        plan = self.planner.generate_plan()
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[1]["tool"], "sql")
        self.assertIn("args", plan[1])

    def test_generate_plan_goal_passthrough(self):
        # The default planner ignores the goal but should not error.
        plan = self.planner.generate_plan(goal="investigate logs")
        self.assertEqual(len(plan), 3)


if __name__ == "__main__":
    unittest.main()
