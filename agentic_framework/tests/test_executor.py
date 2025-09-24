import unittest
from pathlib import Path

from agentic_framework.config import Config
from agentic_framework.executor import Executor
from agentic_framework.planner import Planner


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
        self.config = Config(config_path)
        self.planner = Planner(self.config)
        self.executor = Executor(self.config)

    def test_execute_plan(self):
        plan = self.planner.generate_plan()
        results = self.executor.execute(plan)
        self.assertEqual(len(results), len(plan))
        http_result = results[0]
        self.assertEqual(http_result["output"]["body"], "ok")
        sql_result = results[1]
        self.assertEqual(sql_result["output"]["rowcount"], 2)

    def test_execution_records_duration(self):
        plan = self.planner.generate_plan()
        results = self.executor.execute(plan)
        for result in results:
            self.assertIn("duration", result)
            self.assertGreaterEqual(result["duration"], 0.0)


if __name__ == "__main__":
    unittest.main()
