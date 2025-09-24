import unittest
from pathlib import Path

from agentic_framework.agent import Agent


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"

    def test_agent_run_returns_outputs(self):
        agent = Agent(self.config_path)
        result = agent.run(goal="assess posture")
        self.assertIn("plan", result)
        self.assertIn("results", result)
        self.assertIn("reflection", result)
        self.assertEqual(len(result["plan"]), len(result["results"]))


if __name__ == "__main__":
    unittest.main()
