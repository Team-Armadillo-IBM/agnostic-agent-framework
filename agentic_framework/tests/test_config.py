import unittest
from pathlib import Path

from agentic_framework.config import AgentSettings, Config, LLMConfig, ToolSettings


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
        self.config = Config(self.config_path)

    def test_agent_metadata_loaded(self):
        agent = self.config.agent
        self.assertIsInstance(agent, AgentSettings)
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.llm.provider, "ibm")
        self.assertEqual(agent.llm.model, "granite-base")
        self.assertIn("http", agent.tools)

    def test_tool_settings_loaded(self):
        tools = self.config.tools
        self.assertIn("sql", tools)
        self.assertIsInstance(tools["sql"], ToolSettings)
        self.assertEqual(tools["sql"].class_path, "agentic_framework.tools.SQLTool")

    def test_plan_roundtrip(self):
        plan = self.config.plan()
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0]["tool"], "http")


if __name__ == "__main__":
    unittest.main()
