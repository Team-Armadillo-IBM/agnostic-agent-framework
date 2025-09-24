"""Example demonstrating the quantum tool with a Bell-state circuit."""

from pathlib import Path

from agentic_framework import Agent


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_config.yaml"
    agent = Agent(config_path)
    result = agent.run(goal="optimise max-cut instance")
    print(result["reflection"]["summary"])  # pragma: no cover - demo output


if __name__ == "__main__":
    main()
