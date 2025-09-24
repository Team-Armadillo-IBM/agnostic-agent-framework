"""Example showing SQL tool usage inside the executor."""

from pathlib import Path

from agentic_framework import Agent


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_config.yaml"
    agent = Agent(config_path)
    result = agent.run(goal="list enrolled clients")
    for step in result["results"]:
        if step["tool"] == "sql":
            print(step["output"])  # pragma: no cover - demo output
            break


if __name__ == "__main__":
    main()
