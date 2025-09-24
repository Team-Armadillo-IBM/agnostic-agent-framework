"""Demonstrate the DoclingTool stub."""

from __future__ import annotations

from agentic_framework.tools import DoclingTool


def main() -> None:
    tool = DoclingTool(name="docling")
    spec = {
        "name": tool.name,
        "description": "Parse documents into chunks or lightweight summaries.",
    }
    print("Spec:", spec["name"], "-", spec["description"])

    sample_text = """
# Benefits Overview
Eligibility: Employees and spouses are covered.
Claims: Submit within 60 days.

### Dental Coverage
Includes preventive visits and X-rays.
"""

    output = tool.execute(text=sample_text, mode="qa", query="Are spouses eligible?")
    print("QA Output:", output)


if __name__ == "__main__":
    main()
