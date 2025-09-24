"""Demonstrate the DoclingTool integration."""

from __future__ import annotations

from pprint import pprint

from agentic_framework.tools import DoclingTool


def main() -> None:
    tool = DoclingTool(name="docling")
    print("Spec:", tool.name, "-", "Document parsing via Docling when available")

    sample_text = """
# Benefits Overview
Eligibility: Employees and spouses are covered.
Claims: Submit within 60 days.

### Dental Coverage
Includes preventive visits and X-rays.
"""

    chunk_output = tool.execute(text=sample_text)
    print("\nChunk mode (docling_used=", chunk_output["docling_used"], ")", sep="")
    pprint({"headings": chunk_output["headings"], "sections": chunk_output["sections"][:1]})

    qa_output = tool.execute(text=sample_text, mode="qa", query="Are spouses eligible?")
    print("\nQA answer hint:", qa_output["answer_hint"])

    extract_output = tool.execute(text=sample_text, mode="extract")
    print("\nExtract mode sections:")
    pprint(extract_output["sections"])
    if extract_output["tables"]:
        print("Tables:")
        pprint(extract_output["tables"])


if __name__ == "__main__":
    main()
