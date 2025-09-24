from __future__ import annotations

import tempfile
from pathlib import Path

import unittest

from agentic_framework.tools import DoclingTool


class DoclingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = DoclingTool(name="docling", chunk_size=60, overlap=10)

    def test_chunking_returns_headings(self) -> None:
        text = """
# Policy Overview
Eligibility: Employees and spouses are covered.
## Claims
Submit within 60 days.
"""
        result = self.tool.execute(text=text)
        self.assertEqual(result["mode"], "chunk")
        self.assertGreater(len(result["chunks"]), 0)
        self.assertIn("# Policy Overview", result["headings"])

    def test_qa_mode_returns_answer_hint(self) -> None:
        text = """
# Eligibility
Employees are covered.
Spouses are eligible for benefits as well.
"""
        result = self.tool.execute(text=text, mode="qa", query="Are spouses eligible?")
        self.assertEqual(result["mode"], "qa")
        self.assertIn("spouses", result["answer_hint"].lower())

    def test_summarize_mode_truncates_content(self) -> None:
        text = "# Handbook\n" + "Paragraph about policies. " * 40 + "\n## Appendix\nDetails"  # noqa: E501
        result = self.tool.execute(text=text, mode="summarize", chunk_size=80, overlap=20)
        self.assertEqual(result["mode"], "summarize")
        self.assertIn("...", result["summary"])
        # Should only return a sample of the chunks for the summary output.
        self.assertLessEqual(len(result["chunks"]), 3)

    def test_file_path_is_loaded_when_text_missing(self) -> None:
        text = "# Inline Heading\nBody text for the document."
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(text)
            file_path = Path(handle.name)
        try:
            result = self.tool.execute(file_path=str(file_path))
        finally:
            file_path.unlink(missing_ok=True)
        self.assertEqual(result["mode"], "chunk")
        self.assertGreater(len(result["chunks"]), 0)

    def test_missing_inputs_return_error(self) -> None:
        result = self.tool.execute()
        self.assertEqual(result, {"error": "no text or file provided"})


if __name__ == "__main__":
    unittest.main()
