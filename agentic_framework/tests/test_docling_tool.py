from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import unittest

from agentic_framework.tools import DoclingContent, DoclingTool


class FakeDoclingAdapter:
    def __init__(self, content: Optional[DoclingContent] = None):
        self.calls: List[Dict[str, Optional[str]]] = []
        self._content = content or DoclingContent(
            text="## Outline\nDocling extracted text",
            headings=["## Outline"],
            sections=[{"heading": "Outline", "text": "Docling extracted text"}],
            tables=[{"title": "Table 1", "rows": [["A", "B"]]}],
        )

    def is_available(self) -> bool:
        return True

    def convert(
        self, text: Optional[str] = None, file_path: Optional[str] = None
    ) -> DoclingContent:
        self.calls.append({"text": text, "file_path": file_path})
        return self._content


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
        self.assertFalse(result["docling_used"])
        self.assertGreaterEqual(len(result["sections"]), 2)
        self.assertEqual(result["sections"][0]["heading"], "# Policy Overview")
        self.assertEqual(result["tables"], [])

    def test_qa_mode_returns_answer_hint(self) -> None:
        text = """
# Eligibility
Employees are covered.
Spouses are eligible for benefits as well.
"""
        result = self.tool.execute(text=text, mode="qa", query="Are spouses eligible?")
        self.assertEqual(result["mode"], "qa")
        self.assertIn("spouses", result["answer_hint"].lower())
        self.assertIn("sections", result)
        self.assertEqual(result["tables"], [])

    def test_summarize_mode_truncates_content(self) -> None:
        text = "# Handbook\n" + "Paragraph about policies. " * 40 + "\n## Appendix\nDetails"  # noqa: E501
        result = self.tool.execute(text=text, mode="summarize", chunk_size=80, overlap=20)
        self.assertEqual(result["mode"], "summarize")
        self.assertIn("...", result["summary"])
        # Should only return a sample of the chunks for the summary output.
        self.assertLessEqual(len(result["chunks"]), 3)
        self.assertFalse(result["docling_used"])
        self.assertGreaterEqual(len(result["sections"]), 1)

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
        self.assertFalse(result["docling_used"])

    def test_missing_inputs_return_error(self) -> None:
        result = self.tool.execute()
        self.assertEqual(result, {"error": "no text or file provided"})

    def test_docling_adapter_enriches_output(self) -> None:
        adapter = FakeDoclingAdapter()
        tool = DoclingTool(
            name="docling", chunk_size=40, overlap=5, docling_adapter=adapter
        )
        result = tool.execute(text="Original fallback text")
        self.assertTrue(result["docling_used"])
        self.assertEqual(
            adapter.calls, [{"text": "Original fallback text", "file_path": None}]
        )
        self.assertEqual(result["mode"], "chunk")
        self.assertIn("## Outline", result["headings"][0])
        self.assertEqual(result["sections"][0]["heading"], "Outline")
        self.assertEqual(result["tables"][0]["rows"], [["A", "B"]])

    def test_extract_mode_returns_docling_text(self) -> None:
        content = DoclingContent(
            text="## Outline\nDocling extracted text",
            headings=["## Outline"],
            sections=[{"heading": "Outline", "text": "Docling extracted text"}],
            tables=[{"title": "Numbers", "rows": [["1", "2"]]}],
        )
        adapter = FakeDoclingAdapter(content=content)
        tool = DoclingTool(name="docling", docling_adapter=adapter)
        result = tool.execute(text="raw", mode="extract")
        self.assertEqual(result["mode"], "extract")
        self.assertTrue(result["docling_used"])
        self.assertEqual(result["text"], content.text)
        self.assertEqual(result["tables"][0]["rows"], [["1", "2"]])


if __name__ == "__main__":
    unittest.main()
