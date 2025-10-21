"""Tool registry and built-in tool implementations."""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


_SECTION_PATTERN = re.compile(r"^(#+\s.*|[A-Z][A-Za-z0-9\s]{0,50}:)$", re.MULTILINE)


def _extract_headings_from_text(text: str) -> List[str]:
    """Return Markdown-style or title-case headings detected in ``text``."""

    if not text:
        return []
    return _SECTION_PATTERN.findall(text)


def _split_sections_from_text(text: str) -> List[Dict[str, str]]:
    """Split text into sections using the heading heuristic."""

    if not text:
        return []

    matches = list(_SECTION_PATTERN.finditer(text))
    sections: List[Dict[str, str]] = []
    if not matches:
        stripped = text.strip()
        if stripped:
            sections.append({"heading": None, "text": stripped})
        return sections

    first_start = matches[0].start()
    if first_start > 0:
        lead = text[:first_start].strip()
        if lead:
            sections.append({"heading": None, "text": lead})

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        heading = match.group().strip()
        if heading or body:
            sections.append({"heading": heading or None, "text": body})

    return sections

from .config import Config


class ToolRegistry:
    """Instantiate and cache tools defined in the configuration."""

    def __init__(self, config: Config):
        self._config = config
        self._instances: Dict[str, BaseTool] = {}

    def get(self, name: str) -> "BaseTool":
        if name not in self._instances:
            settings = self._config.get_tool_settings(name)
            tool_cls = resolve_tool_class(settings.class_path)
            kwargs = dict(settings.args)
            self._instances[name] = tool_cls(name=name, **kwargs)
        return self._instances[name]


def resolve_tool_class(class_path: str):
    module_name, _, cls_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class path '{class_path}'")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def get_tool(class_path: str, name: str, **kwargs: Any) -> "BaseTool":
    tool_cls = resolve_tool_class(class_path)
    return tool_cls(name=name, **kwargs)


@dataclass
class BaseTool:
    """Base class for tools with a consistent execute contract."""

    name: str

    def execute(self, **_kwargs: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class HTTPTool(BaseTool):
    """Minimal HTTP abstraction using canned responses for determinism."""

    def __init__(
        self,
        name: str,
        base_url: str = "",
        canned_responses: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name=name)
        self.base_url = base_url.rstrip("/")
        self.canned_responses = dict(canned_responses or {})

    def execute(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        target = self._normalise_url(url)
        body = self.canned_responses.get(target)
        if body is None:
            body = f"{method.upper()} {target}"
        return {
            "url": target,
            "method": method.upper(),
            "params": dict(params or {}),
            "body": body,
        }

    def _normalise_url(self, url: str) -> str:
        if url.startswith("http"):
            return url
        if not self.base_url:
            return url
        return f"{self.base_url}/{url.lstrip('/')}"


class SQLTool(BaseTool):
    """In-memory SQLite tool suitable for deterministic tests."""

    def __init__(self, name: str, setup: Optional[Sequence[str]] = None):
        super().__init__(name=name)
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        if setup:
            cursor = self._connection.cursor()
            for statement in setup:
                cursor.execute(statement)
            self._connection.commit()

    def execute(
        self, query: str, parameters: Optional[Sequence[Any]] = None
    ) -> Dict[str, Any]:
        cursor = self._connection.cursor()
        cursor.execute(query, parameters or [])
        if query.lstrip().upper().startswith("SELECT"):
            rows = [dict(row) for row in cursor.fetchall()]
            return {"rows": rows, "rowcount": len(rows)}
        self._connection.commit()
        return {"rowcount": cursor.rowcount}


class RAGTool(BaseTool):
    """Toy retrieval-augmented generation helper using keyword matching."""

    def __init__(
        self,
        name: str,
        documents: Optional[Iterable[Mapping[str, str]]] = None,
    ):
        super().__init__(name=name)
        self._documents = [dict(doc) for doc in documents or []]

    def execute(self, query: str, top_k: int = 1) -> Dict[str, Any]:
        scored = []
        tokens = self._tokenise(query)
        for doc in self._documents:
            score = self._score(tokens, self._tokenise(doc.get("text", "")))
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_results = [doc for score, doc in scored[:top_k] if score > 0]
        return {"query": query, "results": top_results}

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [token.lower() for token in text.split() if token]

    @staticmethod
    def _score(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
        query_set = set(query_tokens)
        if not query_set:
            return 0.0
        doc_set = set(doc_tokens)
        intersection = query_set & doc_set
        if not intersection:
            return 0.0
        return len(intersection) / math.sqrt(len(query_set) * len(doc_set) or 1)


@dataclass
class DoclingContent:
    """Container for structured content returned by Docling."""

    text: str
    headings: List[str]
    sections: List[Dict[str, Optional[str]]]
    tables: List[Dict[str, Any]]


class DoclingAdapter:
    """Thin wrapper around the optional Docling document converter."""

    def __init__(
        self,
        options: Optional[Mapping[str, Any]] = None,
        converter_factory: Optional[Callable[[], Any]] = None,
    ):
        self._options = dict(options or {})
        self._converter_factory = converter_factory or self._build_factory()

    def is_available(self) -> bool:
        return self._converter_factory is not None

    def convert(
        self, text: Optional[str] = None, file_path: Optional[str] = None
    ) -> DoclingContent:
        if not self.is_available():  # pragma: no cover - guarded by caller
            raise RuntimeError("Docling is not available")

        source, cleanup = self._prepare_source(text=text, file_path=file_path)
        try:
            converter = self._converter_factory()
            if converter is None:
                raise RuntimeError("Docling converter factory returned None")
            result = self._invoke_converter(converter, source)
        finally:
            cleanup()

        document = self._extract_document(result)
        extracted_text = self._extract_text(document) or (text or "")
        headings = self._extract_headings(document)
        sections = self._extract_sections(document)
        tables = self._extract_tables(document)

        if not headings:
            headings = _extract_headings_from_text(extracted_text)
        if not sections:
            sections = _split_sections_from_text(extracted_text)

        return DoclingContent(
            text=extracted_text,
            headings=headings,
            sections=sections,
            tables=tables,
        )

    # -- factory helpers -------------------------------------------------

    def _build_factory(self) -> Optional[Callable[[], Any]]:
        try:
            module = importlib.import_module("docling.document_converter")
        except Exception:  # pragma: no cover - import is environment specific
            return None

        DocumentConverter = getattr(module, "DocumentConverter", None)
        if DocumentConverter is None:
            return None

        config_cls = getattr(module, "DocumentConverterConfig", None)
        options = dict(self._options)

        def factory() -> Any:
            if config_cls is not None:
                with contextlib.suppress(Exception):
                    config = config_cls(**options)
                    return DocumentConverter(config=config)
                with contextlib.suppress(Exception):
                    config = config_cls()
                    for key, value in options.items():
                        setattr(config, key, value)
                    return DocumentConverter(config=config)
            with contextlib.suppress(Exception):
                return DocumentConverter(**options)
            return DocumentConverter()

        return factory

    def _prepare_source(
        self, text: Optional[str], file_path: Optional[str]
    ) -> tuple[str, Callable[[], None]]:
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(file_path)
            return str(path), lambda: None

        if text is None:
            raise ValueError("Docling requires text or a file path")

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        )
        try:
            handle.write(text)
        finally:
            handle.close()

        def cleanup() -> None:
            Path(handle.name).unlink(missing_ok=True)

        return handle.name, cleanup

    def _invoke_converter(self, converter: Any, source: str) -> Any:
        if hasattr(converter, "__enter__") and hasattr(converter, "__exit__"):
            with converter as managed:
                return self._call_convert(managed, source)
        return self._call_convert(converter, source)

    def _call_convert(self, converter: Any, source: str) -> Any:
        convert = getattr(converter, "convert", None)
        if callable(convert):
            with contextlib.suppress(TypeError):
                return convert(source)
            with contextlib.suppress(Exception):
                with open(source, "rb") as handle:
                    return convert(io.BytesIO(handle.read()))

        alt_convert = getattr(converter, "convert_document", None)
        if callable(alt_convert):
            with contextlib.suppress(Exception):
                return alt_convert(source)

        raise AttributeError("Docling converter has no usable convert method")

    # -- extraction helpers ---------------------------------------------

    @staticmethod
    def _extract_document(result: Any) -> Any:
        for attr in ("document", "structured_document", "doc", "output_document"):
            document = getattr(result, attr, None)
            if document is not None:
                return document
        return result

    def _extract_text(self, node: Any) -> str:
        if node is None:
            return ""
        if isinstance(node, str):
            return node
        if isinstance(node, (list, tuple)):
            parts = [self._extract_text(item) for item in node]
            return "\n".join(part for part in parts if part)

        for attr in ("export_to_markdown", "to_markdown", "export_markdown"):
            method = getattr(node, attr, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    value = method()
                    if isinstance(value, str):
                        return value

        for attr in ("export_to_text", "to_text", "get_text"):
            member = getattr(node, attr, None)
            if callable(member):
                with contextlib.suppress(Exception):
                    value = member()
                    if isinstance(value, str):
                        return value
            elif isinstance(member, str):
                return member

        for attr in ("markdown", "text", "content"):
            value = getattr(node, attr, None)
            if isinstance(value, str):
                return value

        pages = getattr(node, "pages", None)
        if isinstance(pages, (list, tuple)):
            parts = [self._extract_text(page) for page in pages]
            parts = [part for part in parts if part]
            if parts:
                return "\n\n".join(parts)

        return ""

    def _extract_headings(self, document: Any) -> List[str]:
        headings: List[str] = []
        for attr in ("headings", "sections", "toc"):
            values = getattr(document, attr, None)
            headings.extend(self._collect_headings(values))

        unique: List[str] = []
        seen = set()
        for heading in headings:
            if not heading:
                continue
            key = heading.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(key)
        return unique

    def _collect_headings(self, values: Any) -> List[str]:
        headings: List[str] = []
        if values is None:
            return headings
        if isinstance(values, (list, tuple, set)):
            for value in values:
                headings.extend(self._collect_headings(value))
            return headings
        if isinstance(values, dict):
            label = values.get("heading") or values.get("title") or values.get("name")
            if isinstance(label, str):
                headings.append(label)
            for key in ("children", "sections", "items"):
                if key in values:
                    headings.extend(self._collect_headings(values[key]))
            return headings

        for attr in ("title", "heading", "name", "label"):
            candidate = getattr(values, attr, None)
            if callable(candidate):
                with contextlib.suppress(Exception):
                    candidate = candidate()
            if isinstance(candidate, str):
                headings.append(candidate)
                break

        for attr in ("children", "sections", "items", "subsections"):
            child = getattr(values, attr, None)
            if child is not None:
                headings.extend(self._collect_headings(child))

        return headings

    def _extract_sections(self, document: Any) -> List[Dict[str, Optional[str]]]:
        sections: List[Dict[str, Optional[str]]] = []
        raw_sections = getattr(document, "sections", None)
        if isinstance(raw_sections, (list, tuple)):
            for section in raw_sections:
                heading = self._stringify(getattr(section, "title", None)) or self._stringify(
                    getattr(section, "heading", None)
                )
                text = self._extract_text(section).strip()
                if heading or text:
                    sections.append({"heading": heading or None, "text": text})
        return sections

    def _extract_tables(self, document: Any) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        raw_tables = getattr(document, "tables", None)
        if isinstance(raw_tables, (list, tuple)):
            for table in raw_tables:
                rows = self._extract_table_rows(table)
                if rows:
                    title = self._stringify(getattr(table, "title", None)) or self._stringify(
                        getattr(table, "heading", None)
                    )
                    tables.append({"title": title or None, "rows": rows})
        return tables

    def _extract_table_rows(self, table: Any) -> List[List[str]]:
        rows: List[List[str]] = []
        raw_rows = getattr(table, "rows", None) or getattr(table, "cells", None)
        if isinstance(raw_rows, (list, tuple)):
            for row in raw_rows:
                if isinstance(row, (list, tuple)):
                    rows.append([self._stringify_cell(cell) for cell in row])
                else:
                    cells = getattr(row, "cells", None)
                    if isinstance(cells, (list, tuple)):
                        rows.append([self._stringify_cell(cell) for cell in cells])
        return rows

    def _stringify_cell(self, cell: Any) -> str:
        text = self._extract_text(cell)
        if text:
            return text
        for attr in ("text", "content", "value"):
            value = getattr(cell, attr, None)
            if isinstance(value, str):
                return value
        return ""

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        if callable(value):
            with contextlib.suppress(Exception):
                value = value()
            if isinstance(value, str):
                return value
        return ""


class DoclingTool(BaseTool):
    """Document parser that optionally delegates to Docling for structure.

    When the Docling library is unavailable the tool falls back to
    deterministic text chunking so tests remain fast and reproducible. The
    adapter exposes Docling's markdown/tables when possible while preserving
    the existing chunking/summary/QA contracts.
    """

    _SUPPORTED_MODES = {"chunk", "summarize", "qa", "extract"}

    def __init__(
        self,
        name: str,
        chunk_size: int = 800,
        overlap: int = 120,
        default_mode: str = "chunk",
        docling_options: Optional[Mapping[str, Any]] = None,
        docling_adapter: Optional[DoclingAdapter] = None,
    ):
        super().__init__(name=name)
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        default_mode = default_mode.lower()
        if default_mode not in self._SUPPORTED_MODES:
            raise ValueError(f"Unsupported default mode '{default_mode}'")
        self.default_mode = default_mode
        self._docling_adapter = (
            docling_adapter if docling_adapter is not None else DoclingAdapter(options=docling_options)
        )

    def execute(
        self,
        text: Optional[str] = None,
        file_path: Optional[str] = None,
        mode: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode_name = (mode or self.default_mode).lower()
        if mode_name not in self._SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode_name}'")

        if text is None and file_path is None:
            return {"error": "no text or file provided"}

        docling_used = False
        docling_content: Optional[DoclingContent] = None
        if self._docling_adapter and self._docling_adapter.is_available():
            with contextlib.suppress(FileNotFoundError, ValueError, AttributeError, RuntimeError):
                docling_content = self._docling_adapter.convert(text=text, file_path=file_path)
                docling_used = bool(docling_content and docling_content.text)

        resolved_text = docling_content.text if docling_content and docling_content.text else None
        if resolved_text is None:
            resolved_text = self._resolve_text(text=text, file_path=file_path)
        if resolved_text is None:
            return {"error": "no text or file provided"}

        chunk_size = int(chunk_size or self.chunk_size)
        overlap = int(overlap if overlap is not None else self.overlap)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be zero or positive")

        headings = list(docling_content.headings) if docling_content else []
        if not headings:
            headings = _extract_headings_from_text(resolved_text)

        sections = list(docling_content.sections) if docling_content else []
        if not sections:
            sections = _split_sections_from_text(resolved_text)

        tables = list(docling_content.tables) if docling_content else []

        chunks = self._chunk(resolved_text, chunk_size=chunk_size, overlap=overlap)

        base_payload: Dict[str, Any] = {
            "mode": mode_name,
            "headings": headings,
            "chunks": chunks,
            "sections": sections,
            "tables": tables,
            "docling_used": docling_used,
        }

        if mode_name == "summarize":
            summary = self._simple_summary(chunks)
            base_payload.update({"summary": summary, "chunks": chunks[:3]})
            return base_payload

        if mode_name == "qa":
            if not query:
                raise ValueError("query is required when mode='qa'")
            answer_hint = self._best_match(chunks, query)
            base_payload.update({"query": query, "answer_hint": answer_hint})
            return base_payload

        if mode_name == "extract":
            base_payload.update({"text": resolved_text})
            return base_payload

        return base_payload

    def _resolve_text(
        self, text: Optional[str], file_path: Optional[str]
    ) -> Optional[str]:
        if text:
            return text
        if not file_path:
            return None
        path = Path(file_path)
        if not path.exists():
            return None
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        return path.read_bytes().decode("utf-8", errors="ignore")

    @staticmethod
    def _chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        step = max(1, chunk_size - overlap)
        index = 0
        text_length = len(text)
        while index < text_length:
            chunks.append(text[index : index + chunk_size])
            index += step
        return chunks

    @staticmethod
    def _best_match(chunks: List[str], query: str) -> str:
        if not chunks:
            return ""
        tokens = DoclingTool._tokenise(query)
        best_chunk = max(
            chunks,
            key=lambda chunk: DoclingTool._overlap_score(tokens, DoclingTool._tokenise(chunk)),
        )
        return best_chunk[:280]

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _overlap_score(query_tokens: List[str], doc_tokens: List[str]) -> int:
        return len(set(query_tokens) & set(doc_tokens))

    @staticmethod
    def _simple_summary(chunks: List[str]) -> str:
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0][:300]
        return f"{chunks[0][:200]} ... {chunks[-1][-200:]}"


