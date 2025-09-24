"""Tool registry and built-in tool implementations."""

from __future__ import annotations

import importlib
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Config

try:  # pragma: no cover - optional dependency
    from qiskit import Aer, QuantumCircuit, execute
except Exception:  # pragma: no cover - gracefully degrade if qiskit is absent
    Aer = None
    QuantumCircuit = None

    def execute(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("qiskit is not available in this environment")


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


class QuantumTool(BaseTool):
    """Wrapper around qiskit with a graceful fallback when unavailable."""

    def __init__(self, name: str, shots: int = 1024, circuit: Optional[Any] = None):
        super().__init__(name=name)
        self.shots = shots
        self.default_circuit = circuit or "bell"

    def execute(
        self, circuit: Optional[Any] = None, shots: Optional[int] = None
    ) -> Dict[str, Any]:
        circuit = circuit or self.default_circuit
        shots = shots or self.shots
        if QuantumCircuit is None:
            return {"counts": {"00": shots}, "backend": "fallback"}

        qc = self._ensure_circuit(circuit)
        backend = Aer.get_backend("aer_simulator")
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return {"counts": counts, "backend": backend.name()}

    def _ensure_circuit(self, circuit: Any) -> "QuantumCircuit":
        if isinstance(circuit, QuantumCircuit):
            return circuit
        if isinstance(circuit, str):
            if circuit == "bell":
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])
                return qc
            raise ValueError(f"Unknown circuit template '{circuit}'")
        raise TypeError("Circuit must be a QuantumCircuit instance or template name")
