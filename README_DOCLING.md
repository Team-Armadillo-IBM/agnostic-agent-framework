# Docling Integration

The framework now ships with a `DoclingTool` that can parse documents with the
[Docling](https://github.com/docling-project/docling) library when it is
installed. When Docling is unavailable the tool gracefully falls back to the
previous deterministic chunking, summarisation, and QA routines so the test
suite remains hermetic.

## Capabilities

- Returns document `chunks` alongside detected `headings` and extracted
  `sections` for downstream RAG ingestion.
- Surfaces detected tables (`tables`) when Docling is available.
- Exposes a simple QA mode that highlights the best matching chunk while also
  reporting whether Docling handled the conversion (`docling_used`).
- Supports `chunk`, `summarize`, `qa`, and `extract` modes. The `extract` mode
  includes the full markdown/text representation when Docling succeeds.

## Files

- `agentic_framework/tools.py` – Contains the `DoclingTool`, a reusable
  `DoclingAdapter`, and shared text helpers.
- `config.yaml` – Registers the tool in the default configuration.
- `agentic_framework/examples/docling.py` – Runnable demo showcasing the
  integration.
- `docs/docling_planner_hint.md` – Notes for updating planner prompts or rules.
- `agentic_framework/tests/test_docling_tool.py` – Coverage for both the Docling
  and fallback execution paths.

## Installation

Docling is an optional dependency. Install it to unlock rich document parsing:

```bash
pip install docling
```

Without the dependency the tool still works using the deterministic fallback.

## Manual registration

The tool is already wired into `config.yaml`. To register manually:

```python
from agentic_framework.tools import DoclingTool, ToolRegistry
from agentic_framework.config import Config

config = Config("config.yaml")
registry = ToolRegistry(config)
registry.get("docling")
```

## Demo

```bash
python -m agentic_framework.examples.docling
```
