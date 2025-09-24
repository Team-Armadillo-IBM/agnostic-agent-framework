# Docling Integration (Stub)

This repository now includes a `DoclingTool` stub so the planner can parse,
summarise, extract, or run lightweight QA over documents before handing the
results to downstream RAG components.

## Files
- `agentic_framework/tools.py` – Adds the `DoclingTool` class with chunking,
  summarisation, and QA helpers.
- `config.yaml` – Registers the tool so it can be resolved via the default
  configuration.
- `agentic_framework/examples/docling.py` – Small runnable demo showcasing the
  QA mode.
- `docs/docling_planner_hint.md` – Notes for updating planner prompts or rules.

## Install
No additional dependencies are required for the stub. Swap in real Granite
Docling bindings when you are ready.

## Register
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
