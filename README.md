# Agnostic Agent Framework

> A lightweight scaffold for experimenting with agentic workflows. Configure a
planner → executor → reflector loop, wire in domain tools, and iterate quickly
with deterministic fixtures.

## Why this framework?

- **Model/provider agnostic.** Select your LLM provider, model, and parameters
  exclusively through YAML configuration—no code edits required.
- **Config-first orchestration.** `config.yaml` houses plans, tool wiring, and
  runtime defaults so you can swap domains or prototype new flows rapidly.
- **Extensible tool registry.** Batteries-included HTTP, SQL, RAG, quantum, and
  Docling integrations illustrate how to compose external capabilities while
  keeping runs deterministic.
- **Observable loop.** The planner emits step lists, the executor streams
  through tools, and the reflector summarises each run for quick insight.
- **Fully tested.** A unittest harness with canned fixtures keeps regressions at
  bay while you iterate.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the agent

Edit `config.yaml` to describe your problem space. Each tool entry accepts a
fully-qualified class path plus optional constructor arguments.

### 3. Run an example loop

```bash
python -m agentic_framework.examples.sql
```

Examples reference the sample fixture configuration to ensure deterministic
output. Swap in your own configuration path for real workloads.

### 4. Execute the test suite

```bash
python -m unittest discover -s agentic_framework/tests
```

## Project layout

```
agentic_framework/
├── agent.py            # High-level façade wiring planner/executor/reflector
├── config.py           # YAML loader with a minimal fallback parser
├── executor.py         # Streams through plan steps, timing each tool call
├── planner.py          # Materialises plans from configuration
├── reflect.py          # Summarises execution results
├── tools.py            # Tool implementations + registry helpers
└── tests/              # Deterministic unit tests & fixtures
```

Additional runnable samples live in `agentic_framework/examples/`.

## Configuration primer

```yaml
agent:
  name: Industry Challenge Agent
  llm:
    provider: ibm
    model: granite-13b-chat-v2
    parameters:
      temperature: 0.1
  tools:
    - http
    - sql
    - rag
    - quantum
    - docling
  plan:
    - tool: http
      args:
        url: "/status"
    - tool: sql
      args:
        query: "SELECT name FROM users"
    - tool: rag
      args:
        query: "collaborative solution design"
        top_k: 2
    - tool: quantum
      args:
        circuit: bell
```

Each tool referenced in `plan` must have a matching entry under `tools`. The
executor resolves tool classes dynamically and caches instances across steps.

### Adding a new tool

1. Subclass `BaseTool` and implement `execute`.
2. Expose the class via a fully-qualified path (for example
   `my_package.tools.CustomTool`).
3. Update `config.yaml` with the tool definition and constructor arguments.

The executor will resolve the class the next time a plan step references the
tool.

### Docling integration

The optional Docling adapter enriches document parsing when the
[Docling](https://github.com/docling-project/docling) dependency is installed.
Without Docling the tool falls back to deterministic chunking so tests remain
hermetic. See [README_DOCLING.md](README_DOCLING.md) for usage details.

## Development tips

- Adjust logging levels via standard Python logging configuration to surface
  executor timing information.
- Extend the unit tests in `agentic_framework/tests/` when adding new tools or
  planner heuristics to preserve deterministic behaviour.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Credits

- Kristopher McCoy, Technical/Team Lead — Initiated the framework to provide an adaptable foundation when the team had not yet selected a topic.
