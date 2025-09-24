# Agnostic Agent Framework

A minimal yet complete scaffold for experimenting with agentic workflows. The
framework exposes a configuration driven planner → executor → reflector loop,
an extensible tool registry, and deterministic fixtures for testing so that new
ideas can be validated quickly.

## Highlights

- **Model/provider agnostic.** The YAML configuration specifies the LLM
  provider, model, and parameters—no code edits required.
- **Config-first orchestration.** Plans, tool wiring, and runtime defaults live
  in `config.yaml`, making it easy to swap domains or prototype new flows.
- **Batteries included tools.** HTTP, SQL, RAG, quantum, and Docling document
  parsing tools demonstrate how to compose external capabilities while keeping
  the suite deterministic.
- **LangGraph-style loop.** The planner emits step lists, the executor streams
  through tools, and the reflector summarises each run for lightweight
  observability.
- **Fully tested.** A unittest harness with canned fixtures ensures changes are
  regression safe.

## Repository layout

```
agentic_framework/
├── agent.py            # High-level façade that wires planner/executor/reflector
├── config.py           # YAML loader with a minimal fallback parser
├── executor.py         # Streams through plan steps, timing each tool call
├── planner.py          # Materialises plans from configuration
├── reflect.py          # Summarises execution results
├── tools.py            # Built-in tool implementations + registry helpers
└── tests/              # Deterministic unit tests & fixtures
```

Additional examples live in `agentic_framework/examples/` and can be run
directly via `python -m ...`.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Review the configuration**

   Update `config.yaml` to describe your domain. Tool definitions accept
   fully-qualified class paths and optional constructor arguments.

3. **Run an example loop**

   ```bash
   python -m agentic_framework.examples.sql
   ```

   Each example references the sample fixture configuration to keep output
   deterministic. Swap in your own configuration path for real workloads.

4. **Execute the test suite**

   ```bash
   python -m unittest discover -s agentic_framework/tests
   ```

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
framework resolves tool classes dynamically and caches instances across steps.

### Adding new tools

1. Implement a subclass of `BaseTool` with an `execute` method.
2. Expose it via a fully-qualified path (for example `my_package.tools.CustomTool`).
3. Update the YAML configuration with the tool definition and any constructor
   arguments.

The executor will automatically resolve the class the next time a plan step
references the tool.

### Docling integration

The optional `DoclingTool` enriches document parsing when the
[Docling](https://github.com/docling-project/docling) dependency is installed.
Without Docling the tool falls back to deterministic chunking so tests remain
hermetic. See [README_DOCILING.md](README_DOCILING.md) for usage details.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
