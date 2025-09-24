# Agnostic Agent Framework

A minimal-yet-complete scaffold for building agentic workflows. The project
ships with a configuration driven planner → executor → reflector loop, an
extensible tool registry, and deterministic fixtures for testing.

## Features

- **Model/provider agnostic.** Configuration specifies the LLM provider,
  model, and parameters without code changes.
- **Config-driven orchestration.** Plans and tool wiring live in YAML, making it
  easy to swap domains or experiment with new flows.
- **Tool-centric execution.** Built-in HTTP, SQL, RAG, and Quantum tools
  demonstrate how to compose external capabilities while keeping tests fast.
- **LangGraph-style loop.** Planner emits a step list, executor streams through
  tools, and the reflector summarises the run for evaluation or logging.
- **Observability aware.** Execution captures per-step timings for lightweight
  performance tracking.
- **Fully tested.** The repository includes a unittest harness and fixture
  configuration for deterministic validation.

## Quick start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Review or edit `config.yaml` to describe your domain. Tool definitions accept
   fully-qualified class paths and optional constructor arguments.

3. Run an agent loop:

   ```bash
   python -m agentic_framework.examples.sql
   ```

   Each example points at the sample fixture configuration to keep output
   deterministic. Swap in your own configuration path for real workloads.

4. Execute the full test suite:

   ```bash
   python -m unittest discover -s agentic_framework/tests
   ```

## Configuration shape

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
framework resolves the class dynamically and caches the instance across steps.

## Adding new tools

1. Implement a subclass of `BaseTool` with an `execute` method.
2. Expose it via a fully-qualified path (for example
   `my_package.tools.CustomTool`).
3. Update the YAML configuration with the tool definition and any constructor
   arguments.

The executor will automatically resolve the class the next time a plan step
references the tool.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
