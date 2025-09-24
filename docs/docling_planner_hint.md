# Planner hints for Docling

Update planner prompts or rules to account for the new document tool:

- Call the `docling` tool whenever the user requests parsing, summarisation,
  extraction, or QA over a document.
- Provide either `text` or `file_path` and select a `mode` of `chunk`,
  `summarize`, `qa`, or `extract`.
- When `mode` is `qa`, include a `query` so the tool can return an
  `answer_hint`.
- Feed returned `chunks` into downstream retrieval tooling or surface the
  summary and `answer_hint` directly to the user.
