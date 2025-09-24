# Planner hints for Docling

Update planner prompts or rules to account for the richer document tool:

- Call the `docling` tool whenever the user requests parsing, summarisation,
  extraction, or QA over a document. Provide either `text` or a `file_path`.
- Select a `mode` of `chunk`, `summarize`, `qa`, or `extract`. The `extract`
  mode mirrors Docling output and includes the full markdown/text payload.
- When `mode` is `qa`, include a `query` so the tool can return an
  `answer_hint`.
- Downstream steps can use the `sections` and `chunks` fields for RAG indexing,
  and the optional `tables` array when Docling surfaces structured data.
- Inspect `docling_used` to determine whether the optional dependency was
  active and decide whether to re-run with fallback logic if needed.
