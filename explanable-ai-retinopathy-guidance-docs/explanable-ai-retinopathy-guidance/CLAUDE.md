# Claude Code Instructions

Read `AGENTS.md` first. It contains the repository-wide rules and takes precedence over task-specific preferences in this file.

## Project context

**Explanable AI for Retinopathy & Guidance** is an educational prototype using APTOS 2019, EfficientNet-B0, Grad-CAM, FAISS retrieval over selected guideline PDFs, Groq-based excerpt summarization, MongoDB Atlas, and Streamlit.

## How to work here

- Inspect the current files before proposing or implementing changes.
- Preserve the architecture unless the user explicitly asks for a redesign.
- Keep preprocessing identical between dataset preparation and inference.
- Treat the test split as final hold-out data.
- Keep the LLM grounded in retrieved excerpts and preserve source/page metadata.
- Keep secrets and raw/private data out of version control.
- Add tests for new functionality and update relevant Markdown docs.
- If a dependency API or model name may have changed, verify it from the installed package/provider configuration rather than guessing.

## Medical and privacy guardrails

- The UI must state that the prototype is not for clinical use.
- Do not claim clinical validation.
- Do not create diagnosis or treatment instructions beyond carefully attributed, retrieved guideline summaries.
- If no relevant source chunks exist, show the fallback.
- Use demo IDs and store the image hash, not the image.
- Never fabricate test results or imply that unexecuted code has passed.

## Response expectations

When completing a task, report:
1. What changed
2. Files changed
3. Tests actually run and their results
4. Remaining risks or steps not verified

Be concise and explicit about uncertainty.
