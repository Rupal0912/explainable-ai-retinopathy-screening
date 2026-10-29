# AGENTS.md — Repository Instructions

These instructions apply to coding agents working in this repository.

## Project identity and purpose

Project name: **Explanable AI for Retinopathy & Guidance**.

Build an educational portfolio prototype for five-class diabetic retinopathy grading from retinal fundus images, Grad-CAM visualization, guideline retrieval, constrained LLM summaries, and demo screening history. It is not a diagnostic device and must not be represented as clinically validated.

## Source of truth

- Follow `README.md`, `ARCHITECTURE.md`, `WORKFLOW.md`, `DATA_CARD.md`, `SECURITY.md`, and `TESTING.md`.
- Treat `DR_Screening_Execution_Blueprint.md` (if present) as the execution-plan reference.
- Do not assume planned code has been implemented or tested. Inspect the repository before describing its status.
- If documentation conflicts, preserve safety and privacy requirements first, then update the documents consistently.

## Non-negotiable safety rules

1. Keep a visible “Educational project — not for clinical use” disclaimer in the UI.
2. Do not claim the model diagnoses, rules out, or treats disease.
3. Do not invent metrics, clinical recommendations, thresholds, or validation results.
4. The LLM may summarize only retrieved guideline excerpts. If retrieval is empty or weak, use the safe fallback.
5. Every displayed source citation must map to an actual retrieved source and page.
6. Use synthetic/demo patient IDs only.
7. Persist a SHA-256 image hash, not the image bytes.
8. Never commit credentials, `.env`, raw dataset files, private images, model caches, or unreviewed generated artifacts.

## Engineering conventions

- Use Python modules and run commands from repository root.
- Keep deterministic preprocessing shared between training and inference.
- Keep training augmentation separate from validation/test/inference preprocessing.
- Never use the held-out test set for tuning, threshold selection, or checkpoint selection.
- Use clear, typed interfaces where practical; validate external inputs at boundaries.
- Handle expected external failures explicitly: image decoding, model loading, retrieval/index access, Groq calls, and MongoDB operations.
- Do not silently convert failures into apparently valid predictions or recommendations.
- Keep secrets in environment variables and document names in `.env.example`.
- Add or update tests whenever behaviour changes.
- Keep dependencies pinned once a known-good environment is established.

## Definition of done

A change is complete only when:
- Its behaviour is implemented, not merely documented.
- Relevant automated/manual tests pass, or unrun tests are explicitly reported.
- Error cases produce understandable messages.
- Security/privacy rules remain intact.
- Documentation and examples match the code.
- No unmeasured performance or clinical claim has been added.

## Preferred change process

1. Inspect relevant files and current status.
2. State the smallest implementation plan.
3. Make a focused change.
4. Run targeted tests and lint/type checks if configured.
5. Review the diff for unintended changes and secrets.
6. Update documentation and summarize test results honestly.

## Avoid

- Broad refactors unrelated to the task.
- Adding authentication, APIs, deployment infrastructure, or new model families without a clear need.
- Treating Grad-CAM as proof of correctness.
- Generating clinical advice without retrieved evidence.
- Storing uploaded images or real identifiers.
