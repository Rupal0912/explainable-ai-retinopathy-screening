# Reusable Development Prompts

Use these prompts with a coding assistant from the repository root. Read `AGENTS.md` first. Prompts are task specifications, not evidence that work has already been done.

## 1. Repository audit

> Inspect this repository against README.md, ARCHITECTURE.md, WORKFLOW.md, AGENTS.md, and TESTING.md. Do not modify files yet. Identify missing components, broken imports, inconsistent preprocessing, unsafe failure paths, privacy risks, and documentation claims unsupported by code. Return a prioritized checklist with exact file paths.

## 2. Preprocessing and quality gate

> Implement the preprocessing and quality-gate contract in ARCHITECTURE.md. Keep deterministic preprocessing identical for training preparation and inference. Validate image shape/type and handle decode failures. Add tests for black borders, too-small images, blur, brightness, and non-fundus-like inputs. Treat thresholds as configurable starting points and do not claim perfect retinal-image detection.

## 3. Dataset preparation

> Implement training-data preparation for the approved APTOS 2019 dataset. Verify file paths and labels, create stratified train/validation/test splits with a fixed seed, and save split CSVs. Prevent test leakage and report class counts. Do not download or commit the dataset automatically unless explicitly requested and terms are reviewed.

## 4. Model training and evaluation

> Implement EfficientNet-B0 five-class training as specified in ARCHITECTURE.md. Use class-weighted loss as a baseline and select checkpoints using validation QWK. Evaluate the untouched test split only after selection. Save actual metrics and confusion matrix; do not fabricate numbers. Make device handling robust and checkpoint to persistent storage where supported.

## 5. Grad-CAM explanation

> Implement Grad-CAM for the predicted class and the configured final feature layer. Ensure tensor normalization and displayed image dimensions are aligned. Add a smoke test for output shape/type and inspect overlays for alignment. Label the overlay as a model-attribution visualization, not proof of clinical reasoning.

## 6. Retrieval index

> Implement the PDF-to-FAISS index pipeline. Read only PDFs documented in rag_corpus/SOURCES.md, preserve source and page metadata, skip empty text safely, and produce a clear error if no usable chunks are found. Add tests for chunk overlap and metadata alignment. Do not invent source citations.

## 7. Grounded LLM guidance

> Implement a constrained LLM wrapper that receives only the retrieved guideline excerpts and the grade label. Require concise output with citations tied to supplied source/page metadata. If no chunks pass the relevance threshold, or the provider call fails, return the documented safe fallback. Do not add free-form medical Q&A, diagnosis, medication advice, or invented follow-up intervals. Add tests with mocked provider responses.

## 8. MongoDB persistence

> Implement persistence for demo screening metadata only. Store demo ID, UTC timestamp, SHA-256 hash of upload bytes, model output, guidance, and source/page/score metadata. Do not store image bytes or real identifiers. Add a compound index for demo ID and timestamp. Handle connection failures without crashing the result screen, and document retention/access assumptions.

## 9. Streamlit integration

> Implement the Screen and Patient History tabs according to ARCHITECTURE.md. Show the not-for-clinical-use disclaimer prominently, reject invalid images, display prediction confidence and low-confidence warnings, show Grad-CAM with a clear caption, display retrieved source citations, and distinguish unsaved results when MongoDB is unavailable. Use demo IDs only.

## 10. Test and harden

> Work through TESTING.md. Add automated tests for deterministic preprocessing, class mapping, no-evidence fallback, citation mapping, database failure, invalid uploads, and history ordering. Mock external APIs. Report only tests actually executed and include exact failures without claiming success prematurely.

## 11. Prepare a portfolio release

> Audit the repository for committed secrets, raw images, dataset files, model files, stale claims, and unverified metrics. Update README.md with measured results only, the split method, dataset limitations, and disclaimer. Produce a concise release checklist and do not push or deploy without explicit approval.
