# Workflow — Explanable AI for Retinopathy & Guidance

## Working rules

- Work from the repository root so module imports resolve.
- Keep raw data, model weights, indexes, `.env`, and private outputs out of Git unless there is an explicit, reviewed reason.
- Run small smoke tests before long training or integration runs.
- Record measured results; never invent metrics.
- The app must visibly state that it is not for clinical use.

## Sprint 1 — Model, preprocessing, inference (Days 1–3)

### Day 1: Data and preprocessing
- Review APTOS 2019 dataset terms and obtain data through the approved route.
- Verify labels, image paths, file integrity, and class counts.
- Implement one deterministic preprocessing function shared by training and inference.
- Calibrate image quality checks against actual images.
- Create stratified train/validation/test splits and save split CSVs.
- Check for duplicate or near-duplicate leakage if practical.

**Exit criteria:** all labelled images are accounted for; split counts are recorded; preprocessing works on sample images; quality-gate rejection rate is inspected.

### Day 2: Train model
- Use Google Colab with a T4 GPU when available.
- Start with pretrained EfficientNet-B0 and a five-class output head.
- Use class-weighted loss as a baseline.
- Save checkpoints to persistent storage after each epoch.
- Select the best checkpoint using validation QWK, not test metrics.

**Exit criteria:** training logs and checkpoint are saved; class distribution and training configuration are recorded.

### Day 3: Evaluate and explain
- Evaluate once on the untouched test split after model selection.
- Record accuracy, QWK, referable sensitivity/specificity, and confusion matrix.
- Inspect per-class metrics, particularly the small severe class.
- Implement Grad-CAM and the Predictor interface.
- Test accepted images and rejected image types.

**Exit criteria:** `models/best.pt` and `models/test_metrics.json` exist; heatmaps overlay the retina; failures return clear messages.

## Sprint 2 — Retrieval, LLM, MongoDB (Days 4–6)

### Day 4: Guideline corpus and index
- Select 3–5 credible, public, text-extractable guideline PDFs.
- Record full provenance and permitted use in `rag_corpus/SOURCES.md`.
- Build the chunk metadata and FAISS index.
- Inspect retrieved chunks for each grade-related query.
- Tune the retrieval threshold using positive and negative examples.

**Exit criteria:** index is non-empty; every chunk maps to a source and page; off-topic queries can return no evidence.

### Day 5: Grounded guidance and persistence
- Implement a constrained LLM wrapper that receives retrieved excerpts only.
- Require concise summaries and source citations.
- Add a no-evidence fallback and provider-error fallback.
- Implement MongoDB persistence of metadata and SHA-256 hash only.
- Add indexes for demo patient ID and descending creation time.

**Exit criteria:** LLM output is bounded by the retrieved text; no-evidence and API-error paths are safe; database writes can be inspected.

### Day 6: Backend smoke test
- Run the complete pipeline from the command line.
- Test a valid image, rejected image, no-retrieval case, unavailable LLM, and unavailable database.
- Confirm history can return multiple records for the same demo ID.
- Fix the first failing boundary before adding features.

**Exit criteria:** smoke tests finish without uncaught errors for expected failure cases.

## Sprint 3 — UI, integration, hardening (Days 7–9)

### Day 7: Streamlit UI
- Build Screen and Patient History tabs.
- Show preprocessed image, Grad-CAM overlay, label, confidence, and warnings.
- Display guideline-grounded guidance and source/page details.
- Show the not-for-clinical-use disclaimer prominently.

### Day 8: Edge cases and history
Manually test:
- Text file renamed with an image extension
- Blurry or over/underexposed image
- Screenshot and non-retinal photograph
- Low-confidence output
- Missing checkpoint/index
- No guideline chunks
- Groq unavailable
- MongoDB unavailable
- Repeated demo ID with two or more records

### Day 9: Release preparation
- Run the full test checklist in `TESTING.md`.
- Review privacy and Git ignore rules in `SECURITY.md`.
- Record actual test metrics and limitations in README.
- Record a short demo with synthetic/demo IDs.
- Inspect staged files and Git history for secrets and private data.
- Push only after tests and documentation match actual behaviour.

## Standard local command sequence

```bash
python -m training.prepare_data
python -m training.train
python -m scripts.build_index
python -m scripts.smoke path/to/retinal_image.png demo-001
streamlit run streamlit_app.py
```

Run only after the relevant code and prerequisites exist. Commands above describe the intended workflow, not a claim that the current repository has already implemented or passed them.

## If time runs short

Cut the history trend chart first, then reduce training epochs only after recording the change. Keep the quality gate, Grad-CAM, evidence-grounded guidance, safe fallbacks, and visible disclaimer.
