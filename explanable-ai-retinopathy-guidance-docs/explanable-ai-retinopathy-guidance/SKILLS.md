# Skills and Engineering Checklist

This file describes the competencies demonstrated by the project. It is not a claim that every item has already been completed.

## Machine learning and computer vision

- [ ] Understand five-class ordinal severity labels.
- [ ] Prepare stratified train/validation/test splits.
- [ ] Apply transfer learning with EfficientNet-B0.
- [ ] Handle class imbalance with a documented baseline.
- [ ] Track training configuration and checkpoints.
- [ ] Evaluate accuracy, QWK, confusion matrix, referable sensitivity, and specificity.
- [ ] Inspect per-class errors and small-class uncertainty.
- [ ] Explain the limits of softmax confidence and Grad-CAM.

## Image processing

- [ ] Decode and validate images safely.
- [ ] Crop black borders and resize consistently.
- [ ] Apply CLAHE in a reproducible way.
- [ ] Separate training augmentations from evaluation preprocessing.
- [ ] Calibrate quality-gate thresholds against real dataset samples.
- [ ] Verify heatmap/image alignment.

## Retrieval-augmented generation

- [ ] Extract PDF text and preserve page provenance.
- [ ] Chunk text with overlap.
- [ ] Embed and index chunks with FAISS.
- [ ] Tune retrieval thresholds with relevant and irrelevant queries.
- [ ] Ground generated summaries exclusively in retrieved passages.
- [ ] Verify citations and implement no-evidence fallback.

## Application engineering

- [ ] Build a Streamlit interface with clear states.
- [ ] Separate inference, retrieval, LLM, and database layers.
- [ ] Manage configuration through environment variables.
- [ ] Handle expected dependency and network failures.
- [ ] Persist screening metadata in MongoDB.
- [ ] Build a demo-ID history view.
- [ ] Add automated tests and a repeatable smoke test.

## Privacy, safety, and reproducibility

- [ ] Use demo IDs only.
- [ ] Store image hashes rather than images.
- [ ] Keep secrets and datasets out of Git.
- [ ] Document dataset terms and guideline sources.
- [ ] Report measured metrics only.
- [ ] Keep the non-clinical disclaimer visible.
- [ ] Record dependency versions and experiment configuration.

## Evidence required before claiming a skill

For every checked item, retain evidence where possible: a test, a run log, a configuration file, a code review note, or a reproducible experiment. Do not mark an item complete merely because the plan or code comments mention it.
