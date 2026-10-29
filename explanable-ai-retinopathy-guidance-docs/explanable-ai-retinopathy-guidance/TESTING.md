# Testing Strategy and Checklist

## Principles

- Test deterministic logic automatically.
- Mock network providers and databases in unit tests.
- Use a small, documented sample for smoke tests.
- Do not use the held-out test set to tune thresholds or choose a model.
- Record exactly which tests ran and which were skipped.
- This is software testing, not clinical validation.

## Suggested test layers

### Unit tests
- Preprocessing returns a valid 256×256 RGB image for valid input.
- Invalid shapes and failed image decodes are rejected clearly.
- Quality-gate outcomes are deterministic for fixed inputs.
- Class IDs map to the documented five labels.
- Low-confidence flag uses the configured threshold.
- Chunking respects size/overlap rules and skips tiny fragments.
- Retrieved chunk IDs, scores, source names, and page numbers remain aligned.
- Empty retrieval returns the safe fallback without calling the LLM.
- LLM output citations map to supplied source/page metadata.
- Image hashing is stable for identical bytes and changes for different bytes.

### Integration tests
- Predictor loads the checkpoint and produces a valid result on a test fixture.
- Grad-CAM overlay matches the image dimensions.
- Index builder creates a non-empty index from a permitted sample PDF.
- Retriever returns expected metadata for known queries.
- MongoDB save/history can be tested against a test database or mocked client.
- Streamlit flow distinguishes result display from successful persistence.

### Manual edge-case checklist

- [ ] A text file renamed to `.jpg`
- [ ] Corrupt image bytes
- [ ] Very small image
- [ ] Blurry image
- [ ] Too-dark or overexposed image
- [ ] Screenshot
- [ ] Non-retinal photo
- [ ] Missing/corrupt model checkpoint
- [ ] Low-confidence prediction
- [ ] Empty or missing FAISS index
- [ ] PDF with no extractable text
- [ ] No retrieved chunks above threshold
- [ ] Groq key absent or API unavailable
- [ ] MongoDB unavailable
- [ ] Two screenings for the same demo ID
- [ ] Invalid/empty demo ID
- [ ] UI disclaimer visible on all relevant screens

## Model evaluation checklist

- [ ] Split is stratified and seed recorded.
- [ ] Test samples were not used for model selection or threshold tuning.
- [ ] Accuracy, QWK, confusion matrix, referable sensitivity, and specificity are calculated from test predictions.
- [ ] Denominators are checked; metrics are not silently treated as reliable when a class has no examples.
- [ ] Per-class sample counts are reported.
- [ ] Metrics are saved with the model/version and evaluation configuration.
- [ ] README reports only measured results.

## Release gate

Do not call the prototype end-to-end complete until upload, quality gate, prediction, Grad-CAM, retrieval, grounded guidance, persistence (or explicit failure notice), and history have been exercised together. Passing software tests does not establish clinical safety or validity.
