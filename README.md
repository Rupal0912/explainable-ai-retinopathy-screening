# Diabetic Retinopathy Screening Demo

An educational five-grade APTOS diabetic-retinopathy classification project using EfficientNet-B0, Grad-CAM, MongoDB screening metadata, and optional evidence retrieval. It is not a medical device and must not be used to diagnose disease or make care decisions.

## Evaluation and artifacts

- The selected checkpoint is `models/best.pt` (EfficientNet-B0, best validation QWK at epoch 4).
- The held-out test set contains 531 samples. Recorded test accuracy is **82.49%** and quadratic weighted kappa is **0.8714**. These figures are in `models/evaluation_summary.json`.
- Epoch-by-epoch training and validation metrics are in `models/training_log.csv`.
- Training and evaluation are complete. Do not run training or evaluation commands to launch the demo.

These are project test-set measurements, not clinical evidence. Grade 3 and grade 4 recall are low, so the model can miss examples in those classes.

## Run the application

Use the existing virtual environment from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run the automated tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The MongoDB CRUD smoke check is an explicit command and is not collected as a pytest test:

```powershell
.\.venv\Scripts\python.exe scripts\mongo_smoke.py
```

## MongoDB setup

The application reads `MONGODB_URI` and optional `MONGODB_DB`. Copy `.env.example` to `.env` yourself and set the values locally; do not commit `.env` or share credentials.

For a local MongoDB server, the non-secret URI format is:

```dotenv
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=retinopathy_demo
```

For MongoDB Atlas, use the URI template supplied by Atlas, for example:

```dotenv
MONGODB_URI=mongodb+srv://<database-user>:<database-password>@<cluster-host>/<database>?retryWrites=true&w=majority
MONGODB_DB=retinopathy_demo
```

Replace placeholders in your local `.env`, use a database-scoped account, and URL-encode reserved characters in credentials. Keep the URI private.

The **Demo history** tab supports Create and Read through screening submission and demo-ID history lookup. It also supports editing a reviewer note and deleting a selected record after confirmation. Grade statistics are calculated with a MongoDB aggregation grouped by grade, including count and average confidence. `metadata_store.py` also provides `stats()` for optional demo-ID filtering.

The storage layer saves an upload SHA-256 hash and screening metadata, not image bytes. Database errors are reported without exposing the URI. `scripts/mongo_smoke.py` performs a separate CRUD check in `screenings_smoke` and cleans up its inserted record.

## Optional educational guidance

Guidance uses a local FAISS index built from permitted, reviewed PDF material. Unless the retrieval index exists and `GROQ_API_KEY` is configured, the app uses a safe fallback and does not generate a model-written summary. Configure `GROQ_MODEL` only if a Groq summary is desired.

## Limitations

- Educational demonstration only; not clinically validated and not suitable for diagnosis, screening decisions, triage, or treatment.
- Grade 3 and grade 4 recall are low; predictions and confidence scores can be wrong or misleading.
- Grad-CAM is an experimental visualization and does not establish why a model prediction is correct.
- Image quality checks are heuristics and do not verify that an upload is a retinal image.
- Guidance is evidence-bounded when sources are available and falls back safely when the retrieval index or `GROQ_API_KEY` is unavailable.
- Demo IDs are not authentication. Do not place patient identifiers or other sensitive information in demo fields or reviewer notes.
