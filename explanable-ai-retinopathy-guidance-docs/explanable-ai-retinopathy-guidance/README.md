# Explanable AI for Retinopathy & Guidance

> Educational portfolio project for diabetic retinopathy (DR) screening research. **Not a diagnostic device and not for clinical use.**

## Overview

This project explores an explainable computer-vision pipeline that classifies retinal fundus images into five diabetic-retinopathy severity grades, visualizes image regions associated with a prediction using Grad-CAM, and presents guidance grounded in retrieved public clinical-guideline excerpts.

The application is designed to make model output easier to inspect—not to replace an ophthalmologist or make treatment decisions.

## Goals

- Classify retinal fundus images into five APTOS 2019 labels.
- Reject unsuitable images before inference using a simple, calibrated image-quality gate.
- Display the predicted grade, model confidence, a low-confidence warning, and a Grad-CAM overlay.
- Retrieve relevant passages from selected public guideline PDFs using sentence-transformers and FAISS.
- Ask an LLM to summarize only retrieved passages and cite them.
- Save a demo screening record and its image hash in MongoDB Atlas, then display demo-ID history.
- Evaluate on an untouched held-out test split and report measured results only.

## Severity labels

| Label ID | Label |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The blueprint defines **referable range** as grade 2 or higher. This is a project convention, not a clinical determination.

## Architecture at a glance

1. Streamlit receives a retinal image and a **demo patient ID only**.
2. The quality gate rejects images that fail calibrated checks.
3. Shared preprocessing crops the black border, resizes to 256×256, and applies CLAHE.
4. EfficientNet-B0 predicts one of five grades.
5. Grad-CAM creates a heatmap overlay for the predicted class.
6. FAISS retrieves guideline chunks relevant to the predicted grade.
7. Groq summarizes only the retrieved excerpts and includes citations; missing evidence triggers a safe fallback.
8. MongoDB stores the screening metadata and SHA-256 image hash, not the uploaded image.
9. Streamlit displays the result, sources, and demo history.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [WORKFLOW.md](WORKFLOW.md).

## Tech stack

- Python, PyTorch, torchvision, EfficientNet-B0
- OpenCV, Albumentations, NumPy, pandas, scikit-learn
- pytorch-grad-cam
- sentence-transformers (`all-MiniLM-L6-v2`), FAISS, pypdf
- Groq API
- MongoDB Atlas with PyMongo
- Streamlit
- Google Colab T4 for training

## Repository layout

```text
explanable-ai-retinopathy-guidance/
├── streamlit_app.py
├── app/
│   └── core/
│       ├── preprocess.py
│       ├── model.py
│       ├── gradcam.py
│       ├── inference.py
│       ├── rag.py
│       ├── llm.py
│       └── db.py
├── training/
│   ├── prepare_data.py
│   └── train.py
├── scripts/
│   ├── build_index.py
│   └── smoke.py
├── rag_corpus/
│   ├── SOURCES.md
│   └── <approved guideline PDFs>
├── index/                 # generated locally; do not commit by default
├── models/                # generated checkpoints/metrics; review before committing
├── data/                  # local data; ignored by Git
├── tests/
├── README.md
├── ARCHITECTURE.md
├── WORKFLOW.md
├── AGENTS.md
├── CLAUDE.md
├── PROMPTS.md
├── SKILLS.md
├── TESTING.md
├── DATA_CARD.md
├── SECURITY.md
├── .env.example
└── .gitignore
```

Create `__init__.py` files in `app/`, `app/core/`, `training/`, and `scripts/`. Run commands from the repository root.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

Create `requirements.txt` from the stack in the execution blueprint, install it, then pin the versions that work in your environment. PyTorch installation can vary by CPU/CUDA platform.

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in credentials locally. Never commit `.env` or paste credentials into source files, issues, or chat.

### 4. Prepare data and train

Follow the APTOS 2019 competition/data terms and download instructions. The raw dataset is not included in this repository.

```bash
python -m training.prepare_data
python -m training.train
```

The training pipeline should create `models/best.pt` and `models/test_metrics.json`. Check that train/validation/test splits are stratified and that the test set is not used for model selection.

### 5. Build the guideline index

Place permitted, text-extractable guideline PDFs in `rag_corpus/` and document their title, publisher, URL, version/date, access date, and usage terms in `rag_corpus/SOURCES.md`.

```bash
python -m scripts.build_index
```

Inspect retrieved passages manually and tune the score threshold against examples before trusting the retrieval stage.

### 6. Smoke-test and launch

```bash
python -m scripts.smoke path/to/retinal_image.png demo-001
streamlit run streamlit_app.py
```

Use demo IDs only. Do not enter real patient identifiers or real patient records.

## Evaluation

Report only values produced by your own untouched test split. Include at minimum:

- Accuracy
- Quadratic weighted kappa (QWK)
- Referable-DR sensitivity and specificity, using the project's grade ≥2 definition
- Confusion matrix and per-class performance where possible
- Test-set class counts and limitations, especially the small severe-class sample count

Do not copy illustrative or expected metrics from tutorials. Record the actual split method, random seed, model checkpoint, package versions, and evaluation date.

## Safety, privacy, and limitations

- This is a student portfolio/research demonstration, not a medical device.
- Predictions, confidence values, and Grad-CAM heatmaps can be wrong or misleading.
- Grad-CAM is a visualization of model sensitivity, not proof of causal reasoning or clinical correctness.
- Model confidence is not a calibrated probability unless calibration has been evaluated.
- APTOS is one public dataset; external validation and clinical validation have not been performed by default.
- Guideline-based text is informational and must not invent diagnoses, medications, thresholds, or follow-up intervals.
- If retrieval returns no relevant passages or the LLM is unavailable, show a safe fallback.
- Store demo IDs and an image hash only; do not persist the uploaded image.
- Keep the not-for-clinical-use disclaimer visible in the interface.

## Development status

The supplied execution blueprint is a plan, not proof that the code has run. Update this section honestly as work progresses:

- [ ] Data preparation and quality-gate calibration
- [ ] Training and held-out test evaluation
- [ ] Grad-CAM inspection
- [ ] Guideline source review and FAISS index
- [ ] Grounded LLM and fallback tests
- [ ] MongoDB persistence and history
- [ ] Streamlit end-to-end flow
- [ ] Edge-case tests and demo recording

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — components and data flow
- [WORKFLOW.md](WORKFLOW.md) — 9-day build plan and operating steps
- [AGENTS.md](AGENTS.md) — repository-wide instructions for coding agents
- [CLAUDE.md](CLAUDE.md) — Claude Code project guidance
- [PROMPTS.md](PROMPTS.md) — reusable development prompts
- [SKILLS.md](SKILLS.md) — required engineering competencies and acceptance criteria
- [TESTING.md](TESTING.md) — test strategy and manual checklist
- [DATA_CARD.md](DATA_CARD.md) — dataset, intended use, and known risks
- [SECURITY.md](SECURITY.md) — secrets, data handling, and reporting issues
- [rag_corpus/SOURCES.md](rag_corpus/SOURCES.md) — provenance of guideline documents
