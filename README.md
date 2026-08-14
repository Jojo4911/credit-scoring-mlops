---
title: Credit Scoring MLOps
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

[English](README.md) · [Français](README.fr.md)

[![CI/CD](https://github.com/Jojo4911/credit-scoring-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Jojo4911/credit-scoring-mlops/actions/workflows/ci-cd.yml)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live-blue)](https://huggingface.co/spaces/JonathanFernandez/pret_a_depenser)

# Credit Scoring MLOps

Production deployment of a credit default scoring model (LightGBM, 20 features) as an API on Hugging Face Spaces : two-stage CI/CD pipeline, 94% pytest coverage, data drift monitoring (Evidently) on simulated production traffic.

**Live demo** : https://huggingface.co/spaces/JonathanFernandez/pret_a_depenser

## Model

- **Algorithm** : LightGBM (sklearn pipeline : SimpleImputer → MinMaxScaler → LGBMClassifier)
- **Features** : 20 numerical variables (external scores, credit history, client data)
- **Decision threshold** : 0.48 (optimised on a business cost function : FN×10 + FP×1)
- **Origin** : Project 6, MLflow `Credit_Scoring_Model@champion` (version 60)

## Repository structure

```
├── src/
│   ├── __init__.py
│   └── app.py                        # Gradio scoring API
├── scripts/
│   ├── extract_demo_examples.py      # Generates data/demo_examples.json
│   ├── simulate_production_data.py
│   └── profile_inference.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── notebooks/
│   └── data_drift_analysis.ipynb
├── models/
│   └── model.joblib                  # sklearn pipeline (5.5 MB)
├── data/                              # Production and drift reference data (not versioned)
│   ├── logging.csv                   # API call logs (auto-generated)
│   └── demo_examples.json            # Preloaded demo examples (versioned)
├── docs/
│   ├── notes.md
│   └── screenshots/
│       ├── pipeline_success.png
│       └── hf_space_running.png
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── .gitignore
├── .dockerignore
├── .env.example
├── pyproject.toml
├── uv.lock
├── README.md
└── README.fr.md
```

## Installation

```bash
git clone https://github.com/Jojo4911/credit-scoring-mlops.git
cd credit-scoring-mlops
uv sync
```

## Usage

Run the API locally :

```bash
uv run python src/app.py
```

The Gradio interface opens in the browser, with three preloaded examples (approved, declined, borderline near the threshold) so the demo returns a meaningful result from the first click. The API accepts the following 20 features (all optional, missing values are median-imputed by the sklearn pipeline) :

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `EXT_SOURCE_2` | float | Normalised external score 2 |
| 2 | `EXT_SOURCE_3` | float | Normalised external score 3 |
| 3 | `EXT_SOURCE_1` | float | Normalised external score 1 |
| 4 | `BUREAU_BUREAU_DEBT_CREDIT_RATIO_MAX` | float | Max debt to credit ratio (credit bureau) |
| 5 | `APP_PAYMENT_RATE` | float | Annual payment rate |
| 6 | `AMT_ANNUITY` | float | Loan annuity amount |
| 7 | `DAYS_EMPLOYED` | int | Days in current job |
| 8 | `AMT_GOODS_PRICE` | float | Price of financed goods |
| 9 | `INSTAL_INSTAL_DAYS_LATE_MAX` | int | Max late payment, in days |
| 10 | `NAME_EDUCATION_TYPE_Higher education` | binary (0/1) | Higher education |
| 11 | `NAME_FAMILY_STATUS_Married` | binary (0/1) | Married |
| 12 | `PREV_PREV_APP_CREDIT_RATIO_MEAN` | float | Mean approved / requested credit ratio |
| 13 | `POS_NB_ENTRIES` | int | Number of POS entries |
| 14 | `DAYS_BIRTH` | int | Age in days (negative value) |
| 15 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | int | Max due date, first version (previous applications) |
| 16 | `POS_CNT_INSTALMENT_FUTURE_MEAN` | float | Mean remaining installments (previous credit) |
| 17 | `INSTAL_INSTAL_PAYMENT_RATIO_MEAN` | float | Mean paid / due ratio |
| 18 | `PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN` | float | Mean share of refused applications |
| 19 | `BUREAU_DAYS_ENDDATE_FACT_MAX` | int | Days since last closed credit (bureau) |
| 20 | `INSTAL_NB_PAYMENTS` | int | Number of past installment payments |

**Outputs** :
- **Verdict** : "Approved" (probability < 0.48) or "Declined" (probability ≥ 0.48)
- **Default probability** : value between 0 and 1

A warning is shown when more than 10 features are missing.

### Demo examples

The 3 examples shown in the interface are extracted from the training dataset, not hand-crafted, so the displayed probabilities are realistic :

```bash
uv run python -m scripts.extract_demo_examples
```

Writes `data/demo_examples.json` (versioned, unlike the rest of `data/`), loaded automatically by `src/app.py` at startup.

## Logging

Every request is automatically recorded in `data/logging.csv` with :
- The 20 input features
- The predicted probability
- The decision (approved/declined)
- The timestamp
- The model's inference time (in seconds)

These logs are the basis for the data drift analysis and performance monitoring.

## Tests

```bash
uv run --group dev pytest --cov=src --cov-report=term-missing
```

Current coverage : 94% (uncovered lines : CSV write error handling, `__main__` entry point).

## Deployment

The API is automatically deployed to Hugging Face Spaces via the CI/CD pipeline.

- **Public Space** : https://huggingface.co/spaces/JonathanFernandez/pret_a_depenser
- **Trigger** : push to `main` via GitHub Actions
- **Pipeline** : tests → Docker build → upload to HF Spaces via `huggingface_hub`

![CI/CD pipeline passing](https://raw.githubusercontent.com/Jojo4911/credit-scoring-mlops/main/docs/screenshots/pipeline_success.png)

The model (5.5 MB) is versioned with Git LFS on GitHub. Deployment to Hugging Face uses the `huggingface_hub.upload_folder()` API rather than a classic Git push, since HF Spaces requires their Xet system for binary files of this size.

To run locally with Docker :

```bash
docker build -t pret-a-depenser-api .
docker run -p 7860:7860 pret-a-depenser-api
```

## Monitoring

![Hugging Face Space running](https://raw.githubusercontent.com/Jojo4911/credit-scoring-mlops/main/docs/screenshots/hf_space_running.png)

### Production data storage

Every API request is logged to `data/logging.csv` with the 20 input features, the predicted probability, the decision, a status (`OK` / `INPUT_INCOMPLETE`), the timestamp and the inference time. This file is the production data store used for drift analysis.

The `INPUT_INCOMPLETE` status is set when more than 10 of the 20 features are missing. Requests with invalid input types (error raised before the probability is computed) are not logged : a known limitation, documented in `docs/notes.md`.

### Simulated production data

A script generates 150 simulated requests to feed the drift analysis :

```bash
uv run --group dev python -m scripts.simulate_production_data
```

Three groups : 80 normal requests, 50 requests with targeted drift on `AMT_ANNUITY` and `DAYS_EMPLOYED`, 20 requests with missing values. Clear `data/logging.csv` before rerunning for a clean run.

### Data drift analysis

The `notebooks/data_drift_analysis.ipynb` notebook compares production data (`data/logging.csv`, filtered on `STATUS == 'OK'`) against the training dataset (`data/app_train_enriched.parquet`) using **Evidently AI** (`DataDriftPreset`).

It also includes operational metrics : predicted score distribution, latency over time (median, P95, max), request status breakdown.

To run the notebook :

```bash
uv run --group dev jupyter notebook notebooks/data_drift_analysis.ipynb
```

The drift report is also exported as HTML to `docs/data_drift_report.html`.

## Performance

Profiled with `scripts/profile_inference.py` (average over 10 calls) :

- Model loading : ~896 ms (once, at startup)
- End-to-end inference (predict_proba + CSV logging) : ~253 ms
- predict_proba alone : ~1.4 ms

The main bottleneck is the synchronous CSV write. The LightGBM model itself is negligible. Optimisation not pursued for this PoC : latency is acceptable for credit scoring without a strict real-time constraint.

To reproduce the profiling :

```bash
uv run python -m scripts.profile_inference
```