# SMS Spam & Phishing Detection – AI Starter (Logistic Regression + TF‑IDF)

> **Quickstart**
>
> ```bash
> # 1) Create & activate a virtualenv (recommended)
> python -m venv .venv
> source .venv/bin/activate  # Windows: .venv\Scripts\activate
>
> # 2) Install deps
> pip install -r requirements.txt
>
> # 3) (Optional) Inspect sample data
> python -m pip install pandas
> python - << 'PY'
> import pandas as pd;print(pd.read_csv('data/sample_sms.csv').head())
> PY
>
> # 4) Train
> python src/train.py --data data/sample_sms.csv --out models/model.joblib
>
> # 5) Predict from CLI
> python src/predict.py --model models/model.joblib --text "Congratulations! You won a prize, click this link"
>
> # 6) Run API (FastAPI + Uvicorn)
> uvicorn src.app:app --reload --port 9000
> # POST http://127.0.0.1:9000/predict  with JSON: {"text": "free gift claim now"}
> ```
>
> **Files**
>
> - `src/train.py` – trains a TF‑IDF + LogisticRegression model
> - `src/predict.py` – loads model & predicts labels/probabilities
> - `src/app.py` – FastAPI server with `/predict` endpoint
> - `src/utils.py` – reusable text cleaning helpers
> - `data/sample_sms.csv` – tiny toy dataset to test the pipeline
> - `models/` – model artifacts will be saved here
>
> ---
>
