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
> ## தமிழ் விளக்கம் (with English keywords kept as‑is)
>
> - இந்த project **SMS spam/phishing detection** க்கு ஒரு **end‑to‑end starter**.
> - **Pipeline**: `TfidfVectorizer` → `LogisticRegression` (binary classification).
> - **train.py**: உங்கள் `CSV` file (columns: `label`, `text`) எடுத்துக்கொண்டு **train/validation split** செய்து, **metrics** (accuracy, precision, recall, f1) print செய்து, **model.joblib** ஆக save பண்ணும்.
> - **predict.py**: ஒரு single message க்கு **label** (ham/spam) மற்றும் **probability** return செய்யும்.
> - **app.py**: **FastAPI** மூலம் `/predict` endpoint. JSON `{"text": "..."} ` அனுப்பினால் prediction தரும்.
>
> ### உங்கள் data format
> CSV with headers like:
> ```csv
> label,text
> ham,I'll call you later
> spam,You have won a FREE lottery! Click here now
> ```
>
> ### Notes
> - `requirements.txt` install பண்ணும் போது ஏதாவது error வந்தால், உங்கள் Python version >=3.9 இருக்கணும்.
> - Accuracy improve செய்ய: **class_weight**, **ngram_range**, **min_df**, **max_df**, **classifier** (e.g., LinearSVC) try பண்ணலாம்.
> - **Tamil/Indian languages** க்காக: `token_pattern`, or `indic-nlp` style tokenizers explore பண்ணலாம்.
>
> ### Next steps (ideas)
> - Add **MLflow** for experiment tracking.
> - Add **Dockerfile** & deploy on cloud (Render/Heroku/Vercel + serverless).
> - Add **streamlit** UI for quick demo.
>
> — Built for Bharath (aka “Jeeva”) to practice ML end‑to‑end 🙂
