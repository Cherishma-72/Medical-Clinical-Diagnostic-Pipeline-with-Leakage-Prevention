# 🫀 Heart Failure Survival Predictor — Streamlit Dashboard

Live, interactive companion to the "Medical Clinical Diagnostic Pipeline with Leakage Prevention"
project. Shows EDA, benchmarks 5 models, demonstrates data leakage live, and lets you enter a
patient's clinical values to get a real-time survival risk prediction.

## Files
- `app.py` — the Streamlit application (single file)
- `requirements.txt` — Python dependencies
- `heart_failure_clinical_records_dataset.csv` — dataset (place in the same folder as `app.py`)

## 1. Run it locally (fastest way to demo it)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`. Nothing needs to be uploaded —
the app automatically loads `heart_failure_clinical_records_dataset.csv` from the same folder. If
that file isn't found, the sidebar will let you upload it manually.

## 2. Deploy it for free (to share a public link with your lecturer)

**Streamlit Community Cloud** (easiest, free):
1. Create a free GitHub repo and push these 3 files into it (`app.py`, `requirements.txt`, the CSV).
2. Go to **share.streamlit.io** → sign in with GitHub → "New app".
3. Select your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. You'll get a public URL like `https://your-app-name.streamlit.app` you can share
   or put in your project report / submit alongside your notebook.

## What's inside the dashboard

| Tab | What it shows |
|---|---|
| 🏠 Overview | Project summary and dataset snapshot |
| 📊 EDA | Target distribution, correlation heatmap, feature distributions by outcome |
| 🤖 Model Comparison | 5 algorithms benchmarked via leak-free cross-validation + statistical significance test |
| 🎯 Final Model | Tuned Random Forest: confusion matrix, ROC/PR curves, bootstrap confidence interval |
| 🔍 Leakage Demo | Live side-by-side comparison of a correct vs. a leaky (scaled-before-split) pipeline |
| 🧬 Feature Importance | Permutation importance + the `time`-feature temporal leakage investigation |
| 🩺 Live Predictor | Enter a patient's values and get a real-time risk prediction with a gauge chart |

## Notes
- The first load takes ~15-30 seconds to train all 5 models and tune the Random Forest — this only
  happens once (results are cached), so switching tabs afterward is instant.
- The Live Predictor tab is clearly labelled as an academic demonstration, not a validated clinical
  tool — this framing is intentional and worth mentioning if your lecturer asks about deployment ethics.
