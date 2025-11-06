# Credit Risk Prediction Platform

A professional Streamlit application for credit risk assessment. It loads pre-trained artifacts, scores single applicants and batches, supports scenario analysis with approval-boundary heatmaps and minimum-change recommendations, and provides a portfolio dashboard for exposure/expected-loss analysis.

## What it does

- Predicts probability of default (PD) using an XGBoost model and a tuned decision threshold
- Single Applicant assessment with a clean form, risk gauge, and decision summary
- Batch Processing of CSV files with aggregate metrics and downloadable results
- Scenario Analysis with a 2D approval boundary heatmap and “minimum changes to approve” guidance
- Portfolio Dashboard with exposure, expected loss, and segment-level analytics

## Requirements

- Windows, macOS, or Linux
- Python 3.13 recommended (artifacts trained in this environment)

## Quick start (Windows PowerShell)

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If models are missing, train and save artifacts
python scripts/train_and_save_model.py

# Launch Streamlit app (default port 8501; use --server.port to change)
python -m streamlit run app.py --server.port 8524
```

Open the local URL printed in the terminal (e.g., http://localhost:8524).

## App modes

1) Single Applicant
- Adjust applicant features via sliders and selectors
- See PD, risk gauge, and Approve/Decline based on the tuned threshold
- Executive summary and sensitivity insights

2) Batch Processing
- Upload a CSV for bulk scoring
- See approval/decline counts, risk distribution, and export the scored file

3) Scenario Analysis
- Explore a 2D approval-boundary heatmap (e.g., Credit Score vs Debt-to-Income)
- Get “minimum changes” recommendations that shift a case from decline to approval

4) Portfolio Dashboard
- Simulate portfolios by risk strategy profile
- View exposure and expected loss metrics, distributions, and segment breakdowns

## Project structure (high level)

```
app.py                      # Streamlit UI (four modes)
scripts/
    train_and_save_model.py   # Train model, tune threshold, persist artifacts
src/
    data/processor.py         # Preprocessing (encoders, scaler, feature names)
    models/credit_model.py    # XGBoost wrapper, metrics, SHAP (optional)
    data/generator.py         # Synthetic data for training
models/
    credit_risk_model.pkl     # Trained model
    model_metadata.json       # Threshold, metrics, feature info
    preprocessor/             # Scaler, encoders, feature lists
data/                       # Sample/processed data (optional)
```

## CSV schema (batch mode)

Your CSV should include the model’s expected feature columns. If unsure, see `models/feature_info.json` (created by the training script) for `feature_names`, and the lists of numerical vs categorical columns.

## Troubleshooting

- Port in use: Run with a different port, e.g. `--server.port 8525`.
- Numpy pickle error (numpy._core): Ensure artifacts are trained and the app runs in the same Python version; run `python scripts/train_and_save_model.py` in your active environment.
- Missing artifacts: Re-run the training script to regenerate `models/`.

## Development

Run tests and linters:

```powershell
pytest -q
```

## License

MIT. See `LICENSE` for details.
