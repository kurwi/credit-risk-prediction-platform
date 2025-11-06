# Requires: Windows PowerShell
# Purpose: One-click setup and launch for Credit Risk Prediction Platform

param(
    [int]$Port = 8524
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Ensuring virtual environment exists..."
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "[2/4] Activating virtual environment and installing dependencies..."
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[3/4] Ensuring trained artifacts are present..."
$ModelPath = Join-Path (Get-Location) "models\credit_risk_model.pkl"
$MetadataPath = Join-Path (Get-Location) "models\model_metadata.json"
if (!(Test-Path $ModelPath) -or !(Test-Path $MetadataPath)) {
    Write-Host "Artifacts missing. Training model..."
    python scripts/train_and_save_model.py
}

Write-Host "[4/4] Launching Streamlit on port $Port..."
python -m streamlit run app.py --server.port $Port --server.headless true
