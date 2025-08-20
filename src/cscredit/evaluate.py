# Minimal evaluate script (uses test.csv created by train.py)
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TARGET = "is_default"

def main():
    model = joblib.load(MODELS / "model.joblib")
    df = pd.read_csv(DATA_PROCESSED / "test.csv")
    y_true = df[TARGET].values
    X = df.drop(columns=[TARGET])
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "holdout_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "holdout_pr_auc": float(average_precision_score(y_true, y_prob)),
        "holdout_brier": float(brier_score_loss(y_true, y_prob)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics_holdout.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)

if __name__ == "__main__":
    main()
