# Minimal training script for UCI Credit Card Default (Taiwan, 2005)
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype

# ---- paths ----
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ---- config ----
FILENAME = "UCI_credit_card.csv"
TARGET   = "default.payment.next.month"



def main():
    # 1) read
    df = pd.read_csv(DATA_RAW / FILENAME)

    # 2) quick clean
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 3) mark categoricals (opsiyonel ama faydalı)
    for c in ["SEX", "EDUCATION", "MARRIAGE"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # 4) split X/y
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 5) column types
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if is_object_dtype(X[c]) or is_categorical_dtype(X[c])]

    # 6) preprocess pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # 7) split train/val/test
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 8) model
    clf = LogisticRegression(max_iter=1000)
    full = Pipeline([("prep", pre), ("clf", clf)])
    full.fit(X_tr, y_tr)

    # 9) metrics (validation)
    val_prob = full.predict_proba(X_val)[:, 1]
    metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, val_prob)),
        "val_pr_auc": float(average_precision_score(y_val, val_prob)),
        "val_brier": float(brier_score_loss(y_val, val_prob)),
    }

    # 10) save model & metrics
    joblib.dump(full, MODELS / "model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 11) save test.csv for evaluate.py
    te = X_te.copy()
    te[TARGET] = y_te.values
    te.to_csv(DATA_PROCESSED / "test.csv", index=False)

    print("Saved:", MODELS / "model.joblib")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
