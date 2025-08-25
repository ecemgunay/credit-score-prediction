# Minimal training script for UCI Credit Card Default (Taiwan, 2005)
from pathlib import Path
import numpy as np
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
    #if "ID" in df.columns:
    #    df = df.drop(columns=["ID"])

    # 3) mark categoricals (opsiyonel ama faydalı)
    #for c in ["SEX", "EDUCATION", "MARRIAGE"]:
    #    if c in df.columns:
    #        df[c] = df[c].astype("category")

     # --- UCI quick clean & lightweight features ---
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # normalize weird category codes
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype("category")
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype("category")
    for c in ["SEX", "EDUCATION", "MARRIAGE"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    
    BILL_COLS  = [f"BILL_AMT{i}" for i in range(1, 7) if f"BILL_AMT{i}" in df.columns]
    PAY_COLS   = [f"PAY_AMT{i}"  for i in range(1, 7) if f"PAY_AMT{i}"  in df.columns]
    DELAY_COLS = [f"PAY_{i}"     for i in range(1, 7) if f"PAY_{i}"     in df.columns]

    # utilization (bills / limit)
    if "LIMIT_BAL" in df.columns and BILL_COLS:
        bills = df[BILL_COLS].fillna(0)
        limit = df["LIMIT_BAL"].replace(0, np.nan).astype(float)
        util = bills.div(limit, axis=0).clip(0, 10)
        df["util_avg"] = util.mean(axis=1)

    # payment ratio (payments / bills)
    if BILL_COLS and PAY_COLS:
        pays = df[PAY_COLS].fillna(0)
        bills = df[BILL_COLS].replace(0, np.nan).abs()
        pay_ratio = pays.div(bills + 1e-6, axis=0).clip(0, 5)
        df["pay_ratio_avg"] = pay_ratio.mean(axis=1)

    # delays
    if DELAY_COLS:
        delays = df[DELAY_COLS].fillna(0)
        df["max_delay"] = delays.max(axis=1)
        df["num_late"]  = (delays > 0).sum(axis=1)
        if "PAY_0" in df.columns:
            df["recent_delay"] = df["PAY_0"]
        elif "PAY_1" in df.columns:
            df["recent_delay"] = df["PAY_1"]

    # bill trend (linear slope)
    if BILL_COLS:
        x = np.arange(1, len(BILL_COLS) + 1)
        df["bill_trend"] = df[BILL_COLS].apply(
            lambda r: np.polyfit(x, r.values.astype(float), 1)[0], axis=1
        )

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
