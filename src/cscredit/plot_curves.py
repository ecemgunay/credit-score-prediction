from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL = joblib.load(ROOT / "models" / "model.joblib")
df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
TARGET = "is_default"

y_true = df[TARGET].values
X = df.drop(columns=[TARGET])
y_prob = MODEL.predict_proba(X)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend()
( ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(ROOT / "reports" / "figures" / "roc_curve.png", dpi=150)

# Calibration
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0,1],[0,1], "--")
plt.xlabel("Predicted probability"); plt.ylabel("Observed rate")
plt.savefig(ROOT / "reports" / "figures" / "calibration.png", dpi=150)

print("Saved plots to reports/figures/")
