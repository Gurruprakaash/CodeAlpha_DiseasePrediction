# ── 1. IMPORTS ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
 accuracy_score, precision_score, recall_score, f1_score,
 confusion_matrix, classification_report, roc_auc_score, roc_curve
)

try:
 from xgboost import XGBClassifier
 XGBOOST_AVAILABLE = True
except ImportError:
 XGBOOST_AVAILABLE = False
 print(" XGBoost not installed. Skipping XGBoost model.")

print("=" * 60)
print(" CodeAlpha | Task 4: Disease Prediction from Medical Data")
print("=" * 60)


# ── 2. LOAD CSV DATASET ──────────────────────────────────────
# heart.csv — UCI Heart Disease dataset
# target: 1 = heart disease present, 0 = no heart disease
CSV_PATH = "heart.csv"
df = pd.read_csv(CSV_PATH)

print(f"\n Loaded dataset: {CSV_PATH}")
print(f" Samples : {df.shape[0]}")
print(f" Features : {df.shape[1] - 1}")
print(f" Disease (1): {(df.target==1).sum()} | "
 f"No Disease (0): {(df.target==0).sum()}")


# ── 3. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("\n── Column Overview ──────────────────────────────────────")
print(df.describe().T[["mean", "std", "min", "max"]].round(2))
print(f"\n Missing values: {df.isnull().sum().sum()}")


# ── 4. FEATURE ENGINEERING ───────────────────────────────────
# Domain-informed derived features
df["age_chol_ratio"] = df["chol"] / df["age"]
df["high_bp"] = (df["trestbps"] > 140).astype(int)
df["high_chol"] = (df["chol"] > 240).astype(int)
df["max_hr_reserve"] = 220 - df["age"] - df["thalach"]

print(" Feature engineering: 4 new features added "
 "(age_chol_ratio, high_bp, high_chol, max_hr_reserve)")


# ── 5. TRAIN / TEST SPLIT ────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\n Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

NEEDS_SCALE = {"Logistic Regression", "SVM"}


# ── 6. TRAIN MULTIPLE MODELS ──────────────────────────────────
models = {
 "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
 "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42),
 "SVM" : SVC(probability=True, random_state=42),
}
if XGBOOST_AVAILABLE:
 models["XGBoost"] = XGBClassifier(
 n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
 )

results = {}

print("\n── Model Training & Evaluation ─────────────────────────")
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
 f"{'F1':>8} {'ROC-AUC':>9}")
print("-" * 70)

for name, model in models.items():
 Xtr = X_train_sc if name in NEEDS_SCALE else X_train
 Xte = X_test_sc if name in NEEDS_SCALE else X_test

 model.fit(Xtr, y_train)
 y_pred = model.predict(Xte)
 y_proba = model.predict_proba(Xte)[:, 1]

 acc = accuracy_score(y_test, y_pred)
 prec = precision_score(y_test, y_pred, zero_division=0)
 rec = recall_score(y_test, y_pred, zero_division=0)
 f1 = f1_score(y_test, y_pred, zero_division=0)
 auc = roc_auc_score(y_test, y_proba)

 results[name] = dict(
 model=model, y_pred=y_pred, y_proba=y_proba,
 acc=acc, prec=prec, rec=rec, f1=f1, auc=auc
 )
 print(f"{name:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} "
 f"{f1:>8.4f} {auc:>9.4f}")


# ── 7. BEST MODEL DETAILED REPORT ────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]

print(f"\n Best Model: {best_name} (ROC-AUC = {best['auc']:.4f})")
print("\nClassification Report:")
print(classification_report(
 y_test, best["y_pred"],
 target_names=["No Disease", "Disease"]
))


# ── 8. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 4 — Disease Prediction Results (Heart Disease)",
 fontsize=14, fontweight="bold")

# (a) Confusion Matrix
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(
 cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
 xticklabels=["No Disease", "Disease"],
 yticklabels=["No Disease", "Disease"]
)
axes[0].set_title(f"Confusion Matrix — {best_name}")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# (b) ROC Curves
for name, res in results.items():
 fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
 axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Random Classifier")
axes[1].set_title("ROC Curves — All Models")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# (c) Feature Importance (Random Forest)
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
axes[2].set_title("Top 10 Feature Importances (Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("task4_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n Visualisation saved task4_results.png")


# ── 9. CROSS-VALIDATION ───────────────────────────────────────
print("\n── 5-Fold Cross-Validation (ROC-AUC) ───────────────────")
for name, model in models.items():
 Xtr = X_train_sc if name in NEEDS_SCALE else X_train
 cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="roc_auc")
 print(f" {name:<22} Mean={cv.mean():.4f} Std=±{cv.std():.4f}")


# ── 10. SAMPLE PREDICTION ─────────────────────────────────────
print("\n── Sample Prediction (first test patient) ──────────────")
sample = X_test.iloc[[0]]
sample_sc = scaler.transform(sample)

for name, res in results.items():
 inp = sample_sc if name in NEEDS_SCALE else sample
 pred = res["model"].predict(inp)[0]
 proba = res["model"].predict_proba(inp)[0][pred]
 label = "Disease " if pred == 1 else "No Disease "
 print(f" {name:<22} {label} (confidence: {proba:.2%})")

actual = "Disease " if y_test.iloc[0] == 1 else "No Disease "
print(f"\n Actual label: {actual}")

print("\n Task 4 complete!\n")
