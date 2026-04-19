"""
=============================================================
  Employee Performance Predictor — Model Module
  Author  : Your Name
  Purpose : Train, evaluate, and save ML models.
            Compares Logistic Regression, Random Forest,
            Gradient Boosting. Best model saved to models/
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

IMG_DIR   = Path("images")
MODEL_DIR = Path("models")
OUT_DIR   = Path("outputs")
for d in [IMG_DIR, MODEL_DIR, OUT_DIR]:
    d.mkdir(exist_ok=True)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
COLORS    = ["#EF4444", "#F59E0B", "#10B981"]


# ─────────────────────────────────────────────────────────────
#  1. Define models
# ─────────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
    }


# ─────────────────────────────────────────────────────────────
#  2. Train & compare all models
# ─────────────────────────────────────────────────────────────
def compare_models(X_train, X_test, y_train, y_test) -> dict:
    """Train all models, return dict of results."""
    models  = get_models()
    results = {}

    print("\n📊 Model Comparison")
    print("─" * 60)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(clf, X_train, y_train,
                                   cv=5, scoring="accuracy").mean()

        results[name] = {
            "model"   : clf,
            "accuracy": round(acc, 4),
            "cv_score": round(cv_score, 4),
            "y_pred"  : y_pred,
        }
        print(f"  {name:<25}  Test Acc: {acc:.4f}  |  CV Acc: {cv_score:.4f}")

    print("─" * 60)
    # Identify best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best model: {best_name}  ({results[best_name]['accuracy']:.4f})\n")
    return results, best_name


# ─────────────────────────────────────────────────────────────
#  3. Detailed evaluation of best model
# ─────────────────────────────────────────────────────────────
def evaluate_best(results: dict, best_name: str,
                  X_test, y_test, feature_names: list):
    clf    = results[best_name]["model"]
    y_pred = results[best_name]["y_pred"]
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

    # ── Classification report ───────────────────────────────
    report = classification_report(y_test, y_pred,
                                   target_names=labels, output_dict=True)
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save report JSON
    with open(OUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Confusion matrix ────────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    fig.savefig(IMG_DIR / "10_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved → images/10_confusion_matrix.png")

    # ── Model comparison bar chart ───────────────────────────
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    cvs   = [results[n]["cv_score"]  for n in names]
    x     = np.arange(len(names))
    w     = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Test Accuracy",  color="#3B82F6", alpha=0.85)
    bars2 = ax.bar(x + w/2, cvs,  w, label="CV Accuracy",    color="#8B5CF6", alpha=0.85)
    for b in bars1 + bars2:
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + 0.003, f"{b.get_height():.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.5, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    fig.savefig(IMG_DIR / "11_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved → images/11_model_comparison.png")

    # ── Feature importance (RF / GB) ────────────────────────
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=feature_names)
        top20 = fi.nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(9, 8))
        bars = ax.barh(top20.index, top20.values,
                       color="#6366F1", alpha=0.85, edgecolor="white")
        ax.set_title(f"Top-20 Feature Importances — {best_name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        for b in bars:
            ax.text(b.get_width() + 0.001,
                    b.get_y() + b.get_height()/2,
                    f"{b.get_width():.3f}",
                    va="center", fontsize=8)
        fig.savefig(IMG_DIR / "12_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  💾 Saved → images/12_feature_importance.png")

        fi.to_csv(OUT_DIR / "feature_importances.csv", header=["importance"])

    return report


# ─────────────────────────────────────────────────────────────
#  4. Save best model + scaler
# ─────────────────────────────────────────────────────────────
def save_model(clf, scaler, feature_names: list, model_name: str):
    artifact = {
        "model"        : clf,
        "scaler"       : scaler,
        "feature_names": feature_names,
        "model_name"   : model_name,
        "label_map"    : LABEL_MAP,
    }
    out = MODEL_DIR / "best_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"✅ Model saved → {out}")


# ─────────────────────────────────────────────────────────────
#  5. Predict on new single employee (HR use-case demo)
# ─────────────────────────────────────────────────────────────
def predict_single(employee_dict: dict, model_path="models/best_model.pkl") -> str:
    """
    Predict performance class for ONE new employee record.

    Parameters
    ----------
    employee_dict : pre-processed feature dict (must match training features)
    """
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    clf           = artifact["model"]
    scaler        = artifact["scaler"]
    feature_names = artifact["feature_names"]
    label_map     = artifact["label_map"]

    row = pd.DataFrame([employee_dict]).reindex(columns=feature_names, fill_value=0)
    row_scaled  = scaler.transform(row)
    pred        = clf.predict(row_scaled)[0]
    proba       = clf.predict_proba(row_scaled)[0]

    label  = label_map[pred]
    conf   = proba[pred] * 100
    return label, conf, dict(zip([label_map[i] for i in range(3)], proba))


# ─────────────────────────────────────────────────────────────
#  6. Save prediction results to CSV
# ─────────────────────────────────────────────────────────────
def save_predictions(y_test, y_pred, out_path="outputs/predictions.csv"):
    df = pd.DataFrame({
        "actual"   : [LABEL_MAP[i] for i in y_test],
        "predicted": [LABEL_MAP[i] for i in y_pred],
        "correct"  : [a == p for a, p in zip(y_test, y_pred)],
    })
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_modeling(X_train, X_test, y_train, y_test,
                 feature_names: list, scaler):
    results, best_name = compare_models(X_train, X_test, y_train, y_test)
    report = evaluate_best(results, best_name, X_test, y_test, feature_names)
    save_model(results[best_name]["model"], scaler, feature_names, best_name)
    save_predictions(y_test, results[best_name]["y_pred"])
    return results, best_name, report
