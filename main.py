"""
=============================================================
  Employee Performance Predictor — Main Pipeline
  Author  : CH S K CHAITANYA
  Run     : python main.py
=============================================================
Orchestrates the complete ML pipeline:
  1. Generate synthetic dataset
  2. Preprocess & feature-engineer
  3. Run EDA (save charts)
  4. Train & evaluate models
  5. Save best model
  6. Print final summary
=============================================================
"""

import sys
from pathlib import Path

# ── Make sure src/ is importable ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generation import generate_dataset
from src.preprocessing   import clean, engineer_features, encode_and_scale, check_quality
from src.eda             import run_eda
from src.model           import run_modeling

import time

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     🏢  EMPLOYEE PERFORMANCE PREDICTOR                  ║
║         Using Data Analytics & Machine Learning          ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    t0 = time.time()

    # ── PHASE 1: Data Generation ─────────────────────────────
    print("━" * 55)
    print("PHASE 1 │ Generating Synthetic HR Dataset …")
    print("━" * 55)
    df_raw = generate_dataset(n=2000)
    df_raw.to_csv("data/employee_data.csv", index=False)
    print(f"✅ Raw data saved → data/employee_data.csv\n")

    # ── PHASE 2: Data Quality ─────────────────────────────────
    print("━" * 55)
    print("PHASE 2 │ Data Quality Check …")
    print("━" * 55)
    check_quality(df_raw)

    # ── PHASE 3: Preprocessing ───────────────────────────────
    print("━" * 55)
    print("PHASE 3 │ Cleaning + Feature Engineering …")
    print("━" * 55)
    df_clean = clean(df_raw)
    df_feat  = engineer_features(df_clean)
    df_feat.to_csv("data/processed_data.csv", index=False)
    print(f"✅ Processed data saved → data/processed_data.csv\n")

    # ── PHASE 4: EDA ─────────────────────────────────────────
    print("━" * 55)
    print("PHASE 4 │ Exploratory Data Analysis …")
    print("━" * 55)
    run_eda(df_clean)

    # ── PHASE 5: Encode & Split ──────────────────────────────
    print("━" * 55)
    print("PHASE 5 │ Encoding + Train/Test Split …")
    print("━" * 55)
    X_train, X_test, y_train, y_test, features, scaler = encode_and_scale(df_feat)

    # ── PHASE 6–8: Model Training & Evaluation ───────────────
    print("\n" + "━" * 55)
    print("PHASE 6 │ Model Training & Evaluation …")
    print("━" * 55)
    results, best_name, report = run_modeling(
        X_train, X_test, y_train, y_test, features, scaler
    )

    # ── Final Summary ────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "═" * 55)
    print("  ✅  PIPELINE COMPLETE")
    print("═" * 55)
    print(f"  Best Model     : {best_name}")
    print(f"  Test Accuracy  : {results[best_name]['accuracy'] * 100:.2f}%")
    print(f"  Cross-Val Acc  : {results[best_name]['cv_score']  * 100:.2f}%")
    print(f"  Runtime        : {elapsed:.1f} seconds")
    print("═" * 55)
    print("\n  📂 Outputs:")
    print("     data/employee_data.csv")
    print("     data/processed_data.csv")
    print("     models/best_model.pkl")
    print("     outputs/predictions.csv")
    print("     outputs/classification_report.json")
    print("     images/ (12 charts saved)")
    print("\n  🚀 Launch dashboard: streamlit run app.py\n")


if __name__ == "__main__":
    main()
