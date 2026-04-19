"""
=============================================================
  Employee Performance Predictor — Preprocessing Module
  Author  : Your Name
  Purpose : Clean raw data, engineer features, encode &
            scale ready for ML training
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ── ordinal mappings (preserve natural order) ────────────────
EDUCATION_ORDER = {
    "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4
}
JOB_LEVEL_ORDER = {
    "Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5
}


def load_data(filepath: str = "data/employee_data.csv") -> pd.DataFrame:
    """Load the CSV and do a quick sanity check."""
    df = pd.read_csv(filepath)
    print(f"📂 Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def check_quality(df: pd.DataFrame) -> None:
    """Print a data quality report."""
    print("\n── Data Quality Report ────────────────────────────")
    print(f"  Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"  Data types     :\n{df.dtypes}")
    print("────────────────────────────────────────────────────\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop duplicates
    2. Fill missing numerical with median, categorical with mode
    3. Remove obvious outliers (salary < 500)
    """
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove rows with suspicious salary values
    df = df[df["monthly_salary"] >= 500]

    print(f"✅ Cleaned: {df.shape[0]} rows remain")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that improve model signal.

    New columns
    -----------
    salary_per_year_experience  : pay relative to tenure (efficiency)
    productivity_index          : projects ÷ years_at_company
    training_to_experience_ratio: training ÷ (years_at_company+1)
    avg_rating                  : mean of manager + peer rating
    loyalty_score               : years_at_company / age
    engagement_score            : mean(job_sat, wlb) / 10
    """
    eps = 1e-6  # avoid division by zero

    df["salary_per_year_experience"] = (
        df["monthly_salary"] / (df["years_at_company"] + eps)
    ).round(2)

    df["productivity_index"] = (
        df["num_projects_completed"] / (df["years_at_company"] + eps)
    ).round(3)

    df["training_to_experience_ratio"] = (
        df["training_hours_per_year"] / (df["years_at_company"] + 1)
    ).round(3)

    df["avg_rating"] = (
        (df["manager_rating"] + df["peer_rating"]) / 2
    ).round(2)

    df["loyalty_score"] = (
        df["years_at_company"] / df["age"]
    ).round(3)

    df["engagement_score"] = (
        (df["job_satisfaction_score"] + df["work_life_balance_score"]) / 20
    ).round(3)

    print(f"✅ Feature engineering done — {df.shape[1]} total features")
    return df


def encode_and_scale(
    df: pd.DataFrame,
    target_col: str = "performance_score",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Encode categoricals, scale numerics, split train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = df.copy()

    # Drop identifier column — not a feature
    df.drop(columns=["employee_id"], errors="ignore", inplace=True)

    # Ordinal encode education & job level (preserve rank)
    df["education_level"] = df["education_level"].map(EDUCATION_ORDER)
    df["job_level"]       = df["job_level"].map(JOB_LEVEL_ORDER)

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, columns=["gender", "department"], drop_first=False)

    # Separate features / target
    X = df.drop(columns=[target_col])
    y = df[target_col] - 1          # shift to 0-indexed (0=Low, 1=Med, 2=High)

    feature_names = X.columns.tolist()

    # Scale numerics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"✅ Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"   Class balance in train: {dict(pd.Series(y_train).value_counts().sort_index())}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def run_preprocessing(filepath="data/employee_data.csv"):
    """Convenience wrapper used by main.py."""
    df_raw  = load_data(filepath)
    check_quality(df_raw)
    df_clean = clean(df_raw)
    df_feat  = engineer_features(df_clean)
    splits   = encode_and_scale(df_feat)
    return df_clean, df_feat, splits


if __name__ == "__main__":
    run_preprocessing()
