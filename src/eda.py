"""
=============================================================
  Employee Performance Predictor — EDA Module
  Author  : Your Name
  Purpose : Exploratory Data Analysis — generate charts &
            statistical insights saved to images/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── global style ─────────────────────────────────────────────
PALETTE   = {1: "#EF4444", 2: "#F59E0B", 3: "#10B981"}   # Low/Med/High
LABEL_MAP = {1: "Low", 2: "Medium", 3: "High"}
COLORS    = list(PALETTE.values())
IMG_DIR   = Path("images")
IMG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, name: str):
    path = IMG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved → {path}")


# ─────────────────────────────────────────────────────────────
#  1. Target distribution
# ─────────────────────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame):
    counts = df["performance_score"].map(LABEL_MAP).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=COLORS, edgecolor="white", linewidth=1.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 10, str(int(b.get_height())),
                ha="center", fontweight="bold")
    ax.set_title("Employee Performance Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Level"); ax.set_ylabel("Count")
    _save(fig, "01_target_distribution.png")


# ─────────────────────────────────────────────────────────────
#  2. Performance by department
# ─────────────────────────────────────────────────────────────
def plot_dept_performance(df: pd.DataFrame):
    pivot = df.groupby(["department", "performance_score"]).size().unstack(fill_value=0)
    pivot.columns = [LABEL_MAP[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white",
               width=0.75)
    ax.set_title("Performance Level by Department", fontsize=14, fontweight="bold")
    ax.set_xlabel("Department"); ax.set_ylabel("Count")
    ax.legend(title="Performance", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "02_dept_performance.png")


# ─────────────────────────────────────────────────────────────
#  3. Correlation heatmap
# ─────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.drop("employee_id", errors="ignore")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, "03_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
#  4. Box plots: key features vs performance
# ─────────────────────────────────────────────────────────────
def plot_feature_boxplots(df: pd.DataFrame):
    features = [
        "training_hours_per_year", "manager_rating", "peer_rating",
        "absenteeism_days", "job_satisfaction_score", "monthly_salary"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat in zip(axes.flat, features):
        data = [df[df["performance_score"] == lvl][feat] for lvl in [1, 2, 3]]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Performance Level")
    fig.suptitle("Key Features vs Performance Level", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "04_feature_boxplots.png")


# ─────────────────────────────────────────────────────────────
#  5. Training hours distribution (KDE)
# ─────────────────────────────────────────────────────────────
def plot_training_kde(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for lvl, color in PALETTE.items():
        subset = df[df["performance_score"] == lvl]["training_hours_per_year"]
        subset.plot.kde(ax=ax, color=color, linewidth=2.5,
                        label=f"{LABEL_MAP[lvl]} Performers")
    ax.set_title("Training Hours Distribution by Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Hours / Year"); ax.set_ylabel("Density")
    ax.legend(); ax.set_xlim(left=0)
    _save(fig, "05_training_kde.png")


# ─────────────────────────────────────────────────────────────
#  6. Salary vs performance (violin)
# ─────────────────────────────────────────────────────────────
def plot_salary_violin(df: pd.DataFrame):
    df2 = df.copy()
    df2["Performance"] = df2["performance_score"].map(LABEL_MAP)
    order = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df2, x="Performance", y="monthly_salary",
                   order=order, palette=COLORS, ax=ax, inner="quartile")
    ax.set_title("Monthly Salary Distribution by Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Level"); ax.set_ylabel("Monthly Salary (₹/USD)")
    _save(fig, "06_salary_violin.png")


# ─────────────────────────────────────────────────────────────
#  7. Absenteeism vs performance
# ─────────────────────────────────────────────────────────────
def plot_absenteeism(df: pd.DataFrame):
    avg_abs = df.groupby("performance_score")["absenteeism_days"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([LABEL_MAP[i] for i in avg_abs.index], avg_abs.values,
           color=COLORS, edgecolor="white", linewidth=1.5)
    for i, (v) in enumerate(avg_abs.values):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_title("Avg Absenteeism Days by Performance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Performance Level"); ax.set_ylabel("Avg Days Absent")
    _save(fig, "07_absenteeism.png")


# ─────────────────────────────────────────────────────────────
#  8. Education level vs performance (stacked %)
# ─────────────────────────────────────────────────────────────
def plot_education(df: pd.DataFrame):
    pivot = (
        df.groupby(["education_level", "performance_score"])
          .size().unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.columns = [LABEL_MAP[c] for c in pivot_pct.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax,
                   color=COLORS, edgecolor="white", width=0.65)
    ax.set_title("Performance Distribution by Education Level (%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Education Level"); ax.set_ylabel("Percentage (%)")
    ax.legend(title="Performance"); plt.xticks(rotation=15)
    _save(fig, "08_education_performance.png")


# ─────────────────────────────────────────────────────────────
#  9. Age distribution
# ─────────────────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for lvl, color in PALETTE.items():
        df[df["performance_score"] == lvl]["age"].plot.hist(
            ax=ax, bins=20, alpha=0.55, color=color, label=LABEL_MAP[lvl]
        )
    ax.set_title("Age Distribution by Performance Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
    _save(fig, "09_age_distribution.png")


# ─────────────────────────────────────────────────────────────
#  RUN ALL
# ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    print("\n🔍 Running EDA …")
    plot_target_distribution(df)
    plot_dept_performance(df)
    plot_correlation_heatmap(df)
    plot_feature_boxplots(df)
    plot_training_kde(df)
    plot_salary_violin(df)
    plot_absenteeism(df)
    plot_education(df)
    plot_age_distribution(df)
    print("✅ All EDA charts saved to images/\n")


if __name__ == "__main__":
    from src.data_generation import generate_dataset
    df = generate_dataset()
    run_eda(df)
