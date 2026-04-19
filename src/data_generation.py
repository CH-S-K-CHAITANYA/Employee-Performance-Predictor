"""
=============================================================
  Employee Performance Predictor — Data Generation Module
  Author  : Your Name
  Purpose : Generate a realistic synthetic HR dataset
=============================================================
No real company data is needed. We simulate 2 000 employees
with features that actually drive performance in real firms.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── reproducible results ───────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── constants ──────────────────────────────────────────────
N_EMPLOYEES = 2_000

DEPARTMENTS   = ["Engineering", "Sales", "HR", "Finance",
                 "Marketing", "Operations", "Product", "Support"]
EDUCATION     = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_LEVELS    = ["Junior", "Mid", "Senior", "Lead", "Manager"]
GENDER        = ["Male", "Female", "Non-binary"]

# ─── helper distributions ───────────────────────────────────
def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n: int = N_EMPLOYEES) -> pd.DataFrame:
    """
    Build a synthetic employee dataset that mimics real HR data.

    Features
    --------
    age, gender, education_level, department, job_level,
    years_at_company, years_in_role, monthly_salary,
    training_hours_per_year, num_projects_completed,
    manager_rating (1-5), peer_rating (1-5),
    absenteeism_days, overtime_hours_per_month,
    work_life_balance_score (1-10), job_satisfaction_score (1-10),
    last_promotion_years_ago, certifications_earned,
    performance_score (target — 1=Low, 2=Medium, 3=High)
    """

    # ── demographics ────────────────────────────────────────
    age              = np.random.randint(22, 60, n)
    gender           = np.random.choice(GENDER, n, p=[0.50, 0.45, 0.05])
    education_level  = np.random.choice(EDUCATION, n,
                                        p=[0.10, 0.55, 0.30, 0.05])
    department       = np.random.choice(DEPARTMENTS, n)
    job_level        = np.random.choice(JOB_LEVELS, n,
                                        p=[0.25, 0.30, 0.25, 0.12, 0.08])

    # ── experience ──────────────────────────────────────────
    years_at_company = _clamp(
        np.random.exponential(scale=5, size=n).astype(int), 0, 35
    )
    years_in_role    = _clamp(
        np.random.randint(0, years_at_company + 1), 0, 20
    )

    # ── compensation (salary influenced by level & dept) ────
    base_salary = {
        "Junior": 30_000, "Mid": 50_000, "Senior": 75_000,
        "Lead": 95_000, "Manager": 115_000
    }
    dept_multiplier = {
        "Engineering": 1.30, "Finance": 1.20, "Product": 1.25,
        "Marketing": 1.05, "Sales": 1.10, "HR": 1.00,
        "Operations": 1.00, "Support": 0.90
    }
    monthly_salary = np.array([
        base_salary[jl] * dept_multiplier[dp] / 12
        for jl, dp in zip(job_level, department)
    ]) * np.random.normal(1.0, 0.10, n)
    monthly_salary = _clamp(monthly_salary, 2_000, 20_000).astype(int)

    # ── work habits ─────────────────────────────────────────
    training_hours_per_year     = np.random.randint(0, 120, n)
    num_projects_completed      = np.random.randint(1, 30, n)
    overtime_hours_per_month    = _clamp(
        np.random.exponential(5, n).astype(int), 0, 40
    )
    absenteeism_days            = _clamp(
        np.random.exponential(4, n).astype(int), 0, 30
    )
    certifications_earned       = np.random.randint(0, 8, n)

    # ── ratings (1–5) ───────────────────────────────────────
    manager_rating = np.random.choice([1, 2, 3, 4, 5], n,
                                      p=[0.05, 0.10, 0.30, 0.35, 0.20])
    peer_rating    = np.random.choice([1, 2, 3, 4, 5], n,
                                      p=[0.05, 0.10, 0.35, 0.35, 0.15])

    # ── satisfaction scores (1–10) ──────────────────────────
    work_life_balance_score  = np.random.randint(1, 11, n)
    job_satisfaction_score   = np.random.randint(1, 11, n)

    # ── promotion history ───────────────────────────────────
    last_promotion_years_ago = _clamp(
        np.random.randint(0, years_at_company + 1), 0, 15
    )

    # ─────────────────────────────────────────────────────────
    # TARGET: performance_score
    # We use a weighted formula so that the target is realistic
    # and not random — this is important for ML to learn patterns
    # ─────────────────────────────────────────────────────────
    score = (
        0.25 * manager_rating
        + 0.20 * peer_rating
        + 0.15 * (training_hours_per_year / 120 * 5)
        + 0.10 * (num_projects_completed / 30 * 5)
        + 0.10 * (job_satisfaction_score / 10 * 5)
        + 0.10 * (work_life_balance_score / 10 * 5)
        - 0.05 * (absenteeism_days / 30 * 5)
        - 0.05 * (last_promotion_years_ago / 15 * 5)
        + np.random.normal(0, 0.3, n)          # realistic noise
    )
    # Map raw score to 3 classes
    p33, p66 = np.percentile(score, [33, 66])
    performance_score = np.where(score <= p33, 1,
                        np.where(score <= p66, 2, 3))
    # Label map: 1=Low, 2=Medium, 3=High

    # ─── assemble DataFrame ─────────────────────────────────
    df = pd.DataFrame({
        "employee_id"              : range(1001, 1001 + n),
        "age"                      : age,
        "gender"                   : gender,
        "education_level"          : education_level,
        "department"               : department,
        "job_level"                : job_level,
        "years_at_company"         : years_at_company,
        "years_in_role"            : years_in_role,
        "monthly_salary"           : monthly_salary,
        "training_hours_per_year"  : training_hours_per_year,
        "num_projects_completed"   : num_projects_completed,
        "overtime_hours_per_month" : overtime_hours_per_month,
        "absenteeism_days"         : absenteeism_days,
        "certifications_earned"    : certifications_earned,
        "manager_rating"           : manager_rating,
        "peer_rating"              : peer_rating,
        "work_life_balance_score"  : work_life_balance_score,
        "job_satisfaction_score"   : job_satisfaction_score,
        "last_promotion_years_ago" : last_promotion_years_ago,
        "performance_score"        : performance_score,
    })

    print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Performance distribution:\n{df['performance_score'].value_counts().sort_index()}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = Path("data/employee_data.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Saved → {out}")
