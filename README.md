# 🏢 Employee Performance Predictor

### AI-Driven HR Analytics using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A complete end-to-end Machine Learning project that predicts employee performance levels (Low / Medium / High) using a synthetic HR dataset, with an interactive Streamlit dashboard for HR insights.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Value](#-business-value)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Features](#-features)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

HR departments struggle to proactively identify employee performance levels due to reliance on annual reviews, gut instincts, and lagging indicators. This results in:

- High performers going unrecognized and leaving (costly attrition)
- Low performers not receiving timely intervention
- Inefficient allocation of training budgets
- Poor promotion decisions

**This system uses Machine Learning to predict employee performance levels before they become critical, enabling proactive HR decisions.**

---

## 💼 Business Value

| Stakeholder | Benefit                                                       |
| ----------- | ------------------------------------------------------------- |
| HR Team     | Identify at-risk employees early, plan targeted interventions |
| Managers    | Data-driven 1:1 feedback, coaching prioritization             |
| Leadership  | Workforce planning, budget allocation for L&D                 |
| Employees   | Fairer, objective performance assessments                     |

---

## 🏗 Architecture

```
📥 RAW DATA (Synthetic HR Dataset — 2,000 employees)
         │
         ▼
┌─────────────────────────────────────────┐
│           DATA PIPELINE                 │
│  • Data Cleaning (nulls, duplicates)    │
│  • Feature Engineering (6 new features) │
│  • Encoding (OHE + ordinal)             │
│  • Scaling (StandardScaler)             │
│  • Train/Test Split (80/20, stratified) │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           ML MODELS                     │
│  ○ Logistic Regression (baseline)       │
│  ○ Random Forest ✓ (best performer)     │
│  ○ Gradient Boosting                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           OUTPUTS                       │
│  • Classification: Low/Medium/High      │
│  • Probability scores per class         │
│  • Feature importance rankings          │
│  • HR action recommendations            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      STREAMLIT DASHBOARD                │
│  📊 Overview · 🔍 Explorer             │
│  🤖 Predictor · 📈 Model Metrics        │
│  📋 HR Insights                         │
└─────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category        | Technology          |
| --------------- | ------------------- |
| Language        | Python 3.10+        |
| ML Framework    | Scikit-learn        |
| Data Processing | Pandas, NumPy       |
| Visualization   | Matplotlib, Seaborn |
| Web Dashboard   | Streamlit           |
| Model Storage   | Pickle              |
| Version Control | Git / GitHub        |

---

## 📊 Dataset

**Synthetic HR dataset generated programmatically — no real company data used.**

| Feature                   | Description                         |
| ------------------------- | ----------------------------------- |
| `age`                     | Employee age (22–60)                |
| `gender`                  | Male / Female / Non-binary          |
| `education_level`         | High School → PhD (ordinal)         |
| `department`              | Engineering, Sales, HR, Finance…    |
| `job_level`               | Junior → Manager (ordinal)          |
| `years_at_company`        | Tenure in years                     |
| `monthly_salary`          | In USD                              |
| `training_hours_per_year` | 0–120 hours                         |
| `manager_rating`          | 1–5 scale                           |
| `peer_rating`             | 1–5 scale                           |
| `absenteeism_days`        | Per year                            |
| `job_satisfaction_score`  | 1–10 scale                          |
| `work_life_balance_score` | 1–10 scale                          |
| `certifications_earned`   | Count                               |
| **`performance_score`**   | **TARGET: 1=Low, 2=Medium, 3=High** |

**Engineered Features:**

- `avg_rating` = (manager_rating + peer_rating) / 2
- `productivity_index` = projects / years_at_company
- `engagement_score` = (job_sat + wlb) / 20
- `salary_per_year_experience` = salary / tenure
- `training_to_experience_ratio` = training_hours / tenure
- `loyalty_score` = tenure / age

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ML Pipeline

```bash
python main.py
```

This will:

- Generate the synthetic dataset (data/)
- Run EDA and save charts (images/)
- Train 3 ML models and evaluate them
- Save the best model (models/best_model.pkl)
- Save predictions (outputs/)

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📈 Results

| Model               | Test Accuracy | CV Accuracy (5-fold) |
| ------------------- | ------------- | -------------------- |
| Logistic Regression | ~72%          | ~71%                 |
| Random Forest       | **~85%**      | **~84%**             |
| Gradient Boosting   | ~83%          | ~82%                 |

**Best Model: Random Forest Classifier**

### Top Predictive Features

1. `manager_rating` — Most impactful predictor
2. `avg_rating` — Combined manager + peer signal
3. `training_hours_per_year` — Training investment
4. `job_satisfaction_score` — Engagement indicator
5. `absenteeism_days` — Attendance signal

---

## 📁 Project Structure

```
Employee-Performance-Predictor/
│
├── data/                          ← Datasets (auto-generated)
│   ├── employee_data.csv          ← Raw synthetic dataset
│   └── processed_data.csv         ← Cleaned + engineered features
│
├── src/                           ← Source modules
│   ├── __init__.py
│   ├── data_generation.py         ← Synthetic data factory
│   ├── preprocessing.py           ← Clean, encode, scale
│   ├── eda.py                     ← EDA + chart generation
│   └── model.py                   ← Train, evaluate, save models
│
├── models/                        ← Saved ML artifacts
│   └── best_model.pkl             ← Trained model + scaler
│
├── outputs/                       ← Prediction results
│   ├── predictions.csv
│   ├── classification_report.json
│   └── feature_importances.csv
│
├── images/                        ← EDA + evaluation charts
│   ├── 01_target_distribution.png
│   ├── 02_dept_performance.png
│   └── ...
│
├── app.py                         ← Streamlit dashboard
├── main.py                        ← Pipeline entry point
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## 🔮 Future Improvements

- [ ] **Employee Attrition Prediction** — Predict who will leave the company
- [ ] **Real-Time API** — Flask/FastAPI endpoint for HR systems integration
- [ ] **Deep Learning** — LSTM for time-series performance trends
- [ ] **SHAP Explainability** — Individual prediction explanations for HR
- [ ] **Clustering** — Unsupervised employee segmentation
- [ ] **NLP Integration** — Analyze performance review text for sentiment
- [ ] **Power BI / Tableau** — Enterprise-grade reporting

---

## 👤 Author

**CH S K CHAITANYA**

- 📧 your.email@email.com
- 🔗 [LinkedIn](https://linkedin.com/in/chskchaitanya)
- 🐙 [GitHub](https://github.com/CH-S-K-CHAITANYA)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> ⭐ If this project helped you, please give it a star on GitHub!
#
