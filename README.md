# 💰 AI-Driven Personal Finance & Expense Tracker

### End-to-End Data Science Pipeline: Synthetic Data Generation → Feature Engineering → Interactive Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.15.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A comprehensive financial intelligence system that simulates, processes, and analyzes 1,200+ transaction records to provide deep behavioral insights and budget tracking.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [The Data Pipeline](#-the-data-pipeline)
- [Key Insights & Analytics](#-key-insights--analytics)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

This project demonstrates the full lifecycle of a data science product. Instead of simple manual logging, it uses automated pipelines to simulate, clean, and visualize complex financial behaviors. 

The system engineers temporal features (weekend vs. weekday spending, monthly volatility) to mirror how modern fintech platforms like **Mint** or **Cred** process user data to provide actionable intelligence.

---

## ❗ Problem Statement

Most individuals struggle with "financial blindness"—the inability to see where money goes beyond surface-level balances. This system solves this by:

* **Automating Data Validation:** Cleaning messy transaction logs and handling duplicates.
* **Behavioral Profiling:** Identifying high-spend days and seasonal trends.
* **Budget Variance Analysis:** Comparing real-time spending against predefined limits to prevent overspending.

---

## 🏗️ System Architecture

📥 DATA LAYER
Synthetic Generator (Faker) / CSV Upload
│
▼
⚙️ PROCESSING MODULE
• Data Cleaning (Pandas)  • Outlier Handling
• Datetime Normalization • Category Mapping
│
▼
🧠 FEATURE ENGINEERING
• Weekend/Weekday Flags  • Rolling Averages
• Budget % Utilization   • Cumulative Spend
│
▼
📊 VISUALIZATION & UI
• Streamlit Dashboard    • Static PNG Reports
• Interactive Plotly     • Automated Insights (.txt)
