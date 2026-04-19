"""
=============================================================
  Employee Performance Predictor — Streamlit Dashboard
  Run: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── Page configuration (MUST be first Streamlit call) ───────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — dark professional theme ─────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary    : #0F172A;
    --bg-secondary  : #1E293B;
    --bg-card       : #1E293B;
    --accent-blue   : #3B82F6;
    --accent-emerald: #10B981;
    --accent-amber  : #F59E0B;
    --accent-red    : #EF4444;
    --accent-purple : #8B5CF6;
    --text-primary  : #F1F5F9;
    --text-secondary: #94A3B8;
    --border        : #334155;
    --gradient      : linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

/* ── Main header ── */
.main-header {
    background: var(--gradient);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(59,130,246,0.25);
}
.main-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem;
    margin: 0.5rem 0 0;
}

/* ── KPI metric cards ── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary) !important;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.82rem;
    color: var(--text-secondary) !important;
    margin-top: 0.35rem;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

/* ── Badge chips ── */
.badge-high   { background:#064E3B; color:#6EE7B7; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-medium { background:#78350F; color:#FDE68A; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-low    { background:#7F1D1D; color:#FECACA; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:600; }

/* ── Prediction result box ── */
.pred-box {
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
}
.pred-box-high   { background:#022C22; border-color:#10B981; }
.pred-box-medium { background:#1C1007; border-color:#F59E0B; }
.pred-box-low    { background:#1F0808; border-color:#EF4444; }
.pred-label { font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:700; }
.pred-conf  { font-size:1rem; margin-top:0.4rem; opacity:0.75; }

/* ── Probability bar ── */
.prob-bar-container { margin: 0.4rem 0; }
.prob-bar-label { display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px; color:var(--text-primary) !important; }
.prob-bar { height:10px; border-radius:999px; transition:width 0.5s; }

/* ── Tables ── */
.dataframe { background:var(--bg-card) !important; color:var(--text-primary) !important; }
thead th { background:var(--bg-secondary) !important; color:var(--text-primary) !important; }

/* ── Streamlit overrides ── */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stRadio label, .stCheckbox label, .stTextInput label {
    color: var(--text-primary) !important;
    font-weight: 500;
}
.stButton>button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.88 !important; }
[data-testid="stMetricValue"]  { color: var(--text-primary) !important; }
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed_data.csv")
    except FileNotFoundError:
        # Auto-generate if missing
        import sys; sys.path.insert(0, ".")
        from src.data_generation import generate_dataset
        from src.preprocessing   import clean, engineer_features
        df = generate_dataset(1500)
        df = clean(df)
        df = engineer_features(df)
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/processed_data.csv", index=False)
        return df


@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def perf_badge(label: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(label, "")
    return f'<span class="{cls}">{label}</span>'


def color_for(label: str) -> str:
    return {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}.get(label, "#94A3B8")


# ─── Load resources ──────────────────────────────────────────
df = load_data()
artifact = load_model()
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# ─── Sidebar navigation ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🏢</div>
        <div style='font-family:Space Grotesk; font-weight:700; font-size:1.1rem;
                    color:#F1F5F9;'>HR Analytics</div>
        <div style='font-size:0.78rem; color:#64748B;'>Powered by Machine Learning</div>
    </div>
    <hr style='border-color:#334155; margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview Dashboard",
         "🔍 Employee Analysis",
         "🤖 Predict Performance",
         "📈 Model Performance",
         "📋 HR Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#64748B; padding:0.5rem;'>
        📂 Dataset: {len(df):,} employees<br>
        🏗️ Features: {df.shape[1]} columns<br>
        🎯 Classes: Low · Medium · High
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Employee Performance Predictor</h1>
        <p>AI-driven HR analytics · Real-time predictions · Actionable workforce insights</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────
    total   = len(df)
    high    = (df["performance_score"] == 3).sum()
    medium  = (df["performance_score"] == 2).sum()
    low     = (df["performance_score"] == 1).sum()
    avg_sal = int(df["monthly_salary"].mean())
    avg_trn = int(df["training_hours_per_year"].mean())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "👥 Total Employees", f"{total:,}", "Synthetic HR dataset"),
        (c2, "🌟 High Performers", f"{high:,}", f"{high/total*100:.1f}% of workforce"),
        (c3, "📊 Medium Performers", f"{medium:,}", f"{medium/total*100:.1f}% of workforce"),
        (c4, "⚠️ Low Performers", f"{low:,}", f"{low/total*100:.1f}% of workforce"),
        (c5, "💰 Avg Monthly Salary", f"${avg_sal:,}", "All departments"),
        (c6, "📚 Avg Training Hours", f"{avg_trn}h", "Per year"),
    ]
    for col, title, value, sub in cards:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Performance donut + Dept breakdown ─────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-heading">Performance Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1E293B")
        sizes  = [low, medium, high]
        labels = ["Low", "Medium", "High"]
        colors = ["#EF4444", "#F59E0B", "#10B981"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor="#0F172A", linewidth=3)
        )
        for t in texts + autotexts:
            t.set_color("#F1F5F9"); t.set_fontsize(12); t.set_fontweight("bold")
        ax.set_facecolor("#1E293B")
        ax.text(0, 0, f"{total:,}\nEmployees",
                ha="center", va="center",
                color="#F1F5F9", fontsize=13, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-heading">Department Performance Breakdown</div>',
                    unsafe_allow_html=True)
        dept_perf = df.groupby(["department", "performance_score"]).size().unstack(fill_value=0)
        dept_perf.columns = ["Low", "Medium", "High"]
        dept_perf_pct = dept_perf.div(dept_perf.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1E293B")
        ax.set_facecolor("#1E293B")
        bottom = np.zeros(len(dept_perf_pct))
        for col_name, color in zip(["Low", "Medium", "High"],
                                   ["#EF4444", "#F59E0B", "#10B981"]):
            vals = dept_perf_pct[col_name].values
            ax.bar(dept_perf_pct.index, vals, bottom=bottom,
                   color=color, label=col_name, edgecolor="#0F172A", linewidth=1.5)
            bottom += vals

        ax.set_xticks(range(len(dept_perf_pct)))
        ax.set_xticklabels(dept_perf_pct.index, rotation=25, ha="right",
                           color="#F1F5F9", fontsize=10)
        ax.set_ylabel("% of employees", color="#94A3B8")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values(): spine.set_color("#334155")
        ax.legend(loc="upper right", facecolor="#1E293B",
                  labelcolor="#F1F5F9", framealpha=0.8)
        ax.set_ylim(0, 110)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Row 2: Salary + Training scatter ──────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-heading">Avg Salary by Job Level</div>',
                    unsafe_allow_html=True)
        level_order = ["Junior", "Mid", "Senior", "Lead", "Manager"]
        level_sal   = df.groupby("job_level")["monthly_salary"].mean().reindex(level_order)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1E293B")
        ax.set_facecolor("#1E293B")
        bars = ax.bar(level_sal.index, level_sal.values,
                      color=["#3B82F6","#6366F1","#8B5CF6","#A855F7","#EC4899"],
                      edgecolor="#0F172A", linewidth=1.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 100,
                    f"${int(b.get_height()):,}", ha="center",
                    color="#F1F5F9", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94A3B8"); ax.set_ylabel("Monthly Salary ($)", color="#94A3B8")
        for spine in ax.spines.values(): spine.set_color("#334155")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_d:
        st.markdown('<div class="section-heading">Training Hours vs Performance</div>',
                    unsafe_allow_html=True)
        colors_map = {1: "#EF4444", 2: "#F59E0B", 3: "#10B981"}
        sample = df.sample(min(500, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1E293B")
        ax.set_facecolor("#1E293B")
        for lvl, color in colors_map.items():
            sub = sample[sample["performance_score"] == lvl]
            ax.scatter(sub["training_hours_per_year"], sub["monthly_salary"],
                       c=color, alpha=0.55, s=25, label=LABEL_MAP[lvl - 1])
        ax.set_xlabel("Training Hours / Year", color="#94A3B8")
        ax.set_ylabel("Monthly Salary ($)", color="#94A3B8")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values(): spine.set_color("#334155")
        ax.legend(facecolor="#1E293B", labelcolor="#F1F5F9", framealpha=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — EMPLOYEE ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Employee Analysis":
    st.markdown('<div class="section-heading" style="font-size:1.5rem;">🔍 Employee Data Explorer</div>',
                unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_dept = st.multiselect("Department", sorted(df["department"].unique()),
                                  default=sorted(df["department"].unique()))
    with col2:
        sel_lvl = st.multiselect("Job Level", ["Junior","Mid","Senior","Lead","Manager"],
                                  default=["Junior","Mid","Senior","Lead","Manager"])
    with col3:
        sel_perf = st.multiselect("Performance", [1, 2, 3],
                                   format_func=lambda x: LABEL_MAP[x-1],
                                   default=[1, 2, 3])

    filtered = df[
        df["department"].isin(sel_dept) &
        df["job_level"].isin(sel_lvl) &
        df["performance_score"].isin(sel_perf)
    ]

    st.markdown(f"**{len(filtered):,} employees match filters**")

    # ── Stats table ──────────────────────────────────────────
    display_cols = [
        "department", "job_level", "monthly_salary",
        "training_hours_per_year", "manager_rating",
        "absenteeism_days", "performance_score"
    ]
    st.dataframe(
        filtered[display_cols]
        .rename(columns={"performance_score": "perf_label"})
        .assign(perf_label=lambda d: d["perf_label"].map({1:"Low",2:"Medium",3:"High"}))
        .head(200),
        use_container_width=True, height=350
    )

    # ── Stat cards ───────────────────────────────────────────
    st.markdown('<div class="section-heading">Summary Statistics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Manager Rating",    f"{filtered['manager_rating'].mean():.2f} / 5")
    c2.metric("Avg Training Hours",    f"{filtered['training_hours_per_year'].mean():.0f} h")
    c3.metric("Avg Absenteeism",       f"{filtered['absenteeism_days'].mean():.1f} days")
    c4.metric("Avg Monthly Salary",    f"${filtered['monthly_salary'].mean():,.0f}")


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICT PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "🤖 Predict Performance":
    st.markdown('<div class="section-heading" style="font-size:1.5rem;">🤖 Predict Employee Performance</div>',
                unsafe_allow_html=True)
    st.markdown("Fill in the employee details below and click **Predict** to get an AI-powered performance forecast.")

    if artifact is None:
        st.warning("⚠️ No trained model found. Run `python main.py` first, then refresh.")
        st.stop()

    clf           = artifact["model"]
    scaler        = artifact["scaler"]
    feature_names = artifact["feature_names"]

    # ── Input form ───────────────────────────────────────────
    st.markdown('<div class="section-heading">Employee Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age       = st.slider("Age", 22, 60, 32)
        dept      = st.selectbox("Department",
                                 ["Engineering","Sales","HR","Finance",
                                  "Marketing","Operations","Product","Support"])
        edu       = st.selectbox("Education Level",
                                 ["High School","Bachelor's","Master's","PhD"])
        job_lvl   = st.selectbox("Job Level",
                                 ["Junior","Mid","Senior","Lead","Manager"])
    with c2:
        yrs_co    = st.slider("Years at Company", 0, 35, 4)
        yrs_role  = st.slider("Years in Current Role", 0, 20, 2)
        salary    = st.number_input("Monthly Salary ($)", 2000, 20000, 5500, step=500)
        training  = st.slider("Training Hours / Year", 0, 120, 40)
    with c3:
        projects  = st.slider("Projects Completed", 1, 30, 8)
        overtime  = st.slider("Overtime Hours / Month", 0, 40, 5)
        absent    = st.slider("Absenteeism Days / Year", 0, 30, 3)
        certs     = st.slider("Certifications Earned", 0, 8, 1)

    c4, c5 = st.columns(2)
    with c4:
        mgr_rating = st.slider("Manager Rating (1–5)", 1, 5, 3)
        peer_rating = st.slider("Peer Rating (1–5)", 1, 5, 3)
    with c5:
        wlb     = st.slider("Work-Life Balance Score (1–10)", 1, 10, 6)
        job_sat = st.slider("Job Satisfaction Score (1–10)", 1, 10, 6)
        promo   = st.slider("Years Since Last Promotion", 0, 15, 2)
        gender  = st.selectbox("Gender", ["Male", "Female", "Non-binary"])

    predict_btn = st.button("🔮  Predict Performance")

    if predict_btn:
        # ── Build feature dict (must match training columns) ──
        edu_map  = {"High School":1,"Bachelor's":2,"Master's":3,"PhD":4}
        lvl_map  = {"Junior":1,"Mid":2,"Senior":3,"Lead":4,"Manager":5}
        dept_col = {f"department_{d}": 0 for d in
                    ["Engineering","Finance","HR","Marketing","Operations","Product","Sales","Support"]}
        gen_col  = {f"gender_{g}": 0 for g in ["Female","Male","Non-binary"]}

        dept_col[f"department_{dept}"] = 1
        gen_col[f"gender_{gender}"]    = 1

        eps = 1e-6
        feat = {
            "age"                          : age,
            "education_level"              : edu_map[edu],
            "job_level"                    : lvl_map[job_lvl],
            "years_at_company"             : yrs_co,
            "years_in_role"                : yrs_role,
            "monthly_salary"               : salary,
            "training_hours_per_year"      : training,
            "num_projects_completed"       : projects,
            "overtime_hours_per_month"     : overtime,
            "absenteeism_days"             : absent,
            "certifications_earned"        : certs,
            "manager_rating"               : mgr_rating,
            "peer_rating"                  : peer_rating,
            "work_life_balance_score"      : wlb,
            "job_satisfaction_score"       : job_sat,
            "last_promotion_years_ago"     : promo,
            "salary_per_year_experience"   : salary / (yrs_co + eps),
            "productivity_index"           : projects / (yrs_co + eps),
            "training_to_experience_ratio" : training / (yrs_co + 1),
            "avg_rating"                   : (mgr_rating + peer_rating) / 2,
            "loyalty_score"                : yrs_co / age,
            "engagement_score"             : (job_sat + wlb) / 20,
            **dept_col,
            **gen_col,
        }

        row = pd.DataFrame([feat]).reindex(columns=feature_names, fill_value=0)
        row_scaled = scaler.transform(row)
        pred       = clf.predict(row_scaled)[0]
        proba      = clf.predict_proba(row_scaled)[0]
        label      = LABEL_MAP[pred]
        conf       = proba[pred] * 100

        # ── Result ───────────────────────────────────────────
        st.markdown("---")
        box_cls = {"High":"pred-box-high","Medium":"pred-box-medium","Low":"pred-box-low"}[label]
        color   = color_for(label)
        emoji   = {"High":"🌟","Medium":"📊","Low":"⚠️"}[label]

        col_res, col_prob = st.columns([1, 1])
        with col_res:
            st.markdown(f"""
            <div class="pred-box {box_cls}">
                <div style='font-size:3rem;'>{emoji}</div>
                <div class="pred-label" style='color:{color};'>{label} Performer</div>
                <div class="pred-conf" style='color:#94A3B8;'>
                    {conf:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            # HR Recommendations
            recs = {
                "High"  : ["🏆 Fast-track promotion consideration",
                           "💡 Assign to high-impact projects",
                           "🌍 Consider leadership development program",
                           "💰 Review compensation competitiveness"],
                "Medium": ["📚 Enroll in targeted skill training",
                           "👥 Pair with a high-performer mentor",
                           "🎯 Set clear quarterly OKRs",
                           "📋 Schedule monthly 1:1 check-ins"],
                "Low"   : ["🚨 Schedule performance improvement plan",
                           "🔍 Identify root-cause blockers",
                           "🤝 Increase manager support cadence",
                           "📖 Mandatory upskilling workshops"],
            }
            st.markdown(f"**📋 HR Recommendations:**")
            for r in recs[label]:
                st.markdown(f"- {r}")

        with col_prob:
            st.markdown('<div class="section-heading">Probability Breakdown</div>',
                        unsafe_allow_html=True)
            for i, (lbl, prob) in enumerate(zip(["Low","Medium","High"], proba)):
                clr = ["#EF4444","#F59E0B","#10B981"][i]
                st.markdown(f"""
                <div class="prob-bar-container">
                  <div class="prob-bar-label">
                    <span>{lbl}</span><span style="color:{clr}; font-weight:700;">{prob*100:.1f}%</span>
                  </div>
                  <div style='background:#334155; border-radius:999px; height:12px;'>
                    <div class="prob-bar" style='width:{prob*100:.1f}%; background:{clr};'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge-like radar for attributes
            categories = ["Manager\nRating", "Peer\nRating", "Training", "Satisfaction", "WLB"]
            values_norm = [
                mgr_rating / 5,
                peer_rating / 5,
                training / 120,
                job_sat / 10,
                wlb / 10,
            ]
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values_norm += values_norm[:1]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True),
                                   facecolor="#1E293B")
            ax.set_facecolor("#1E293B")
            ax.plot(angles, values_norm, color=color, linewidth=2)
            ax.fill(angles, values_norm, color=color, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, color="#F1F5F9", fontsize=9)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["25%","50%","75%","100%"],
                               color="#64748B", fontsize=7)
            ax.spines["polar"].set_color("#334155")
            ax.grid(color="#334155")
            ax.set_title("Employee Profile Radar", color="#F1F5F9",
                         fontsize=11, fontweight="bold", pad=15)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="section-heading" style="font-size:1.5rem;">📈 Model Performance Dashboard</div>',
                unsafe_allow_html=True)

    if artifact is None:
        st.warning("⚠️ Run `python main.py` first to train the model.")
        st.stop()

    # Load saved report
    report_path = Path("outputs/classification_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", f"{report['accuracy']*100:.2f}%")
        c2.metric("Low — F1 Score",    f"{report['Low']['f1-score']*100:.1f}%")
        c3.metric("Medium — F1 Score", f"{report['Medium']['f1-score']*100:.1f}%")
        c4.metric("High — F1 Score",   f"{report['High']['f1-score']*100:.1f}%")

    # Images
    img_map = {
        "Confusion Matrix"     : "images/10_confusion_matrix.png",
        "Model Comparison"     : "images/11_model_comparison.png",
        "Feature Importances"  : "images/12_feature_importance.png",
    }
    cols = st.columns(len(img_map))
    for col, (title, path) in zip(cols, img_map.items()):
        if Path(path).exists():
            col.markdown(f"**{title}**")
            col.image(path, use_container_width=True)

    # Feature importance table
    fi_path = Path("outputs/feature_importances.csv")
    if fi_path.exists():
        st.markdown('<div class="section-heading">Top Feature Importances</div>',
                    unsafe_allow_html=True)
        fi_df = pd.read_csv(fi_path, index_col=0).sort_values("importance", ascending=False)
        fi_df["importance"] = (fi_df["importance"] * 100).round(3)
        fi_df.columns = ["Importance (%)"]
        st.dataframe(fi_df.head(15), use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 5 — HR INSIGHTS
# ═════════════════════════════════════════════════════════════
elif page == "📋 HR Insights":
    st.markdown('<div class="section-heading" style="font-size:1.5rem;">📋 HR Strategic Insights</div>',
                unsafe_allow_html=True)

    # ── Retention risk table ──────────────────────────────────
    st.markdown('<div class="section-heading">⚠️ At-Risk Employees (Low Performers)</div>',
                unsafe_allow_html=True)
    low_perf = df[df["performance_score"] == 1].nsmallest(10, "job_satisfaction_score")[
        ["department","job_level","monthly_salary",
         "absenteeism_days","job_satisfaction_score","manager_rating"]
    ]
    st.dataframe(low_perf, use_container_width=True)

    # ── High performer insights ───────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-heading">🌟 High Performer Traits</div>',
                    unsafe_allow_html=True)
        high = df[df["performance_score"] == 3]
        low  = df[df["performance_score"] == 1]
        traits = {
            "Avg Training Hours"    : (high["training_hours_per_year"].mean(),
                                       low["training_hours_per_year"].mean()),
            "Avg Manager Rating"    : (high["manager_rating"].mean(),
                                       low["manager_rating"].mean()),
            "Avg Projects"          : (high["num_projects_completed"].mean(),
                                       low["num_projects_completed"].mean()),
            "Avg Job Satisfaction"  : (high["job_satisfaction_score"].mean(),
                                       low["job_satisfaction_score"].mean()),
            "Avg Absenteeism"       : (high["absenteeism_days"].mean(),
                                       low["absenteeism_days"].mean()),
        }
        trait_df = pd.DataFrame(traits, index=["High","Low"]).T
        trait_df["Difference"] = (trait_df["High"] - trait_df["Low"]).round(2)
        st.dataframe(trait_df.round(2), use_container_width=True)

    with col_b:
        st.markdown('<div class="section-heading">💡 Department Risk Score</div>',
                    unsafe_allow_html=True)
        risk = (
            df.groupby("department")["performance_score"]
            .apply(lambda x: (x == 1).sum() / len(x) * 100)
            .sort_values(ascending=False)
            .reset_index()
        )
        risk.columns = ["Department", "Low Performer %"]
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1E293B")
        ax.set_facecolor("#1E293B")
        colors_risk = ["#EF4444" if v > 40 else "#F59E0B" if v > 30 else "#10B981"
                       for v in risk["Low Performer %"]]
        ax.barh(risk["Department"], risk["Low Performer %"],
                color=colors_risk, edgecolor="#0F172A")
        ax.axvline(33, color="#94A3B8", linestyle="--", alpha=0.5)
        ax.set_xlabel("Low Performers (%)", color="#94A3B8")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values(): spine.set_color("#334155")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Recommendations ──────────────────────────────────────
    st.markdown('<div class="section-heading">📌 Strategic HR Recommendations</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    recs = [
        ("🎓 Training Investment",
         "Departments with low training hours show 2× higher low-performer rates. Mandate minimum 60h/year.",
         "#3B82F6"),
        ("💬 Manager Quality",
         "Manager rating is the #1 predictor. Invest in manager coaching programs and 360° feedback cycles.",
         "#8B5CF6"),
        ("🏆 Recognition Programs",
         "Employees without a promotion in 5+ years show declining performance. Implement structured career paths.",
         "#10B981"),
    ]
    for col, (title, text, color) in zip(cols, recs):
        col.markdown(f"""
        <div style='background:#1E293B; border:1px solid {color};
                    border-left: 4px solid {color};
                    border-radius:12px; padding:1.2rem;'>
            <div style='font-weight:700; font-size:1rem;
                        color:#F1F5F9; margin-bottom:0.5rem;'>{title}</div>
            <div style='font-size:0.88rem; color:#94A3B8; line-height:1.6;'>{text}</div>
        </div>
        """, unsafe_allow_html=True)
