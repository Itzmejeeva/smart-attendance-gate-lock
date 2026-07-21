"""
Social Credit Scoring Framework — ML-Powered Risk & Trust Assessment
A production-style Streamlit application.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Credit Scoring Framework",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
PRIMARY = "#1B3B6F"        # deep civic navy
ACCENT = "#2E7DD1"         # trust blue
ACCENT_SOFT = "#E7F0FB"
GOOD = "#1E8E5A"
AVERAGE = "#C98A1B"
POOR = "#C0392B"
INK = "#101828"
MUTED = "#5B6472"
CANVAS = "#F5F7FA"
CARD = "#FFFFFF"
LINE = "#E3E8EF"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    html, body {{
        overflow-x: hidden;
    }}
    .stApp {{
        background: {CANVAS};
        overflow-x: hidden;
    }}
    .block-container {{
        padding-top: 1.6rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
        width: 100%;
        margin: 0 auto;
    }}
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
    }}
    [data-testid="column"] {{
        min-width: 320px;
    }}
    /* Streamlit dims/greys out the page slightly on every widget interaction
       while it recalculates -- this is expected behavior, not a bug, but we
       keep it subtle so the UI doesn't look "broken" mid-update. */
    [data-stale="true"] {{
        opacity: 0.85 !important;
        transition: opacity 0.15s ease;
    }}

    /* ---- Masthead ---- */
    .masthead {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #142E57 55%, #0E223F 100%);
        border-radius: 18px;
        padding: 2.1rem 2.4rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px -12px rgba(16,24,40,0.35);
    }}
    .masthead::after {{
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(46,125,209,0.35), transparent 70%);
    }}
    .masthead-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        color: #8FB8EA;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .masthead-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 2.05rem;
        color: #FFFFFF;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
    }}
    .masthead-sub {{
        color: #C4D5EC;
        font-size: 0.98rem;
        max-width: 640px;
        line-height: 1.5;
        margin: 0;
    }}

    /* ---- Cards ---- */
    .panel {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
    }}
    .panel-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.02rem;
        color: {INK};
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.2rem;
    }}
    .panel-title .dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {ACCENT};
        display: inline-block;
    }}
    .panel-desc {{
        color: {MUTED};
        font-size: 0.84rem;
        margin-bottom: 1rem;
    }}

    /* ---- Section label ---- */
    .section-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {MUTED};
        margin: 0.2rem 0 0.6rem 2px;
    }}

    /* ---- Verdict badge ---- */
    .verdict-wrap {{
        text-align: center;
        padding: 0.4rem 0 0.2rem 0;
    }}
    .verdict-badge {{
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 999px;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
    }}
    .verdict-caption {{
        color: {MUTED};
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }}

    /* ---- Metric row ---- */
    .metric-card {{
        background: {ACCENT_SOFT};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }}
    .metric-card .val {{
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: {PRIMARY};
    }}
    .metric-card .lab {{
        font-size: 0.72rem;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.15rem;
    }}

    /* ---- Factor bars ---- */
    .factor-row {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.42rem 0;
        border-bottom: 1px dashed {LINE};
        font-size: 0.85rem;
    }}
    .factor-row:last-child {{ border-bottom: none; }}
    .factor-name {{ flex: 0 0 190px; color: {INK}; font-weight: 500; }}
    .factor-track {{
        flex: 1; height: 7px; background: #EEF1F5; border-radius: 5px; overflow: hidden;
    }}
    .factor-fill {{ height: 100%; border-radius: 5px; background: {ACCENT}; }}
    .factor-pct {{ flex: 0 0 44px; text-align: right; color: {MUTED}; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;}}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid {LINE};
        min-width: 300px !important;
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
    section[data-testid="stSidebar"] h3 {{
        color: {INK} !important;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
    }}
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label {{
        color: {INK} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] strong {{
        color: {PRIMARY} !important;
        font-size: 0.9rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}

    /* ---- Buttons ---- */
    .stButton>button {{
        background: {PRIMARY};
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.15s ease;
    }}
    .stButton>button:hover {{
        background: {ACCENT};
        transform: translateY(-1px);
    }}

    /* ---- History table ---- */
    .hist-tag {{
        padding: 2px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 600;
    }}

    .footer-note {{
        text-align: center;
        color: {MUTED};
        font-size: 0.76rem;
        padding: 1.4rem 0 0.6rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CONSTANTS — must mirror the training notebook's preprocessing exactly
# ----------------------------------------------------------------------------
CLASS_LABELS = {0: "Poor", 1: "Average", 2: "Good"}
CLASS_COLORS = {"Poor": POOR, "Average": AVERAGE, "Good": GOOD}

MODEL_FEATURES = [
    "Age", "Annual_Income", "Loan_Repayment", "Tax_Payment", "Utility_Bill_Payment",
    "Traffic_Violations", "Criminal_Record", "Community_Service_Hours",
    "Online_Fraud_Complaints", "Employment_Status_Part-Time",
    "Employment_Status_Self-Employed", "Employment_Status_Unemployed",
    "Education_Graduate", "Education_High School", "Education_Postgraduate",
]

BINARY_MAPS = {
    "Loan_Repayment": {"Late": 0, "On Time": 1},
    "Tax_Payment": {"Paid": 0, "Unpaid": 1},
    "Utility_Bill_Payment": {"Late": 0, "Paid": 1},
    "Criminal_Record": {"No": 0, "Yes": 1},
}

EMPLOYMENT_OPTIONS = ["Full-Time", "Part-Time", "Self-Employed", "Unemployed"]
EDUCATION_OPTIONS = ["Diploma", "Graduate", "High School", "Postgraduate"]


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("model.pkl")


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Reproduce the exact training-time encoding for a single record."""
    row = {
        "Age": raw["Age"],
        "Annual_Income": raw["Annual_Income"],
        "Loan_Repayment": BINARY_MAPS["Loan_Repayment"][raw["Loan_Repayment"]],
        "Tax_Payment": BINARY_MAPS["Tax_Payment"][raw["Tax_Payment"]],
        "Utility_Bill_Payment": BINARY_MAPS["Utility_Bill_Payment"][raw["Utility_Bill_Payment"]],
        "Traffic_Violations": raw["Traffic_Violations"],
        "Criminal_Record": BINARY_MAPS["Criminal_Record"][raw["Criminal_Record"]],
        "Community_Service_Hours": raw["Community_Service_Hours"],
        "Online_Fraud_Complaints": raw["Online_Fraud_Complaints"],
        "Employment_Status_Part-Time": 1 if raw["Employment_Status"] == "Part-Time" else 0,
        "Employment_Status_Self-Employed": 1 if raw["Employment_Status"] == "Self-Employed" else 0,
        "Employment_Status_Unemployed": 1 if raw["Employment_Status"] == "Unemployed" else 0,
        "Education_Graduate": 1 if raw["Education"] == "Graduate" else 0,
        "Education_High School": 1 if raw["Education"] == "High School" else 0,
        "Education_Postgraduate": 1 if raw["Education"] == "Postgraduate" else 0,
    }
    return pd.DataFrame([row])[MODEL_FEATURES]


def gauge_chart(score_pct: float, label: str) -> go.Figure:
    color = CLASS_COLORS[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pct,
        number={"suffix": "%", "font": {"size": 40, "family": "Sora", "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": MUTED, "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#FCEAE8"},
                {"range": [40, 70], "color": "#FBF1DE"},
                {"range": [70, 100], "color": "#E5F3EC"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR — Applicant Profile Input
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ◆ Applicant Profile")
    st.caption("Enter the individual's data to run the assessment.")

    st.markdown("**Demographics**")
    age = st.slider("Age", 18, 80, 34)
    income = st.number_input("Annual Income (₹)", min_value=0, max_value=10_000_000,
                              value=650_000, step=10_000)
    employment = st.selectbox("Employment Status", EMPLOYMENT_OPTIONS, index=0)
    education = st.selectbox("Education Level", EDUCATION_OPTIONS, index=0)

    st.markdown("**Financial Conduct**")
    loan_repay = st.selectbox("Loan Repayment History", ["On Time", "Late"], index=0)
    tax_payment = st.selectbox("Tax Payment Status", ["Paid", "Unpaid"], index=0)
    utility_payment = st.selectbox("Utility Bill Payment", ["Paid", "Late"], index=0)

    st.markdown("**Civic & Behavioral Record**")
    traffic_violations = st.slider("Traffic Violations (count)", 0, 15, 1)
    criminal_record = st.selectbox("Criminal Record", ["No", "Yes"], index=0)
    community_service = st.slider("Community Service Hours", 0, 200, 40)
    fraud_complaints = st.slider("Online Fraud Complaints", 0, 10, 0)

    st.markdown("---")
    run_btn = st.button("Run Assessment →", use_container_width=True)
    st.caption("Model: Random Forest Classifier · 200 trees · depth 10")

# ----------------------------------------------------------------------------
# MASTHEAD
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="masthead">
    <div class="masthead-eyebrow">ML-BASED RISK & TRUST ASSESSMENT</div>
    <div class="masthead-title">Social Credit Scoring Framework</div>
    <p class="masthead-sub">A machine-learning framework that evaluates financial reliability,
    civic conduct, and behavioral history to classify individuals into Poor, Average, or Good
    credit standing — trained on a Random Forest ensemble for stable, interpretable predictions.</p>
</div>
""", unsafe_allow_html=True)

# session history
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
col_left, col_right = st.columns([1.15, 1], gap="medium")

raw_input = {
    "Age": age, "Annual_Income": income, "Employment_Status": employment,
    "Education": education, "Loan_Repayment": loan_repay, "Tax_Payment": tax_payment,
    "Utility_Bill_Payment": utility_payment, "Traffic_Violations": traffic_violations,
    "Criminal_Record": criminal_record, "Community_Service_Hours": community_service,
    "Online_Fraud_Complaints": fraud_complaints,
}

model = load_model()

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Assessment Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">Prediction generated by the trained classifier for the profile in the sidebar.</div>', unsafe_allow_html=True)

    X_input = build_feature_row(raw_input)
    proba = model.predict_proba(X_input)[0]
    pred_class = int(np.argmax(proba))
    pred_label = CLASS_LABELS[pred_class]
    confidence = proba[pred_class] * 100

    if run_btn:
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "label": pred_label,
            "confidence": round(confidence, 1),
            "income": income,
            "age": age,
        })
        st.session_state.history = st.session_state.history[:8]

    fig = gauge_chart(confidence, pred_label)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    color = CLASS_COLORS[pred_label]
    st.markdown(f"""
    <div class="verdict-wrap">
        <span class="verdict-badge" style="background:{color}18; color:{color};">
            ● {pred_label} Standing
        </span>
        <div class="verdict-caption">Model confidence: {confidence:.1f}% · Random Forest Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, lab, val in zip(
        [m1, m2, m3],
        ["Poor", "Average", "Good"],
        [proba[0] * 100, proba[1] * 100, proba[2] * 100],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val" style="color:{CLASS_COLORS[lab]}">{val:.1f}%</div>
                <div class="lab">{lab}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Feature importance / driving factors ----
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Key Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">Global feature importance from the trained Random Forest — what the model weighs most heavily across all predictions.</div>', unsafe_allow_html=True)

    importances = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False).head(7)
    max_imp = importances.max()
    display_names = {
        "Annual_Income": "Annual Income", "Traffic_Violations": "Traffic Violations",
        "Community_Service_Hours": "Community Service Hours", "Age": "Age",
        "Online_Fraud_Complaints": "Online Fraud Complaints",
        "Loan_Repayment": "Loan Repayment History", "Tax_Payment": "Tax Payment Status",
        "Utility_Bill_Payment": "Utility Bill Payment", "Criminal_Record": "Criminal Record",
        "Employment_Status_Part-Time": "Employment: Part-Time",
        "Employment_Status_Self-Employed": "Employment: Self-Employed",
        "Employment_Status_Unemployed": "Employment: Unemployed",
        "Education_Graduate": "Education: Graduate", "Education_High School": "Education: High School",
        "Education_Postgraduate": "Education: Postgraduate",
    }
    for feat, imp in importances.items():
        pct_width = (imp / max_imp) * 100
        st.markdown(f"""
        <div class="factor-row">
            <div class="factor-name">{display_names.get(feat, feat)}</div>
            <div class="factor-track"><div class="factor-fill" style="width:{pct_width:.0f}%;"></div></div>
            <div class="factor-pct">{imp*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Profile Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">The record currently being evaluated.</div>', unsafe_allow_html=True)

    summary_rows = [
        ("Age", f"{age} yrs"), ("Annual Income", f"₹{income:,.0f}"),
        ("Employment", employment), ("Education", education),
        ("Loan Repayment", loan_repay), ("Tax Payment", tax_payment),
        ("Utility Bill Payment", utility_payment), ("Traffic Violations", traffic_violations),
        ("Criminal Record", criminal_record), ("Community Service Hours", f"{community_service} hrs"),
        ("Online Fraud Complaints", fraud_complaints),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Attribute", "Value"])
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        height=390,
        column_config={
            "Attribute": st.column_config.TextColumn(width="medium"),
            "Value": st.column_config.TextColumn(width="medium"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span class="dot"></span>Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">Most recent assessments run in this session.</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No assessments run yet. Click **Run Assessment** in the sidebar to begin.")
    else:
        for h in st.session_state.history:
            c = CLASS_COLORS[h["label"]]
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.5rem 0; border-bottom:1px solid {LINE}; font-size:0.85rem;">
                <span style="color:{MUTED}; font-family:'JetBrains Mono',monospace; font-size:0.75rem;">{h['time']}</span>
                <span>Age {h['age']} · ₹{h['income']:,.0f}</span>
                <span class="hist-tag" style="background:{c}18; color:{c};">{h['label']} · {h['confidence']}%</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="footer-note">
    Social Credit Scoring Framework · Random Forest Classifier (n_estimators=200, max_depth=10)
    · Built for research &amp; educational purposes — not a real-world credit decision system.
</div>
""", unsafe_allow_html=True)