import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="PSO-IDS | Network Intrusion Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART = Path(__file__).parent

# ============================================================
# THEME / CSS  — control-room console aesthetic
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
:root{
  --bg-deep:#0A0E17;
  --bg-panel:#121826;
  --bg-panel-alt:#182031;
  --border:rgba(255,255,255,0.08);
  --text-primary:#E7ECF5;
  --text-muted:#8592AC;
  --safe:#2DD4BF;
  --safe-dim:rgba(45,212,191,0.12);
  --alert:#FB5B6E;
  --alert-dim:rgba(251,91,110,0.12);
  --optim:#F5B93D;
  --optim-dim:rgba(245,185,61,0.12);
}

html, body, [class*="css"]{
  font-family:'Inter', sans-serif;
}

.stApp{
  background:
    radial-gradient(circle at 15% 0%, rgba(45,212,191,0.06), transparent 45%),
    radial-gradient(circle at 85% 15%, rgba(245,185,61,0.05), transparent 40%),
    var(--bg-deep);
  color:var(--text-primary);
}

section[data-testid="stSidebar"]{
  background:#0D1220;
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label{
  font-family:'IBM Plex Mono', monospace;
  font-size:0.92rem;
}

h1,h2,h3,h4{
  font-family:'IBM Plex Mono', monospace;
  letter-spacing:-0.01em;
  color:var(--text-primary);
}

.terminal-tag{
  font-family:'IBM Plex Mono', monospace;
  font-size:0.72rem;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:var(--safe);
  border:1px solid rgba(45,212,191,0.35);
  background:var(--safe-dim);
  padding:3px 10px;
  border-radius:3px;
  display:inline-block;
}

.hero-wrap{
  display:flex;
  align-items:center;
  gap:2.5rem;
  padding:1.2rem 0 2.2rem 0;
  border-bottom:1px solid var(--border);
  margin-bottom:1.8rem;
}
.hero-title{
  font-size:2.5rem;
  font-weight:700;
  line-height:1.15;
  margin:0.6rem 0 0.7rem 0;
}
.hero-sub{
  color:var(--text-muted);
  font-size:1.02rem;
  max-width:640px;
  line-height:1.55;
}
.hero-accent{color:var(--safe);}

/* Radar / swarm signature element */
.radar-box{
  position:relative;
  width:190px; height:190px;
  min-width:190px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(45,212,191,0.10) 0%, rgba(10,14,23,0) 70%);
  border:1px solid rgba(45,212,191,0.25);
}
.radar-ring{
  position:absolute; border-radius:50%;
  border:1px solid rgba(45,212,191,0.18);
}
.r1{inset:14px;} .r2{inset:38px;} .r3{inset:62px;} .r4{inset:86px;}
.radar-sweep{
  position:absolute; inset:0; border-radius:50%;
  background:conic-gradient(from 0deg, rgba(45,212,191,0.55), transparent 32%);
  animation:spin 3.4s linear infinite;
}
@keyframes spin{ to{ transform:rotate(360deg);} }
.particle{
  position:absolute; width:6px; height:6px; border-radius:50%;
  background:var(--optim);
  box-shadow:0 0 8px rgba(245,185,61,0.9);
  animation:converge 3.6s ease-in-out infinite;
}
@keyframes converge{
  0%{ transform:translate(var(--sx),var(--sy)); opacity:0.9;}
  55%{ transform:translate(calc(var(--sx)*0.25),calc(var(--sy)*0.25)); opacity:1;}
  100%{ transform:translate(0,0); opacity:0.15;}
}
.radar-core{
  position:absolute; left:50%; top:50%; width:8px; height:8px;
  margin:-4px 0 0 -4px; border-radius:50%;
  background:var(--alert); box-shadow:0 0 12px rgba(251,91,110,0.85);
}

.stat-row{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.6rem; }
.stat-card{
  flex:1; min-width:170px;
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:8px;
  padding:1rem 1.2rem;
}
.stat-label{
  font-family:'IBM Plex Mono', monospace;
  font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--text-muted); margin-bottom:0.35rem;
}
.stat-value{
  font-family:'IBM Plex Mono', monospace;
  font-size:1.9rem; font-weight:600; color:var(--text-primary);
}
.stat-delta{ font-size:0.78rem; color:var(--safe); margin-top:0.15rem;}

.panel{
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:10px;
  padding:1.4rem 1.6rem;
  margin-bottom:1.3rem;
}
.panel-title{
  font-family:'IBM Plex Mono', monospace;
  font-size:0.95rem; color:var(--text-primary);
  margin-bottom:0.9rem; font-weight:600;
}

.verdict-safe{
  border:1px solid rgba(45,212,191,0.4); background:var(--safe-dim);
  color:var(--safe); font-family:'IBM Plex Mono', monospace;
  padding:0.9rem 1.2rem; border-radius:8px; font-size:1.05rem; font-weight:600;
}
.verdict-alert{
  border:1px solid rgba(251,91,110,0.45); background:var(--alert-dim);
  color:var(--alert); font-family:'IBM Plex Mono', monospace;
  padding:0.9rem 1.2rem; border-radius:8px; font-size:1.05rem; font-weight:600;
}

hr{ border-color:var(--border); }

.small-muted{ color:var(--text-muted); font-size:0.85rem; }

[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono', monospace; color:var(--text-primary); }
.stButton>button{
  font-family:'IBM Plex Mono', monospace;
  background:var(--safe); color:#04211D; border:none; font-weight:600;
  border-radius:6px;
}
.stButton>button:hover{ background:#22b8a4; color:#04211D; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open(ART / "rf_pso_model.pkl", "rb") as f: rf_pso = pickle.load(f)
    with open(ART / "rf_baseline.pkl", "rb") as f: rf_baseline = pickle.load(f)
    with open(ART / "dt_baseline.pkl", "rb") as f: dt_baseline = pickle.load(f)
    with open(ART / "xgb_baseline.pkl", "rb") as f: xgb_baseline = pickle.load(f)
    with open(ART / "rf_multiclass_model.pkl", "rb") as f: rf_multi = pickle.load(f)
    with open(ART / "label_encoder.pkl", "rb") as f: le = pickle.load(f)
    with open(ART / "selected_features.pkl", "rb") as f: selected_features = pickle.load(f)
    with open(ART / "train_columns.pkl", "rb") as f: train_columns = pickle.load(f)
    meta = json.loads((ART / "meta.json").read_text())
    results = json.loads((ART / "results_summary.json").read_text())
    return dict(rf_pso=rf_pso, rf_baseline=rf_baseline, dt_baseline=dt_baseline,
                xgb_baseline=xgb_baseline, rf_multi=rf_multi, le=le,
                selected_features=selected_features, train_columns=train_columns,
                meta=meta, results=results)

ARTIFACTS = load_artifacts()
META = ARTIFACTS["meta"]
RESULTS = ARTIFACTS["results"]

KDD_COLUMNS_41 = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
]
CAT_COLS = ["protocol_type", "service", "flag"]

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Take raw NSL-KDD-format rows (41 feature columns) and produce the
    one-hot-encoded, PSO-feature-selected matrix the model expects."""
    df = df_raw.copy()
    enc = pd.get_dummies(df, columns=CAT_COLS)
    enc = enc.reindex(columns=ARTIFACTS["train_columns"], fill_value=0)
    return enc[ARTIFACTS["selected_features"]]

def predict(df_raw: pd.DataFrame):
    X = preprocess(df_raw)
    proba = ARTIFACTS["rf_pso"].predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    multi_pred = ARTIFACTS["rf_multi"].predict(X)
    multi_label = ARTIFACTS["le"].inverse_transform(multi_pred)
    return pred, proba, multi_label

# ============================================================
# SIDEBAR NAV
# ============================================================
with st.sidebar:
    st.markdown("### ◈ PSO&#8209;IDS")
    st.markdown('<span class="small-muted">Intrusion Detection Console</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("navigate", [
        "Overview", "Live Detection", "Model Comparison", "PSO Insights", "Attack Categories", "About"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f'<span class="small-muted">Dataset: NSL-KDD<br>Train rows: {META["n_train_rows"]:,}<br>'
        f'Test rows: {META["n_test_rows"]:,}</span>', unsafe_allow_html=True
    )

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "Overview":
    pso_row = RESULTS["Random Forest (PSO-Optimized)"]
    rf_row = RESULTS["Random Forest (Baseline)"]
    reduction_pct = round(100 * (1 - META["n_features_selected"] / META["n_features_total"]))

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<span class="terminal-tag">$ system.status — optimized</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Particle Swarm<br>Optimization&#8209;Enhanced<br>'
            '<span class="hero-accent">Intrusion Detection</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">A hybrid machine learning framework that lets a swarm of '
            'candidate solutions search for the optimal feature subset and Random Forest '
            'hyperparameters — cutting the feature space and matching hand-tuned baselines, '
            'trained and evaluated on the NSL-KDD network traffic dataset.</div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="radar-box">
          <div class="radar-ring r1"></div><div class="radar-ring r2"></div>
          <div class="radar-ring r3"></div><div class="radar-ring r4"></div>
          <div class="radar-sweep"></div>
          <div class="particle" style="--sx:70px; --sy:-40px; top:50%; left:50%; animation-delay:0s;"></div>
          <div class="particle" style="--sx:-60px; --sy:60px; top:50%; left:50%; animation-delay:0.6s;"></div>
          <div class="particle" style="--sx:50px; --sy:65px; top:50%; left:50%; animation-delay:1.2s;"></div>
          <div class="particle" style="--sx:-70px; --sy:-30px; top:50%; left:50%; animation-delay:1.8s;"></div>
          <div class="particle" style="--sx:20px; --sy:-75px; top:50%; left:50%; animation-delay:2.4s;"></div>
          <div class="radar-core"></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    stats = [
        ("Detection Accuracy", f'{pso_row["Accuracy"]*100:.1f}%', "PSO-optimized Random Forest"),
        ("ROC-AUC", f'{pso_row["ROC-AUC"]:.3f}', "Test set"),
        ("Features Selected", f'{META["n_features_selected"]}/{META["n_features_total"]}', f'{reduction_pct}% reduction'),
        ("PSO Validation Acc", f'{META["pso_best_val_accuracy"]*100:.2f}%', f'{len(META["cost_history"])} iterations'),
    ]
    cols = st.columns(4)
    for c, (label, val, delta) in zip(cols, stats):
        with c:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-label">{label}</div>
              <div class="stat-value">{val}</div>
              <div class="stat-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
- **Ingest** raw NSL-KDD traffic records (41 features)
- **Encode** categorical protocol / service / flag fields
- **PSO search** selects the best feature subset + Random Forest hyperparameters
- **Classify** Normal vs Attack, then break attacks into DoS / Probe / R2L / U2R
- **Serve** predictions live through this console
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-title">Why PSO?</div>', unsafe_allow_html=True)
        st.markdown(f"""
Random Forest alone needs someone to guess `n_estimators`, `max_depth`, and which
of the {META['n_features_total']} engineered traffic features actually matter.

PSO treats each guess as a *particle* in a swarm, evaluates it against a validation
set, and lets particles pull toward whatever the swarm has found best so far —
landing on **{META['n_features_selected']} features** and
**n_estimators={META['best_n_estimators']}, max_depth={META['best_max_depth']}**
without manual grid search.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: LIVE DETECTION
# ============================================================
elif page == "Live Detection":
    st.markdown('<span class="terminal-tag">$ traffic.scan</span>', unsafe_allow_html=True)
    st.markdown("## Live Detection")
    st.markdown('<span class="small-muted">Classify network traffic records as Normal or an Attack, '
                'and identify the likely attack category.</span>', unsafe_allow_html=True)
    st.write("")

    tab1, tab2 = st.tabs(["Upload traffic log (CSV)", "Quick scan (manual)"])

    with tab1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload a CSV of NSL-KDD-format traffic records</div>', unsafe_allow_html=True)
        st.markdown('<span class="small-muted">41 columns, comma-separated, no header, in standard KDD order '
                     '(same layout as KDDTest+.txt). Label/difficulty columns, if present, are ignored.</span>',
                     unsafe_allow_html=True)
        up = st.file_uploader("traffic_log.csv", type=["csv", "txt"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if up is not None:
            raw = pd.read_csv(up, header=None)
            if raw.shape[1] >= 41:
                raw = raw.iloc[:, :41]
            raw.columns = KDD_COLUMNS_41
            st.markdown(f'<span class="small-muted">Loaded {len(raw):,} records.</span>', unsafe_allow_html=True)

            if st.button("Run detection", key="run_csv"):
                pred, proba, multi_label = predict(raw)
                out = raw[["protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes"]].copy()
                out["verdict"] = np.where(pred == 1, "Attack", "Normal")
                out["attack_probability"] = np.round(proba, 4)
                out["attack_category"] = multi_label
                out.loc[out["verdict"] == "Normal", "attack_category"] = "normal"

                n_attack = int(pred.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Records scanned", f"{len(raw):,}")
                c2.metric("Flagged as Attack", f"{n_attack:,}")
                c3.metric("Attack rate", f"{100*n_attack/len(raw):.1f}%")

                st.dataframe(out, use_container_width=True, height=380)
                st.download_button("Download results as CSV", out.to_csv(index=False).encode(),
                                    file_name="detection_results.csv", mime="text/csv")

    with tab2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Describe one connection</div>', unsafe_allow_html=True)
        st.markdown('<span class="small-muted">Set the fields that matter most for detection — everything else '
                     'is filled with typical baseline values.</span>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
            service = st.selectbox("Service", ["http", "ftp_data", "private", "smtp", "domain_u",
                                                 "ftp", "telnet", "other", "eco_i", "finger"])
            flag = st.selectbox("Connection flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH"])
        with c2:
            duration = st.number_input("Duration (s)", 0, 60000, 0)
            src_bytes = st.number_input("Src bytes", 0, 2_000_000, 200)
            dst_bytes = st.number_input("Dst bytes", 0, 2_000_000, 0)
            logged_in = st.selectbox("Logged in?", [1, 0])
        with c3:
            count = st.slider("Count (conns to same host, last 2s)", 0, 511, 5)
            srv_count = st.slider("Srv count (conns to same service)", 0, 511, 5)
            serror_rate = st.slider("SYN error rate", 0.0, 1.0, 0.0)
            same_srv_rate = st.slider("Same-service rate", 0.0, 1.0, 1.0)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Run detection", key="run_manual"):
            row = {c: 0 for c in KDD_COLUMNS_41}
            row.update(dict(
                protocol_type=protocol_type, service=service, flag=flag,
                duration=duration, src_bytes=src_bytes, dst_bytes=dst_bytes,
                logged_in=logged_in, count=count, srv_count=srv_count,
                serror_rate=serror_rate, same_srv_rate=same_srv_rate,
                srv_serror_rate=serror_rate, diff_srv_rate=1 - same_srv_rate,
                dst_host_count=count, dst_host_srv_count=srv_count,
                dst_host_same_srv_rate=same_srv_rate,
            ))
            single = pd.DataFrame([row])[KDD_COLUMNS_41]
            pred, proba, multi_label = predict(single)
            p = float(proba[0])

            gcol, vcol = st.columns([1, 1.3])
            with gcol:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p * 100,
                    number={'suffix': "%", 'font': {'family': 'IBM Plex Mono', 'color': '#E7ECF5'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#8592AC'},
                        'bar': {'color': '#FB5B6E' if p >= 0.5 else '#2DD4BF'},
                        'bgcolor': '#121826',
                        'borderwidth': 1, 'bordercolor': 'rgba(255,255,255,0.08)',
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(45,212,191,0.12)'},
                            {'range': [50, 100], 'color': 'rgba(251,91,110,0.12)'},
                        ],
                    },
                    title={'text': "Attack probability", 'font': {'family': 'Inter', 'size': 14, 'color': '#8592AC'}}
                ))
                fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10),
                                   paper_bgcolor='rgba(0,0,0,0)', font_color="#E7ECF5")
                st.plotly_chart(fig, use_container_width=True)
            with vcol:
                st.write("")
                st.write("")
                if p >= 0.5:
                    st.markdown(f'<div class="verdict-alert">⚠ ATTACK DETECTED — likely category: '
                                 f'{multi_label[0].upper()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-safe">✓ NORMAL TRAFFIC</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<span class="small-muted">Confidence: {max(p, 1-p)*100:.1f}% · '
                             f'Model: Random Forest (PSO-optimized, {META["n_features_selected"]} features)'
                             f'</span>', unsafe_allow_html=True)

# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.markdown('<span class="terminal-tag">$ models.benchmark</span>', unsafe_allow_html=True)
    st.markdown("## Model Comparison")
    st.markdown('<span class="small-muted">Decision Tree and Random Forest baselines vs XGBoost vs the '
                'PSO-optimized Random Forest, all evaluated on the same held-out NSL-KDD test set.</span>',
                unsafe_allow_html=True)
    st.write("")

    df = pd.DataFrame(RESULTS).T.reset_index().rename(columns={"index": "Model"})
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

    fig = go.Figure()
    colors = ['#3B4A66', '#5B6C8F', '#F5B93D', '#2DD4BF']
    for i, model in enumerate(df["Model"]):
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=[df.loc[df["Model"] == model, m].values[0] for m in metrics],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode='group', height=440,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E7ECF5", legend=dict(orientation="h", y=1.15),
        yaxis=dict(range=[0, 1.05], gridcolor='rgba(255,255,255,0.06)'),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Full results table</div>', unsafe_allow_html=True)
    st.dataframe(df.set_index("Model").style.format("{:.4f}").background_gradient(
        cmap="BuGn", subset=metrics), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    best_model = df.loc[df["Accuracy"].idxmax(), "Model"]
    st.markdown(f'<span class="small-muted">Highest test accuracy: <b>{best_model}</b>. The PSO-optimized model '
                f'reaches comparable performance to XGBoost while training on only '
                f'{META["n_features_selected"]} of {META["n_features_total"]} features.</span>',
                unsafe_allow_html=True)

# ============================================================
# PAGE: PSO INSIGHTS
# ============================================================
elif page == "PSO Insights":
    st.markdown('<span class="terminal-tag">$ pso.trace</span>', unsafe_allow_html=True)
    st.markdown("## PSO Insights")
    st.write("")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">Convergence — fitness per iteration</div>', unsafe_allow_html=True)
        hist = META["cost_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[1 - c for c in hist], mode="lines+markers",
            line=dict(color="#2DD4BF", width=3), marker=dict(size=6, color="#F5B93D"),
            name="Validation accuracy"
        ))
        fig.update_layout(
            height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="#E7ECF5",
            xaxis=dict(title="Iteration", gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(title="Best validation accuracy", gridcolor='rgba(255,255,255,0.06)'),
            margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title">Feature space reduction</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Selected by PSO", "Dropped"],
            values=[META["n_features_selected"], META["n_features_total"] - META["n_features_selected"]],
            hole=0.62, marker=dict(colors=["#2DD4BF", "#1E2637"]),
            textfont=dict(color="#E7ECF5", family="IBM Plex Mono"),
        ))
        fig2.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)', font_color="#E7ECF5",
            showlegend=True, legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=10),
            annotations=[dict(text=f'{META["n_features_selected"]}/{META["n_features_total"]}',
                               font=dict(size=20, color="#E7ECF5", family="IBM Plex Mono"),
                               showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Top feature importances — PSO-optimized Random Forest</div>', unsafe_allow_html=True)
    importances = pd.Series(
        ARTIFACTS["rf_pso"].feature_importances_, index=ARTIFACTS["selected_features"]
    ).sort_values(ascending=False).head(15)
    fig3 = px.bar(x=importances.values[::-1], y=importances.index[::-1], orientation="h",
                   color=importances.values[::-1], color_continuous_scale=["#182031", "#2DD4BF"])
    fig3.update_layout(
        height=440, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E7ECF5", coloraxis_showscale=False,
        xaxis_title="Importance", yaxis_title="", margin=dict(t=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="panel">
      <div class="panel-title">Tuned hyperparameters</div>
      <span class="small-muted">
      PSO searched jointly over the feature mask and these Random Forest hyperparameters —
      <b>n_estimators = {META['best_n_estimators']}</b>, <b>max_depth = {META['best_max_depth']}</b> —
      converging to <b>{META['pso_best_val_accuracy']*100:.2f}%</b> validation accuracy over
      {len(META['cost_history'])} iterations.
      </span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: ATTACK CATEGORIES
# ============================================================
elif page == "Attack Categories":
    st.markdown('<span class="terminal-tag">$ attacks.classify --multiclass</span>', unsafe_allow_html=True)
    st.markdown("## Attack Category Breakdown")
    st.markdown('<span class="small-muted">Beyond Normal vs Attack, the PSO-selected features also drive a '
                'multi-class model that separates traffic into DoS, Probe, R2L, and U2R.</span>',
                unsafe_allow_html=True)
    st.write("")

    report = META["multiclass_report"]
    labels = [l for l in META["multiclass_labels"]]
    rows = []
    for l in labels:
        if l in report:
            rows.append({"Category": l, **{k: v for k, v in report[l].items() if k != "support"},
                         "Support": int(report[l]["support"])})
    rdf = pd.DataFrame(rows).set_index("Category")

    fig = go.Figure()
    for metric, color in [("precision", "#2DD4BF"), ("recall", "#F5B93D"), ("f1-score", "#FB5B6E")]:
        fig.add_trace(go.Bar(name=metric.capitalize(), x=rdf.index, y=rdf[metric],
                              marker_color=color))
    fig.update_layout(
        barmode='group', height=420,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E7ECF5", legend=dict(orientation="h", y=1.12),
        yaxis=dict(range=[0, 1.05], gridcolor='rgba(255,255,255,0.06)'),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel"><div class="panel-title">Per-category metrics</div>', unsafe_allow_html=True)
    st.dataframe(rdf.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                 use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="small-muted">R2L and U2R are rare in NSL-KDD\'s training data, so recall on '
                'those categories is typically the hardest to push up — a known, documented limitation of '
                'this dataset rather than the modeling approach.</span>', unsafe_allow_html=True)

# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "About":
    st.markdown('<span class="terminal-tag">$ project.readme</span>', unsafe_allow_html=True)
    st.markdown("## About This Project")
    st.write("")

    st.markdown("""
    <div class="panel">
    <div class="panel-title">Particle Swarm Optimization-Enhanced Machine Learning Framework
    for Network Intrusion Detection</div>
    <span class="small-muted">
    A hybrid ML security system that classifies network traffic as Normal or Attack, and further
    breaks attacks into DoS, Probe, R2L, and U2R categories. Particle Swarm Optimization (PSO) — a
    nature-inspired algorithm modeled on flocking behavior — searches jointly over the feature space
    and Random Forest hyperparameters, removing the need for manual feature selection or grid search.
    </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="panel">
        <div class="panel-title">Tech stack</div>
        <span class="small-muted">
        <b>Frontend</b> — Streamlit + Plotly<br>
        <b>ML models</b> — Decision Tree, Random Forest, XGBoost (baselines)<br>
        <b>Optimization</b> — Particle Swarm Optimization (pyswarms)<br>
        <b>Data</b> — Pandas, NumPy, scikit-learn<br>
        <b>Dataset</b> — NSL-KDD
        </span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="panel">
        <div class="panel-title">Limitations</div>
        <span class="small-muted">
        NSL-KDD reflects late-1990s traffic patterns and doesn't include modern attack types
        (e.g. encrypted C2 traffic, modern DDoS botnets). For production relevance, retrain the
        same pipeline on CICIDS2017/2018.
        </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="small-muted">Built as an academic ML systems project — PSO optimization, '
                'ensemble learning, and a live detection console in one pipeline.</span>', unsafe_allow_html=True)
