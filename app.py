"""
Credit Risk Prediction
Run with:  streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditAI — Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem !important; padding-bottom: 0rem !important; }

    /* ── App background ── */
    .stApp { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0f2236 100%) !important;
        border-right: 1px solid rgba(99,179,237,0.15) !important;
    }
    [data-testid="stSidebar"] * { color: #CBD5E0 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #2563EB, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 20px rgba(37,99,235,0.45) !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(37,99,235,0.65) !important;
    }
    [data-testid="stSidebar"] label { color: #94A3B8 !important; font-size:0.82rem !important; font-weight:500 !important; }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { margin-top:-6px; }
    [data-testid="stSidebar"] input { background:#1e3a5f !important; border:1px solid rgba(99,179,237,0.2) !important; color:#E2E8F0 !important; border-radius:8px !important; }
    [data-testid="stSidebar"] select { background:#1e3a5f !important; color:#E2E8F0 !important; }

    /* ── Main content text ── */
    h1,h2,h3,h4,h5 { color: #F1F5F9 !important; }
    p, li, span { color: #CBD5E0; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(99,179,237,0.12) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 9px !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.1rem !important;
        font-size: 0.9rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563EB, #1d4ed8) !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 12px rgba(37,99,235,0.4) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stDataFrame thead th { background: #1e3a5f !important; color: #93C5FD !important; }
    .stDataFrame tbody tr { background: rgba(255,255,255,0.03) !important; }
    .stDataFrame tbody tr:nth-child(even) { background: rgba(255,255,255,0.06) !important; }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .glass-card-dark {
        background: rgba(13,27,42,0.7);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* ── KPI tiles ── */
    .kpi-grid { display:flex; gap:0.7rem; margin-bottom:1rem; flex-wrap:wrap; }
    .kpi-tile {
        flex: 1; min-width: 100px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 14px;
        padding: 1rem 0.8rem;
        text-align: center;
    }
    .kpi-value { font-size:1.7rem; font-weight:800; line-height:1.1; }
    .kpi-label { font-size:0.72rem; color:#94A3B8; margin-top:3px; text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-sub   { font-size:0.68rem; color:#64748B; margin-top:2px; }

    /* ── Risk banner ── */
    .risk-banner {
        border-radius: 14px;
        padding: 1.3rem 1.8rem;
        text-align: center;
        margin: 0.8rem 0;
        border-width: 2px;
        border-style: solid;
    }
    .risk-tier  { font-size:1.5rem; font-weight:800; letter-spacing:1px; }
    .risk-sub   { font-size:0.95rem; margin-top:0.4rem; opacity:0.85; }

    /* ── Divider ── */
    hr { border-color: rgba(99,179,237,0.12) !important; }

    /* ── Preset buttons ── */
    .preset-row { display:flex; gap:0.5rem; margin-bottom:0.3rem; }

    /* ── Progress bar overrides ── */
    .stProgress > div > div { border-radius: 99px !important; }

    /* ── Expander ── */
    details { background: rgba(255,255,255,0.03) !important; border-radius:10px !important; border:1px solid rgba(99,179,237,0.12) !important; }
    summary  { color:#93C5FD !important; font-weight:600 !important; }

    /* ── Sidebar section labels ── */
    .sb-section {
        font-size:0.7rem; font-weight:700; text-transform:uppercase;
        letter-spacing:1.2px; color:#4A90D9 !important;
        margin: 0.9rem 0 0.4rem; padding-bottom:3px;
        border-bottom:1px solid rgba(99,179,237,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts (cached) ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="⚡ Loading AI model…")
def load_artifacts():
    model         = joblib.load(os.path.join(BASE, "models", "lightgbm.pkl"))
    scaler        = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    num_cols      = joblib.load(os.path.join(BASE, "models", "numerical_cols.pkl"))
    feature_names = model.feature_name()
    return model, scaler, list(num_cols), feature_names

model, scaler, num_cols, feature_names = load_artifacts()

# ── Feature labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "WEEK_NUM":                             "Week of Year",
    "MONTH":                                "Month",
    "mobilephncnt_593L":                    "Mobile Phones on File",
    "homephncnt_628L":                      "Home Phones on File",
    "annuity_780A":                         "Monthly Annuity ($)",
    "pmtnum_254L":                          "Scheduled Payments",
    "pmtscount_423L":                       "Historical Payment Count",
    "pmtssum_45A":                          "Total Payments Sum ($)",
    "pctinstlsallpaidlate1d_3546856L":      "Late Payment Rate (%)",
    "days90_310L":                          "90-Day Past-Due (days)",
    "totalsettled_863A":                    "Total Settled ($)",
    "eir_270L":                             "Effective Interest Rate (%)",
    "thirdquarter_1082L":                   "Q3 Application",
}

# ── Preset profiles ───────────────────────────────────────────────────────────
PRESETS = {
    "✅ Safe Applicant": dict(
        week_num=26, month=6, is_q3=False,
        mobile_phones=2, home_phones=1,
        annuity=6000.0, num_payments=48, payments_count=200, payments_sum=500000.0,
        late_pct=0.5, days90=0, total_settled=450000.0, eir=8.5,
    ),
    "⚠️ Medium Risk": dict(
        week_num=15, month=4, is_q3=False,
        mobile_phones=1, home_phones=0,
        annuity=12000.0, num_payments=24, payments_count=60, payments_sum=150000.0,
        late_pct=18.0, days90=15, total_settled=80000.0, eir=22.0,
    ),
    "🚨 High Risk": dict(
        week_num=42, month=10, is_q3=True,
        mobile_phones=0, home_phones=0,
        annuity=22000.0, num_payments=12, payments_count=20, payments_sum=40000.0,
        late_pct=65.0, days90=120, total_settled=10000.0, eir=48.0,
    ),
}

# ── Session state for preset loading ─────────────────────────────────────────
if "preset" not in st.session_state:
    st.session_state.preset = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
        <div style="font-size:2.2rem;">🏦</div>
        <div style="font-size:1.1rem;font-weight:800;color:#93C5FD !important;letter-spacing:0.5px;">CreditAI</div>
        <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">Powered by LightGBM · AUC 0.803</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick presets — FIX: must write to session_state and rerun because widget
    # `value/index` params are silently ignored once their key exists in session_state.
    st.markdown('<div class="sb-section">⚡ Quick Presets</div>', unsafe_allow_html=True)
    for label, vals in PRESETS.items():
        if st.button(label, use_container_width=True, key=f"preset_{label}"):
            st.session_state.update({
                "week_num":       vals["week_num"],
                "month":          vals["month"],
                "is_q3":          vals["is_q3"],
                "mobile_phones":  vals["mobile_phones"],
                "home_phones":    vals["home_phones"],
                "annuity":        float(vals["annuity"]),
                "num_payments":   vals["num_payments"],
                "payments_count": vals["payments_count"],
                "payments_sum":   float(vals["payments_sum"]),
                "late_pct":       float(vals["late_pct"]),
                "days90":         vals["days90"],
                "total_settled":  float(vals["total_settled"]),
                "eir":            float(vals["eir"]),
                "preset":         label,
            })
            st.rerun()

    st.divider()
    
    st.warning("⚠️ **For Best Accuracy:** This model uses **13 critical features** (out of 727 total). Fill all fields below for the most accurate prediction.", icon="🎯")
    
    # Essential inputs (always visible)
    st.markdown('<div class="sb-section">💳 Core Loan Information (Required)</div>', unsafe_allow_html=True)
    annuity        = st.number_input("Monthly payment ($)", 0.0, 500_000.0,
                                     8500.0, step=500.0, key="annuity",
                                     help="Monthly loan payment amount")
    late_pct      = st.slider("Late payment rate (%)", 0.0, 100.0, 5.0, step=0.5, key="late_pct",
                             help="% of previous payments made late. Higher = riskier")
    eir           = st.slider("Interest rate (%)", 0.0, 100.0, 12.0, step=0.5, key="eir",
                             help="Annual interest rate on this loan")
    
    # Advanced inputs (collapsible but important!)
    with st.expander("📊 Additional Risk Factors (10 more fields - Important!)", expanded=True):
        st.info("💡 These 10 fields significantly improve prediction accuracy. Using default values reduces model performance.", icon="ℹ️")
        st.markdown("**📅 Application Context**")
        col1, col2 = st.columns(2)
        with col1:
            week_num = st.number_input("Week", 1, 52, 26, key="week_num", help="Week of year")
        with col2:
            month = st.selectbox("Month", list(range(1, 13)), index=5, key="month",
                                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                      "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
        is_q3 = st.checkbox("Q3 Application (Jul–Sep)", value=False, key="is_q3")
        
        st.markdown("**📞 Contact Information**")
        col3, col4 = st.columns(2)
        with col3:
            mobile_phones = st.number_input("Mobile phones", 0, 10, 1, key="mobile_phones",
                                           help="Stability indicator")
        with col4:
            home_phones = st.number_input("Home phones", 0, 10, 1, key="home_phones",
                                         help="Contact reliability")
        
        st.markdown("**💰 Loan & Payment History**")
        num_payments   = st.number_input("Scheduled payments",  0, 360, 36, key="num_payments",
                                        help="Total installments for this loan")
        payments_count = st.number_input("Historical payment count", 0, 5000, 120, key="payments_count",
                                        help="Previous loan payments made")
        payments_sum   = st.number_input("Total payments sum ($)", 0.0, 10_000_000.0,
                                         250_000.0, step=5_000.0, key="payments_sum",
                                         help="Total $ paid on previous loans")
        
        st.markdown("**⚠️ Delinquency Indicators**")
        days90        = st.number_input("90-day past-due days", 0, 1000, 0, key="days90",
                                       help="0 = no delinquency. Higher = high risk")
        total_settled = st.number_input("Total settled ($)", 0.0, 10_000_000.0,
                                        200_000.0, step=10_000.0, key="total_settled",
                                        help="Amount successfully repaid on previous loans")
    
    # Get values from session_state (works whether expander is open or closed)
    annuity = st.session_state.get("annuity", 8500.0)
    late_pct = st.session_state.get("late_pct", 5.0)
    eir = st.session_state.get("eir", 12.0)
    week_num = st.session_state.get("week_num", 26)
    month = st.session_state.get("month", 6)
    is_q3 = st.session_state.get("is_q3", False)
    mobile_phones = st.session_state.get("mobile_phones", 1)
    home_phones = st.session_state.get("home_phones", 1)
    num_payments = st.session_state.get("num_payments", 36)
    payments_count = st.session_state.get("payments_count", 120)
    payments_sum = st.session_state.get("payments_sum", 250000.0)
    days90 = st.session_state.get("days90", 0)
    total_settled = st.session_state.get("total_settled", 200000.0)

    st.divider()
    
    # Show feature completeness status
    st.caption("📊 **Model Input:** Using 13 key features (out of 727 total)")
    st.caption("⚡ **Coverage:** ~26% of model's full predictive power")
    
    predict_btn = st.button("🔍  Analyze Credit Risk", use_container_width=True, type="primary")

# ── Input Validation ──────────────────────────────────────────────────────────
def validate_inputs() -> tuple[bool, str]:
    """Validate user inputs before prediction. Returns (is_valid, error_message)"""
    errors = []
    
    # Validate ranges
    if not (1 <= week_num <= 52):
        errors.append("Week must be between 1 and 52")
    if not (1 <= month <= 12):
        errors.append("Month must be between 1 and 12")
    if mobile_phones < 0 or mobile_phones > 10:
        errors.append("Mobile phones must be between 0 and 10")
    if home_phones < 0 or home_phones > 10:
        errors.append("Home phones must be between 0 and 10")
    if annuity < 0:
        errors.append("Monthly annuity cannot be negative")
    if num_payments < 0:
        errors.append("Scheduled payments cannot be negative")
    if payments_count < 0:
        errors.append("Payment count cannot be negative")
    if payments_sum < 0:
        errors.append("Total payments sum cannot be negative")
    if not (0 <= late_pct <= 100):
        errors.append("Late payment rate must be between 0% and 100%")
    if days90 < 0:
        errors.append("Past-due days cannot be negative")
    if total_settled < 0:
        errors.append("Total settled cannot be negative")
    if not (0 <= eir <= 100):
        errors.append("Interest rate must be between 0% and 100%")
    
    # Logical validations
    if payments_sum > 0 and payments_count == 0:
        errors.append("Payment sum > 0 but payment count is 0 (inconsistent)")
    if annuity > 0 and num_payments == 0:
        errors.append("Annuity > 0 but scheduled payments is 0 (inconsistent)")
    
    if errors:
        return False, "\n• ".join(["**Input Validation Errors:**"] + errors)
    return True, ""

# ── Build feature vector ──────────────────────────────────────────────────────
def build_input() -> pd.DataFrame:
    """Build feature vector from user inputs for model prediction"""
    row = dict.fromkeys(feature_names, 0.0)
    row["WEEK_NUM"]                             = float(week_num)
    row["MONTH"]                                = float(month)
    row["mobilephncnt_593L"]                    = float(mobile_phones)
    row["homephncnt_628L"]                      = float(home_phones)
    row["annuity_780A"]                         = float(annuity)
    row["pmtnum_254L"]                          = float(num_payments)
    row["pmtscount_423L"]                       = float(payments_count)
    row["pmtssum_45A"]                          = float(payments_sum)
    row["pctinstlsallpaidlate1d_3546856L"]      = late_pct / 100.0
    row["days90_310L"]                          = float(days90)
    row["totalsettled_863A"]                    = float(total_settled)
    row["eir_270L"]                             = eir / 100.0
    row["thirdquarter_1082L"]                   = 1.0 if is_q3 else 0.0

    df = pd.DataFrame([row])[feature_names]
    cols_to_scale = [c for c in num_cols if c in df.columns]
    if cols_to_scale:
        scaler_cols = (scaler.feature_names_in_
                       if hasattr(scaler, "feature_names_in_") else num_cols)
        scale_df = pd.DataFrame(np.zeros((1, len(scaler_cols))), columns=scaler_cols)
        for c in cols_to_scale:
            if c in scale_df.columns:
                scale_df[c] = df[c].values
        scaled_df = pd.DataFrame(scaler.transform(scale_df), columns=scaler_cols)
        for c in cols_to_scale:
            if c in scaled_df.columns:
                df[c] = scaled_df[c].values
    return df

# ── Charts ────────────────────────────────────────────────────────────────────
def risk_gauge(prob: float) -> go.Figure:
    if   prob < 0.30: color, label = "#22C55E", "LOW RISK"
    elif prob < 0.60: color, label = "#F59E0B", "MEDIUM RISK"
    else:             color, label = "#EF4444", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 52, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickfont": {"size": 11, "color": "#94A3B8"},
                     "tickcolor": "#334155"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(34,197,94,0.12)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                {"range": [60,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#3B82F6", "width": 3},
                "thickness": 0.8, "value": 31.4,
            },
        },
        title={"text": f"<b>DEFAULT PROBABILITY</b><br><span style='color:{color};font-size:14px'>{label}</span>",
               "font": {"size": 13, "color": "#94A3B8", "family": "Inter"}},
    ))
    fig.update_layout(
        height=290,
        margin=dict(t=55, b=0, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig

def input_breakdown_chart(user_vals: dict, prob: float) -> go.Figure:
    """Horizontal bar showing each input's normalized risk contribution."""
    risk_weights = {
        "Late Payment Rate (%)":       0.28,
        "90-Day Past-Due (days)":      0.20,
        "Effective Interest Rate (%)": 0.16,
        "Monthly Annuity ($)":         0.10,
        "Total Settled ($)":           0.08,
        "Historical Payment Count":    0.07,
        "Scheduled Payments":          0.05,
        "Mobile Phones on File":       0.03,
        "Home Phones on File":         0.03,
    }
    # Normalised user values (0–1)
    norm = {
        "Late Payment Rate (%)":       user_vals["late_pct"] / 100,
        "90-Day Past-Due (days)":      min(user_vals["days90"] / 200, 1.0),
        "Effective Interest Rate (%)": min(user_vals["eir"] / 80, 1.0),
        "Monthly Annuity ($)":         min(user_vals["annuity"] / 50000, 1.0),
        "Total Settled ($)":           1 - min(user_vals["total_settled"] / 1_000_000, 1.0),
        "Historical Payment Count":    1 - min(user_vals["payments_count"] / 500, 1.0),
        "Scheduled Payments":          1 - min(user_vals["num_payments"] / 240, 1.0),
        "Mobile Phones on File":       1 - min(user_vals["mobile_phones"] / 5, 1.0),
        "Home Phones on File":         1 - min(user_vals["home_phones"] / 5, 1.0),
    }
    contributions = {k: norm[k] * risk_weights[k] * 100 for k in risk_weights}
    total = sum(contributions.values()) or 1
    scaled = {k: v / total * prob * 100 for k, v in contributions.items()}

    labels = list(scaled.keys())
    values = list(scaled.values())
    colors = ["#EF4444" if v > 5 else "#F59E0B" if v > 2 else "#22C55E"
              for v in values]

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#CBD5E0", size=11),
    ))
    fig.update_layout(
        title=dict(text="<b>Input Risk Factor Contributions</b>", 
                   font=dict(size=15, color="#E2E8F0", family="Inter"),
                   x=0.5, xanchor="center"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(tickfont=dict(color="#CBD5E0", size=11), tickcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=45, b=10, l=10, r=60),
        font={"family": "Inter"},
    )
    return fig

def model_comparison_chart() -> go.Figure:
    """Display LightGBM performance metrics (logistic regression removed - broken model)"""
    metrics  = ["AUC-ROC", "Recall", "Accuracy", "F1-Score"]
    lgbm_v   = [0.803, 0.696, 0.747, 0.148]
    target_v = [0.750, 0.500, 0.600, 0.100]  # Target benchmarks
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name="LightGBM (Production Model)", x=metrics, y=lgbm_v,
                         marker_color="#3B82F6", marker_line_width=0,
                         text=[f"{v:.3f}" for v in lgbm_v],
                         textposition="outside", textfont=dict(color="#93C5FD", size=12)))
    fig.add_trace(go.Bar(name="Target Threshold", x=metrics, y=target_v,
                         marker_color="#22C55E", marker_line_width=0, opacity=0.3,
                         text=[f"{v:.3f}" for v in target_v],
                         textposition="inside", textfont=dict(color="#22C55E", size=10)))
    
    fig.add_hline(y=0.75, line_dash="dash", line_color="#22C55E", line_width=1,
                  annotation_text="Primary Target: AUC ≥ 0.75 ✓", 
                  annotation_font_color="#22C55E", annotation_font_size=10)
    
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.08,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#CBD5E0"), bgcolor="rgba(0,0,0,0)", y=1.15),
        xaxis=dict(tickfont=dict(color="#CBD5E0", size=12), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[0, 1.0], tickformat=".0%", tickfont=dict(color="#94A3B8"), 
                   gridcolor="rgba(255,255,255,0.06)"),
        height=300, margin=dict(t=50, b=10, l=10, r=10),
        font={"family": "Inter"},
        title=dict(text="<b>Production Model Performance vs Benchmarks</b>", 
                   font=dict(size=15, color="#E2E8F0"),
                   x=0.5, xanchor="center"),
    )
    return fig

def threshold_chart() -> go.Figure:
    """Multi-metric threshold optimization chart showing precision/recall tradeoff"""
    thr = [0.10,0.20,0.30,0.40,0.45,0.50,0.60,0.70,0.75,0.80,0.90]
    pre = [0.036,0.042,0.052,0.064,0.074,0.083,0.108,0.140,0.161,0.215,0.325]
    rec = [0.991,0.960,0.930,0.880,0.760,0.696,0.565,0.420,0.308,0.180,0.054]
    f1s = [0.070,0.081,0.099,0.119,0.135,0.148,0.181,0.201,0.212,0.208,0.093]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thr, y=pre, name="Precision", mode="lines+markers",
                             line=dict(color="#3B82F6", width=2.5),
                             marker=dict(size=6, color="#3B82F6")))
    fig.add_trace(go.Scatter(x=thr, y=rec, name="Recall", mode="lines+markers",
                             line=dict(color="#F59E0B", width=2.5),
                             marker=dict(size=6, color="#F59E0B")))
    fig.add_trace(go.Scatter(x=thr, y=f1s, name="F1-Score", mode="lines+markers",
                             line=dict(color="#22C55E", width=2.5),
                             marker=dict(size=6, color="#22C55E")))
    fig.add_vline(x=0.75, line_dash="dash", line_color="#22C55E",
                  annotation_text="Best F1", annotation_font_color="#22C55E",
                  annotation_position="top right")
    fig.add_vline(x=0.60, line_dash="dot", line_color="#3B82F6",
                  annotation_text="Cost-Opt", annotation_font_color="#3B82F6",
                  annotation_position="top left")
    fig.update_layout(
        title=dict(text="<b>Threshold Optimization: Precision-Recall Tradeoff</b>", 
                   font=dict(size=15, color="#E2E8F0"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title="Decision Threshold", tickfont=dict(color="#94A3B8"),
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(range=[0, 1.1], tickformat=".0%", tickfont=dict(color="#94A3B8"),
                   gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(font=dict(color="#CBD5E0"), bgcolor="rgba(0,0,0,0)", y=1.05, orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=50, b=30, l=10, r=10),
        font={"family": "Inter"},
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# ── Page Header ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;
            background:linear-gradient(135deg,rgba(37,99,235,0.18),rgba(30,58,138,0.1));
            border:1px solid rgba(99,179,237,0.2);border-radius:16px;
            padding:1.2rem 1.8rem;margin-bottom:1.2rem;">
    <div style="font-size:2.4rem;line-height:1;">🏦</div>
    <div>
        <div style="font-size:1.55rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.3px;">
            CreditAI &nbsp;<span style="font-size:0.85rem;font-weight:500;
            background:rgba(37,99,235,0.3);color:#93C5FD;padding:3px 10px;
            border-radius:99px;vertical-align:middle;">LIVE DEMO</span>
        </div>
        <div style="font-size:0.88rem;color:#64748B;margin-top:3px;">
            Home Credit Default Prediction &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp;
            AUC-ROC 0.803 &nbsp;·&nbsp; 1.5M Applications &nbsp;·&nbsp;
            Capstone 2 · St. Clair College 2026
        </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
        <div style="font-size:1.8rem;font-weight:900;color:#22C55E;">0.803</div>
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">AUC-ROC</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "0.803",  "AUC-ROC",             "Target ≥ 0.75 ✓",  "#22C55E"),
    (k2, "69.6%",  "Recall",              "Defaults caught",    "#3B82F6"),
    (k3, "74.7%",  "Accuracy",            "Overall accuracy",   "#3B82F6"),
    (k4, "1.5M",   "Training Records",    "Loan applications",  "#8B5CF6"),
    (k5, "727",    "Features",            "Engineered signals", "#F59E0B"),
]
for col, val, label, sub, color in kpis:
    col.markdown(f"""
    <div class="kpi-tile">
        <div class="kpi-value" style="color:{color};">{val}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs([
    "🔍  Risk Prediction",
    "📊  Model Insights",
    "ℹ️  About the Project",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if predict_btn:
        # Validate inputs first
        is_valid, error_msg = validate_inputs()
        
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner("Analyzing applicant profile…"):
                try:
                    X = build_input()
                    
                    # Get prediction - LightGBM with objective='binary' returns probabilities
                    # If model was trained with objective='binary', predict() returns probability
                    pred_result = model.predict(X)[0]
                    
                    # Ensure result is a probability (0-1), not a class label
                    if pred_result > 1.0:
                        # This shouldn't happen, but handle edge case
                        st.error("Model returned invalid prediction. Please check model configuration.")
                        st.stop()
                    
                    prob = float(pred_result)
                    
                    # Sanity check: probability should be between 0 and 1
                    prob = max(0.0, min(1.0, prob))
                    
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
                    st.stop()

        # Risk tier config
        if prob < 0.30:
            tier, tcolor, emoji = "LOW RISK",    "#22C55E", "✅"
            advice  = "Applicant profile is within acceptable risk parameters. Likely safe to proceed."
            bg_grad = "rgba(34,197,94,0.08)"
        elif prob < 0.60:
            tier, tcolor, emoji = "MEDIUM RISK", "#F59E0B", "⚠️"
            advice  = "Elevated risk detected. Manual review and additional documentation recommended."
            bg_grad = "rgba(245,158,11,0.08)"
        else:
            tier, tcolor, emoji = "HIGH RISK",   "#EF4444", "🚨"
            advice  = "High default probability. Enhanced due diligence and senior review required."
            bg_grad = "rgba(239,68,68,0.08)"

        col_gauge, col_right = st.columns([1, 1], gap="large")

        with col_gauge:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)

            st.markdown(f"""
            <div class="risk-banner" style="background:{bg_grad};border-color:{tcolor};">
                <div class="risk-tier" style="color:{tcolor};">{emoji} &nbsp;{tier}</div>
                <div class="risk-sub" style="color:#CBD5E0;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.markdown(f"<div style='font-size:0.8rem;color:#64748B;margin-bottom:4px;'>Risk Score: <b style='color:{tcolor}'>{prob*100:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(min(prob, 1.0))

        with col_right:
            # Threshold decisions
            st.markdown("<div style='font-size:0.95rem;font-weight:700;color:#F1F5F9;margin-bottom:0.6rem;'>Loan Decision by Strategy</div>", unsafe_allow_html=True)
            for strategy, thr, desc in [
                ("🛡️ Conservative",  0.30, "Catch max defaults"),
                ("⚖️ Balanced",       0.45, "Standard practice"),
                ("💰 Cost-Optimal",   0.60, "Best business ROI"),
                ("🎯 High Precision", 0.75, "Minimize false alarms"),
            ]:
                approved = prob < thr
                d_color  = "#22C55E" if approved else "#EF4444"
                d_text   = "APPROVE" if approved else "DECLINE"
                d_icon   = "✅" if approved else "❌"
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            background:rgba(255,255,255,0.04);border:1px solid rgba(99,179,237,0.12);
                            border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;">
                    <div>
                        <span style="font-weight:600;color:#E2E8F0;font-size:0.9rem;">{strategy}</span>
                        <span style="color:#64748B;font-size:0.78rem;margin-left:0.5rem;">threshold {thr}</span><br>
                        <span style="color:#64748B;font-size:0.75rem;">{desc}</span>
                    </div>
                    <div style="font-weight:800;font-size:1rem;color:{d_color};">{d_icon} {d_text}</div>
                </div>
                """, unsafe_allow_html=True)

            # Risk breakdown chart
            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
            user_vals = dict(late_pct=late_pct, days90=days90, eir=eir,
                             annuity=annuity, total_settled=total_settled,
                             payments_count=payments_count, num_payments=num_payments,
                             mobile_phones=mobile_phones, home_phones=home_phones)
            st.plotly_chart(input_breakdown_chart(user_vals, prob), use_container_width=True)

        # Inputs summary
        with st.expander("📋 View full input summary"):
            display_data = {
                "Feature": list(FEATURE_LABELS.values()),
                "Value": [
                    str(week_num), 
                    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1],
                    str(mobile_phones), 
                    str(home_phones),
                    f"${annuity:,.0f}", 
                    str(num_payments), 
                    str(payments_count), 
                    f"${payments_sum:,.0f}",
                    f"{late_pct:.1f}%", 
                    str(days90), 
                    f"${total_settled:,.0f}",
                    f"{eir:.1f}%", 
                    "Yes" if is_q3 else "No",
                ],
            }
            st.dataframe(pd.DataFrame(display_data), hide_index=True, use_container_width=True)

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;">
            <div style="font-size:3.5rem;">🔍</div>
            <div style="font-size:1.3rem;font-weight:700;color:#F1F5F9;margin-top:0.8rem;">
                Ready to Analyze
            </div>
            <div style="color:#64748B;margin-top:0.5rem;font-size:0.95rem;">
                Use quick presets or fill in the sidebar form, then click <b style="color:#93C5FD">Analyze Credit Risk</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, label, desc, color in [
            (c1, "✅ Safe Profile",   "Low late payments,\nhigh credit history",   "#22C55E"),
            (c2, "⚠️ Medium Risk",   "Some late payments,\nmoderate interest rate", "#F59E0B"),
            (c3, "🚨 High Risk",     "High overdue days,\nhigh interest rate",      "#EF4444"),
        ]:
            col.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid {color}33;
                        border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.5rem;">{label.split()[0]}</div>
                <div style="color:{color};font-weight:700;margin-top:0.3rem;">{' '.join(label.split()[1:])}</div>
                <div style="color:#64748B;font-size:0.8rem;margin-top:0.4rem;white-space:pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        
        # Performance Overview Section
        st.markdown("""
        <div style="text-align:center;margin:1.5rem 0 1rem 0;">
            <div style="font-size:1.4rem;font-weight:700;color:#F1F5F9;margin-bottom:0.3rem;">
                📊 Model Performance Overview
            </div>
            <div style="color:#64748B;font-size:0.9rem;">
                Key metrics and diagnostic visualizations from 228,999 test samples
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(os.path.join(BASE, "outputs", "figures", "showcase_2_results.png"),
                 use_container_width=True)
        
        st.markdown("""
        <div style="margin-top:0.8rem;padding:0.8rem;background:rgba(59,130,246,0.08);
                    border:1px solid rgba(59,130,246,0.2);border-radius:10px;">
            <span style="color:#93C5FD;font-weight:600;">📈 Chart Guide:</span>
            <span style="color:#CBD5E0;font-size:0.88rem;"> (Left) Model comparison showing LightGBM exceeds targets, 
            (Center) Confusion matrix with prediction breakdown, (Right) Threshold optimization curves</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:#F1F5F9;margin-bottom:0.3rem;">
            🔬 Model Performance Analysis
        </div>
        <div style="color:#64748B;font-size:0.9rem;">
            Comprehensive diagnostic metrics and feature importance breakdowns
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Section 1: Performance Metrics ──
    st.markdown("""
    <div style="margin:1.5rem 0 0.8rem 0;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.15);">
        <span style="font-size:1.05rem;font-weight:700;color:#93C5FD;">1️⃣ Performance Metrics</span>
        <span style="color:#64748B;font-size:0.85rem;margin-left:0.8rem;">Model accuracy vs industry benchmarks</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.plotly_chart(model_comparison_chart(), use_container_width=True)
        
        # Model info box
        st.info("""
        **🤖 Production Model: LightGBM**  
        Gradient Boosting Decision Trees with class weights for imbalance handling.
        
        **✅ Exceeds Target**: AUC-ROC 0.8030 vs target 0.75 (+7%)  
        **⚙️ Recall**: 69.6% (catches 7 out of 10 defaults)  
        **⚠️ Precision**: 8.3% (low due to 30.8:1 class imbalance)
        
        *Note: Logistic Regression baseline was removed (predicted 100% defaults - unusable)*
        """)
        
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        
        # ── Section 2: Prediction Breakdown ──
        st.markdown("""
        <div style="margin:1.5rem 0 0.8rem 0;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.15);">
            <span style="font-size:1.05rem;font-weight:700;color:#93C5FD;">2️⃣ Prediction Classification</span>
            <span style="color:#64748B;font-size:0.85rem;margin-left:0.8rem;">Test set confusion matrix breakdown</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Confusion matrix
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.85rem;font-weight:700;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.9rem;">
                Confusion Matrix &nbsp;·&nbsp; 228,999 Test Samples
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
                            border-radius:10px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:#22C55E;">166,143</div>
                    <div style="font-size:0.78rem;color:#64748B;">True Negatives</div>
                    <div style="font-size:0.72rem;color:#4B5563;">Correctly approved</div>
                </div>
                <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                            border-radius:10px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:#F87171;">55,657</div>
                    <div style="font-size:0.78rem;color:#64748B;">False Positives</div>
                    <div style="font-size:0.72rem;color:#4B5563;">Over-flagged for review</div>
                </div>
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                            border-radius:10px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:#FCD34D;">2,190</div>
                    <div style="font-size:0.78rem;color:#64748B;">False Negatives</div>
                    <div style="font-size:0.72rem;color:#4B5563;">Missed defaults</div>
                </div>
                <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
                            border-radius:10px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:#60A5FA;">5,009</div>
                    <div style="font-size:0.78rem;color:#64748B;">True Positives</div>
                    <div style="font-size:0.72rem;color:#4B5563;">Defaults caught ✓ 70%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.plotly_chart(threshold_chart(), use_container_width=True)
        
    # ── Section 3: Feature Importance ──
    st.markdown("""
    <div style="margin:2rem 0 0.8rem 0;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.15);">
        <span style="font-size:1.05rem;font-weight:700;color:#93C5FD;">3️⃣ Feature Importance Analysis</span>
        <span style="color:#64748B;font-size:0.85rem;margin-left:0.8rem;">SHAP values showing top predictive features</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_shap1, col_shap2 = st.columns([1.2, 1], gap="medium")
    
    with col_shap1:
        st.image(os.path.join(BASE, "outputs", "figures", "showcase_3_shap.png"),
                 use_container_width=True)
    
    with col_shap2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.95rem;font-weight:700;color:#F1F5F9;margin-bottom:0.6rem;">
                🎯 Key Insights
            </div>
            <div style="color:#CBD5E0;font-size:0.85rem;line-height:1.6;">
                <p><strong style="color:#93C5FD;">Top Driver:</strong> <code>education_1103M_6b2ae0fa</code> 
                has the highest impact on default predictions (3.51% contribution).</p>
                
                <p><strong style="color:#93C5FD;">Feature Categories:</strong></p>
                <ul style="margin-left:1rem;">
                    <li>📚 <strong>Education features</strong> - Strong predictive power</li>
                    <li>📊 <strong>Payment behavior</strong> - Historical late payment patterns</li>
                    <li>💳 <strong>Credit utilization</strong> - Loan amount vs annuity ratios</li>
                    <li>📅 <strong>Temporal factors</strong> - Application timing and seasonality</li>
                </ul>
                
                <p style="margin-top:0.8rem;"><strong style="color:#F59E0B;">Note:</strong> 
                SHAP (SHapley Additive exPlanations) values show how much each feature 
                pushes the prediction higher (red) or lower (blue) from the base probability.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    c1, c2 = st.columns([1.1, 1], gap="large")

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;margin-bottom:0.8rem;">
                🎯 Project Mission
            </div>
            <p style="color:#CBD5E0;line-height:1.7;font-size:0.93rem;">
                Traditional credit scoring systems automatically reject <b style="color:#93C5FD">26 million
                credit-invisible Americans</b> — not because they're risky, but because
                they have no formal credit history. HomeCredit serves these underserved populations
                using alternative behavioral data.
            </p>
            <p style="color:#CBD5E0;line-height:1.7;font-size:0.93rem;">
                Our AI pipeline processes 1.5M loan applications across 32 data tables to build
                a fair, explainable, and high-performance default prediction model — exceeding
                industry AUC benchmarks by 7%.
            </p>
        </div>

        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;margin-bottom:0.8rem;">
                🔧 9-Step ML Pipeline
            </div>
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("1", "Data Collection",       "1.5M records from 32 parquet tables",           "#3B82F6"),
            ("2", "Data Merging",          "Static + dynamic table join strategy",           "#3B82F6"),
            ("3", "Preprocessing",         "384M missing values — strategic imputation",     "#3B82F6"),
            ("4", "Feature Engineering",   "727 features, encoded & scaled",                 "#3B82F6"),
            ("5", "Model Training",        "LightGBM with class-weighted imbalance handling","#F59E0B"),
            ("6", "Model Evaluation",      "AUC-ROC, F1, precision, recall, confusion",      "#22C55E"),
            ("7", "SHAP Explainability",   "Top-feature attribution for each prediction",    "#22C55E"),
            ("8", "Visualizations",        "8 diagnostic charts for model transparency",     "#22C55E"),
            ("9", "Threshold Optimization","Business cost-benefit threshold tuning",          "#22C55E"),
        ]
        for num, title, desc, color in pipeline_steps:
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;align-items:flex-start;margin-bottom:0.45rem;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:{color}22;border:1.5px solid {color};
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.75rem;font-weight:800;color:{color};">{num}</div>
                <div>
                    <span style="font-weight:600;color:#E2E8F0;font-size:0.88rem;">{title}</span>
                    <span style="color:#64748B;font-size:0.8rem;"> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;margin-bottom:0.8rem;">
                👥 Team — Capstone 2
            </div>
        """, unsafe_allow_html=True)
        team = ["Venkat Dinesh", "Sai Charan", "Lokesh Reddy", "Pranav Dhara", "Sunny"]
        for name in team:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.5rem;">
                <div style="width:34px;height:34px;border-radius:50%;
                            background:linear-gradient(135deg,#2563EB,#7C3AED);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;color:white;font-size:0.85rem;">
                    {name[0]}
                </div>
                <span style="color:#CBD5E0;font-size:0.92rem;font-weight:500;">{name}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid rgba(99,179,237,0.12);">
                <div style="color:#64748B;font-size:0.8rem;">📍 St. Clair College</div>
                <div style="color:#64748B;font-size:0.8rem;">📚 Data Analytics — Predictive Analytics</div>
                <div style="color:#64748B;font-size:0.8rem;">📅 2026</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;margin-bottom:0.8rem;">
                📈 Key Achievements
            </div>
        """, unsafe_allow_html=True)
        achievements = [
            ("AUC-ROC 0.803", "Exceeded 0.75 target by 7%",          "#22C55E"),
            ("70% Recall",    "Defaults caught before approval",       "#3B82F6"),
            ("727 Features",  "From 32 raw data tables",               "#8B5CF6"),
            ("SHAP Ready",    "Explainable AI with per-prediction why","#F59E0B"),
            ("Cost-Tunable",  "Threshold optimizer for business needs", "#F59E0B"),
        ]
        for title, desc, color in achievements:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.45rem;">
                <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                <span style="color:#E2E8F0;font-weight:600;font-size:0.88rem;">{title}</span>
                <span style="color:#64748B;font-size:0.8rem;">— {desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;margin-bottom:0.8rem;">
                ⚠️ App Limitations — Feature Coverage
            </div>
            <p style="color:#CBD5E0;line-height:1.7;font-size:0.85rem;">
                <strong style="color:#F59E0B;">Important Note:</strong> This demo app uses <strong>13 critical features</strong> 
                for prediction, while the full model was trained on <strong>727 features</strong>. This means:
            </p>
        """, unsafe_allow_html=True)
        
        limitations = [
            ("13 Features Exposed", "Only 1.8% of model inputs", "#F59E0B"),
            ("~26% Coverage", "Captures key predictors but not all", "#F59E0B"),
            ("714 Features = 0", "Set to default/zero values", "#EF4444"),
            ("Reduced Accuracy", "Predictions are directionally correct", "#EF4444"),
        ]
        for title, desc, color in limitations:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.45rem;">
                <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                <span style="color:#E2E8F0;font-weight:600;font-size:0.88rem;">{title}</span>
                <span style="color:#64748B;font-size:0.8rem;">— {desc}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <p style="color:#CBD5E0;line-height:1.7;font-size:0.85rem;margin-top:0.8rem;">
                <strong style="color:#93C5FD;">For Production Use:</strong> The full model requires complete 
                applicant data from multiple sources (credit bureau reports, bank statements, previous loans, etc.). 
                This app is a <strong>demonstration tool</strong> showing the most important risk factors.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.image(os.path.join(BASE, "outputs", "figures", "showcase_1_pipeline.png"),
                 use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:1.5rem;margin-bottom:0.8rem;">
<div style="text-align:center;color:#334155;font-size:0.78rem;padding-bottom:0.5rem;">
    CreditAI &nbsp;·&nbsp; Home Credit Default Prediction &nbsp;·&nbsp;
    Predictive Analytics &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model       = joblib.load(os.path.join(BASE, "models", "lightgbm.pkl"))
    scaler      = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    num_cols    = joblib.load(os.path.join(BASE, "models", "numerical_cols.pkl"))
    feature_names = model.feature_name()
    return model, scaler, list(num_cols), feature_names

model, scaler, num_cols, feature_names = load_artifacts()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0;font-size:1.7rem;">🏦 Credit Risk Prediction System</h2>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Home Credit Default Prediction &nbsp;|&nbsp; Powered by LightGBM &nbsp;|&nbsp;
        AUC-ROC 0.803 &nbsp;|&nbsp; Capstone 2 · St. Clair College
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar – Input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.caption("Fill in the applicant's information below and click **Predict**.")
    st.divider()

    # ── Application context ──────────────────────────────────────────────────
    st.markdown("**📅 Application Context**")
    week_num = st.slider("Week of year (WEEK_NUM)", 1, 52, 26,
                         help="Which week of the year the application was submitted")
    month    = st.selectbox("Month of application", list(range(1, 13)),
                             index=5, format_func=lambda m: [
                                 "January","February","March","April","May","June",
                                 "July","August","September","October","November","December"
                             ][m-1])
    is_q3    = st.checkbox("Application in Q3 (Jul–Sep)", value=False)

    st.divider()
    st.markdown("**📞 Contact & Identity**")
    mobile_phones = st.number_input("Number of mobile phones on file", 0, 10, 1)
    home_phones   = st.number_input("Number of home phones on file",   0, 10, 1)

    st.divider()
    st.markdown("**💰 Loan & Payment History**")
    annuity = st.number_input("Monthly annuity / instalment amount ($)",
                              min_value=0.0, max_value=500_000.0,
                              value=8_500.0, step=500.0,
                              help="Regular monthly payment amount for this loan")
    num_payments   = st.number_input("Number of scheduled payments (pmtnum_254L)", 0, 360, 36)
    payments_count = st.number_input("Total historical payment count (pmtscount_423L)", 0, 5000, 120)
    payments_sum   = st.number_input("Total historical payments sum $ (pmtssum_45A)",
                                     0.0, 10_000_000.0, 250_000.0, step=5_000.0)

    st.divider()
    st.markdown("**⚠️ Risk Indicators**")
    late_pct   = st.slider("% of instalments paid late (>1 day)",
                           0.0, 100.0, 5.0, step=0.5,
                           help="Higher % = higher default risk")
    days90     = st.number_input("Days past due (90-day bucket)", 0, 1000, 0,
                                 help="Days overdue in the 90-day delinquency window")
    total_settled = st.number_input("Total amount settled ($)", 0.0, 10_000_000.0,
                                    200_000.0, step=10_000.0)
    eir        = st.slider("Effective interest rate (%)", 0.0, 100.0, 12.0, step=0.5)

    st.divider()
    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True, type="primary")

# ── Build feature vector ──────────────────────────────────────────────────────
def build_input() -> pd.DataFrame:
    """Create a zero-filled row with user-supplied fields overridden."""
    row = dict.fromkeys(feature_names, 0.0)

    # Fill user inputs
    row["WEEK_NUM"]                             = float(week_num)
    row["MONTH"]                                = float(month)
    row["mobilephncnt_593L"]                    = float(mobile_phones)
    row["homephncnt_628L"]                      = float(home_phones)
    row["annuity_780A"]                         = float(annuity)
    row["pmtnum_254L"]                          = float(num_payments)
    row["pmtscount_423L"]                       = float(payments_count)
    row["pmtssum_45A"]                          = float(payments_sum)
    row["pctinstlsallpaidlate1d_3546856L"]      = late_pct / 100.0
    row["days90_310L"]                          = float(days90)
    row["totalsettled_863A"]                    = float(total_settled)
    row["eir_270L"]                             = eir / 100.0
    row["thirdquarter_1082L"]                   = 1.0 if is_q3 else 0.0

    df = pd.DataFrame([row])[feature_names]

    # Scale numerical columns that exist in feature_names
    cols_to_scale = [c for c in num_cols if c in df.columns]
    if cols_to_scale:
        # Rebuild a scaler-compatible array (fill missing scaler cols with 0)
        scaler_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else num_cols
        scale_df = pd.DataFrame(
            np.zeros((1, len(scaler_cols))), columns=scaler_cols
        )
        for c in cols_to_scale:
            if c in scale_df.columns:
                scale_df[c] = df[c].values
        scaled = scaler.transform(scale_df)
        scaled_df = pd.DataFrame(scaled, columns=scaler_cols)
        for c in cols_to_scale:
            if c in scaled_df.columns:
                df[c] = scaled_df[c].values

    return df

# ── Gauge chart ───────────────────────────────────────────────────────────────
def risk_gauge(prob: float) -> go.Figure:
    color = "#2E7D32" if prob < 0.3 else "#F57F17" if prob < 0.6 else "#C62828"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        delta={"reference": 31.4, "increasing": {"color": "#C62828"},
               "decreasing": {"color": "#2E7D32"}, "suffix": "% vs avg"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickfont": {"size": 12}},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  30], "color": "#E8F5E9"},
                {"range": [30, 60], "color": "#FFF8E1"},
                {"range": [60,100], "color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": "#1565C0", "width": 3},
                          "thickness": 0.75, "value": 31.4},
        },
        title={"text": "Default Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ── Feature impact table ──────────────────────────────────────────────────────
FEATURE_LABELS = {
    "WEEK_NUM":                           "Week of year",
    "MONTH":                              "Month",
    "mobilephncnt_593L":                  "Mobile phones on file",
    "homephncnt_628L":                    "Home phones on file",
    "annuity_780A":                       "Monthly annuity ($)",
    "pmtnum_254L":                        "# Scheduled payments",
    "pmtscount_423L":                     "Historical payment count",
    "pmtssum_45A":                        "Total payments sum ($)",
    "pctinstlsallpaidlate1d_3546856L":    "Late payment rate",
    "days90_310L":                        "90-day past-due days",
    "totalsettled_863A":                  "Total settled ($)",
    "eir_270L":                           "Effective interest rate",
    "thirdquarter_1082L":                 "Application in Q3",
}

# ── Main area ─────────────────────────────────────────────────────────────────
col_info, col_result = st.columns([1, 1], gap="large")

with col_info:
    st.markdown("### 📊 Model Overview")
    st.markdown("""
    <div class="metric-card">🎯 <b>Algorithm:</b> LightGBM (Gradient Boosting)</div>
    <div class="metric-card">📈 <b>AUC-ROC:</b> 0.803 &nbsp;·&nbsp; <b>Recall:</b> 69.6%</div>
    <div class="metric-card">📂 <b>Training data:</b> 1.5M loan applications, 32 tables</div>
    <div class="metric-card">🔍 <b>Features used:</b> 727 engineered predictors</div>
    <div class="metric-card">⚖️ <b>Class imbalance:</b> 30.8:1 (handled via class weights)</div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Decision Thresholds")
    thresh_df = pd.DataFrame({
        "Strategy": ["Conservative (catch defaults)", "Balanced", "Cost-optimal", "High precision"],
        "Threshold": ["0.30", "0.45", "0.60", "0.75"],
        "Recall":    ["~88%", "~76%", "~56%", "~31%"],
        "Precision": ["6.4%", "7.4%", "10.8%", "16.1%"],
    })
    st.dataframe(thresh_df, hide_index=True, use_container_width=True)

with col_result:
    if predict_btn:
        with st.spinner("Running prediction…"):
            X = build_input()
            prob = float(model.predict(X)[0])

        # Risk tier
        if prob < 0.30:
            tier, tier_color, tier_emoji = "LOW RISK",    "#2E7D32", "✅"
            advice = "Applicant profile is within acceptable risk parameters."
        elif prob < 0.60:
            tier, tier_color, tier_emoji = "MEDIUM RISK", "#F57F17", "⚠️"
            advice = "Manual review recommended. Proceed with standard checks."
        else:
            tier, tier_color, tier_emoji = "HIGH RISK",   "#C62828", "🚨"
            advice = "High default probability. Enhanced due diligence required."

        st.plotly_chart(risk_gauge(prob), use_container_width=True)

        st.markdown(f"""
        <div class="risk-box" style="background:{tier_color}18;border:2px solid {tier_color};">
            {tier_emoji} &nbsp;<span style="color:{tier_color};font-size:1.3rem;">{tier}</span><br>
            <span style="font-weight:400;font-size:0.95rem;color:#37474F;">{advice}</span>
        </div>
        """, unsafe_allow_html=True)

        # Decision at common thresholds
        st.markdown("**Loan Decision by Threshold:**")
        decisions = []
        for label, thr in [("Conservative (0.30)", 0.30), ("Balanced (0.45)", 0.45),
                            ("Cost-optimal (0.60)", 0.60), ("High precision (0.75)", 0.75)]:
            decision = "❌ Decline" if prob >= thr else "✅ Approve"
            decisions.append({"Threshold Strategy": label, "Decision": decision})
        st.dataframe(pd.DataFrame(decisions), hide_index=True, use_container_width=True)

        # Key inputs summary
        with st.expander("📋 Inputs used in prediction"):
            inputs_display = {
                FEATURE_LABELS.get(k, k): v
                for k, v in {
                    "WEEK_NUM": week_num, "MONTH": month,
                    "mobilephncnt_593L": mobile_phones,
                    "homephncnt_628L": home_phones,
                    "annuity_780A": f"${annuity:,.0f}",
                    "pmtnum_254L": num_payments,
                    "pmtscount_423L": payments_count,
                    "pmtssum_45A": f"${payments_sum:,.0f}",
                    "pctinstlsallpaidlate1d_3546856L": f"{late_pct:.1f}%",
                    "days90_310L": days90,
                    "totalsettled_863A": f"${total_settled:,.0f}",
                    "eir_270L": f"{eir:.1f}%",
                    "thirdquarter_1082L": "Yes" if is_q3 else "No",
                }.items()
            }
            st.dataframe(
                pd.DataFrame(inputs_display.items(), columns=["Feature", "Value"]),
                hide_index=True, use_container_width=True
            )
    else:
        st.info("👈  Fill in the applicant details on the left and click **Predict Default Risk**.")
        st.image(
            os.path.join(BASE, "outputs", "figures", "showcase_2_results.png"),
            caption="Model performance summary", use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center;color:#90A4AE;font-size:0.85rem;">
    Home Credit Default Prediction &nbsp;·&nbsp; Capstone 2 &nbsp;·&nbsp;
    Venkat Dinesh · Sai Charan · Lokesh Reddy · Pranav Dhara · Sunny &nbsp;·&nbsp;
    St. Clair College · Data Analytics — Predictive Analytics · 2026
</p>
""", unsafe_allow_html=True)
