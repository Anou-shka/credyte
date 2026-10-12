# app.py — Credyte: Credit Risk & Explainability Dashboard
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from src.features import (
    build_single_from_minimal,
    compute_features,
    read_feature_schema,
    FEATURE_COLUMNS_DEFAULT,
    FEATURE_LABELS,
    HALFLIFE_DEFAULT,
    MONTHS_DEFAULT,
)
from src.score import default_bands

# ---------- Page config ----------
st.set_page_config(
    page_title="Credyte",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
  --bg: #0d1117;
  --surface: #161b27;
  --border: #21293d;
  --muted: #8b96b4;
  --text: #e2e8f0;
  --accent: #6366f1;
  --accent2: #8b5cf6;
  --green: #10b981;
  --yellow: #f59e0b;
  --red: #ef4444;
}

.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

/* Hero */
.hero {
  background: linear-gradient(135deg, rgba(99,102,241,.14) 0%, rgba(139,92,246,.10) 60%, rgba(16,185,129,.06) 100%);
  border: 1px solid rgba(99,102,241,.30);
  border-radius: 20px;
  padding: 32px 36px;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -40px; right: -40px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(99,102,241,.18) 0%, transparent 70%);
  pointer-events: none;
}
.hero-icon {
  font-size: 36px;
  line-height: 1;
  margin-bottom: 8px;
  display: block;
}
.hero h1 {
  margin: 0;
  font-size: 32px;
  font-weight: 800;
  letter-spacing: -.8px;
  background: linear-gradient(90deg, #e2e8f0 0%, #a5b4fc 60%, #c4b5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 14.5px;
  line-height: 1.6;
  max-width: 560px;
}
.hero-pills {
  display: flex;
  gap: 8px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.hero-pill {
  font-size: 11.5px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(99,102,241,.12);
  border: 1px solid rgba(99,102,241,.25);
  color: #a5b4fc;
  letter-spacing: .04em;
}

/* Section header */
.section-header {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 8px 0 6px;
  border-bottom: 1px solid var(--border);
  margin: 1.4rem 0 .8rem;
}

/* KPI cards */
.kpi-row { display: flex; gap: 14px; margin: 1rem 0; }
.kpi {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  text-align: center;
}
.kpi .label { font-size: 11.5px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .07em; }
.kpi .value { font-size: 32px; font-weight: 700; line-height: 1; }
.kpi.risk-low  .value { color: var(--green); }
.kpi.risk-med  .value { color: var(--yellow); }
.kpi.risk-high .value { color: var(--red); }
.kpi .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

/* Risk badge */
.risk-badge {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  margin-top: 8px;
}
.badge-low  { background: rgba(16,185,129,.15); color: var(--green); border: 1px solid rgba(16,185,129,.3); }
.badge-med  { background: rgba(245,158,11,.15); color: var(--yellow); border: 1px solid rgba(245,158,11,.3); }
.badge-high { background: rgba(239,68,68,.15); color: var(--red); border: 1px solid rgba(239,68,68,.3); }

/* Ratio grid */
.ratio-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: .6rem 0 1rem; }
.ratio-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.ratio-card .rk { font-size: 11.5px; color: var(--muted); margin-bottom: 4px; }
.ratio-card .rv { font-size: 22px; font-weight: 700; }
.ratio-card .rbar { height: 3px; border-radius: 2px; margin-top: 8px; background: var(--border); }
.ratio-card .rfill { height: 100%; border-radius: 2px; }

/* Driver cards */
.driver-row { display: flex; gap: 10px; flex-wrap: wrap; margin: .5rem 0 1rem; }
.driver-card {
  flex: 1; min-width: 130px; max-width: 200px;
  background: var(--surface);
  border-radius: 10px;
  padding: 10px 12px;
  text-align: center;
}
.driver-card .dk { font-size: 11px; color: var(--muted); margin-bottom: 3px; }
.driver-card .dv { font-size: 18px; font-weight: 700; }
.driver-card.risk-up   { border: 1px solid rgba(239,68,68,.4); }
.driver-card.risk-down { border: 1px solid rgba(16,185,129,.4); }
.driver-card.risk-up   .dv { color: var(--red); }
.driver-card.risk-down .dv { color: var(--green); }

/* Expander tweaks */
details summary { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
DATA_PATH      = Path("data/final_dataset.csv")
MODELS_DIR     = Path("models")
FEATURES_TXT   = MODELS_DIR / "feature_columns.txt"
BEST_PATH      = MODELS_DIR / "best.joblib"
BASE_PATH      = MODELS_DIR / "baseline.joblib"
CALIBRATED_PATH= MODELS_DIR / "calibrated_best.joblib"
METRICS_PATH   = MODELS_DIR / "metrics.json"
CAL_METRICS_PATH = MODELS_DIR / "calibration_metrics.json"
SHAP_DIR       = Path("images/shap")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ---------- Schema ----------
def load_feature_schema() -> List[str]:
    return read_feature_schema(FEATURES_TXT) or FEATURE_COLUMNS_DEFAULT


def load_feature_params() -> tuple:
    """Feature-engineering params the models were trained with (must match at inference)."""
    params = _read_json(METRICS_PATH).get("params", {})
    return int(params.get("months", MONTHS_DEFAULT)), int(params.get("halflife", HALFLIFE_DEFAULT))


# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_models_and_threshold():
    if CALIBRATED_PATH.exists():
        scoring_model = joblib.load(CALIBRATED_PATH); scoring_tag = "calibrated"
    elif BEST_PATH.exists():
        scoring_model = joblib.load(BEST_PATH); scoring_tag = "best"
    elif BASE_PATH.exists():
        scoring_model = joblib.load(BASE_PATH); scoring_tag = "baseline"
    else:
        return None

    shap_path  = BEST_PATH if BEST_PATH.exists() else BASE_PATH
    shap_model = joblib.load(shap_path)
    shap_tag   = "best" if shap_path == BEST_PATH else "baseline"

    (default_low, default_high), op = default_bands(MODELS_DIR)
    return scoring_model, scoring_tag, shap_model, shap_tag, (default_low, default_high), op


@st.cache_data(show_spinner=False)
def load_background(schema: tuple, months: int, halflife: int, n: int = 800) -> pd.DataFrame:
    """Random (not head-of-file) sample of engineered training rows, aligned to the schema."""
    df = pd.read_csv(DATA_PATH).drop(columns=["DEFAULT"], errors="ignore")
    df = df.sample(n=min(n, len(df)), random_state=17)
    X = compute_features(df, months_default=months, halflife=halflife)
    return X.reindex(columns=list(schema)).fillna(0.0)


@st.cache_resource(show_spinner=False)
def get_explainer(_model, model_key: str, schema: tuple, months: int, halflife: int):
    if DATA_PATH.exists():
        X_bg = load_background(schema, months, halflife)
    else:
        X_bg = None
    return make_explainer(_model, X_bg, list(schema))


# ---------- Helpers ----------
def band_from(p: float, low: float, high: float) -> str:
    return "Low" if p < low else ("Medium" if p < high else "High")

def band_css(band: str) -> str:
    return {"Low": "risk-low", "Medium": "risk-med", "High": "risk-high"}.get(band, "")

def badge_css(band: str) -> str:
    return {"Low": "badge-low", "Medium": "badge-med", "High": "badge-high"}.get(band, "")

def align_series_to_schema(s: pd.Series, schema: List[str]) -> pd.Series:
    return pd.Series({c: float(s.get(c, 0.0)) for c in schema}, index=schema, dtype=float)


# ---------- SHAP explainer ----------
def make_explainer(model, X_bg, feature_cols: List[str]):
    """Returns (explainer, preprocessor or None, kind). X_bg may be None only for tree models."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    name = type(model).__name__.lower()
    if "xgb" in name or "lgb" in name:
        try:
            # path-dependent TreeExplainer needs no background data
            return shap.TreeExplainer(model), None, "tree"
        except Exception:
            pass
    if X_bg is None:
        raise RuntimeError(f"{DATA_PATH} is needed as SHAP background data for this model type.")

    if isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        est = model.steps[-1][1]
        if isinstance(est, LogisticRegression) and pre is not None:
            X_bg_np = pre.transform(X_bg)
            X_bg_df = pd.DataFrame(X_bg_np, index=X_bg.index, columns=feature_cols)
            try:
                return shap.LinearExplainer(est, X_bg_df), pre, "linear"
            except Exception:
                return shap.Explainer(est, X_bg_df), pre, "generic"
        return shap.Explainer(model, X_bg), None, "pipeline-generic"

    return shap.Explainer(model, X_bg), None, "generic"


# ---------- Gauge plot ----------
def render_gauge(proba: float, band: str, low: float, high: float) -> None:
    color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    color = color_map.get(band, "#6366f1")

    fig, ax = plt.subplots(figsize=(2.6, 1.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#161b27")
    ax.set_facecolor("#161b27")

    n = 300
    # Background track
    theta_bg = np.linspace(np.pi, 0, n)
    ax.plot(theta_bg, [1]*n, color="#21293d", linewidth=14, solid_capstyle="round")

    # Zone colouring follows the active band thresholds: low=green, med=amber, high=red
    for zone_start, zone_end, zone_color in [
        (np.pi,                 np.pi*(1 - low),  "#10b981"),
        (np.pi*(1 - low),       np.pi*(1 - high), "#f59e0b"),
        (np.pi*(1 - high),      0,                "#ef4444"),
    ]:
        zone_theta = np.linspace(zone_start, zone_end, 100)
        ax.plot(zone_theta, [1]*100, color=zone_color, linewidth=14, alpha=0.20, solid_capstyle="butt")

    # Filled arc
    fill_end = np.pi - proba * np.pi
    theta_fill = np.linspace(np.pi, fill_end, n)
    ax.plot(theta_fill, [1]*n, color=color, linewidth=14, solid_capstyle="round")

    # Needle
    angle = np.pi - proba * np.pi
    ax.annotate("", xy=(angle, 0.88), xytext=(angle, 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#e2e8f0", lw=1.5, mutation_scale=10))
    ax.plot([angle], [0.05], "o", color="#e2e8f0", markersize=4)

    ax.set_ylim(0, 1.25)
    ax.axis("off")
    ax.set_thetamin(0); ax.set_thetamax(180)

    # Labels
    ax.text(0.5, 0.18, f"{proba:.1%}", ha="center", va="center", transform=ax.transAxes,
            fontsize=17, fontweight="bold", color=color)
    ax.text(0.5, 0.02, band + " Risk", ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color="#8b96b4", fontweight="600")
    ax.text(0.04, 0.00, "0%", ha="center", transform=ax.transAxes, fontsize=7, color="#4a5568")
    ax.text(0.96, 0.00, "100%", ha="center", transform=ax.transAxes, fontsize=7, color="#4a5568")

    plt.tight_layout(pad=0)
    st.pyplot(fig, clear_figure=True)


# ---------- Waterfall SHAP ----------
def render_waterfall(explainer, shap_row: np.ndarray, X_one_raw: pd.DataFrame) -> None:
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).ravel()[-1])

    exp = shap.Explanation(
        values=shap_row,
        base_values=base_val,
        data=X_one_raw.iloc[0].values,          # show the borrower's real (unscaled) values
        feature_names=[PRETTY_NAMES.get(c, c) for c in X_one_raw.columns],
    )

    with plt.style.context("dark_background"):
        plt.figure(figsize=(9, 4.5), facecolor="#161b27")
        shap.plots.waterfall(exp, max_display=12, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#161b27")
        for ax in fig.axes:
            ax.set_facecolor("#161b27")
            for spine in ax.spines.values():
                spine.set_edgecolor("#21293d")
            ax.tick_params(colors="#8b96b4")
            ax.xaxis.label.set_color("#8b96b4")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    plt.close("all")


# ---------- Ratio cards ----------
PRETTY_NAMES = FEATURE_LABELS


def _ratio_cards(ui: dict) -> None:
    def bar(val: float, color: str) -> str:
        pct = min(100, max(0, val * 100))
        return (
            f'<div class="rbar"><div class="rfill" style="width:{pct:.0f}%;background:{color}"></div></div>'
        )

    items = [
        ("avg_pay_amt",        f"NT$ {ui['avg_pay_amt']:,.0f}",           "", "#6366f1"),
        ("avg_bill_amt",       f"NT$ {ui['avg_bill_amt']:,.0f}",           "", "#6366f1"),
        ("credit_utilization", f"{ui['credit_utilization']:.1%}",          ui["credit_utilization"], "#ef4444"),
        ("repay_ratio_avg",    f"{ui['repay_ratio_avg']:.1%}",             ui["repay_ratio_avg"], "#10b981"),
        ("cur_recent",         f"{ui['cur_recent']:.1%}",                  ui["cur_recent"], "#10b981"),
        ("payment_trend",      f"{ui['payment_trend']:+.2f}",              (ui["payment_trend"]+1)/2, "#f59e0b"),
    ]

    cards = ""
    for key, val, bar_val, color in items:
        bar_html = bar(float(bar_val), color) if bar_val != "" else ""
        cards += (
            f'<div class="ratio-card">'
            f'<div class="rk">{PRETTY_NAMES.get(key, key)}</div>'
            f'<div class="rv">{val}</div>'
            f'{bar_html}'
            f'</div>'
        )

    st.markdown(
        '<div class="section-header">Computed Financial Ratios</div>'
        f'<div class="ratio-grid">{cards}</div>',
        unsafe_allow_html=True
    )


# ---------- Driver chips ----------
def _driver_chips(shap_row: np.ndarray, schema: List[str], X_one_row: pd.Series) -> None:
    s = pd.Series(shap_row, index=schema)
    ups   = s[s > 0].nlargest(3)
    downs = s[s < 0].nsmallest(3)

    def make_cards(vals: pd.Series, cls: str) -> str:
        if vals.empty:
            return '<div style="font-size:12px;color:#8b96b4;">None for this borrower.</div>'
        html = ""
        for k in vals.index:
            raw = X_one_row.get(k, 0)
            try:
                v = float(raw)
                display_val = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
            except Exception:
                display_val = str(raw)
            html += (
                f'<div class="driver-card {cls}">'
                f'<div class="dk">{PRETTY_NAMES.get(k, k)}</div>'
                f'<div class="dv">{display_val}</div>'
                f'</div>'
            )
        return html

    st.markdown('<div class="section-header">Key Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.18);border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#9aa6d1;line-height:1.6;">
  <b style="color:#a5b4fc;">How to read this:</b>&nbsp;
  These are up to three features with the <b>largest positive</b> (↑) and <b>largest negative</b> (↓) SHAP values for this borrower — the ones that pushed the predicted default risk up or down the most.
  The displayed number is the <b>actual feature value</b> for this borrower, not the SHAP impact.
  A high <em>credit utilization</em> borrower will appear on the risk side even if they paid on time, because utilization alone carries significant predictive weight.
</div>
""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="font-size:12px;color:#ef4444;font-weight:600;margin-bottom:6px;">↑ Pushing Risk Up</div>'
            f'<div class="driver-row">{make_cards(ups, "risk-up")}</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div style="font-size:12px;color:#10b981;font-weight:600;margin-bottom:6px;">↓ Pulling Risk Down</div>'
            f'<div class="driver-row">{make_cards(downs, "risk-down")}</div>',
            unsafe_allow_html=True
        )


# ---------- Load artifacts ----------
schema = load_feature_schema()
months, halflife = load_feature_params()
_loaded = load_models_and_threshold()
if _loaded is None:
    st.error("No model found in models/. Run `python -m src.train` first.")
    st.stop()
scoring_model, scoring_tag, shap_model, shap_tag, (def_low, def_high), op_meta = _loaded
_metrics = _read_json(METRICS_PATH)
_cal_meta = _read_json(CAL_METRICS_PATH)
_model_label = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "logreg": "Logistic Regression"}.get(
    _metrics.get("best", {}).get("name", ""), "ML model")
_cal_label = {"isotonic": "Isotonic", "platt": "Platt"}.get(
    _cal_meta.get("chosen", {}).get("method", ""), "")
_stack_label = f"{_model_label} + {_cal_label}" if scoring_tag == "calibrated" and _cal_label else _model_label

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <span class="hero-icon">⬡</span>
  <h1>Credyte</h1>
  <p class="hero-sub">
    Credit risk scoring for credit-card borrowers — estimate the probability of default with a
    calibrated gradient-boosted model, see exactly why with SHAP, and review key financial ratios.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">Calibrated PD Scoring</span>
    <span class="hero-pill">SHAP Explainability</span>
    <span class="hero-pill">Financial Ratio Engine</span>
    <span class="hero-pill">{_stack_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("**Risk Band Thresholds**")
    low_thr  = st.slider("Low / Medium cutoff", 0.0, 1.0, round(def_low, 2),  0.01)
    high_thr = st.slider("Medium / High cutoff", 0.0, 1.0, round(def_high, 2), 0.01)
    if low_thr >= high_thr:
        st.error("The Low/Medium cutoff must be below the Medium/High cutoff. Using the defaults.")
        low_thr, high_thr = def_low, def_high
    st.divider()
    st.caption(f"Scoring: **{scoring_tag}** · SHAP base: **{shap_tag}**")
    st.caption(f"Features: **{len(schema)}** · history **{months}** months · EMA half-life **{halflife}**")
    if op_meta:
        st.caption(f"Operating point ({op_meta.get('criterion', 'f1').upper()}): **t = {op_meta['threshold']:.3f}**")

tabs = st.tabs(["🔮 Predict", "🌍 Global SHAP", "📐 Calibration"])

# =========================================================
# Tab 1 — Predict
# =========================================================
with tabs[0]:
    # UCI repayment status codes (PAY_0 … PAY_6)
    pay_map = {
        "No consumption (-2)": -2, "Paid in full (-1)": -1, "Revolving credit, min. paid (0)": 0,
        "1 month late": 1, "2 months late": 2, "3 months late": 3, "4 months late": 4,
        "5 months late": 5, "6 months late": 6, "7 months late": 7, "8+ months late": 8,
    }
    edu_map = {"Graduate school": 1, "University": 2, "High school": 3, "Other": 4}
    mar_map = {"Married": 1, "Single": 2, "Other": 3}

    st.markdown('<div class="section-header">Borrower Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    LIMIT_BAL  = c1.number_input("Credit Limit (NT$)", value=120_000.0, min_value=1_000.0, step=5_000.0, format="%.0f")
    AGE        = c2.number_input("Age", value=32, min_value=18, max_value=100, step=1)
    edu_label  = c3.selectbox("Education", list(edu_map.keys()), index=1)
    mar_label  = c4.selectbox("Marital Status", list(mar_map.keys()), index=1)
    EDU = int(edu_map[edu_label]); MAR = int(mar_map[mar_label])

    st.markdown('<div class="section-header">Latest Month</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    pay_label      = c5.selectbox("Payment Status", list(pay_map.keys()), index=3)
    PAY_STATUS     = pay_map[pay_label]
    PAY_AMT_LATEST = c6.number_input("Payment Amount (NT$)", value=1_800.0, min_value=0.0, step=100.0, format="%.0f")
    BILL_AMT_LATEST= c7.number_input("Bill Amount (NT$)", value=22_000.0, step=500.0, format="%.0f",
                                     help="Negative means a credit balance (nothing owed).")

    st.markdown(f'<div class="section-header">Prior History — averages over the {months - 1} months before the latest</div>', unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    AVG_PAY_AMT_PRIOR  = c8.number_input("Prior Avg Payment (NT$)", value=1_500.0, min_value=0.0, step=100.0, format="%.0f")
    AVG_BILL_AMT_PRIOR = c9.number_input("Prior Avg Bill (NT$)", value=20_000.0, step=500.0, format="%.0f")
    PRIOR_PAY_AVG      = c10.number_input(
        "Prior Pay Status Avg", value=float(max(-2, PAY_STATUS - 1)), min_value=-2.0, max_value=8.0, step=0.5,
        help="Average of past payment status codes (−2 to 8)"
    )

    predict_btn = st.button("Predict Default Risk", use_container_width=True, type="primary")

    if predict_btn:
        feature_row, ui = build_single_from_minimal(
            row_dict={
                "LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "EDUCATION": EDU, "MARRIAGE": MAR,
                "PAY_STATUS": PAY_STATUS,
                "PAY_AMT_LATEST": PAY_AMT_LATEST, "BILL_AMT_LATEST": BILL_AMT_LATEST,
                "AVG_PAY_AMT_PRIOR": AVG_PAY_AMT_PRIOR, "AVG_BILL_AMT_PRIOR": AVG_BILL_AMT_PRIOR,
                "PRIOR_PAY_AVG": PRIOR_PAY_AVG, "MONTHS": months,
            },
            months=months, halflife=halflife, feature_order=schema,
        )
        feature_row = align_series_to_schema(feature_row, schema)
        X_one = feature_row.to_frame().T

        proba = float(scoring_model.predict_proba(X_one)[:, 1][0])
        band  = band_from(proba, low_thr, high_thr)

        st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

        gauge_col, pd_col, thresh_col = st.columns([1, 1, 2])
        with gauge_col:
            render_gauge(proba, band, low_thr, high_thr)

        with pd_col:
            st.markdown(f"""
            <div class="kpi" style="height:100%;justify-content:center;">
              <div class="label">Probability of Default</div>
              <div class="value {band_css(band)}">{proba:.1%}</div>
              <span class="risk-badge {badge_css(band)}">{band} Risk</span>
            </div>
            """, unsafe_allow_html=True)

        with thresh_col:
            st.markdown(f"""
            <div class="kpi" style="height:100%;justify-content:center;">
              <div class="label">Risk Band Thresholds</div>
              <div style="display:flex;gap:10px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#10b981;font-weight:600;margin-bottom:2px;">LOW</div>
                  <div style="font-size:18px;font-weight:700;color:#10b981;">PD &lt; {low_thr:.0%}</div>
                </div>
                <div style="color:#21293d;font-size:22px;padding-top:6px;">·</div>
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#f59e0b;font-weight:600;margin-bottom:2px;">MEDIUM</div>
                  <div style="font-size:18px;font-weight:700;color:#f59e0b;">PD &lt; {high_thr:.0%}</div>
                </div>
                <div style="color:#21293d;font-size:22px;padding-top:6px;">·</div>
                <div style="text-align:center;">
                  <div style="font-size:11px;color:#ef4444;font-weight:600;margin-bottom:2px;">HIGH</div>
                  <div style="font-size:18px;font-weight:700;color:#ef4444;">PD ≥ {high_thr:.0%}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Ratio cards
        _ratio_cards(ui)

        # SHAP computation
        shap_row = None
        explainer = None
        try:
            explainer, preproc, kind = get_explainer(shap_model, shap_tag, tuple(schema), months, halflife)
            X_one_shap = (
                pd.DataFrame(preproc.transform(X_one), index=X_one.index, columns=schema)
                if preproc is not None else X_one
            )
            vals = explainer(X_one_shap)
            shap_row = vals.values[0] if hasattr(vals, "values") else np.array(vals)[0]
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

        if shap_row is not None:
            _driver_chips(shap_row, schema, X_one.iloc[0])

            st.markdown('<div class="section-header">Why this prediction? — SHAP Waterfall</div>', unsafe_allow_html=True)
            st.markdown("""
<div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.18);border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#9aa6d1;line-height:1.6;">
  <b style="color:#a5b4fc;">How SHAP works:</b>&nbsp;
  SHAP (SHapley Additive exPlanations) decomposes the model's output into per-feature contributions, grounded in cooperative game theory.
  The waterfall starts from the <b>baseline</b> — the average model output across all borrowers (E[f(x)]) — and each bar shows how much a single feature <b>raises</b> (red, →) or <b>lowers</b> (blue, ←) the final score.
  The bars stack to reach <b>f(x)</b>, the uncalibrated model's raw score (log-odds) for this borrower — the PD above is that score after calibration.
  Features are sorted by impact magnitude — the top bar has the biggest influence on this specific prediction.
</div>
""", unsafe_allow_html=True)
            render_waterfall(explainer, shap_row, X_one)

            with st.expander("All feature contributions (|SHAP|)"):
                contrib = (
                    pd.Series(np.abs(shap_row), index=schema)
                    .sort_values(ascending=False)
                    .rename(PRETTY_NAMES)
                    .to_frame("Absolute SHAP")
                )
                st.dataframe(contrib, use_container_width=True)

# =========================================================
# Tab 2 — Global SHAP
# =========================================================
with tabs[1]:
    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

    img_bar = SHAP_DIR / f"summary_{shap_tag}_bar.png"
    img_bee = SHAP_DIR / f"summary_{shap_tag}_beeswarm.png"

    if img_bar.exists():
        c1, c2 = st.columns(2)
        c1.image(str(img_bar), caption="Mean |SHAP| — Feature Importance", use_container_width=True)
        if img_bee.exists():
            c2.image(str(img_bee), caption="SHAP Beeswarm — Direction & Distribution", use_container_width=True)
    else:
        if DATA_PATH.exists():
            try:
                df_raw = pd.read_csv(DATA_PATH).drop(columns=["DEFAULT"], errors="ignore")
                X_all  = compute_features(df_raw, months_default=months, halflife=halflife)
                X_all  = X_all.reindex(columns=schema).fillna(0.0)
                X_samp = X_all.sample(n=min(1500, len(X_all)), random_state=17)

                explainer_g, preproc_g, kind_g = make_explainer(shap_model, X_samp, schema)
                X_plot = X_samp
                if preproc_g is not None:
                    X_plot = pd.DataFrame(preproc_g.transform(X_samp), index=X_samp.index, columns=schema)
                vals_g = explainer_g(X_plot)
                sv = vals_g.values if hasattr(vals_g, "values") else np.array(vals_g)

                c1, c2 = st.columns(2)
                with c1:
                    fig1 = plt.figure(facecolor="#161b27")
                    shap.summary_plot(sv, X_plot, plot_type="bar", show=False)
                    st.pyplot(fig1, clear_figure=True)
                with c2:
                    fig2 = plt.figure(facecolor="#161b27")
                    shap.summary_plot(sv, X_plot, show=False)
                    st.pyplot(fig2, clear_figure=True)
            except Exception as e:
                st.info(f"Could not compute global SHAP: {e}")
        else:
            st.info("Run `python -m src.explain` to precompute SHAP images, or place data/final_dataset.csv here.")

    st.markdown("#### Reading these charts")
    st.markdown("""
- **Bar chart** — mean absolute SHAP per feature: bigger bar = more influence overall.
- **Beeswarm** — each dot is one sample. Position left/right shows whether the feature pushes PD **down** or **up**. Color encodes the feature value (pink = high, blue = low).
- **SHAP uses the uncalibrated base model** for stable explanations. The Predict tab scores with the calibrated model.
    """)

    GLOSSARY = {
        "pay_status_current":  "Latest payment code (−2 no consumption, −1 paid in full, 0 revolving credit, 1–8 months late).",
        "pay_status_ema":      "Exponential moving average of recent payment codes — tracks the trend.",
        "pay_status_avg":      "Simple running average of past payment codes.",
        "credit_utilization":  "Average monthly bill as a fraction of credit limit — how much credit is being used.",
        "repay_ratio_avg":     "Avg payment / avg bill, capped at 100% — overall payment coverage rate (100% when nothing is owed).",
        "cur_recent":          "Latest payment / latest bill, capped at 100% — most recent month's coverage (100% when nothing is owed).",
        "paydown_ratio":       "Avg payment / credit limit — how aggressively the balance is being paid down.",
        "payment_trend":       "(Latest pay − prior avg pay) / (prior avg pay + 1), capped to ±1 — is payment behaviour improving or worsening?",
        "pay_is_delinquent":   "Binary: 1 if current pay code ≥ 1 (any lateness).",
        "pay_severity":        "max(0, pay_status_current) / 8 — normalized lateness severity.",
        "avg_bill_amt":        "Updated average monthly bill (including latest).",
        "avg_pay_amt":         "Updated average monthly payment (including latest).",
        "LIMIT_BAL":           "Credit limit in NT dollars.",
        "AGE":                 "Borrower age.",
        "EDUCATION":           "1=Grad school, 2=University, 3=High school, 4=Other.",
        "MARRIAGE":            "1=Married, 2=Single, 3=Other.",
    }
    with st.expander("Feature glossary"):
        for k in [f for f in schema if f in GLOSSARY]:
            st.markdown(f"- **{PRETTY_NAMES.get(k, k)}** (`{k}`) — {GLOSSARY[k]}")


# =========================================================
# Tab 3 — Calibration
# =========================================================
with tabs[2]:
    st.markdown('<div class="section-header">Reliability Diagrams</div>', unsafe_allow_html=True)

    cal_dir = Path("images/calibration")
    items   = [
        (cal_dir / "reliability_uncalibrated.png", "Uncalibrated"),
        (cal_dir / "reliability_platt.png",         "Platt (sigmoid)"),
        (cal_dir / "reliability_isotonic.png",       "Isotonic"),
    ]
    c1, c2, c3 = st.columns(3)
    for (p, caption), col in zip(items, [c1, c2, c3]):
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)
        else:
            col.info(f"{caption}: not generated yet — run `python -m src.calibrate`")

    st.markdown("""
<div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.18);border-radius:10px;padding:10px 14px;margin:10px 0 14px;font-size:13px;color:#9aa6d1;line-height:1.6;">
  <b style="color:#a5b4fc;">How to read reliability diagrams:</b>&nbsp;
  Each plot groups borrowers into bins by predicted probability and compares the average predicted PD against the <em>actual observed default rate</em> in that bin.
  The <b>dashed diagonal</b> is perfect calibration — predicted = observed.
  Points <b>above</b> the line mean the model <em>underestimates</em> risk (it said 20%, but 30% defaulted).
  Points <b>below</b> mean it <em>overestimates</em> (it said 60%, but only 40% defaulted).
  A well-calibrated model's points hug the diagonal closely.
</div>
""", unsafe_allow_html=True)

    if _cal_meta:
        meta         = _cal_meta
        base_m       = meta.get("uncalibrated_eval", {})
        platt_m      = meta.get("platt_eval", {})
        iso_m        = meta.get("isotonic_eval", {})
        chosen_block = meta.get("chosen", {})
        chosen_m     = chosen_block.get("metrics", {})
        chosen_method= chosen_block.get("method", "—")
        chosen_op    = meta.get("threshold", {})
        crit         = meta.get("criterion", "f1")

        n_eval = meta.get("splits", {}).get("n_eval")
        st.markdown("#### Calibration metrics — held-out evaluation")
        if n_eval:
            st.caption(f"Computed on {n_eval:,} borrowers that were used neither for training nor for fitting the calibrators.")

        # Metric definition cards
        metric_defs = [
            ("ROC-AUC ↑", "#6366f1",
             "Area Under the ROC Curve",
             "Measures how well the model <b>ranks</b> defaulters above non-defaulters across all thresholds. "
             "1.0 = perfect separation, 0.5 = random. <em>Higher is better.</em> "
             "Not affected by calibration — purely about ranking."),
            ("PR-AUC ↑", "#8b5cf6",
             "Area Under the Precision-Recall Curve",
             "Like ROC-AUC but focuses on the <b>minority class (defaults)</b>. "
             "More informative than ROC-AUC when defaults are rare. <em>Higher is better.</em> "
             "Rewards models that find more defaulters without too many false alarms."),
            ("Brier Score ↓", "#f59e0b",
             "Mean Squared Probability Error",
             "Average squared difference between predicted probability and true label (0 or 1). "
             "Formula: <code>mean((p̂ᵢ − yᵢ)²)</code>. Range 0–1. <em>Lower is better.</em> "
             "Directly measures calibration quality — a well-calibrated model will have a low Brier score."),
            ("Log-Loss ↓", "#10b981",
             "Binary Cross-Entropy",
             "Penalises confident wrong predictions exponentially. "
             "Formula: <code>−mean(y·log(p̂) + (1−y)·log(1−p̂))</code>. <em>Lower is better.</em> "
             "More sensitive than Brier to extreme mispredictions (e.g. predicting 95% chance of no default when default occurs)."),
        ]

        def_cols = st.columns(4)
        for (label, color, title, body), col in zip(metric_defs, def_cols):
            with col:
                with st.popover(f"ⓘ {label}", use_container_width=True):
                    st.markdown(f"**{title}**")
                    st.markdown(body, unsafe_allow_html=True)

        key_cols = ["roc_auc", "pr_auc", "brier", "log_loss"]
        col_labels = ["ROC-AUC ↑", "PR-AUC ↑", "Brier ↓", "Log-Loss ↓"]
        table = pd.DataFrame(
            [
                [base_m.get(k, np.nan)  for k in key_cols],
                [platt_m.get(k, np.nan) for k in key_cols],
                [iso_m.get(k, np.nan)   for k in key_cols],
                [chosen_m.get(k, np.nan) for k in key_cols],
            ],
            index=["Uncalibrated", "Platt", "Isotonic", f"Chosen ({chosen_method})"],
            columns=col_labels,
        )
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

        # Calibration method definitions
        st.markdown("#### Calibration methods — what they do")
        chosen_note = " <b>Chosen here</b> — it achieved the lower Brier score on the evaluation split."
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("""
<div style="background:#161b27;border:1px solid #21293d;border-radius:12px;padding:14px 16px;">
<div style="font-size:12px;font-weight:700;color:#8b96b4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Uncalibrated</div>
<div style="font-size:13.5px;color:#e2e8f0;line-height:1.6;">
The raw model output. XGBoost outputs probabilities that often <b>underestimate or overestimate</b> true default rates — especially in the tails. Use as a relative rank, not an absolute probability.
</div>
</div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
<div style="background:#161b27;border:1px solid #21293d;border-radius:12px;padding:14px 16px;">
<div style="font-size:12px;font-weight:700;color:#8b96b4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Platt Scaling (Sigmoid)</div>
<div style="font-size:13.5px;color:#e2e8f0;line-height:1.6;">
Fits a <b>logistic regression on top</b> of the model scores using a held-out calibration set. Effective when the miscalibration is systematic and monotone. Fast, interpretable, but assumes S-shaped distortion.{chosen_note if chosen_method == "platt" else ""}
</div>
</div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
<div style="background:#161b27;border:1px solid #21293d;border-radius:12px;padding:14px 16px;">
<div style="font-size:12px;font-weight:700;color:#8b96b4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Isotonic Regression</div>
<div style="font-size:13.5px;color:#e2e8f0;line-height:1.6;">
Fits a <b>non-parametric monotone function</b> to the calibration set. More flexible than Platt — can correct any shape of miscalibration. Requires more data to avoid overfitting.{chosen_note if chosen_method == "isotonic" else ""}
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Operating threshold")
        if chosen_op:
            st.info(
                f"**{chosen_method.title()}** calibrated model, selected by **{crit}**:  \n"
                f"Threshold **t = {chosen_op.get('threshold', 0):.3f}**  —  "
                f"F1 = {chosen_op.get('f1', 0):.3f}, "
                f"Precision = {chosen_op.get('precision', 0):.3f}, "
                f"Recall = {chosen_op.get('recall', 0):.3f}"
            )
            st.caption("Band defaults in the sidebar are centred around this threshold (±10%).")

# ---------- Footer ----------
st.markdown("---")
st.caption("Credyte · Calibrated PD scoring · SHAP explanations from the base model · Built with Streamlit")
