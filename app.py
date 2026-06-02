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
    FEATURE_COLUMNS_DEFAULT,
)

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
  background: linear-gradient(135deg, rgba(99,102,241,.12) 0%, rgba(139,92,246,.08) 100%);
  border: 1px solid rgba(99,102,241,.25);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -.5px; }
.hero p  { margin: 6px 0 0; color: var(--muted); font-size: 14px; }

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
FEATURES_TXT   = Path("models/feature_columns.txt")
BEST_PATH      = Path("models/best.joblib")
BASE_PATH      = Path("models/baseline.joblib")
CALIBRATED_PATH= Path("models/calibrated_best.joblib")
OP_POINT_PATH  = Path("models/operating_point.json")
SHAP_DIR       = Path("images/shap")


# ---------- Schema ----------
def load_feature_schema() -> List[str]:
    if FEATURES_TXT.exists():
        cols = [ln.strip() for ln in FEATURES_TXT.read_text().splitlines() if ln.strip()]
        if cols:
            return cols
    return FEATURE_COLUMNS_DEFAULT


# ---------- Model loading ----------
def load_models_and_threshold():
    if CALIBRATED_PATH.exists():
        scoring_model = joblib.load(CALIBRATED_PATH); scoring_tag = "calibrated"
    elif BEST_PATH.exists():
        scoring_model = joblib.load(BEST_PATH); scoring_tag = "best"
    elif BASE_PATH.exists():
        scoring_model = joblib.load(BASE_PATH); scoring_tag = "baseline"
    else:
        st.error("No model found in models/. Run src/train.py first.")
        st.stop()

    shap_path  = BEST_PATH if BEST_PATH.exists() else BASE_PATH
    shap_model = joblib.load(shap_path)
    shap_tag   = "best" if shap_path == BEST_PATH else "baseline"

    if OP_POINT_PATH.exists():
        op = json.loads(OP_POINT_PATH.read_text())
        t  = float(op["threshold"])
        default_low  = max(0.0, t - 0.10)
        default_high = min(1.0, t + 0.10)
    else:
        op = None
        default_low, default_high = 0.25, 0.50

    return scoring_model, scoring_tag, shap_model, shap_tag, (default_low, default_high), op


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
def make_explainer(model, X_bg: pd.DataFrame, feature_cols: List[str]):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

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

    name = type(model).__name__.lower()
    if "xgb" in name or "lgb" in name:
        try:
            return shap.TreeExplainer(model), None, "tree"
        except Exception:
            pass
    return shap.Explainer(model, X_bg), None, "generic"


# ---------- Gauge plot ----------
def render_gauge(proba: float, band: str) -> None:
    color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    color = color_map.get(band, "#6366f1")

    fig, ax = plt.subplots(figsize=(3.5, 2.0), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#161b27")
    ax.set_facecolor("#161b27")

    # Background arc (grey)
    theta_start, theta_end = np.pi, 0
    n = 200
    theta = np.linspace(theta_start, theta_end, n)
    ax.plot(theta, [1]*n, color="#21293d", linewidth=18, solid_capstyle="round")

    # Filled arc proportional to proba
    fill_end = np.pi - proba * np.pi
    theta_fill = np.linspace(np.pi, fill_end, n)
    ax.plot(theta_fill, [1]*n, color=color, linewidth=18, solid_capstyle="round")

    # Needle
    angle = np.pi - proba * np.pi
    ax.annotate("", xy=(angle, 0.95), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#e2e8f0", lw=2, mutation_scale=12))

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    ax.set_thetamin(0); ax.set_thetamax(180)

    ax.text(0.5, 0.22, f"{proba:.1%}", ha="center", va="center", transform=ax.transAxes,
            fontsize=24, fontweight="bold", color=color)
    ax.text(0.5, 0.05, band + " Risk", ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="#8b96b4")
    ax.text(0.07, 0.05, "0%", ha="center", transform=ax.transAxes, fontsize=9, color="#8b96b4")
    ax.text(0.93, 0.05, "100%", ha="center", transform=ax.transAxes, fontsize=9, color="#8b96b4")

    plt.tight_layout(pad=0)
    st.pyplot(fig, clear_figure=True)


# ---------- Waterfall SHAP ----------
def render_waterfall(explainer, shap_row: np.ndarray, X_one_shap: pd.DataFrame) -> None:
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).ravel()[0])

    exp = shap.Explanation(
        values=shap_row,
        base_values=base_val,
        data=X_one_shap.iloc[0].values,
        feature_names=list(X_one_shap.columns),
    )

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(9, 4.5), facecolor="#161b27")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#161b27")

    shap.plots.waterfall(exp, max_display=12, show=False)

    fig.patch.set_facecolor("#161b27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21293d")
    ax.tick_params(colors="#8b96b4")
    ax.xaxis.label.set_color("#8b96b4")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.style.use("default")


# ---------- Ratio cards ----------
PRETTY_NAMES = {
    "avg_pay_amt":        "Avg Monthly Payment",
    "avg_bill_amt":       "Avg Monthly Bill",
    "credit_utilization": "Credit Utilization",
    "repay_ratio_avg":    "Payment Coverage",
    "cur_recent":         "Latest Pay Coverage",
    "paydown_ratio":      "Paydown Rate",
    "payment_trend":      "Payment Trend",
    "pay_status_current": "Pay Status (Current)",
    "pay_status_avg":     "Pay Status (Avg)",
    "pay_status_ema":     "Pay Status (EMA)",
    "pay_status_ema":     "Pay Status (Trend)",
    "LIMIT_BAL":          "Credit Limit",
    "AGE":                "Age",
    "EDUCATION":          "Education",
    "MARRIAGE":           "Marital Status",
    "MONTHS":             "History (Months)",
    "pay_is_delinquent":  "Delinquent",
    "pay_severity":       "Delinquency Severity",
    "paydown_ratio":      "Paydown Rate",
}


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
    ups   = s.nlargest(3)
    downs = s.nsmallest(3)

    def make_cards(vals: pd.Series, cls: str) -> str:
        html = ""
        for k in vals.index:
            raw = X_one_row.get(k, 0)
            try:
                display_val = f"{float(raw):.3g}"
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="font-size:12px;color:#ef4444;font-weight:600;margin-bottom:6px;">↑ Increasing Risk</div>'
            f'<div class="driver-row">{make_cards(ups, "risk-up")}</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div style="font-size:12px;color:#10b981;font-weight:600;margin-bottom:6px;">↓ Protective Factors</div>'
            f'<div class="driver-row">{make_cards(downs, "risk-down")}</div>',
            unsafe_allow_html=True
        )


# ---------- Load artifacts ----------
schema = load_feature_schema()
scoring_model, scoring_tag, shap_model, shap_tag, (def_low, def_high), op_meta = load_models_and_threshold()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>⬡ Credyte</h1>
  <p>Credit risk analytics — calibrated probability of default, financial ratio analysis, and SHAP explainability.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("**Risk Band Thresholds**")
    low_thr  = st.slider("Low / Medium cutoff", 0.0, 1.0, def_low,  0.01)
    high_thr = st.slider("Medium / High cutoff", 0.0, 1.0, def_high, 0.01)
    st.divider()
    st.markdown("**Feature Engineering**")
    months   = st.slider("Months in history", 2, 12, 6, 1, help="History length for running averages")
    halflife = st.slider("PAY EMA half-life", 1, 12, 3, 1, help="Memory for payment status trend")
    st.divider()
    st.caption(f"Scoring: **{scoring_tag}** · SHAP base: **{shap_tag}**")
    st.caption(f"Features: **{len(schema)}**")
    if op_meta:
        st.caption(f"Operating point (F1): **t = {op_meta['threshold']:.3f}**")

tabs = st.tabs(["🔮 Predict", "🌍 Global SHAP", "📐 Calibration"])

# =========================================================
# Tab 1 — Predict
# =========================================================
with tabs[0]:
    pay_map = {
        "Paid in full (-1)": -1, "On time (0)": 0,
        "1 month late": 1, "2 months late": 2, "3 months late": 3,
        "4 months late": 4, "5 months late": 5,
    }
    edu_map = {"Graduate school": 1, "University": 2, "High school": 3, "Other": 4}
    mar_map = {"Married": 1, "Single": 2, "Other": 3}

    st.markdown('<div class="section-header">Borrower Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    LIMIT_BAL  = c1.number_input("Credit Limit (NT$)", value=120_000.0, step=5_000.0, format="%.0f")
    AGE        = c2.number_input("Age", value=32, min_value=18, max_value=100, step=1)
    edu_label  = c3.selectbox("Education", list(edu_map.keys()), index=1)
    mar_label  = c4.selectbox("Marital Status", list(mar_map.keys()), index=1)
    EDU = int(edu_map[edu_label]); MAR = int(mar_map[mar_label])

    st.markdown('<div class="section-header">Latest Month</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    pay_label      = c5.selectbox("Payment Status", list(pay_map.keys()), index=2)
    PAY_STATUS     = pay_map[pay_label]
    PAY_AMT_LATEST = c6.number_input("Payment Amount (NT$)", value=1_800.0, step=100.0, format="%.0f")
    BILL_AMT_LATEST= c7.number_input("Bill Amount (NT$)", value=22_000.0, step=500.0, format="%.0f")

    st.markdown('<div class="section-header">Prior History (Running Averages)</div>', unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    AVG_PAY_AMT_PRIOR  = c8.number_input("Prior Avg Payment (NT$)", value=1_500.0, step=100.0, format="%.0f")
    AVG_BILL_AMT_PRIOR = c9.number_input("Prior Avg Bill (NT$)", value=20_000.0, step=500.0, format="%.0f")
    PRIOR_PAY_AVG      = c10.number_input(
        "Prior Pay Status Avg", value=float(max(-1, PAY_STATUS - 1)), step=0.5,
        help="Running average of past PAY codes (−1 to 5)"
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

        gauge_col, detail_col = st.columns([1, 2])
        with gauge_col:
            render_gauge(proba, band)

        with detail_col:
            st.markdown(f"""
            <div class="kpi" style="margin-bottom:12px;">
              <div class="label">Probability of Default</div>
              <div class="value {band_css(band)}">{proba:.1%}</div>
              <span class="risk-badge {badge_css(band)}">{band} Risk</span>
            </div>
            <div class="kpi">
              <div class="label">Decision Threshold</div>
              <div class="value" style="font-size:20px;color:#8b96b4;">
                Low &lt; {low_thr:.0%} &nbsp;·&nbsp; Medium &lt; {high_thr:.0%} &nbsp;·&nbsp; High ≥ {high_thr:.0%}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Ratio cards
        _ratio_cards(ui)

        # SHAP computation
        shap_row = None
        explainer = None
        X_one_shap = X_one
        try:
            if DATA_PATH.exists():
                df_bg = pd.read_csv(DATA_PATH).head(800).drop(columns=["DEFAULT"], errors="ignore")
                X_bg = compute_features(df_bg, months_default=months, halflife=halflife, write_schema_to=None)
                X_bg = X_bg[[c for c in schema if c in X_bg.columns]].fillna(0.0)
            else:
                X_bg = pd.concat([X_one]*64, ignore_index=True)

            X_bg = X_bg.reindex(columns=schema).fillna(0.0)
            explainer, preproc, kind = make_explainer(shap_model, X_bg, schema)
            X_one_shap = (
                pd.DataFrame(preproc.transform(X_one), index=X_one.index, columns=schema)
                if preproc is not None and kind in ("linear", "generic") else X_one
            )
            vals = explainer(X_one_shap)
            shap_row = vals.values[0] if hasattr(vals, "values") else np.array(vals)[0]
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

        if shap_row is not None:
            _driver_chips(shap_row, schema, X_one_shap.iloc[0])

            st.markdown('<div class="section-header">Why this prediction? (SHAP Waterfall)</div>', unsafe_allow_html=True)
            render_waterfall(explainer, shap_row, X_one_shap)

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
                X_all  = compute_features(df_raw, months_default=months, halflife=halflife, write_schema_to=None)
                X_all  = X_all[[c for c in schema if c in X_all.columns]].fillna(0.0)
                X_all  = X_all.reindex(columns=schema).fillna(0.0)
                X_samp = X_all.sample(n=min(1500, len(X_all)), random_state=17)

                explainer_g, preproc_g, kind_g = make_explainer(shap_model, X_samp, schema)
                X_plot = X_samp
                if preproc_g is not None and kind_g in ("linear", "generic"):
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
        "pay_status_current":  "Latest payment code (−1 paid in full, 0 on time, 1–5 months late).",
        "pay_status_ema":      "Exponential moving average of recent payment codes — tracks the trend.",
        "pay_status_avg":      "Simple running average of past payment codes.",
        "credit_utilization":  "Average monthly bill as a fraction of credit limit — how much credit is being used.",
        "repay_ratio_avg":     "Avg payment / avg bill — overall payment coverage rate.",
        "cur_recent":          "Latest payment / latest bill — most recent month's coverage.",
        "paydown_ratio":       "Avg payment / credit limit — how aggressively the balance is being paid down.",
        "payment_trend":       "(Latest pay − prior avg) / prior avg — is payment behaviour improving or worsening?",
        "pay_is_delinquent":   "Binary: 1 if current pay code ≥ 1 (any lateness).",
        "pay_severity":        "pay_status_current / 8 — normalized lateness severity.",
        "avg_bill_amt":        "Updated average monthly bill (including latest).",
        "avg_pay_amt":         "Updated average monthly payment (including latest).",
        "LIMIT_BAL":           "Credit limit in NT dollars.",
        "AGE":                 "Borrower age.",
        "EDUCATION":           "1=Grad school, 2=University, 3=High school, 4=Other.",
        "MARRIAGE":            "1=Married, 2=Single, 3=Other.",
        "MONTHS":              "Number of months of history used for running averages.",
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

    st.markdown("#### Reading reliability diagrams")
    st.markdown("""
- **X-axis** = predicted PD; **Y-axis** = observed default rate in that bin.
- **Dashed diagonal** = perfect calibration (predicted ≈ actual).
- Points **above** the line → model underestimates risk. Points **below** → overestimates.
- Isotonic regression typically gives the best calibration given enough data.
    """)

    met_path = Path("models/calibration_metrics.json")
    if met_path.exists():
        meta         = json.loads(met_path.read_text())
        base_m       = meta.get("uncalibrated_eval", {})
        platt_m      = meta.get("platt_eval", {})
        iso_m        = meta.get("isotonic_eval", {})
        chosen_block = meta.get("chosen", {})
        chosen_m     = chosen_block.get("metrics", {})
        chosen_method= chosen_block.get("method", "—")
        chosen_op    = meta.get("threshold", {})
        crit         = meta.get("criterion", "f1")

        st.markdown("#### Calibration metrics (held-out evaluation)")
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
