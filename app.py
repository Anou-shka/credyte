# app.py — Calibrated prediction + pretty ratios + SHAP
import os
import io
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# === Your feature helpers ===
from src.features import (
    build_single_from_minimal,
    compute_features,
    FEATURE_COLUMNS_DEFAULT,
)

# ---------- Page config ----------
st.set_page_config(
    page_title="Credyte • Risk & SHAP",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Base styles (dark UI) ----------

st.markdown("""
<style>
.force-card{
  background:#ffffff;             /* light background so SHAP's black text is readable */
  border:1px solid #e8ebf5;
  border-radius:14px;
  padding:10px 14px;
  margin-top:.35rem;
}
.force-wrap{
  width:100%;
  display:flex;      /* center horizontally -  justify-content:center;    */ 
  overflow:hidden;                 /* hide overflow when zoomed */
}
.force-inner{
  transform-origin: center;    /* zoom from the center */
}
.section-kicker{
  background:linear-gradient(135deg,#1b2033,#1b2033);
  border:1px solid #2a3350;
  color:#c9d2ea;
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
  padding:10px 12px;
  border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{
  --accent:#6b8afd; --accent2:#a76bff;
  --card:#14182a; --border:#23283b; --muted:#9aa6d1;
  --ok:#16c784; --warn:#f5a524; --bad:#ef4e4e;
}
.main .block-container{padding-top:1.1rem;padding-bottom:3rem}
.hero{padding:18px;border-radius:14px;background:
      linear-gradient(135deg,rgba(107,138,253,.15),rgba(167,107,255,.12));
      border:1px solid var(--border)}
/* KPI cards (PD / Band / Thresholds) */
.kpi {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:14px;
  display:flex;
  flex-direction:column;
  justify-content:center;  /* vertical centering */
  align-items:center;      /* horizontal centering */
  text-align:center;       /* center the text */
}
.kpi .label {
  color:var(--muted);
  font-size:12.5px;
  margin-bottom:4px;       /* small gap above the value */
}
.kpi .value {
  font-size:28px;
  font-weight:600;
}

.chip{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid var(--border);font-size:12.5px}
.chip.low{color:var(--ok);border-color:rgba(22,199,132,.4)}
.chip.med{color:var(--warn);border-color:rgba(245,165,36,.4)}
.chip.high{color:var(--bad);border-color:rgba(239,78,78,.4)}
/* Pretty ratios + drivers + force plot */
.sec{font-size:14px;color:#9aa6d1;text-transform:uppercase;letter-spacing:.12em;margin:.9rem 0 .45rem}
.sec.accent{background:linear-gradient(90deg,rgba(107,138,253,.18),rgba(167,107,255,.10));
            border:1px solid #23283b;padding:6px 10px;border-radius:10px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
.stat{background:#14182a;border:1px solid #23283b;border-radius:14px;padding:12px}
.stat.emph{background:linear-gradient(180deg,rgba(107,138,253,.08),rgba(167,107,255,.06));
           border-color:rgba(107,138,253,.35)}
.stat .k{color:#9aa6d1;font-size:12px}
.stat .v{font-size:26px;font-weight:700;margin-top:.1rem}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{padding:6px 10px;border-radius:999px;border:1px solid #23283b;font-size:12.5px;background:rgba(255,255,255,.02)}
.pill.risk{color:#ff6b6b;border-color:rgba(255,107,107,.45)}
.pill.prot{color:#49d394;border-color:rgba(73,211,148,.45)}
.pill .sub{opacity:.65}
.force{background:#14182a;border:1px solid #23283b;border-radius:16px;padding:8px 10px}
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
DATA_PATH = Path("data/final_dataset.csv")        # optional for global SHAP
FEATURES_TXT = Path("models/feature_columns.txt")
BEST_PATH = Path("models/best.joblib")
BASE_PATH = Path("models/baseline.joblib")
CALIBRATED_PATH = Path("models/calibrated_best.joblib")
OP_POINT_PATH = Path("models/operating_point.json")
SHAP_DIR = Path("images/shap")

# ---------- Schema helper ----------
def load_feature_schema() -> List[str]:
    if FEATURES_TXT.exists():
        cols = [ln.strip() for ln in FEATURES_TXT.read_text().splitlines() if ln.strip()]
        if cols:
            return cols
    return FEATURE_COLUMNS_DEFAULT

# ---------- Bands / UI helpers ----------
def band_from(p: float, low: float, high: float) -> str:
    return "Low" if p < low else ("Medium" if p < high else "High")

def chip(band: str) -> str:
    return {"Low":"low","Medium":"med","High":"high"}.get(band, "")

def align_series_to_schema(s: pd.Series, schema: List[str]) -> pd.Series:
    aligned = {c: float(s.get(c, 0.0)) for c in schema}
    return pd.Series(aligned, index=schema, dtype=float)

# ---------- Calibrated scoring + SHAP base ----------
def load_models_and_threshold():
    # scoring: calibrated if available, else best, else baseline
    if CALIBRATED_PATH.exists():
        scoring_model = joblib.load(CALIBRATED_PATH); scoring_tag = "calibrated"
    elif BEST_PATH.exists():
        scoring_model = joblib.load(BEST_PATH); scoring_tag = "best"
    elif BASE_PATH.exists():
        scoring_model = joblib.load(BASE_PATH); scoring_tag = "baseline"
    else:
        st.error("No model found in models/. Train and/or calibrate first.")
        st.stop()

    # SHAP uses uncalibrated base (best if present else baseline)
    shap_path = BEST_PATH if BEST_PATH.exists() else BASE_PATH
    shap_model = joblib.load(shap_path)
    shap_tag = "best" if shap_path == BEST_PATH else "baseline"

    # operating point (for default band sliders)
    if OP_POINT_PATH.exists():
        op = json.loads(OP_POINT_PATH.read_text())
        t = float(op["threshold"])
        default_low  = max(0.0, t - 0.10)
        default_high = min(1.0, t + 0.10)
    else:
        op = None
        default_low, default_high = 0.30, 0.60

    return scoring_model, scoring_tag, shap_model, shap_tag, (default_low, default_high), op

# ---------- SHAP helpers ----------
def make_explainer(model, X_bg: pd.DataFrame, feature_cols: List[str]):
    """
    Build a SHAP explainer that works for both Pipeline(LogReg) and XGB/LGBM.
    We transform for pipelines so LinearExplainer can attach to the LR head.
    """
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
        # fallback for odd pipelines
        return shap.Explainer(model, X_bg), None, "pipeline-generic"

    name = type(model).__name__.lower()
    if "xgb" in name or "lgb" in name:
        try:
            return shap.TreeExplainer(model), None, "tree"
        except Exception:
            pass
    return shap.Explainer(model, X_bg), None, "generic"


import streamlit.components.v1 as components

def render_force_plot_centered(
    explainer, shap_row, series_one, *, scale: float = 1.8, height: int = 320
):
    """SHAP force plot without the unreadable bubble badge (we’ll add our own text)."""
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray, pd.Series)):
        base_val = float(np.array(base_val).ravel()[0])

    # shap_values sum + base = prediction
    fx = base_val + float(np.sum(shap_row))

    plot = shap.force_plot(
        base_value=base_val,
        shap_values=shap_row,
        features=series_one,
        matplotlib=False
    )

    css = """
    <style>
    /* make all labels white for dark theme */
    .shap text, text { fill:#e6e9f5 !important; }
    .shap, .shap * { color:#e6e9f5 !important; }
    .tick text { fill:#e6e9f5 !important; }
    .legend text, .baseValue { color:#e6e9f5 !important; fill:#e6e9f5 !important; }

    /* --- remove number from prediction bubble --- */
    .shap g.value text, 
    .shap g.value tspan {
        display: none !important;   /* hide everything inside value badge */
    }

    /* --- BUT keep the "f(x)" label --- */
    .shap text:contains("f(x)"),
    .shap tspan:contains("f(x)") {
        display: inline !important;
        fill: #e6e9f5 !important;
        font-weight: 600 !important;
    }

    body { margin:0; background:transparent; }
    </style>

    """

    html = f"""
      <head>{shap.getjs()}{css}</head>
      <body>
        <div style="display:flex;justify-content:center;margin-top:10px;margin-bottom:10px;">
          <div style="color:#fff;font-size:18px;font-weight:600;">
            f(x) = {fx:.2f}
          </div>
        </div>
        <div style="display:flex;justify-content:center;overflow:hidden;
                    margin-top:10px;margin-bottom:24px;">
          <div style="display:inline-block;transform:scale({scale});
                      transform-origin: top center;min-width:1100px;">
            {plot.html()}
          </div>
        </div>
      </body>
    """

    components.html(html, height=150, scrolling=False)



# ---------- Pretty Predict-tab helpers ----------
def _ratio_cards(ui: dict):
    """Render ratio grid with centered content."""
    items = [
        ("Average Pay Amount",  f"{ui['avg_pay_amt']:,.0f}", True),
        ("Average Bill Amount", f"{ui['avg_bill_amt']:,.0f}", True),
        ("Average Repay Ratio",      f"{ui['repay_ratio_avg']:.3f}", True),
        ("Receny",           f"{ui['cur_recent']:.3f}", True),
        ("Paydown Ratio",        f"{ui['paydown_ratio']:.3f}", True),
        ("Minimum Pay Ratio",                  f"{ui['mpr']:.3f}", True),
    ]

    cards_html = ""
    for k, v, emph in items:
        cls = "stat emph" if emph else "stat"
        cards_html += (
            f'<div class="{cls}" '
            'style="text-align:center;display:flex;flex-direction:column;'
            'justify-content:center;align-items:center;">'
            f'<div class="k">{k}</div>'
            f'<div class="v">{v}</div>'
            '</div>'
        )

    st.markdown(
        '<div class="sec">Financial ratios (computed)</div>'
        f'<div class="stats">{cards_html}</div>',
        unsafe_allow_html=True
    )


def _top_driver_chips(shap_row: np.ndarray, schema: list[str], X_one_row: pd.Series):
    """Show 3 strongest ↑risk and 3 strongest ↓risk as mini-cards with spacing."""
    pretty = {
        "pay_status_ema":"PAY EMA","pay_status_current":"Current Pay Status","pay_status_avg":"Average Pay Status",
        "avg_bill_amt":"Average Bill Amount","avg_pay_amt":"Average Pay Amount","repay_ratio_avg":"Repay Ratio",
        "paydown_ratio":"Paydown Ratio","cur_recent":"Recency","mpr":"Min pay ratio",
        "LIMIT_BAL":"Limit Balance","AGE":"Age","MARRIAGE":"Marriage","EDUCATION":"Education","MONTHS":"Months",
    }

    s = pd.Series(shap_row, index=schema)
    ups   = s.nlargest(3)
    downs = s.nsmallest(3)

    def card_row(vals: pd.Series, border_color: str, title: str) -> str:
        cards = []
        for k in vals.index:
            cards.append(
                (
                    '<div style="flex:1;background:#14182a;border:1px solid {bc};'
                    'border-radius:10px;padding:8px 10px;text-align:center;min-width:110px;">'
                    '<div style="font-size:11px;color:#9aa6d1;">{name}</div>'
                    '<div style="font-size:16px;font-weight:700;color:#fff;margin-top:2px;">{val:.3g}</div>'
                    '</div>'
                ).format(bc=border_color, name=pretty.get(k, k), val=X_one_row.get(k, 0))
            )
        # IMPORTANT: no leading spaces/newlines at line starts → avoids Markdown code block
        row = (
            '<div style="margin:.5rem 0;">'
            '<div style="font-size:12px;color:#9aa6d1;text-transform:uppercase;'
            'letter-spacing:.07em;margin-bottom:.25rem">{title}</div>'
            '<div style="display:flex;gap:10px;flex-wrap:wrap">{cards}</div>'
            '</div>'
        ).format(title=title, cards="".join(cards))
        return row

    st.markdown('<div class="sec accent">Key drivers</div>', unsafe_allow_html=True)
    st.markdown(card_row(ups,   "#ef4e4e", "↑ Risk Drivers"), unsafe_allow_html=True)
    st.markdown(card_row(downs, "#16c784", "↓ Protective Drivers"), unsafe_allow_html=True)



# ---------- Load artifacts ----------
schema = load_feature_schema()
scoring_model, scoring_tag, shap_model, shap_tag, (def_low, def_high), op_meta = load_models_and_threshold()

# ---------- Header ----------
st.markdown("""
<div class="hero" style="text-align:center;">
  <h2 style="margin:0">Credyte — Prediction & Explainability</h2>
  <p style="color:#a8b3cf;margin:.4rem 0 0">
    Enter the latest status & amounts + prior averages.<br>
    We update averages, compute ratios, predict calibrated PD, band the risk, and show SHAP from the base model.
  </p>
</div>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Settings")
    low_thr  = st.slider("Low / Medium cut", 0.0, 1.0, def_low, 0.01, help="Risk band thresholds")
    high_thr = st.slider("Medium / High cut", 0.0, 1.0, def_high, 0.01, help="Risk band thresholds")
    months   = st.slider("Months in history (for averages)", 2, 12, 6, 1, help="History length for averages")
    halflife = st.slider("Half-life for PAY EMA", 1, 12, 3, 1, help="PAY trend memory (months)")
    st.caption(f"Scoring model: **{scoring_tag}**, SHAP base: **{shap_tag}** • {len(schema)} features")
    if op_meta:
        st.caption(f"Operating point ({op_meta['criterion']}): t = **{op_meta['threshold']:.3f}**")

tabs = st.tabs(["Predict", "Global SHAP", "Calibration"])

# =========================================================
# Tab 1 — Predict (single borrower)
# =========================================================
with tabs[0]:
    st.markdown('<div class="sec">Inputs</div>', unsafe_allow_html=True)

    # PAY status selector
    pay_map = {
        "Paid in full": -1, "On time": 0,
        "1 month late": 1, "2 months late": 2, "3 months late": 3,
        "4 months late": 4, "5 months late": 5
    }

    c1,c2,c3 = st.columns(3)
    LIMIT_BAL = c1.number_input("LIMIT_BAL", value=120000.0, step=1000.0, format="%.0f", help="Credit limit")
    AGE       = c2.number_input("AGE", value=32, min_value=18, max_value=100, step=1, help="Select the borrower's age")
    PAY_STATUS = pay_map[c3.selectbox("Latest pay status", list(pay_map.keys()), index=2, help="Select latest PAY status")]

    # EDUCATION (1=grad school, 2=university, 3=high school, 4=other)
    edu_map = {
        "Graduate school": 1, "University": 2, "High school": 3, "Other": 4,
    }
    # MARRIAGE (1=married, 2=single, 3=other)
    mar_map = {"Married": 1, "Single": 2, "Other": 3}

    c4,c5,c6 = st.columns(3)
    edu_label = c4.selectbox("Education", list(edu_map.keys()), index=1, help="Please select highest education attained")
    EDU = int(edu_map[edu_label])
    mar_label = c5.selectbox("Marital status", list(mar_map.keys()), index=1, help="Please select marital status")
    MAR = int(mar_map[mar_label])
    PRIOR_PAY_AVG = c6.number_input(
        "PRIOR_PAY_AVG",
        value=float(PAY_STATUS),   # default to latest PAY_STATUS
        step=0.5,
        help="Prior PAY average over the months in history (can be fractional)"
    )


    c7,c8,c9 = st.columns(3)
    PAY_AMT_LATEST  = c7.number_input("Latest pay amount", value=1800.0, step=100.0, format="%.0f", help="Most recent amount paid")
    BILL_AMT_LATEST = c8.number_input("Latest bill amount", value=2200.0, step=100.0, format="%.0f", help="Most recent bill amount")
    AVG_PAY_AMT_PRIOR  = c9.number_input("Prior average pay amount ", value=1500.0, step=100.0, format="%.0f", help="Prior average PAY amount over the months in history")

    c10,c11,_ = st.columns(3)
    AVG_BILL_AMT_PRIOR = c10.number_input("Prior average bill amount", value=2100.0, step=100.0, format="%.0f")
    _ = c11.markdown("")  # spacer

    go = st.button("Predict", use_container_width=True)

    if go:
        # --- Build engineered feature row + UI stats
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

        # --- Predict with calibrated/best/baseline scoring model
        proba = float(scoring_model.predict_proba(X_one)[:, 1][0])
        band  = band_from(proba, low_thr, high_thr)

        k1,k2,k3 = st.columns([1,1,2])
        with k1:
            st.markdown(f'<div class="kpi"><div class="label">PD (Probability of Default)</div><div class="value">{proba:.3f}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi"><div class="label">Risk Band</div><div class="value">{band}</div></div>', unsafe_allow_html=True)
        with k3:
            # Build the thresholds text as HTML
            thresh_html = (
                f'Low &lt; <b>{low_thr:.2f}</b> &nbsp;•&nbsp; '
                f'Medium &lt; <b>{high_thr:.2f}</b> &nbsp;•&nbsp; '
                f'High ≥ <b>{high_thr:.2f}</b>'
            )

            # Put it inside the KPI card
            st.markdown(
                f'''
                <div class="kpi">
                <div class="label">Thresholds</div>
                <div class="value"">{thresh_html}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        # --- Pretty ratios grid
        _ratio_cards(ui)

        # --- SHAP setup (base model) and explanation
        shap_row = None
        try:
            if DATA_PATH.exists():
                df_bg = pd.read_csv(DATA_PATH).head(800).drop(columns=["DEFAULT"], errors="ignore")
                X_bg = compute_features(df_bg, months_default=months, halflife=halflife, write_schema_to=None)
                X_bg = X_bg[[c for c in schema if c in X_bg.columns]].fillna(0.0)
            else:
                X_bg = pd.concat([X_one]*64, ignore_index=True)

            X_bg = X_bg.reindex(columns=schema).fillna(0.0)
            explainer, preproc, kind = make_explainer(shap_model, X_bg, schema)
            X_one_shap = (pd.DataFrame(preproc.transform(X_one), index=X_one.index, columns=schema)
                          if preproc is not None and kind in ("linear","generic") else X_one)
            vals = explainer(X_one_shap)
            shap_row = vals.values[0] if hasattr(vals, "values") else np.array(vals)[0]
        except Exception as e:
            shap_row = None
            st.warning(f"Couldn’t compute SHAP: {e}")

        # --- Key driver chips
        if shap_row is not None:
            _top_driver_chips(shap_row, schema, X_one.iloc[0])

        #---- the graph----
        st.markdown('<div class="section-kicker">Why this prediction?</div>', unsafe_allow_html=True)
       
       
        # If you want simple defaults (no right-side controls):
        use_waterfall = False     # set True to show the fallback waterfall
        zoom = 1             # tweak to taste (1.0 – 1.6 is a good range)

        if shap_row is None:
            st.info("No SHAP to display.")
        elif not use_waterfall:
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            render_force_plot_centered(explainer, shap_row, X_one_shap.iloc[0], scale=zoom, height=260)
        else:
            # Waterfall fallback (still readable on dark themes)
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(np.array(base_val).ravel()[0])
            exp = shap.Explanation(
                values=shap_row,
                base_values=base_val,
                data=X_one_shap.iloc[0].values,
                feature_names=list(X_one_shap.columns),
            )
            fig = plt.figure(figsize=(9.5, 4.2))
            shap.plots.waterfall(exp, max_display=12, show=False)
            st.pyplot(fig, clear_figure=True)

        # --- Top feature contributions (|SHAP|)
        with st.expander("Top feature contributions (|SHAP|)"):
            if shap_row is not None:
                contrib = pd.Series(np.abs(shap_row), index=schema).sort_values(ascending=False).head(12)
                st.dataframe(contrib.to_frame("abs_SHAP"), use_container_width=True)
            else:
                st.write("—")

# =========================================================
# Tab 2 — Global SHAP (from base model)
# =========================================================
with tabs[1]:
    st.markdown('<div class="sec">Global Explainability</div>', unsafe_allow_html=True)

    # Prefer precomputed images from src/explain.py if present
    img_bar = SHAP_DIR / f"summary_{shap_tag}_bar.png"
    img_bee = SHAP_DIR / f"summary_{shap_tag}_beeswarm.png"
    if img_bar.exists():
        c1,c2 = st.columns(2)
        c1.image(str(img_bar), caption="Feature Importance (bar)", use_container_width=True)
        if img_bee.exists():
            c2.image(str(img_bee), caption="SHAP Beeswarm", use_container_width=True)
    else:
        if DATA_PATH.exists():
            try:
                df_raw = pd.read_csv(DATA_PATH).drop(columns=["DEFAULT"], errors="ignore")
                X_all = compute_features(df_raw, months_default=months, halflife=halflife, write_schema_to=None)
                X_all = X_all[[c for c in schema if c in X_all.columns]].fillna(0.0)
                X_all = X_all.sample(n=min(2000, len(X_all)), random_state=17)

                explainer, preproc, kind = make_explainer(shap_model, X_all, schema)
                X_plot = X_all
                if preproc is not None and kind in ("linear","generic"):
                    X_np = preproc.transform(X_all)
                    X_plot = pd.DataFrame(X_np, index=X_all.index, columns=schema)
                vals = explainer(X_plot)
                shap_vals = vals.values if hasattr(vals, "values") else np.array(vals)

                c1,c2 = st.columns(2)
                with c1:
                    fig1 = plt.figure()
                    shap.summary_plot(shap_vals, X_plot, plot_type="bar", show=False)
                    st.pyplot(fig1, clear_figure=True)
                with c2:
                    fig2 = plt.figure()
                    shap.summary_plot(shap_vals, X_plot, show=False)
                    st.pyplot(fig2, clear_figure=True)
            except Exception as e:
                st.info(f"Couldn’t compute global SHAP from data: {e}")
        else:
            st.info("Run `python -m src.explain` to precompute SHAP images, or place data/final_dataset.csv to compute here.")

    # Explanatory text
    st.markdown("#### How to read these charts")
    st.markdown(
        """
- **Left (bar)** shows overall importance: mean absolute SHAP for each feature (bigger bar = bigger global impact).
- **Right (beeswarm)** adds **direction** and **distribution**:
  - Points to the **right** push PD **up**; to the **left** push it **down**.
  - **Color** encodes feature value (pink = high, blue = low). If pink clusters on the right → higher values **increase risk**; on the left → **decrease risk**.
- SHAP explains the **base model** (uncalibrated). The Predict tab still **scores** with the calibrated model so PDs are usable.
        """
    )

    FEATURE_GLOSSARY = {
        "pay_status_current": "Latest PAY code (−1 paid in full, 0 on time, 1..8 months late).",
        "pay_status_ema": "Exponential moving average of recent PAY (half-life from sidebar).",
        "pay_status_avg": "Longer-term mean of past PAY statuses.",
        "pay_is_delinquent": "Indicator from PAY (e.g., PAY ≥ 1).",
        "pay_severity": "Optional numeric encoding of delinquency severity.",
        "avg_bill_amt": "Updated average monthly bill (prior avg + latest).",
        "avg_pay_amt": "Updated average monthly payment (prior avg + latest).",
        "repay_ratio_avg": "Average payment/bill ratio.",
        "paydown_ratio": "Portion of prior bill paid down.",
        "cur_recent": "Recency proxy from payments/bills.",
        "mpr": "Minimum payment ratio proxy.",
        "LIMIT_BAL": "Credit limit.",
        "AGE": "Borrower age.",
        "EDUCATION": "1=Grad, 2=Univ, 3=HS, 4=Other.",
        "MARRIAGE": "1=Married, 2=Single, 3=Other.",
        "MONTHS": "History length used in averages.",
    }
    with st.expander("Feature glossary (what each term means)"):
        for k in [f for f in schema if f in FEATURE_GLOSSARY]:
            st.markdown(f"- **{k}** — {FEATURE_GLOSSARY[k]}")

# =========================================================
# Tab 3 — Calibration (if generated by src/calibrate.py)
# =========================================================
with tabs[2]:
    st.markdown('<div class="sec">Calibration (Reliability)</div>', unsafe_allow_html=True)
    cal_dir = Path("images/calibration")
    c1,c2,c3 = st.columns(3)
    items = [
        (cal_dir/"reliability_uncalibrated.png", "Uncalibrated"),
        (cal_dir/"reliability_platt.png", "Platt (sigmoid)"),
        (cal_dir/"reliability_isotonic.png", "Isotonic"),
    ]
    for (p, caption), col in zip(items, [c1,c2,c3]):
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)
        else:
            col.info(f"{caption}: not found")

    st.markdown("#### How to read these")
    st.markdown(
        """
- **X-axis** = predicted PD; **Y-axis** = *empirical* default rate in that bin.
- The **dashed diagonal** is **perfect calibration** (predicted ≈ observed).
- Points **above** the line → the model **underestimates** risk; **below** → **overestimates** risk.
- **Platt** is a smooth sigmoid remap; **Isotonic** is flexible and monotonic and often calibrates best with enough data.
        """
    )

    met_path = Path("models/calibration_metrics.json")
    if met_path.exists():
        meta = json.loads(met_path.read_text())
        base = meta.get("uncalibrated_eval", {})
        platt = meta.get("platt_eval", {})
        iso = meta.get("isotonic_eval", {})
        chosen_block = meta.get("chosen", {})
        chosen_method = chosen_block.get("method", "—")
        chosen_metrics = chosen_block.get("metrics", {})
        chosen_op = meta.get("threshold", {})
        crit = meta.get("criterion", "f1")

        st.markdown("#### Calibration metrics (held-out eval)")
        key_cols = ["roc_auc", "pr_auc", "brier", "log_loss"]
        table = pd.DataFrame(
            [ [base.get(k, np.nan) for k in key_cols],
              [platt.get(k, np.nan) for k in key_cols],
              [iso.get(k, np.nan) for k in key_cols],
              [chosen_metrics.get(k, np.nan) for k in key_cols] ],
            index=["Uncalibrated", "Platt", "Isotonic", f"Chosen ({chosen_method})"],
            columns=key_cols,
        )
        st.dataframe(table.style.format({c: "{:.4f}" for c in key_cols}), use_container_width=True)

        st.markdown("#### Operating threshold")
        if chosen_op:
            st.write(
                f"Selected by **{crit}** for **{chosen_method}**: "
                f"**t = {chosen_op.get('threshold', float('nan')):.3f}**  "
                f"(F1={chosen_op.get('f1', float('nan')):.3f}, "
                f"Precision={chosen_op.get('precision', float('nan')):.3f}, "
                f"Recall={chosen_op.get('recall', float('nan')):.3f})"
            )
            st.caption(
                "In the Predict tab, the Low/Medium/High defaults are centered around this t (±0.10). "
                "Adjust to match business costs or portfolio targets."
            )
        else:
            st.info("No operating point found. Re-run `python -m src.calibrate`.")
    else:
        st.info("No metrics file found. Run `python -m src.calibrate` to generate `models/calibration_metrics.json`.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Scoring uses calibrated probabilities when available. SHAP explanations come from the underlying base model.")
