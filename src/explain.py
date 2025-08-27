# src/explain.py
"""
Build SHAP summaries + per-borrower force plots for your models.

Usage:
  python -m src.explain --input data/final_dataset.csv
  # or if you already have engineered features:
  python -m src.explain --features data/features.parquet

Outputs (default):
  images/shap/
    summary_baseline_bar.png
    summary_baseline_beeswarm.png
    summary_best_bar.png
    summary_best_beeswarm.png
    force_baseline_<i>.html
    force_best_<i>.html
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

# our feature builder
from src.features import compute_features, FEATURE_COLUMNS_DEFAULT


# ------------------------- Paths & Defaults -------------------------
DEFAULT_DATA = Path("data/final_dataset.csv")
DEFAULT_FEATURES = None  # Path("data/features.parquet") if you prefer
DEFAULT_OUTDIR = Path("images/shap")
FEATURES_TXT = Path("models/feature_columns.txt")
BASELINE_PATH = Path("models/baseline.joblib")
BEST_PATH = Path("models/best.joblib")


# ------------------------- Small utilities -------------------------
def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def ensure_features(
    raw_or_features: pd.DataFrame,
    months_default: int,
    halflife: int,
) -> pd.DataFrame:
    """
    If engineered columns aren't present, build them using compute_features().
    Leaves DEFAULT in place if it exists.
    """
    df = raw_or_features.copy()
    target = None
    if "DEFAULT" in df.columns:
        target = df["DEFAULT"]
        df = df.drop(columns=["DEFAULT"])

    # if our engineered columns aren't there, build them
    key_cols = {"avg_pay_amt", "avg_bill_amt", "repay_ratio_avg", "pay_status_current"}
    if not key_cols.issubset(df.columns):
        df = compute_features(df, months_default=months_default, halflife=halflife, write_schema_to=None)

    # clean, then reattach target (if present)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if target is not None:
        df["DEFAULT"] = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    return df


def load_feature_list() -> Optional[list[str]]:
    if FEATURES_TXT.exists():
        with open(FEATURES_TXT) as f:
            cols = [ln.strip() for ln in f if ln.strip()]
        if cols:
            print(f"[schema] Loaded {len(cols)} feature names from {FEATURES_TXT}")
            return cols
    return None


def align_X(X: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    miss = [c for c in feature_cols if c not in X.columns]
    if miss:
        raise ValueError(f"Missing required feature columns: {miss[:12]}{'...' if len(miss)>12 else ''}")
    return X.loc[:, list(feature_cols)].copy()


def sample_background(X: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if len(X) <= size:
        return X.copy()
    return X.sample(n=size, random_state=seed)


def save_summary_plots(shap_values, X_like: pd.DataFrame, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)
    # bar
    plt.figure()
    shap.summary_plot(shap_values, X_like, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"summary_{prefix}_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_like, show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"summary_{prefix}_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def _expected_value(explainer) -> float:
    base = getattr(explainer, "expected_value", None)
    if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0:
        return float(np.array(base).ravel()[0])
    return float(base)


def save_force_plots(
    explainer,
    shap_values: np.ndarray,
    X_rows: pd.DataFrame,
    outdir: Path,
    prefix: str,
    n_force: int = 10,
    id_series: Optional[pd.Series] = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    base = _expected_value(explainer)
    n = min(n_force, len(X_rows))
    for i in range(n):
        # shap.force_plot expects a 1D array for one sample
        sv_i = shap_values[i]
        row = X_rows.iloc[i]
        label = f"{i}" if id_series is None else str(id_series.iloc[i])
        force = shap.force_plot(base, sv_i, row, matplotlib=False)
        shap.save_html(str(outdir / f"force_{prefix}_{label}.html"), force)


def make_linear_explainer_for_pipeline(pipe, X_trans_df: pd.DataFrame):
    """
    LinearExplainer for a Pipeline with a final LogisticRegression.
    X_trans_df must already be transformed to the estimator's input space.
    """
    from sklearn.linear_model import LogisticRegression
    est = pipe[-1]
    if not isinstance(est, LogisticRegression):
        # Not a plain LR — fall back to shap.Explainer
        return shap.Explainer(est, X_trans_df)
    try:
        return shap.LinearExplainer(est, X_trans_df, feature_perturbation="interventional")
    except Exception:
        # Newer SHAP prefers maskers; fallback
        return shap.Explainer(est, X_trans_df)


def explainer_for_model(model, X_bg: pd.DataFrame, transformed: bool = False):
    """
    Build a SHAP explainer best-suited to the model.
    - sklearn Pipeline (scaler + LR): LinearExplainer on the final estimator (with transformed X)
    - XGBoost/LightGBM: TreeExplainer
    - Other sklearn models: shap.Explainer
    """
    from sklearn.pipeline import Pipeline
    # Pipelines (we expect scaler + lr)
    if isinstance(model, Pipeline):
        return make_linear_explainer_for_pipeline(model, X_bg)

    # XGBoost / LightGBM → TreeExplainer
    mcls = type(model).__name__.lower()
    if "xgb" in mcls or "xgboost" in mcls or "lgbm" in mcls or "lightgbm" in mcls:
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass

    # Generic fallback
    return shap.Explainer(model, X_bg)


def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Return (values, ok_dataframe_to_plot).
    SHAP can return Explanation or raw numpy — normalize to numpy.
    """
    vals = explainer(X)
    if hasattr(vals, "values"):
        return vals.values, X
    # Older API
    return np.array(vals), X


# ------------------------- Main explain logic -------------------------
def explain_one_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    outdir: Path,
    bg_size: int,
    force_n: int,
    ids: Optional[pd.Series] = None,
) -> dict:
    """
    Compute & save summary + force plots for one model.
    Handles pipelines by transforming the background & X appropriately.
    """
    from sklearn.pipeline import Pipeline

    prefix = "baseline" if model_name == "baseline" else "best"

    # If Pipeline: transform X to estimator space (names preserved)
    if isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        if pre is not None:
            # background and the full X in the same transformed space
            X_bg_raw = sample_background(X, bg_size, seed=42)
            X_bg_np = pre.transform(X_bg_raw)
            X_bg = pd.DataFrame(X_bg_np, index=X_bg_raw.index, columns=X.columns)  # scaler keeps order
            X_np = pre.transform(X)
            X_for_shap = pd.DataFrame(X_np, index=X.index, columns=X.columns)
            expl = explainer_for_model(model, X_bg, transformed=True)
        else:
            X_bg = sample_background(X, bg_size, seed=42)
            X_for_shap = X
            expl = explainer_for_model(model, X_bg, transformed=False)
    else:
        # non-pipeline models work on raw feature space
        X_bg = sample_background(X, bg_size, seed=42)
        X_for_shap = X
        expl = explainer_for_model(model, X_bg)

    # Compute values on a capped sample for speed in summaries
    X_sum = sample_background(X_for_shap, min(len(X_for_shap), max(2000, bg_size)), seed=17)
    shap_vals_sum, _ = compute_shap_values(expl, X_sum)

    # Save global summary plots
    save_summary_plots(shap_vals_sum, X_sum, outdir, prefix=prefix)

    # Per-borrower force plots (first `force_n`)
    X_force = X_for_shap.iloc[:force_n].copy()
    shap_vals_force, _ = compute_shap_values(expl, X_force)
    save_force_plots(expl, shap_vals_force, X_force, outdir, prefix=prefix, n_force=force_n, id_series=ids)

    # Also dump a small info JSON
    payload = {
        "model": model_name,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "bg_size": int(len(X_bg)),
        "force_n": int(min(force_n, len(X))),
        "outdir": str(outdir),
        "feature_sample": list(X.columns[:10]),
    }
    with open(outdir / f"explain_{prefix}.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main(
    input_csv: Optional[Path],
    features_path: Optional[Path],
    outdir: Path,
    months: int,
    halflife: int,
    bg_size: int,
    force_n: int,
    which: str,  # "both" | "baseline" | "best"
):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    if features_path:
        df_in = read_table(features_path)
    elif input_csv:
        df_in = read_table(input_csv)
    else:
        raise ValueError("Provide either --input (raw CSV) or --features (engineered).")

    # 2) Build features if needed (keep DEFAULT/id if present)
    target = df_in["DEFAULT"] if "DEFAULT" in df_in.columns else None
    ids = df_in["ID"] if "ID" in df_in.columns else None
    feats = ensure_features(df_in, months_default=months, halflife=halflife)

    # 3) Align to feature schema
    feature_cols = load_feature_list() or FEATURE_COLUMNS_DEFAULT
    X = align_X(feats, feature_cols)

    # 4) Load models
    models = {}
    if which in ("both", "baseline"):
        if not BASELINE_PATH.exists():
            raise FileNotFoundError(f"Baseline model not found: {BASELINE_PATH}")
        models["baseline"] = joblib.load(BASELINE_PATH)

    if which in ("both", "best"):
        if not BEST_PATH.exists():
            raise FileNotFoundError(f"Best model not found: {BEST_PATH}")
        models["best"] = joblib.load(BEST_PATH)

    # 5) Explain each
    summaries = {}
    for name, model in models.items():
        print(f"[explain] {name}…")
        subdir = outdir  # single folder (prefix handles naming)
        summaries[name] = explain_one_model(
            model_name=name,
            model=model,
            X=X,
            outdir=subdir,
            bg_size=bg_size,
            force_n=force_n,
            ids=ids,
        )

    # 6) Save a manifest
    with open(outdir / "manifest.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"✅ Done. Plots saved in {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_DATA,
                    help="Raw cleaned CSV (e.g., data/final_dataset.csv). Will featureize if needed.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help="Precomputed features parquet/csv. If provided, --input is ignored.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--months", type=int, default=6, help="History length for running averages if building features.")
    ap.add_argument("--halflife", type=int, default=3, help="Half-life (months) for PAY EMA if building features.")
    ap.add_argument("--bg_size", type=int, default=512, help="Background sample size for SHAP.")
    ap.add_argument("--force_n", type=int, default=10, help="How many per-borrower force plots to export.")
    ap.add_argument("--which", type=str, default="both", choices=["both", "baseline", "best"],
                    help="Which model(s) to explain.")
    args = ap.parse_args()

    main(
        input_csv=args.input if args.features is None else None,
        features_path=args.features,
        outdir=args.outdir,
        months=args.months,
        halflife=args.halflife,
        bg_size=args.bg_size,
        force_n=args.force_n,
        which=args.which,
    )
