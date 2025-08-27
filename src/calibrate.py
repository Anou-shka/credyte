# src/calibrate.py
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import compute_features, FEATURE_COLUMNS_DEFAULT


# -------------------- I/O helpers --------------------
def _load_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def _ensure_features(df: pd.DataFrame, months: int, halflife: int) -> pd.DataFrame:
    """If engineered cols aren't present, build them; keep DEFAULT if present."""
    has_eng = {"avg_pay_amt","avg_bill_amt","repay_ratio_avg","pay_status_current"}.issubset(df.columns)
    y = df["DEFAULT"] if "DEFAULT" in df.columns else None
    X = df.drop(columns=["DEFAULT"], errors="ignore")
    if not has_eng:
        X = compute_features(X, months_default=months, halflife=halflife, write_schema_to=None)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if y is not None:
        X["DEFAULT"] = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    return X

def _split_xy(df: pd.DataFrame, target: str = "DEFAULT") -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' missing.")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def _load_model_path(model_choice: str, models_dir: Path) -> Path:
    if model_choice == "best":
        p = models_dir / "best.joblib"
    elif model_choice == "baseline":
        p = models_dir / "baseline.joblib"
    else:
        p = Path(model_choice)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}")
    return p


# -------------------- Metrics & thresholding --------------------
def eval_probs(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p_clip)),
    }

def threshold_sweep(y_true: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    ts = np.linspace(0, 1, 101)
    rows = []
    for t in ts:
        yhat = (p >= t).astype(int)
        prec = precision_score(y_true, yhat, zero_division=0)
        rec  = recall_score(y_true, yhat, zero_division=0)
        f1   = f1_score(y_true, yhat, zero_division=0)
        tn = int(((y_true == 0) & (yhat == 0)).sum())
        fp = int(((y_true == 0) & (yhat == 1)).sum())
        fn = int(((y_true == 1) & (yhat == 0)).sum())
        tp = int(((y_true == 1) & (yhat == 1)).sum())
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        bal_acc = 0.5 * (tpr + (1 - fpr))
        youden = tpr - fpr
        rows.append(dict(t=t, precision=prec, recall=rec, f1=f1,
                         tpr=tpr, fpr=fpr, balanced_acc=bal_acc, youden=youden,
                         tp=tp, fp=fp, tn=tn, fn=fn))
    return pd.DataFrame(rows)

def pick_threshold(curve: pd.DataFrame, criterion: str = "f1") -> Dict[str, float]:
    if criterion == "youden":
        i = int(curve["youden"].values.argmax())
    elif criterion == "balanced_acc":
        i = int(curve["balanced_acc"].values.argmax())
    else:
        i = int(curve["f1"].values.argmax())
    row = curve.iloc[i]
    return {"threshold": float(row.t), "criterion": criterion,
            "f1": float(row.f1), "precision": float(row.precision), "recall": float(row.recall),
            "tpr": float(row.tpr), "fpr": float(row.fpr),
            "balanced_acc": float(row.balanced_acc), "youden": float(row.youden),
            "tp": int(row.tp), "fp": int(row.fp), "tn": int(row.tn), "fn": int(row.fn)}


# -------------------- Plotting --------------------
def plot_reliability(y: np.ndarray, p: np.ndarray, title: str, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
    plt.figure(figsize=(4.8, 4.2))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    plt.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# -------------------- Calibrator compat helper --------------------
def _calibrator_prefit(estimator, method: str):
    """
    Create CalibratedClassifierCV for a **pre-fit** estimator.
    sklearn>=1.1 uses 'estimator', older versions used 'base_estimator'.
    """
    try:
        # new API
        return CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:
        # old API
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")


# -------------------- Main --------------------
def main(
    input_path: Path,
    features_path: Path | None,
    models_dir: Path,
    model_choice: str,
    out_models_dir: Path,
    months: int,
    halflife: int,
    calib_frac: float,
    seed: int,
    threshold_criterion: str,
):
    # 1) Load data -> features
    if features_path:
        df_in = _load_any(features_path)
    else:
        df_in = _load_any(input_path)
    feats = _ensure_features(df_in, months=months, halflife=halflife)
    X_all, y_all = _split_xy(feats, "DEFAULT")

    # 2) Align to training schema if available
    schema_path = models_dir / "feature_columns.txt"
    if schema_path.exists():
        with open(schema_path) as f:
            cols = [ln.strip() for ln in f if ln.strip()]
        X_all = X_all.reindex(columns=cols).fillna(0.0)

    # 3) Load pre-fit model
    model_path = _load_model_path(model_choice, models_dir)
    base_model = joblib.load(model_path)

    # 4) Create eval / calibration split (eval gets the larger chunk by default)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
    train_idx, calib_idx = next(sss.split(X_all, y_all))
    eval_idx = train_idx             # larger split → used for evaluation/threshold selection
    X_eval, y_eval = X_all.iloc[eval_idx], y_all.iloc[eval_idx].values
    X_cal,  y_cal  = X_all.iloc[calib_idx], y_all.iloc[calib_idx].values

    # 5) Uncalibrated baseline metrics (on eval)
    p_base_eval = base_model.predict_proba(X_eval)[:, 1]
    base_metrics = eval_probs(y_eval, p_base_eval)
    base_curve   = threshold_sweep(y_eval, p_base_eval)
    base_op      = pick_threshold(base_curve, threshold_criterion)

    # 6) Fit calibrators on calibration split (prefit mode)
    platt = _calibrator_prefit(base_model, method="sigmoid")
    platt.fit(X_cal, y_cal)
    iso = _calibrator_prefit(base_model, method="isotonic")
    iso.fit(X_cal, y_cal)

    # 7) Evaluate calibrators on eval split
    p_platt_eval = platt.predict_proba(X_eval)[:, 1]
    p_iso_eval   = iso.predict_proba(X_eval)[:, 1]
    platt_metrics = eval_probs(y_eval, p_platt_eval)
    iso_metrics   = eval_probs(y_eval, p_iso_eval)

    # choose better by Brier (tie→logloss)
    choose_iso = (
        (iso_metrics["brier"] < platt_metrics["brier"]) or
        (np.isclose(iso_metrics["brier"], platt_metrics["brier"]) and iso_metrics["log_loss"] < platt_metrics["log_loss"])
    )
    chosen_name   = "isotonic" if choose_iso else "platt"
    chosen_model  = iso if choose_iso else platt
    p_chosen_eval = p_iso_eval if choose_iso else p_platt_eval

    # 8) Pick operating threshold on eval for the chosen calibrator
    chosen_curve = threshold_sweep(y_eval, p_chosen_eval)
    chosen_op    = pick_threshold(chosen_curve, threshold_criterion)

    # 9) Save calibrated models + metrics + operating point
    out_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(platt, out_models_dir / "calibrated_platt.joblib")
    joblib.dump(iso,   out_models_dir / "calibrated_isotonic.joblib")
    joblib.dump(chosen_model, out_models_dir / "calibrated_best.joblib")

    metrics_payload = {
        "base_model": str(model_path),
        "splits": {"n_total": int(len(X_all)), "n_eval": int(len(X_eval)), "n_calib": int(len(X_cal))},
        "uncalibrated_eval": base_metrics,
        "platt_eval": platt_metrics,
        "isotonic_eval": iso_metrics,
        "chosen": {"method": chosen_name, "metrics": eval_probs(y_eval, p_chosen_eval)},
        "base_threshold": base_op,
        "threshold": chosen_op,
        "criterion": threshold_criterion,
    }
    with open(out_models_dir / "calibration_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(out_models_dir / "operating_point.json", "w") as f:
        json.dump({"method": chosen_name, **chosen_op}, f, indent=2)

    # 10) Reliability plots
    cal_dir = Path("images/calibration")
    plot_reliability(y_eval, p_base_eval,   "Uncalibrated",      cal_dir / "reliability_uncalibrated.png")
    plot_reliability(y_eval, p_platt_eval,  "Platt (sigmoid)",   cal_dir / "reliability_platt.png")
    plot_reliability(y_eval, p_iso_eval,    "Isotonic",          cal_dir / "reliability_isotonic.png")

    print(f"[done] Saved calibrated models to {out_models_dir}")
    print(f"[done] Operating point: {chosen_op} (method={chosen_name})")
    print(f"[info] Reliability plots → images/calibration/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/final_dataset.csv"),
                    help="Clean CSV with DEFAULT; ignored if --features is given.")
    ap.add_argument("--features", type=Path, default=None,
                    help="Precomputed features parquet/csv including DEFAULT.")
    ap.add_argument("--models_dir", type=Path, default=Path("models"))
    ap.add_argument("--model", type=str, default="best", help="'best' | 'baseline' | /path/to/model.joblib")
    ap.add_argument("--out_models", type=Path, default=Path("models"))
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--halflife", type=int, default=3)
    ap.add_argument("--calib_frac", type=float, default=0.5, help="Fraction as calibration split; rest is eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--criterion", type=str, default="f1", choices=["f1","youden","balanced_acc"])
    args = ap.parse_args()

    main(
        input_path=args.input,
        features_path=args.features,
        models_dir=args.models_dir,
        model_choice=args.model,
        out_models_dir=args.out_models,
        months=args.months,
        halflife=args.halflife,
        calib_frac=args.calib_frac,
        seed=args.seed,
        threshold_criterion=args.criterion,
    )
