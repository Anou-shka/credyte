# src/train.py
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

# Optional boosters
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# your feature builder
from src.features import compute_features, FEATURE_COLUMNS_DEFAULT


# ---------------- I/O helpers ----------------
def _load_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def _ensure_features(df: pd.DataFrame, months_default: int, halflife: int) -> pd.DataFrame:
    """
    If engineered keys aren't present, build them from minimal inputs.
    Otherwise just clean dtypes/NaNs.
    """
    need_cols = {"avg_pay_amt", "avg_bill_amt", "repay_ratio_avg", "pay_status_current"}
    if not need_cols.issubset(df.columns):
        df = compute_features(df, months_default=months_default, halflife=halflife)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _split_xy(df: pd.DataFrame, target: str = "DEFAULT") -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features.")
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[target])
    return X, y


# ---------------- Models ----------------
def train_logreg(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=10_000, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe

def train_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    if xgb is None:
        return None
    pos, neg = int(y.sum()), int((y == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    model = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=seed, tree_method="hist", objective="binary:logistic",
        eval_metric="auc", scale_pos_weight=spw, n_jobs=0,
    )
    model.fit(X, y)
    return model

def train_lgbm(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    if lgb is None:
        return None
    pos, neg = int(y.sum()), int((y == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    model = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, objective="binary", scale_pos_weight=spw, metric="auc",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    proba_clipped = np.clip(proba, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba_clipped)),
    }


# ---------------- Main ----------------
def main(
    input_path: Path,
    outdir: Path = Path("models"),
    seed: int = 42,
    test_size: float = 0.2,
    months: int = 6,
    halflife: int = 3,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw (may be minimal CSV or already-featureized)
    raw = _load_any(input_path)

    # 2) Capture target BEFORE featureizing, then drop from inputs
    y_series = None
    if "DEFAULT" in raw.columns:
        y_series = pd.to_numeric(raw["DEFAULT"], errors="coerce").fillna(0).astype(int)
        raw = raw.drop(columns=["DEFAULT"])

    # 3) Ensure features and then reattach target (if we had it)
    feats = _ensure_features(raw, months_default=months, halflife=halflife)
    if y_series is not None:
        feats = feats.copy()
        feats["DEFAULT"] = y_series.values
    elif "DEFAULT" not in feats.columns:
        # If neither raw nor feats contain DEFAULT, we can't train
        raise ValueError("Target 'DEFAULT' not found in input. Make sure your CSV includes it.")

    # 4) Split
    X, y = _split_xy(feats, target="DEFAULT")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # Persist schema for app/inference
    with open(outdir / "feature_columns.txt", "w") as f:
        for c in X.columns:
            f.write(c + "\n")

    # 5) Train baseline LR
    print("[train] Logistic Regression…")
    lr = train_logreg(X_tr, y_tr)
    m_lr_tr = evaluate(lr, X_tr, y_tr)
    m_lr_te = evaluate(lr, X_te, y_te)

    best_name, best_model, best_te = "logreg", lr, m_lr_te

    # 6) Train XGBoost
    xgbm, m_xgb_tr, m_xgb_te = None, None, None
    if xgb is not None:
        print("[train] XGBoost…")
        xgbm = train_xgb(X_tr, y_tr, seed=seed)
        if xgbm is not None:
            m_xgb_tr = evaluate(xgbm, X_tr, y_tr)
            m_xgb_te = evaluate(xgbm, X_te, y_te)

    # 7) Train LightGBM (optional)
    lgbm, m_lgb_tr, m_lgb_te = None, None, None
    if lgb is not None:
        print("[train] LightGBM…")
        lgbm = train_lgbm(X_tr, y_tr, seed=seed)
        if lgbm is not None:
            m_lgb_tr = evaluate(lgbm, X_tr, y_tr)
            m_lgb_te = evaluate(lgbm, X_te, y_te)

    # 8) Pick best by ROC-AUC
    candidates = [("logreg", lr, m_lr_te)]
    if m_xgb_te is not None: candidates.append(("xgboost", xgbm, m_xgb_te))
    if m_lgb_te is not None: candidates.append(("lightgbm", lgbm, m_lgb_te))
    best_name, best_model, best_te = sorted(candidates, key=lambda t: t[2]["roc_auc"], reverse=True)[0]

    # 9) Save models + metrics
    joblib.dump(lr, outdir / "baseline.joblib")
    joblib.dump(best_model, outdir / "best.joblib")

    metrics_payload = {
        "data": {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "n_features": int(X.shape[1]), "features": list(X.columns),
        },
        "baseline_lr": {"train": m_lr_tr, "test": m_lr_te},
        "xgboost": {"train": m_xgb_tr, "test": m_xgb_te} if m_xgb_te else None,
        "lightgbm": {"train": m_lgb_tr, "test": m_lgb_te} if m_lgb_te else None,
        "best": {"name": best_name, "test": best_te},
        "params": {"seed": seed, "test_size": test_size, "months": months, "halflife": halflife},
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[done] Baseline → {outdir / 'baseline.joblib'}")
    print(f"[done] Best ({best_name}) → {outdir / 'best.joblib'}")
    print(f"[done] Metrics → {outdir / 'metrics.json'}")
    print(f"[info] Features ({X.shape[1]}): {list(X.columns)[:8]}{'...' if X.shape[1] > 8 else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Clean CSV (e.g., data/final_dataset.csv) or features parquet/csv")
    ap.add_argument("--outdir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--months", type=int, default=6, help="History length when building features from raw CSV")
    ap.add_argument("--halflife", type=int, default=3, help="Half-life (months) for PAY EMA when building features")
    args = ap.parse_args()
    main(args.input, args.outdir, args.seed, args.test_size, args.months, args.halflife)
