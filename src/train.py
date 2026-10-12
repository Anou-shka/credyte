# src/train.py
"""
Train the baseline (scaled logistic regression) and gradient-boosted models.

Split layout (all stratified, fixed seed):
  data ─┬─ train (1 - test_size)   → model fitting (XGBoost uses 10% of it for early stopping)
        └─ holdout (test_size)     → model selection here; later split by src/calibrate.py
                                     into calibration / evaluation halves
The holdout row positions are written to models/splits.json so calibration
never touches rows the model was trained on.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

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

from src.features import featurize_table, read_table, write_feature_schema


# ---------------- Models ----------------
def train_logreg(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=10_000, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe

def _pos_weight(y: pd.Series) -> float:
    pos, neg = int(y.sum()), int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0

def train_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    if xgb is None:
        return None
    X_fit, X_es, y_fit, y_es = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=seed, tree_method="hist", objective="binary:logistic",
        eval_metric="auc", early_stopping_rounds=100,
        scale_pos_weight=_pos_weight(y_fit), n_jobs=0,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model

def train_lgbm(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    if lgb is None:
        return None
    X_fit, X_es, y_fit, y_es = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, objective="binary", scale_pos_weight=_pos_weight(y_fit),
        n_jobs=-1, verbose=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
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

    # 1) Load + featurize
    raw = read_table(input_path)
    X, y = featurize_table(raw, months=months, halflife=halflife)
    if y is None:
        raise ValueError("Target 'DEFAULT' not found in input. Make sure your CSV includes it.")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 2) Split (positions are relative to the input file's row order)
    positions = np.arange(len(X))
    tr_pos, ho_pos = train_test_split(positions, test_size=test_size, random_state=seed, stratify=y)
    X_tr, y_tr = X.iloc[tr_pos], y.iloc[tr_pos]
    X_te, y_te = X.iloc[ho_pos], y.iloc[ho_pos]

    write_feature_schema(X.columns, outdir / "feature_columns.txt")
    with open(outdir / "splits.json", "w") as f:
        json.dump({
            "input": str(input_path), "n_rows": int(len(X)), "seed": seed, "test_size": test_size,
            "holdout_positions": sorted(int(i) for i in ho_pos),
        }, f)

    # 3) Train candidates
    print("[train] Logistic Regression…")
    lr = train_logreg(X_tr, y_tr)
    candidates = {"logreg": lr}
    if xgb is not None:
        print("[train] XGBoost…")
        candidates["xgboost"] = train_xgb(X_tr, y_tr, seed=seed)
    if lgb is not None:
        print("[train] LightGBM…")
        candidates["lightgbm"] = train_lgbm(X_tr, y_tr, seed=seed)

    results = {name: {"train": evaluate(m, X_tr, y_tr), "test": evaluate(m, X_te, y_te)}
               for name, m in candidates.items()}

    # 4) Pick best by holdout ROC-AUC
    best_name = max(results, key=lambda n: results[n]["test"]["roc_auc"])
    best_model = candidates[best_name]

    # 5) Save models + metrics
    joblib.dump(lr, outdir / "baseline.joblib")
    joblib.dump(best_model, outdir / "best.joblib")

    extra = {}
    if best_name == "xgboost":
        extra["best_iteration"] = int(best_model.best_iteration)
    metrics_payload = {
        "data": {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "default_rate": float(y.mean()),
            "n_features": int(X.shape[1]), "features": list(X.columns),
        },
        "baseline_lr": results["logreg"],
        "xgboost": results.get("xgboost"),
        "lightgbm": results.get("lightgbm"),
        "best": {"name": best_name, "test": results[best_name]["test"], **extra},
        "params": {"seed": seed, "test_size": test_size, "months": months, "halflife": halflife},
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    for name, r in results.items():
        print(f"[metrics] {name:9s} train AUC={r['train']['roc_auc']:.4f}  test AUC={r['test']['roc_auc']:.4f}")
    print(f"\n[done] Baseline → {outdir / 'baseline.joblib'}")
    print(f"[done] Best ({best_name}) → {outdir / 'best.joblib'}")
    print(f"[done] Metrics → {outdir / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/final_dataset.csv"),
                    help="Clean CSV (e.g., data/final_dataset.csv) or features parquet/csv")
    ap.add_argument("--outdir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--months", type=int, default=6, help="History length when building features from raw CSV")
    ap.add_argument("--halflife", type=int, default=3, help="Half-life (months) for PAY EMA when building features")
    args = ap.parse_args()
    main(args.input, args.outdir, args.seed, args.test_size, args.months, args.halflife)
