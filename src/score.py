# src/score.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from src.features import compute_features

def _load_any(p: Path) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() in (".parquet",".pq"): return pd.read_parquet(p)
    if p.suffix.lower() in (".csv",".txt"):    return pd.read_csv(p)
    raise ValueError(f"Unsupported: {p.suffix}")

def load_scoring_and_schema(models_dir: Path):
    models_dir = Path(models_dir)
    # model
    path = models_dir/"calibrated_best.joblib"
    tag = "calibrated"
    if not path.exists():
        path = models_dir/"best.joblib"; tag = "best" if path.exists() else tag
    if not path.exists():
        path = models_dir/"baseline.joblib"; tag = "baseline"
    model = joblib.load(path)
    # schema
    schema = None
    sc = models_dir/"feature_columns.txt"
    if sc.exists():
        schema = [ln.strip() for ln in sc.read_text().splitlines() if ln.strip()]
    return model, tag, schema

def main(input_path: Path, features_path: Path|None, out_csv: Path,
         models_dir: Path, months: int, halflife: int):
    # load data
    df = _load_any(features_path or input_path)
    y = df["DEFAULT"] if "DEFAULT" in df.columns else None
    X = df.drop(columns=["DEFAULT"], errors="ignore")
    # ensure features
    need = {"avg_pay_amt","avg_bill_amt","repay_ratio_avg","pay_status_current"}
    if not need.issubset(X.columns):
        X = compute_features(X, months_default=months, halflife=halflife, write_schema_to=None)
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)

    model, tag, schema = load_scoring_and_schema(models_dir)
    if schema:
        X = X.reindex(columns=[c for c in schema if c in X.columns]).fillna(0.0)

    p = model.predict_proba(X)[:,1]
    out = df.copy()
    out["pd"] = p

    # risk bands from operating point
    op_path = Path(models_dir)/"operating_point.json"
    if op_path.exists():
        op = json.loads(op_path.read_text())
        t = float(op["threshold"])
        low, high = max(0.0,t-0.10), min(1.0,t+0.10)
    else:
        low, high = 0.30, 0.60
    out["risk_band"] = np.where(out["pd"]<low,"Low", np.where(out["pd"]<high,"Medium","High"))
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} with {len(out)} rows using model={tag}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/final_dataset.csv"))
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/scored.csv"))
    ap.add_argument("--models_dir", type=Path, default=Path("models"))
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--halflife", type=int, default=3)
    args = ap.parse_args()
    main(args.input, args.features, args.out, args.models_dir, args.months, args.halflife)
