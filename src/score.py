# src/score.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, pandas as pd, joblib
from src.features import featurize_table, read_table, align_to_schema, read_feature_schema, FEATURE_COLUMNS_DEFAULT

# Band edges when no operating point is available, and the half-width of the
# Medium band around the operating threshold when one is. Shared with app.py.
FALLBACK_BANDS = (0.25, 0.50)
BAND_HALF_WIDTH = 0.10


def default_bands(models_dir: Path = Path("models")) -> Tuple[Tuple[float, float], Optional[dict]]:
    """(low, high) band edges centred on the calibrated operating threshold, plus the operating point."""
    op_path = Path(models_dir) / "operating_point.json"
    if not op_path.exists():
        return FALLBACK_BANDS, None
    op = json.loads(op_path.read_text())
    t = float(op["threshold"])
    return (max(0.0, t - BAND_HALF_WIDTH), min(1.0, t + BAND_HALF_WIDTH)), op


def risk_band(p, low: float, high: float):
    return np.where(np.asarray(p) < low, "Low", np.where(np.asarray(p) < high, "Medium", "High"))


def load_scoring_and_schema(models_dir: Path):
    models_dir = Path(models_dir)
    for name, tag in (("calibrated_best.joblib", "calibrated"), ("best.joblib", "best"), ("baseline.joblib", "baseline")):
        path = models_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No model found in {models_dir}. Run `python -m src.train` first.")
    model = joblib.load(path)
    schema = read_feature_schema(models_dir / "feature_columns.txt") or FEATURE_COLUMNS_DEFAULT
    return model, tag, schema


def main(input_path: Path, features_path: Path | None, out_csv: Path,
         models_dir: Path, months: int, halflife: int):
    df = read_table(features_path or input_path)
    X, _ = featurize_table(df, months=months, halflife=halflife)

    model, tag, schema = load_scoring_and_schema(models_dir)
    X = align_to_schema(X, schema)

    out = df.copy()
    out["pd"] = model.predict_proba(X)[:, 1]

    (low, high), _ = default_bands(models_dir)
    out["risk_band"] = risk_band(out["pd"], low, high)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} with {len(out)} rows using model={tag}; bands: Low<{low:.2f}≤Medium<{high:.2f}≤High")


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
