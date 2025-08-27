# src/features.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Public constants (import in app.py if you want)
# ---------------------------------------------------------------------
MINIMAL_INPUT_KEYS: List[str] = [
    # demographics / limits
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    # latest status & amounts
    "PAY_STATUS", "PAY_AMT_LATEST", "BILL_AMT_LATEST",
    # prior running averages you maintain
    "AVG_PAY_AMT_PRIOR", "AVG_BILL_AMT_PRIOR",
    # prior average of status (if you computed it from PAY_2..PAY_6)
    "PRIOR_PAY_AVG",
    # optional: MONTHS (total months AFTER adding latest); if not provided we use a default
    # "MONTHS",
]

# The canonical order of engineered features for modeling
FEATURE_COLUMNS_DEFAULT: List[str] = [
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    "avg_pay_amt", "avg_bill_amt",
    "repay_ratio_avg", "cur_recent", "paydown_ratio", "mpr",
    "pay_status_current", "pay_status_avg", "pay_status_ema",
    "pay_is_delinquent", "pay_severity",
    "MONTHS",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _to_num(s: pd.Series | float | int, default: float = 0.0) -> pd.Series:
    """Coerce to numeric; NaN -> default."""
    if isinstance(s, (int, float, np.number)):
        return pd.Series([s], dtype=float)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def alpha_from_halflife(halflife: int = 3) -> float:
    """Half-life (months) → EMA alpha."""
    h = max(1, int(halflife))
    return 1.0 - 2.0 ** (-1.0 / h)


def update_avg_vec(prior: pd.Series | float,
                   latest: pd.Series | float,
                   months_total: pd.Series | int) -> pd.Series:
    """
    Vectorized running average update:
      new_avg = (prior_avg * (months_total - 1) + latest) / months_total
    Assumes months_total is the count AFTER adding the latest month.
    """
    prior = _to_num(prior, 0.0)
    latest = _to_num(latest, 0.0)

    if isinstance(months_total, (int, float, np.number)):
        months = pd.Series([months_total] * max(len(prior), len(latest)))
    else:
        months = pd.to_numeric(months_total, errors="coerce").fillna(1).astype(int)
    months = months.clip(lower=1)

    return (prior * (months - 1) + latest) / months


def _clip01(s: pd.Series | float) -> pd.Series:
    if isinstance(s, (int, float, np.number)):
        return pd.Series([float(np.clip(s, 0.0, 1.0))])
    return s.clip(0.0, 1.0)


def _compute_ratios(limit_bal: pd.Series,
                    avg_pay_amt: pd.Series,
                    avg_bill_amt: pd.Series,
                    latest_pay_amt: pd.Series,
                    latest_bill_amt: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute the four ratios you want to show on the UI."""
    # avoid divide-by-zero
    denom_avg_bill = avg_bill_amt.replace(0, np.nan)
    denom_limit = limit_bal.replace(0, np.nan)
    denom_latest_bill = latest_bill_amt.replace(0, np.nan)

    repay_ratio_avg = _clip01(avg_pay_amt / denom_avg_bill).fillna(0.0)
    cur_recent = _clip01(latest_pay_amt / denom_latest_bill).fillna(0.0)
    paydown_ratio = _clip01(avg_pay_amt / denom_limit).fillna(0.0)
    mpr = paydown_ratio.copy()  # alias

    return repay_ratio_avg, cur_recent, paydown_ratio, mpr


# ---------------------------------------------------------------------
# Core: build features for a whole DataFrame
# ---------------------------------------------------------------------
def compute_features(
    df: pd.DataFrame,
    months_default: int = 6,
    halflife: int = 3,
    feature_order: Iterable[str] = FEATURE_COLUMNS_DEFAULT,
    write_schema_to: str | Path | None = "models/feature_columns.txt",
) -> pd.DataFrame:
    """
    Given a DataFrame with columns like:
      ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_STATUS',
       'BILL_AMT_LATEST','PAY_AMT_LATEST','AVG_PAY_AMT_PRIOR',
       'AVG_BILL_AMT_PRIOR','PRIOR_PAY_AVG', ...]
    compute the engineered features that the app + model will use.

    Returns a clean DataFrame with columns in `feature_order`
    (missing ones are dropped if truly unavailable).
    """
    df = df.copy()

    # 1) Resolve MONTHS (if column missing, use a constant)
    if "MONTHS" in df.columns:
        months_total = pd.to_numeric(df["MONTHS"], errors="coerce").fillna(months_default).astype(int)
    else:
        months_total = pd.Series(months_default, index=df.index, dtype=int)

    # 2) Updated running averages for amounts (include latest)
    df["avg_pay_amt"] = update_avg_vec(df.get("AVG_PAY_AMT_PRIOR", 0.0),
                                       df.get("PAY_AMT_LATEST", 0.0),
                                       months_total)

    df["avg_bill_amt"] = update_avg_vec(df.get("AVG_BILL_AMT_PRIOR", 0.0),
                                        df.get("BILL_AMT_LATEST", 0.0),
                                        months_total)

    # 3) Payment status summaries
    df["pay_status_current"] = _to_num(df.get("PAY_STATUS", 0.0), 0.0)

    # Running average of PAY including latest.
    # If PRIOR_PAY_AVG missing, we treat it as 0.0 for the "prior months".
    df["pay_status_avg"] = update_avg_vec(df.get("PRIOR_PAY_AVG", 0.0),
                                          df["pay_status_current"],
                                          months_total)

    # EMA of PAY (using prior average as baseline if you don't have a prior EMA)
    a = alpha_from_halflife(halflife)
    prior_for_ema = _to_num(df.get("PRIOR_PAY_AVG", np.nan)).fillna(df["pay_status_current"])
    df["pay_status_ema"] = a * df["pay_status_current"] + (1 - a) * prior_for_ema

    df["pay_is_delinquent"] = (df["pay_status_current"] > 0).astype(int)
    df["pay_severity"] = np.maximum(0.0, df["pay_status_current"].astype(float)) / 8.0

    # 4) Ratios for UI/model
    limit_bal = _to_num(df.get("LIMIT_BAL", 0.0), 0.0)
    repay_ratio_avg, cur_recent, paydown_ratio, mpr = _compute_ratios(
        limit_bal=limit_bal,
        avg_pay_amt=df["avg_pay_amt"],
        avg_bill_amt=df["avg_bill_amt"],
        latest_pay_amt=_to_num(df.get("PAY_AMT_LATEST", 0.0), 0.0),
        latest_bill_amt=_to_num(df.get("BILL_AMT_LATEST", 0.0), 0.0),
    )
    df["repay_ratio_avg"] = repay_ratio_avg
    df["cur_recent"] = cur_recent
    df["paydown_ratio"] = paydown_ratio
    df["mpr"] = mpr

    # Provide MONTHS explicitly (in case it wasn’t present)
    df["MONTHS"] = months_total

    # 5) Pick final feature order (keep only columns that exist)
    feature_order = list(feature_order)
    exist = [c for c in feature_order if c in df.columns]
    out = df.loc[:, exist].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Optionally write schema
    if write_schema_to:
        Path(write_schema_to).parent.mkdir(parents=True, exist_ok=True)
        with open(write_schema_to, "w") as f:
            for c in exist:
                f.write(c + "\n")

    return out


# ---------------------------------------------------------------------
# Single-row builder (for Streamlit form)
# ---------------------------------------------------------------------
def build_single_from_minimal(
    row_dict: Dict[str, float | int | str],
    months: int = 6,
    halflife: int = 3,
    feature_order: Iterable[str] = FEATURE_COLUMNS_DEFAULT,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Turn a minimal input dict into a full feature vector + UI metrics.

    Expected keys in `row_dict` (strings are fine; we coerce to float):
      LIMIT_BAL, AGE, EDUCATION, MARRIAGE,
      PAY_STATUS, PAY_AMT_LATEST, BILL_AMT_LATEST,
      AVG_PAY_AMT_PRIOR, AVG_BILL_AMT_PRIOR, PRIOR_PAY_AVG
      (MONTHS is optional; if provided it overrides `months`)

    Returns:
      - feature_row: pd.Series with the engineered features (ordered)
      - ui: dict with computed ratios/averages you can show in the app
    """
    # Prepare single-row Series
    s = pd.Series(row_dict, dtype="object")

    # Resolve months
    months_total = int(s.get("MONTHS", months)) if pd.notna(s.get("MONTHS", np.nan)) else months

    # Amount averages (updated)
    avg_pay_amt = update_avg_vec(pd.Series([s.get("AVG_PAY_AMT_PRIOR", 0.0)]),
                                 pd.Series([s.get("PAY_AMT_LATEST", 0.0)]),
                                 pd.Series([months_total]))[0]

    avg_bill_amt = update_avg_vec(pd.Series([s.get("AVG_BILL_AMT_PRIOR", 0.0)]),
                                  pd.Series([s.get("BILL_AMT_LATEST", 0.0)]),
                                  pd.Series([months_total]))[0]

    # Payment status summaries
    pay_status_current = float(pd.to_numeric(pd.Series([s.get("PAY_STATUS", 0.0)]), errors="coerce").fillna(0.0)[0])
    prior_pay_avg = float(pd.to_numeric(pd.Series([s.get("PRIOR_PAY_AVG", 0.0)]), errors="coerce").fillna(0.0)[0])

    pay_status_avg = update_avg_vec(pd.Series([prior_pay_avg]),
                                    pd.Series([pay_status_current]),
                                    pd.Series([months_total]))[0]

    a = alpha_from_halflife(halflife)
    prior_for_ema = prior_pay_avg  # use prior average as EMA baseline if no prior EMA
    pay_status_ema = a * pay_status_current + (1 - a) * prior_for_ema

    pay_is_delinquent = int(pay_status_current > 0)
    pay_severity = max(0.0, pay_status_current) / 8.0

    # Ratios
    limit_bal = float(pd.to_numeric(pd.Series([s.get("LIMIT_BAL", 0.0)]), errors="coerce").fillna(0.0)[0])
    latest_pay = float(pd.to_numeric(pd.Series([s.get("PAY_AMT_LATEST", 0.0)]), errors="coerce").fillna(0.0)[0])
    latest_bill = float(pd.to_numeric(pd.Series([s.get("BILL_AMT_LATEST", 0.0)]), errors="coerce").fillna(0.0)[0])

    repay_ratio_avg, cur_recent, paydown_ratio, mpr = _compute_ratios(
        limit_bal=pd.Series([limit_bal]),
        avg_pay_amt=pd.Series([avg_pay_amt]),
        avg_bill_amt=pd.Series([avg_bill_amt]),
        latest_pay_amt=pd.Series([latest_pay]),
        latest_bill_amt=pd.Series([latest_bill]),
    )
    repay_ratio_avg = float(repay_ratio_avg.iloc[0])
    cur_recent = float(cur_recent.iloc[0])
    paydown_ratio = float(paydown_ratio.iloc[0])
    mpr = float(mpr.iloc[0])

    # Build feature row (include EDUCATION/MARRIAGE/AGE if present)
    feat = {
        "LIMIT_BAL": float(limit_bal),
        "AGE": float(pd.to_numeric(pd.Series([s.get("AGE", 0.0)]), errors="coerce").fillna(0.0)[0]),
        "EDUCATION": float(pd.to_numeric(pd.Series([s.get("EDUCATION", 0.0)]), errors="coerce").fillna(0.0)[0]),
        "MARRIAGE": float(pd.to_numeric(pd.Series([s.get("MARRIAGE", 0.0)]), errors="coerce").fillna(0.0)[0]),

        "avg_pay_amt": float(avg_pay_amt),
        "avg_bill_amt": float(avg_bill_amt),

        "repay_ratio_avg": repay_ratio_avg,
        "cur_recent": cur_recent,
        "paydown_ratio": paydown_ratio,
        "mpr": mpr,

        "pay_status_current": pay_status_current,
        "pay_status_avg": float(pay_status_avg),
        "pay_status_ema": float(pay_status_ema),
        "pay_is_delinquent": pay_is_delinquent,
        "pay_severity": float(pay_severity),

        "MONTHS": float(months_total),
    }

    # Order and drop any missing columns not relevant for your model
    order = [c for c in feature_order if c in feat]
    feature_row = pd.Series({c: feat[c] for c in order}, index=order, dtype=float)

    # For UI display
    ui = {
        "avg_pay_amt": float(avg_pay_amt),
        "avg_bill_amt": float(avg_bill_amt),
        "repay_ratio_avg": repay_ratio_avg,
        "cur_recent": cur_recent,
        "paydown_ratio": paydown_ratio,
        "mpr": mpr,
        "pay_status_current": pay_status_current,
        "pay_status_avg": float(pay_status_avg),
        "pay_status_ema": float(pay_status_ema),
    }
    return feature_row, ui


# ---------------------------------------------------------------------
# Helper: split X/y and save schema
# ---------------------------------------------------------------------
def split_xy(features_df: pd.DataFrame, target_col: str = "DEFAULT") -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) if target exists; else return (features_df, None)."""
    if target_col in features_df.columns:
        y = features_df[target_col].astype(int)
        X = features_df.drop(columns=[target_col])
        return X, y
    return features_df, None


def write_feature_schema(feature_cols: Iterable[str], path: str | Path = "models/feature_columns.txt") -> None:
    """Persist the feature order for inference/app."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")


# ---------------------------------------------------------------------
# CLI: transform a CSV → engineered features parquet
# ---------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Clean CSV with minimal columns")
    ap.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    ap.add_argument("--months", type=int, default=6, help="History length used to update averages")
    ap.add_argument("--halflife", type=int, default=3, help="Half-life (months) for PAY EMA")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    feats = compute_features(df, months_default=args.months, halflife=args.halflife)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"Saved features to {args.out} with shape {feats.shape}")


if __name__ == "__main__":
    _cli()
