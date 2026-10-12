# src/features.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------
MINIMAL_INPUT_KEYS: List[str] = [
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    "PAY_STATUS", "PAY_AMT_LATEST", "BILL_AMT_LATEST",
    "AVG_PAY_AMT_PRIOR", "AVG_BILL_AMT_PRIOR",
    "PRIOR_PAY_AVG",
]

# Feature-engineering parameters the shipped models were trained with.
# Changing them at inference time changes what the features mean, so the app
# reads the values recorded in models/metrics.json instead of exposing knobs.
MONTHS_DEFAULT = 6
HALFLIFE_DEFAULT = 3

FEATURE_COLUMNS_DEFAULT: List[str] = [
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    "avg_pay_amt", "avg_bill_amt",
    "credit_utilization",   # avg_bill / limit_bal — how much credit is being used
    "repay_ratio_avg",      # avg_pay / avg_bill — overall payment coverage
    "cur_recent",           # latest_pay / latest_bill — most recent payment coverage
    "paydown_ratio",        # avg_pay / limit_bal — payments as fraction of limit
    "payment_trend",        # (latest_pay - prior_avg_pay) / (|prior_avg_pay| + 1) — improving?
    "pay_status_current",
    "pay_status_avg",
    "pay_status_ema",
    "pay_is_delinquent",
    "pay_severity",
]

# Human-readable names for charts and the app.
FEATURE_LABELS: Dict[str, str] = {
    "LIMIT_BAL":          "Credit Limit",
    "AGE":                "Age",
    "EDUCATION":          "Education",
    "MARRIAGE":           "Marital Status",
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
    "pay_is_delinquent":  "Delinquent",
    "pay_severity":       "Delinquency Severity",
}

# Highest PAY code in the source data ("8 months or more late").
MAX_PAY_CODE = 8.0


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Numeric column aligned to df.index; missing column or NaN -> default."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def alpha_from_halflife(halflife: int = HALFLIFE_DEFAULT) -> float:
    h = max(1, int(halflife))
    return 1.0 - 2.0 ** (-1.0 / h)


def update_avg_vec(prior: pd.Series, latest: pd.Series, months_total: pd.Series) -> pd.Series:
    """Running average update: new_avg = (prior*(n-1) + latest) / n, with n >= 1."""
    n = months_total.clip(lower=1)
    return (prior * (n - 1) + latest) / n


def _coverage(paid: pd.Series, owed: pd.Series) -> pd.Series:
    """paid / owed in [0, 1]. Nothing owed (zero or credit balance) counts as fully covered."""
    ratio = paid / owed.where(owed > 0)
    return ratio.clip(0.0, 1.0).fillna(1.0)


def _share_of_limit(amount: pd.Series, limit_bal: pd.Series) -> pd.Series:
    return (amount / limit_bal.where(limit_bal > 0)).clip(0.0, 1.0).fillna(0.0)


# ---------------------------------------------------------------------
# Core: build features for a whole DataFrame
# ---------------------------------------------------------------------
def compute_features(
    df: pd.DataFrame,
    months_default: int = MONTHS_DEFAULT,
    halflife: int = HALFLIFE_DEFAULT,
    feature_order: Iterable[str] = FEATURE_COLUMNS_DEFAULT,
    write_schema_to: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build model features from the minimal inputs (see MINIMAL_INPUT_KEYS).
    The *_PRIOR / PRIOR_* inputs are averages over the months *before* the
    latest one; MONTHS (optional column) is the history length including the
    latest month.
    """
    months = _col(df, "MONTHS", months_default)
    months = months.where(months >= 1, months_default)

    limit_bal = _col(df, "LIMIT_BAL")
    pay_latest = _col(df, "PAY_AMT_LATEST")
    bill_latest = _col(df, "BILL_AMT_LATEST")
    pay_prior = _col(df, "AVG_PAY_AMT_PRIOR")
    bill_prior = _col(df, "AVG_BILL_AMT_PRIOR")
    status = _col(df, "PAY_STATUS")
    status_prior = _col(df, "PRIOR_PAY_AVG", np.nan).fillna(status)

    feats = pd.DataFrame(index=df.index)
    feats["LIMIT_BAL"] = limit_bal
    feats["AGE"] = _col(df, "AGE")
    feats["EDUCATION"] = _col(df, "EDUCATION")
    feats["MARRIAGE"] = _col(df, "MARRIAGE")

    feats["avg_pay_amt"] = update_avg_vec(pay_prior, pay_latest, months)
    feats["avg_bill_amt"] = update_avg_vec(bill_prior, bill_latest, months)

    feats["credit_utilization"] = _share_of_limit(feats["avg_bill_amt"], limit_bal)
    feats["repay_ratio_avg"] = _coverage(feats["avg_pay_amt"], feats["avg_bill_amt"])
    feats["cur_recent"] = _coverage(pay_latest, bill_latest)
    feats["paydown_ratio"] = _share_of_limit(feats["avg_pay_amt"], limit_bal)
    feats["payment_trend"] = ((pay_latest - pay_prior) / (pay_prior.abs() + 1.0)).clip(-1.0, 1.0)

    a = alpha_from_halflife(halflife)
    feats["pay_status_current"] = status
    feats["pay_status_avg"] = update_avg_vec(status_prior, status, months)
    feats["pay_status_ema"] = a * status + (1 - a) * status_prior
    feats["pay_is_delinquent"] = (status > 0).astype(int)
    feats["pay_severity"] = status.clip(lower=0.0) / MAX_PAY_CODE

    order = [c for c in feature_order if c in feats.columns]
    missing = [c for c in feature_order if c not in feats.columns]
    if missing:
        raise ValueError(f"Unknown feature(s) requested: {missing}")
    out = feats.loc[:, order].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if write_schema_to:
        write_feature_schema(order, write_schema_to)
    return out


# ---------------------------------------------------------------------
# Single-row builder (for Streamlit form)
# ---------------------------------------------------------------------
UI_KEYS = [
    "avg_pay_amt", "avg_bill_amt", "credit_utilization", "repay_ratio_avg",
    "cur_recent", "paydown_ratio", "payment_trend",
    "pay_status_current", "pay_status_avg", "pay_status_ema",
]


def build_single_from_minimal(
    row_dict: Dict[str, float | int | str],
    months: int = MONTHS_DEFAULT,
    halflife: int = HALFLIFE_DEFAULT,
    feature_order: Iterable[str] = FEATURE_COLUMNS_DEFAULT,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Same code path as compute_features, so app and training can't drift apart."""
    all_feats = compute_features(
        pd.DataFrame([row_dict]), months_default=months, halflife=halflife,
        feature_order=FEATURE_COLUMNS_DEFAULT,
    ).iloc[0].astype(float)
    order = list(feature_order)
    feature_row = all_feats.reindex(order)
    if feature_row.isna().any():
        raise ValueError(f"Schema asks for unknown features: {list(feature_row[feature_row.isna()].index)}")
    ui = {k: float(all_feats[k]) for k in UI_KEYS}
    return feature_row, ui


# ---------------------------------------------------------------------
# Helper: split X/y and save schema
# ---------------------------------------------------------------------
def split_xy(features_df: pd.DataFrame, target_col: str = "DEFAULT") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if target_col in features_df.columns:
        y = features_df[target_col].astype(int)
        X = features_df.drop(columns=[target_col])
        return X, y
    return features_df, None


def write_feature_schema(feature_cols: Iterable[str], path: str | Path = "models/feature_columns.txt") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")


def read_feature_schema(path: str | Path = "models/feature_columns.txt") -> Optional[List[str]]:
    path = Path(path)
    if not path.exists():
        return None
    cols = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return cols or None


def featurize_table(
    df: pd.DataFrame,
    months: int = MONTHS_DEFAULT,
    halflife: int = HALFLIFE_DEFAULT,
    target_col: str = "DEFAULT",
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Raw minimal table (or already-engineered table) -> (X, y).
    y is None when the target column is absent.
    """
    y = None
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        if y.isna().any():
            raise ValueError(f"{int(y.isna().sum())} rows have a missing/non-numeric '{target_col}'.")
        y = y.astype(int)
    X = df.drop(columns=[target_col], errors="ignore")
    if not set(FEATURE_COLUMNS_DEFAULT).issubset(X.columns):
        X = compute_features(X, months_default=months, halflife=halflife)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def align_to_schema(X: pd.DataFrame, schema: Iterable[str]) -> pd.DataFrame:
    schema = list(schema)
    missing = [c for c in schema if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return X.loc[:, schema]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    ap.add_argument("--months", type=int, default=MONTHS_DEFAULT)
    ap.add_argument("--halflife", type=int, default=HALFLIFE_DEFAULT)
    args = ap.parse_args()

    df = read_table(args.input)
    X, y = featurize_table(df, months=args.months, halflife=args.halflife)
    if y is not None:
        X = X.assign(DEFAULT=y.values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() in (".parquet", ".pq"):
        X.to_parquet(args.out, index=False)
    else:
        X.to_csv(args.out, index=False)
    print(f"Saved features to {args.out} with shape {X.shape}")


if __name__ == "__main__":
    _cli()
