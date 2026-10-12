# src/prepare_data.py
"""
Reproducible cleaning of the raw UCI "default of credit card clients" file
(data/data.csv) into the minimal-input table used by the rest of the pipeline
(data/final_dataset.csv).

Output columns
--------------
ID, LIMIT_BAL, EDUCATION, MARRIAGE, AGE,
PAY_STATUS          latest repayment status (PAY_0)
BILL_AMT_LATEST     latest bill (BILL_AMT1)
PAY_AMT_LATEST      latest payment (PAY_AMT1)
AVG_PAY_AMT_PRIOR   mean of the 5 *prior* payments   (PAY_AMT2..6)
AVG_BILL_AMT_PRIOR  mean of the 5 *prior* bills      (BILL_AMT2..6)
PRIOR_PAY_AVG       mean of the 5 *prior* statuses   (PAY_2..6)
DEFAULT

All three "prior" columns exclude the latest month, which is exactly what the
app asks the user for. src/features.py then folds the latest month in.

Usage:
  python -m src.prepare_data --input data/data.csv --out data/final_dataset.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIOR_MONTHS = [2, 3, 4, 5, 6]

OUTPUT_COLUMNS = [
    "ID", "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_STATUS", "BILL_AMT_LATEST", "PAY_AMT_LATEST", "DEFAULT",
    "AVG_PAY_AMT_PRIOR", "AVG_BILL_AMT_PRIOR", "PRIOR_PAY_AVG",
]


def load_raw(path: Path) -> pd.DataFrame:
    # The UCI export has a dummy "X1, X2, ..." row above the real header.
    df = pd.read_csv(path, header=0)
    if "ID" not in df.columns:
        df = pd.read_csv(path, header=1)
    return df.apply(pd.to_numeric, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "default payment next month": "DEFAULT",
        "PAY_0": "PAY_STATUS",
        "BILL_AMT1": "BILL_AMT_LATEST",
        "PAY_AMT1": "PAY_AMT_LATEST",
    })

    df["AVG_PAY_AMT_PRIOR"] = df[[f"PAY_AMT{i}" for i in PRIOR_MONTHS]].mean(axis=1)
    df["AVG_BILL_AMT_PRIOR"] = df[[f"BILL_AMT{i}" for i in PRIOR_MONTHS]].mean(axis=1)
    df["PRIOR_PAY_AVG"] = df[[f"PAY_{i}" for i in PRIOR_MONTHS]].mean(axis=1)

    # Undocumented category codes are folded into "Other" instead of being
    # dropped, so the app's "Other" options are inside the training data.
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)

    out = df[OUTPUT_COLUMNS].dropna(subset=["DEFAULT"]).copy()
    out["DEFAULT"] = out["DEFAULT"].astype(int)
    return out.reset_index(drop=True)


def main(input_path: Path, out_path: Path) -> None:
    out = clean(load_raw(input_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} with shape {out.shape}; default rate = {out['DEFAULT'].mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/data.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/final_dataset.csv"))
    args = ap.parse_args()
    main(args.input, args.out)
