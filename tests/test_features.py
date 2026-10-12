import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS_DEFAULT, build_single_from_minimal, compute_features

ROW = {
    "LIMIT_BAL": 100_000, "AGE": 30, "EDUCATION": 2, "MARRIAGE": 1,
    "PAY_STATUS": 2, "PAY_AMT_LATEST": 1_000, "BILL_AMT_LATEST": 20_000,
    "AVG_PAY_AMT_PRIOR": 2_000, "AVG_BILL_AMT_PRIOR": 10_000, "PRIOR_PAY_AVG": 0.0,
}


def test_single_row_matches_batch():
    df = pd.DataFrame([ROW, {**ROW, "PAY_STATUS": -1, "BILL_AMT_LATEST": -50}], index=[10, 20])
    batch = compute_features(df)
    for i, (_, r) in enumerate(df.iterrows()):
        single, _ = build_single_from_minimal(r.to_dict())
        pd.testing.assert_series_equal(single, batch.iloc[i].astype(float), check_names=False)


def test_formulas():
    f = compute_features(pd.DataFrame([ROW])).iloc[0]
    assert list(f.index) == FEATURE_COLUMNS_DEFAULT
    assert f.avg_pay_amt == pytest.approx((2_000 * 5 + 1_000) / 6)
    assert f.avg_bill_amt == pytest.approx((10_000 * 5 + 20_000) / 6)
    assert f.cur_recent == pytest.approx(1_000 / 20_000)
    assert f.payment_trend == pytest.approx((1_000 - 2_000) / 2_001)
    assert f.pay_is_delinquent == 1
    assert f.pay_severity == pytest.approx(2 / 8)


def test_nothing_owed_counts_as_fully_covered():
    f = compute_features(pd.DataFrame([{**ROW, "BILL_AMT_LATEST": 0, "AVG_BILL_AMT_PRIOR": -100}])).iloc[0]
    assert f.cur_recent == 1.0
    assert f.repay_ratio_avg == 1.0
    assert f.credit_utilization == 0.0


def test_missing_columns_and_non_default_index():
    df = pd.DataFrame({"LIMIT_BAL": [50_000, 80_000], "PAY_STATUS": [1, 0]}, index=[7, 3])
    out = compute_features(df)
    assert out.notna().all().all()
    assert list(out.index) == [7, 3]
    # missing PRIOR_PAY_AVG falls back to the current status
    assert out.loc[7, "pay_status_ema"] == pytest.approx(1.0)
    assert np.isfinite(out.to_numpy()).all()
