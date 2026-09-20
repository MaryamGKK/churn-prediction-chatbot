import numpy as np
import pandas as pd
import pytest

from src.config import DATASET_PATH
from src.preprocessing import build_pipeline, clean_and_engineer, encode_target


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(DATASET_PATH)


def test_clean_strips_column_names(raw_df):
    cleaned = clean_and_engineer(raw_df)
    for col in cleaned.columns:
        assert col == col.strip()


def test_clean_drops_customer_id(raw_df):
    cleaned = clean_and_engineer(raw_df)
    assert "customerID" not in cleaned.columns


def test_clean_drops_churn(raw_df):
    cleaned = clean_and_engineer(raw_df)
    assert "Churn" not in cleaned.columns


def test_clean_fills_total_charges(raw_df):
    cleaned = clean_and_engineer(raw_df)
    assert cleaned["Total_Charges"].isna().sum() == 0
    assert (cleaned["Total_Charges"] >= 0).all()


def test_tenure_bins_no_nan(raw_df):
    cleaned = clean_and_engineer(raw_df)
    assert cleaned["tenure_bins"].isna().sum() == 0


def test_service_count_range(raw_df):
    cleaned = clean_and_engineer(raw_df)
    assert cleaned["service_count"].min() >= 0
    assert cleaned["service_count"].max() <= 6


def test_pipeline_output_shape(raw_df):
    pipeline = build_pipeline()
    X = pipeline.fit_transform(raw_df)
    assert X.shape[0] == len(raw_df)
    assert X.shape[1] > 0
    assert not np.isnan(X).any()


def test_encode_target(raw_df):
    y = encode_target(raw_df["Churn"])
    assert set(np.unique(y)) == {0, 1}
    assert len(y) == len(raw_df)
