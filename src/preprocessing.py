import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BINARY_COLS, DROP_COLS, MULTI_CAT_COLS, NUMERIC_COLS, SERVICE_COLS


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce").fillna(0.0)

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["Senior_Citizen"] = df["Senior_Citizen"].astype(str).replace({"0": "No", "1": "Yes"})

    bins = [-1, 12, 24, 48, np.inf]
    labels = ["0-12", "13-24", "25-48", "49+"]
    df["tenure_bins"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

    df["service_count"] = df[SERVICE_COLS].apply(lambda row: (row == "Yes").sum(), axis=1)
    df["avg_monthly_charge"] = df["Total_Charges"] / df["tenure"].clip(lower=1)
    df["charge_per_service"] = df["Monthly_Charges"] / df["service_count"].clip(lower=1)

    return df


def _clean_transform(X: pd.DataFrame) -> pd.DataFrame:
    return clean_and_engineer(X)


def build_pipeline() -> Pipeline:
    cleaner = FunctionTransformer(_clean_transform, validate=False)

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="infrequent_if_exist"), BINARY_COLS),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist"), MULTI_CAT_COLS),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("clean", cleaner),
        ("encode", encoder),
    ])


def encode_target(y: pd.Series) -> np.ndarray:
    return (y == "Yes").astype(int).values
