"""Dataset loading and preprocessing (shared by training and prediction)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

# Columns that are category *codes* (not quantities) -> one-hot encoded.
NOMINAL_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]


def load_dataset(path=config.DATASET_PATH):
    """Read the UCI CSV (semicolon separated, BOM, stray tabs in headers)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if df.shape[1] == 1:                      # comma separated variant
        df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(str).str.strip()
    return df


def split_xy(df):
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    return X, y


def prepare_frame(X):
    """Make dtypes consistent between training and prediction."""
    X = X.copy()
    for c in NOMINAL_COLS:
        X[c] = X[c].astype(float).round().astype(int)
    return X


def build_preprocessor(feature_cols):
    numeric = [c for c in feature_cols if c not in NOMINAL_COLS]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS),
            ("num", StandardScaler(), numeric),
        ]
    )