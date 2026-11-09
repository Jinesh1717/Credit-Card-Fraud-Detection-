from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE

EXPECTED = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]

def make_synthetic(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed); fraud = rng.random(n) < .03
    x = rng.normal(size=(n, 28)); x[fraud, :3] += 2.5
    return pd.DataFrame({"Time": rng.uniform(0, 172800, n), **{f"V{i}": x[:, i-1] for i in range(1,29)}, "Amount": rng.exponential(80, n), "Class": fraud.astype(int)})

def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, bool]:
    if path.exists():
        df = pd.read_csv(path); synthetic = False
    else:
        logging.warning("%s not found; using a clearly labeled synthetic dataset. Download creditcard.csv from Kaggle to reproduce real results.", path)
        df = make_synthetic(); synthetic = True
    validate_schema(df); return df, synthetic

def validate_schema(df: pd.DataFrame) -> None:
    missing = set(EXPECTED) - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[EXPECTED].isna().any().any(): raise ValueError("Dataset contains nulls")
    if not set(df.Class.unique()).issubset({0, 1}): raise ValueError("Class must be binary")

def split_data(df: pd.DataFrame, time_ordered: bool = False):
    X, y = df.drop(columns="Class"), df.Class.astype(int)
    if time_ordered:
        order = np.argsort(X.Time); cut = int(len(df)*(1-TEST_SIZE)); idx1, idx2 = order[:cut], order[cut:]
        return X.iloc[idx1], X.iloc[idx2], y.iloc[idx1], y.iloc[idx2]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
