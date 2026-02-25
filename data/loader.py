"""Singleton DataFrame loader for synthetic_data.csv."""

from pathlib import Path

import pandas as pd

_df: pd.DataFrame | None = None

DATA_PATH = Path(__file__).parent.parent / "synthetic_data.csv"


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df
