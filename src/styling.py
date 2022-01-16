import pandas as pd
from src.constants import params, Constants
from typing import Any

__all__ = ["highlight_col"]


def highlight_col(x: pd.DataFrame, color="yellow") -> str:
    return f"background-color: {color}"
