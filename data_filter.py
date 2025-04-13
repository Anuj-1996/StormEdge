# data_filter.py

import pandas as pd

def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter the dataframe between start_date and end_date."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df[(df.index >= start_date) & (df.index <= end_date)]
