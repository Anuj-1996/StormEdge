import pandas as pd
import numpy as np
def generate_signals(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = predictions
    df['Signal'] = df['Signal'].shift(1)  # Avoid look-ahead bias
    df.dropna(inplace=True)
    return df
