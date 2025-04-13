# label_generator.py
import pandas as pd
import numpy as np

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    return df
