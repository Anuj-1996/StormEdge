# position_sizing.py

import pandas as pd

def calculate_position_size(df: pd.DataFrame, capital: float, risk_pct: float = 0.02) -> pd.Series:
    """
    Risk-based position sizing using ATR-based stop-loss concept.
    """
    df = df.copy()
    df['ATR'] = df['ATR_norm'] * df['Close']  # convert normalized ATR to actual price units
    df['Risk_per_trade'] = df['ATR'] * df['Signal'].abs()  # only consider active trades
    df['PositionSize'] = (capital * risk_pct) / df['Risk_per_trade']
    df['PositionSize'].fillna(0, inplace=True)
    df['PositionSize'] = df['PositionSize'].clip(upper=1.0)  # No leverage for now
    return df['PositionSize']
