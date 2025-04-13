# transaction_costs.py

import pandas as pd
from config import CAPITAL

def apply_transaction_costs(df: pd.DataFrame, signal_col: str = 'Signal', capital: float = CAPITAL,
                            commission_pct: float = 0.001, slippage_pct: float = 0.0005) -> pd.DataFrame:
    """
    Adds columns to simulate commission and slippage effects.
    """
    df = df.copy()

    df['PositionSize'] = df.get('PositionSize', 1)
    df['Effective_Position'] = df[signal_col].shift(1) * df['PositionSize'].shift(1)

    # Slippage: penalty on entry/exit
    df['Slippage_Adjustment'] = df['Effective_Position'].diff().fillna(0).apply(lambda x: -slippage_pct if x != 0 else 0)

    # Commission: cost on trade size
    df['Commission_Cost'] = df['Effective_Position'].diff().abs().fillna(0) * commission_pct

    # Adjust Strategy Return
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = (df['Effective_Position'] * df['Daily_Return']) + df['Slippage_Adjustment'] - df['Commission_Cost']
    df['Strategy_Return'].fillna(0, inplace=True)
    df['Portfolio'] = (1 + df['Strategy_Return']).cumprod() * capital

    return df
