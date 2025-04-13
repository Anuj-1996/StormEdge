# garch_volatility.py

import pandas as pd
from arch import arch_model

def add_garch_volatility(df: pd.DataFrame, return_col: str = 'LogReturn') -> pd.DataFrame:
    """
    Fit a GARCH(1,1) model on return_col and add predicted volatility as 'GARCH_Vol'.
    """
    df = df.copy()
    returns = df[return_col].dropna() * 100  # convert to percentage returns for GARCH

    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')

    forecasts = res.forecast(horizon=1, reindex=False)
    df['GARCH_Vol'] = forecasts.variance.iloc[:, 0]**0.5 / 100  # convert back to decimal
    df['GARCH_Vol'].fillna(method='bfill', inplace=True)

    return df
