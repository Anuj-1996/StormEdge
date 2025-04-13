import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from sklearn.preprocessing import OneHotEncoder
import ta 
def add_technical_indicators(df):
    df = df.copy()

    # Your existing features
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_5'] = df['Close'].rolling(window=5).std()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Return'].rolling(window=10).std()

    # === New features required for HMM model ===

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # MACD Histogram
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD_Hist'] = macd.macd_diff()

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['BB_Width'] = bb.bollinger_wband()

    # Normalized ATR
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    df['ATR_norm'] = df['ATR'] / df['Close']

    # Drop NaNs created by indicators
    df.dropna(inplace=True)

    return df

def add_lag_features(df, lags=[1, 2, 3]):
    df = df.copy()
    for lag in lags:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
    return df

def add_garch_volatility(df):
    df = df.copy()
    returns = df['LogReturn'].dropna() * 100
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp="off")
    df.loc[returns.index, 'GARCH_vol'] = res.conditional_volatility
    return df

def add_hmm_regime_features(df, regimes, regime_proba):
    df = df.copy()
    df['Regime'] = regimes
    proba_df = pd.DataFrame(regime_proba, columns=[f'Regime_Prob_{i}' for i in range(regime_proba.shape[1])])
    df = pd.concat([df.reset_index(drop=True), proba_df], axis=1)
    return df

def build_features(df, regimes, regime_proba):
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_garch_volatility(df)
    df = add_hmm_regime_features(df, regimes, regime_proba)
    df.dropna(inplace=True)
    return df
