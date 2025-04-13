# ml_model.py

import numpy as np
import pandas as pd
from stacked_classifier import StackedClassifier
from sklearn.preprocessing import StandardScaler

def generate_ml_signals(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    # Define features and target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 if next day close is higher (UP), else 0
    df.dropna(inplace=True)

    # Define feature columns - adjust based on your indicators
    feature_cols = [col for col in df.columns if col in [
        'SMA', 'EMA', 'RSI', 'MACD', 'Volatility',
        'UpperBand', 'LowerBand', 'ADX', 'CCI', 'ATR'
    ]]

    if not feature_cols:
        raise ValueError("No valid feature columns found in the DataFrame.")

    X = df[feature_cols]
    y = df['Target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Train stacked model
    model = StackedClassifier()
    model.fit(X_scaled, y)

    # Predict
    df['ML_Signal'] = model.predict(X_scaled)

    return df['ML_Signal']
