# regime_classifier.py
import pandas as pd

class RegimeClassifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def classify(self) -> pd.DataFrame:
        self.df['Regime'] = self.df.apply(self._determine_regime, axis=1)
        return self.df

    def _determine_regime(self, row) -> str:
        if row['ATR_norm'] > 0.015 and abs(row['MACD_Hist']) > 20:
            return 'Trending'
        elif row['ATR_norm'] < 0.01 and row['BB_Width'] < 0.03:
            return 'Low Volatility'
        elif row['RSI'] > 50 and row['MACD_Hist'] > 0:
            return 'Bullish Bias'
        elif row['RSI'] < 50 and row['MACD_Hist'] < 0:
            return 'Bearish Bias'
        else:
            return 'Sideways'
