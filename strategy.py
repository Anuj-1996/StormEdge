# strategy.py
import pandas as pd
from ml_model import generate_ml_signals

class RegimeBasedStrategy:
    def __init__(self):
        pass  # No need for self.position now

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using a stacked ML classifier:
        - 1 = Buy signal (model predicts UP)
        - -1 = Sell signal (model predicts DOWN)
        """
        ml_signals = generate_ml_signals(df)

        # Convert ML output (0 or 1) to trading signals:
        # 1 = Buy, -1 = Sell, 0 = Hold
        signals = ml_signals.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        return signals
