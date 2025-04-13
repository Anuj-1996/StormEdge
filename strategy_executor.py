# strategy_executor.py
import pandas as pd
import yfinance as yf
from feature_engineering import add_technical_indicators
from regime_classifier import classify_regimes

class RegimeBasedStrategy:
    def __init__(self, ticker: str = "^NSEI", start_date: str = "2020-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, start=self.start_date, interval="1d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        return df

    def run(self):
        print("[INFO] Adding indicators...")
        self.data = add_technical_indicators(self.data)

        print("[INFO] Classifying regimes...")
        self.data = classify_regimes(self.data)

        print("[INFO] Strategy ready. Sample output:")
        print(self.data[['Close', 'Regime']].tail(10))

if __name__ == "__main__":
    strategy = RegimeBasedStrategy()
    strategy.run()
