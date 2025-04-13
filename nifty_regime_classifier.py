import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class RegimeBasedStrategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None

    def preprocess(self):
        self.df['log_ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['vol'] = self.df['log_ret'].rolling(5).std()
        self.df.dropna(inplace=True)

    def fit_hmm(self, n_components=2):
        X = self.df[['log_ret', 'vol']].values
        model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000, random_state=42)
        model.fit(X)
        self.df['Regime'] = model.predict(X)
        self.model = model

        # Regime mapping: 1 for trending (higher mean return), 0 for low-vol
        means = self.df.groupby('Regime')['log_ret'].mean()
        trending_regime = means.idxmax()
        self.df['Regime'] = self.df['Regime'].apply(lambda x: 1 if x == trending_regime else 0)

    def generate_signals(self):
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        self.df['Position'] = 0

        condition = (self.df['SMA_20'] > self.df['SMA_50']) & (self.df['Regime'] == 1)
        self.df.loc[condition, 'Position'] = 1

    def calculate_performance(self):
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Strategy'] = self.df['Position'].shift(1) * self.df['Returns']
        self.df.dropna(inplace=True)

    def sharpe_ratio(self, series):
        return (series.mean() / series.std()) * np.sqrt(252)

    def plot_results(self):
        cumulative = (1 + self.df[['Returns', 'Strategy']]).cumprod()
        cumulative.plot(figsize=(14, 6), title="Strategy vs Buy & Hold")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.preprocess()
        self.fit_hmm()
        self.generate_signals()
        self.calculate_performance()
        print(f"Buy & Hold Sharpe: {self.sharpe_ratio(self.df['Returns']):.2f}")
        print(f"Strategy Sharpe: {self.sharpe_ratio(self.df['Strategy']):.2f}")
        self.plot_results()


def load_nifty_data(path: str):
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    df.columns = df.columns.get_level_values(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    return df


if __name__ == "__main__":
    data_path = "nifty.csv"  # 🔁 your CSV path
    nifty_df = load_nifty_data(data_path)

    strategy = RegimeBasedStrategy(nifty_df)
    strategy.run()
