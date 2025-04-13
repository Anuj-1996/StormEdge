import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

class HMMRegimeClassifier:
    def __init__(self, n_states: int = 5):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        features = df[['ATR_norm', 'RSI', 'MACD_Hist', 'BB_Width']].copy()
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        self.fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        features = df[['ATR_norm', 'RSI', 'MACD_Hist', 'BB_Width']].copy()
        features_scaled = self.scaler.transform(features)
        regimes = self.model.predict(features_scaled)
        return regimes

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        features = df[['ATR_norm', 'RSI', 'MACD_Hist', 'BB_Width']].copy()
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)
        return probabilities

    def plot_regimes(self, df: pd.DataFrame, regimes: np.ndarray, save_path: str = "plots/regime_plot.png") -> None:
        """
        Plot NIFTY price with color-coded regimes and save to file.
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must include 'Close' column for plotting.")
        
        plt.figure(figsize=(14, 6))
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            regime_indices = np.where(regimes == regime)[0]
            plt.plot(df.index[regime_indices], df['Close'].iloc[regime_indices], label=f'Regime {regime}')
        
        plt.title("Market Regimes Detected by HMM")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Regime plot saved to: {save_path}")
