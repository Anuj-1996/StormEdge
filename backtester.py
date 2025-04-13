# backtester.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from config import CAPITAL


class BacktestEngine:
    def __init__(self, initial_capital: float = CAPITAL, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct


    def run_backtest(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        logging.info("Starting backtest engine with ML signals, position sizing, and transaction costs...")

        df = df.copy()
        df['Signal'] = signals
        df['Daily_Return'] = df['Close'].pct_change()
        df['PositionSize'] = df.get('PositionSize', 1)

        df['Effective_Position'] = df['Signal'].shift(1) * df['PositionSize'].shift(1)

        # Apply slippage (higher price paid on buy, lower received on sell)
        df['Slippage_Adjustment'] = df['Effective_Position'].diff().fillna(0).apply(lambda x: -self.slippage_pct if x != 0 else 0)
        
        # Apply commission (only when trade happens — signal changes)
        df['Commission_Cost'] = df['Effective_Position'].diff().abs().fillna(0) * self.commission_pct

        # Adjust return
        df['Strategy_Return'] = (df['Effective_Position'] * df['Daily_Return']) + df['Slippage_Adjustment'] - df['Commission_Cost']
        df['Strategy_Return'].fillna(0, inplace=True)

        df['Portfolio'] = (1 + df['Strategy_Return']).cumprod() * self.initial_capital

        self.plot_portfolio(df)

        final_value = df['Portfolio'].iloc[-1]
        logging.info(f"Backtest complete. Final portfolio value: {final_value:.2f}")
        return df[['Signal', 'PositionSize', 'Strategy_Return', 'Portfolio']]



    def plot_portfolio(self, df: pd.DataFrame, save_path: str = "plots/portfolio_plot.png") -> None:
        df = df.copy()
        df['Returns'] = df['Portfolio'].pct_change().fillna(0)
        df['Cumulative'] = (1 + df['Returns']).cumprod()

        # Performance metrics
        sharpe = self._sharpe_ratio(df['Returns'])
        sortino = self._sortino_ratio(df['Returns'])
        max_dd = self._max_drawdown(df['Cumulative'])

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Portfolio'], label='Portfolio Value', color='blue')
        plt.title("Backtest Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (INR)")
        plt.grid(True)

        # Overlay metrics as text
        plt.text(df.index[int(len(df) * 0.01)], df['Portfolio'].max()*0.98,
                 f"Sharpe Ratio: {sharpe:.2f}\nSortino Ratio: {sortino:.2f}\nMax Drawdown: {max_dd:.2%}",
                 fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Plot saved to: {save_path}")
        logging.info(f"Plot saved to: {save_path}")
        logging.info("Backtest completed and plot saved.")
        logging.info(f"Sharpe Ratio: {sharpe:.2f}")
        logging.info(f"Sortino Ratio: {sortino:.2f}")
        logging.info(f"Max Drawdown: {max_dd:.2%}")
        logging.info("Backtest completed and plot saved.")

    def _sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        excess = returns - risk_free_rate / 252  # annual to daily
        return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

    def _sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        excess = returns - risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / (downside_std + 1e-9)

    def _max_drawdown(self, cumulative_returns: pd.Series) -> float:
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1.0
        return drawdown.min()
