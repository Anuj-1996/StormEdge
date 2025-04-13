# main.py
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
from data_ingestion import get_nifty_data
from feature_engineering import add_technical_indicators
from hmm_regime_classifier import HMMRegimeClassifier
from model_trainer import TrendPredictor  # Real ML model class
from backtester import BacktestEngine
from performance_metrics import evaluate_performance

from data_filter import filter_by_date
from garch_volatility import add_garch_volatility
from transaction_costs import apply_transaction_costs
from position_sizing import calculate_position_size
from config import CAPITAL

import warnings
warnings.filterwarnings("ignore")

# Create necessary folders
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# For full reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler()
    ]
)

def plot_signals(df):
    """
    Plot Buy/Sell signals on the price chart.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Close Price', alpha=0.7)

    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='red')

    plt.title("ML Trading Signals on NIFTY Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/signals_plot.png")
    logging.info("Signal plot saved to: plots/signals_plot.png")
    plt.close()

def main():
    logging.info("Loading NIFTY data...")
    nifty_df = get_nifty_data()

    logging.info("Adding technical indicators...")
    nifty_df = add_technical_indicators(nifty_df)
    # Add GARCH-based volatility
    nifty_df = add_garch_volatility(nifty_df)
    
    logging.info("Fitting HMM model for regime detection...")
    hmm_model = HMMRegimeClassifier(n_states=5)
    hmm_model.fit(nifty_df)
    regimes = hmm_model.predict(nifty_df)
    nifty_df['Regime'] = regimes

    # Add daily returns and target signal
    nifty_df['Return'] = nifty_df['Close'].pct_change()
    nifty_df['Signal'] = (nifty_df['Return'].shift(-1) > 0).astype(int)  # 1 = Buy, 0 = Hold/Sell


    # Calculate Sharpe-like score for each regime
    regime_stats = nifty_df.groupby('Regime')['Return'].agg(['mean', 'std'])
    regime_stats['sharpe'] = regime_stats['mean'] / regime_stats['std']

    # Select top 2 regimes as tradable
    ranked = regime_stats.sort_values(by='sharpe', ascending=False)
    allowed_regimes = ranked.head(5).index.tolist()
    
    logging.info(f"Auto-selected regimes based on Sharpe: {allowed_regimes}")
    logging.info("\nRegime ranking based on Sharpe:\n" + str(ranked))


    logging.info("Plotting HMM regimes...")
    hmm_model.plot_regimes(nifty_df, regimes, save_path="plots/regime_plot.png")
    logging.info("Regime plot saved.")

    logging.info("Training ML classifier for signal generation...")
    feature_cols = ['RSI', 'MACD_Hist', 'Volatility', 'Momentum', 'ATR_norm']

    target_col = 'Signal'

    # Stress test: Filter COVID crash period (Feb to May 2020)
    # nifty_df = filter_by_date(nifty_df, start_date='2020-02-01', end_date='2020-05-31')
    # For now, maybe test post-COVID rally
    # nifty_df = filter_by_date(nifty_df, start_date='2021-06-01', end_date='2023-12-31')

    today_date = datetime.today().strftime('%Y-%m-%d')

    # Apply the filter with today's date as start and end
    nifty_df = filter_by_date(nifty_df, start_date='2022-01-01', end_date=today_date)



    predictor = TrendPredictor(tune=False)
    predictor.train(nifty_df.dropna(), feature_cols, target_col)

    # Get probabilities
    nifty_df = nifty_df.dropna().copy()
    probs = predictor.predict_proba(nifty_df, feature_cols)[:, 1]
    nifty_df['ML_Proba'] = probs
    nifty_df['ML_Signal'] = (nifty_df['ML_Proba'] > 0.60).astype(int)

    # Convert 0s to -1 for sell/neutral
    nifty_df['Signal'] = nifty_df['ML_Signal'].replace(0, -1)

    logging.info("Displaying frequency of (Signal, Regime) combinations:")
    print(nifty_df[['Signal', 'Regime']].value_counts(dropna=False))

    plot_signals(nifty_df)

    logging.info("Backtesting strategy...")
    backtester = BacktestEngine()

    # Filter signal based on allowed regimes
    nifty_df['Filtered_Signal'] = nifty_df.apply(
        lambda row: row['Signal'] if row['Regime'] in allowed_regimes else 0, axis=1
    )

    nifty_df['PositionSize'] = calculate_position_size(nifty_df, capital=CAPITAL, risk_pct=0.02)
    
    # Apply realistic transaction costs and slippage using modular logic
    nifty_df['Signal'] = nifty_df['Filtered_Signal']  # reuse for cost function

    nifty_df = apply_transaction_costs(
        nifty_df,
        signal_col='Signal',
        capital=backtester.initial_capital,
        commission_pct=0.001,  # 0.1%
        slippage_pct=0.0005    # 0.05%
    )

    # Then pass to the backtester to plot only (not to recalculate returns again)
    results = nifty_df[['Signal', 'Strategy_Return', 'Portfolio']]
    backtester.plot_portfolio(nifty_df)

    print(nifty_df[['Signal', 'Regime']].value_counts(dropna=False))
    logging.info("\n" + str(nifty_df[['Signal', 'Regime']].value_counts(dropna=False)))




    # Evaluate performance
    evaluate_performance(results)

if __name__ == "__main__":
    main()
