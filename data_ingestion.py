# data_ingestion.py
import yfinance as yf
import pandas as pd

def get_nifty_data(start_date="2020-01-01", interval="1d"):
    ticker = '^NSEI'
    df = yf.download(ticker, start=start_date, interval=interval)
    df.columns = df.columns.droplevel(0)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    df.dropna(inplace=True)
    # df.columns.name = None  # remove multi-index if present
    print(df.head(2))
    return df
