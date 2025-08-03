import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_dir):
    """
    Fetches OHLCV data for a list of tickers from Yahoo Finance and saves each to a separate CSV.

    Args:
        tickers (list): A list of stock tickers.
        start_date (str): The start date for the data in YYYY-MM-DD format.
        end_date (str): The end date for the data in YYYY-MM-DD format.
        output_dir (str): The absolute path to the directory to save the CSV files.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download data for each ticker individually to handle potential errors
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"Warning: No data was downloaded for {ticker}. It might be delisted or the ticker is incorrect.")
                continue

            file_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"Saved data for {ticker} to {file_path}")

        except Exception as e:
            print(f"Could not download data for {ticker}. Error: {e}")

if __name__ == '__main__':
    # List of 10 prominent Indian stocks
    indian_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS'
    ]
    
    # Set the date range for 20 years
    start_date = '2005-08-01'
    end_date = '2025-08-01'
    
    # Define the output directory
    output_directory = 'D:/gaf-vitrade/data/indian_stocks/'

    fetch_data(indian_tickers, start_date, end_date, output_directory)

    print("\nData collection process complete.")
