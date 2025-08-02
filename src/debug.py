
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField

def debug_single_ticker_gaf():
    """A focused debugging script to trace the data transformation for a single ticker."""
    try:
        # 1. Load the data
        print("--- Loading Data ---")
        labeled_data = pd.read_csv('D:/GOLDMAN SACHS PROJECT/gaf-vitrade/data/labeled_indian_stocks.csv', index_col=0)
        print("Data loaded successfully.")
        print("Data columns:", labeled_data.columns)
        print("Data head:\n", labeled_data.head())

        # 2. Select a single ticker
        print("\n--- Selecting Ticker ---")
        ticker_name = 'BHARTIARTL.NS'
        ticker_data = labeled_data[labeled_data['ticker'] == ticker_name]
        print(f"Data for {ticker_name} selected.")
        print("Ticker data columns:", ticker_data.columns)
        print("Ticker data head:\n", ticker_data.head())

        # 3. Select only the 'Close' column
        print("\n--- Selecting 'Close' Column ---")
        close_data = ticker_data['Close']
        print("'Close' column selected.")
        print("Close data head:\n", close_data.head())

        # 4. Create a single window
        print("\n--- Creating a Single Window ---")
        window_size = 30
        window = close_data.head(window_size)
        print("Single window created.")
        print("Window data:\n", window)

        # 5. Convert window to NumPy array
        print("\n--- Converting to NumPy Array ---")
        window_np = window.to_numpy()
        print("Conversion successful.")
        print("NumPy array:\n", window_np)

        # 6. Reshape for GAF
        print("\n--- Reshaping Array ---")
        window_reshaped = window_np.reshape(1, -1)
        print("Reshaping successful.")
        print("Reshaped array shape:", window_reshaped.shape)

        # 7. Apply GAF transformation
        print("\n--- Applying GAF Transformation ---")
        gaf = GramianAngularField(image_size=30, method='summation')
        gaf_image = gaf.fit_transform(window_reshaped)
        print("GAF transformation successful.")
        print("GAF image shape:", gaf_image.shape)

        print("\n--- Debugging Successful ---")

    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_single_ticker_gaf()
