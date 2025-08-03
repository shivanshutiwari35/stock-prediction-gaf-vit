import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def create_gaf_images_for_ticker(data, ticker_name, window_size, image_size, base_output_dir):
    """
    Creates and saves Gramian Angular Field (GAF) images for a single ticker's data.

    Args:
        data (pd.DataFrame): The DataFrame for a single ticker.
        ticker_name (str): The name of the ticker (e.g., 'RELIANCE.NS').
        window_size (int): The size of the rolling window.
        image_size (int): The desired size of the output GAF images.
        base_output_dir (str): The base directory to save the GAF images (e.g., 'gaf_images/train').
    """
    # Create rolling windows from the 'Close' price
    windows = []
    labels = []
    for i in range(len(data) - window_size + 1):
        windows.append(data['Close'].iloc[i:i+window_size].values)
        labels.append(data['label'].iloc[i+window_size-1])

    if not windows:
        print(f"No windows created for {ticker_name}. Skipping.")
        return

    # Create GAF images
    gaf = GramianAngularField(image_size=image_size, method='summation')
    X_gaf = gaf.fit_transform(np.array(windows))

    # Save images
    for i, (image, label) in enumerate(zip(X_gaf, labels)):
        # Create separate directories for each label (0 or 1)
        label_dir = os.path.join(base_output_dir, str(int(label)))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Include ticker name in the image file to ensure uniqueness
        image_filename = f"{ticker_name}_gaf_{i}.png"
        plt.imsave(os.path.join(label_dir, image_filename), image, cmap='rainbow', origin='lower')

if __name__ == '__main__':
    # Load the consolidated labeled data
    labeled_data = pd.read_csv('D:gaf-vitrade/data/labeled_indian_stocks.csv', index_col=0)

    # Group data by ticker
    grouped = labeled_data.groupby('ticker')

    print(f"Processing data for {len(grouped)} tickers.")

    # Define base output directories
    train_dir = 'D:/gaf-vitrade/gaf_images/train'
    test_dir = 'D:/gaf-vitrade/gaf_images/test'

    # Clear out old images
    for directory in [train_dir, test_dir]:
        if os.path.exists(directory):
            for label_folder in ['0', '1']:
                folder_path = os.path.join(directory, label_folder)
                if os.path.exists(folder_path):
                    for file in os.listdir(folder_path):
                        os.remove(os.path.join(folder_path, file))
    print("Cleared old GAF images.")

    for ticker_name, ticker_data in grouped:
        print(f"\nProcessing ticker: {ticker_name}")
        
        # Split data for each ticker into training and testing sets
        train_data, test_data = train_test_split(ticker_data[['Close', 'label']], test_size=0.2, shuffle=False) # shuffle=False is important for time-series
        
        print(f"  - Creating {len(train_data)} training images...")
        create_gaf_images_for_ticker(train_data, ticker_name, window_size=30, image_size=30, base_output_dir=train_dir)
        
        print(f"  - Creating {len(test_data)} testing images...")
        create_gaf_images_for_ticker(test_data, ticker_name, window_size=30, image_size=30, base_output_dir=test_dir)

    print("\nSuccessfully created all GAF images.")
