import pandas as pd
import os

def generate_labels(data, horizon=3):

    # Calculate the future price and generate the label
    data['label'] = (data['Close'].shift(-horizon) > data['Close']).astype(int)
    # Drop the rows where the label cannot be calculated (the last `horizon` rows)
    data.dropna(subset=['label'], inplace=True)
    return data

if __name__ == '__main__':
    input_dir = 'D:/gaf-vitrade/data/indian_stocks/'
    output_file = 'D:/gaf-vitrade/data/labeled_indian_stocks.csv'
    
    all_labeled_data = []

    # Get all the CSV files from the input directory
    stock_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    print(f"Found {len(stock_files)} stock data files to process.")

    for file_name in stock_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            print(f"Processing {file_name}...")
            stock_data = pd.read_csv(file_path, index_col=0, parse_dates=True, header=0)
            
            # Ensure the data has the 'Close' column
            if 'Close' not in stock_data.columns:
                print(f"Warning: 'Close' column not found in {file_name}. Skipping.")
                continue

            labeled_df = generate_labels(stock_data)
            
            # Add a column for the ticker symbol to keep track of the source
            ticker = file_name.replace('.csv', '')
            labeled_df['ticker'] = ticker
            
            all_labeled_data.append(labeled_df)
            print(f"Successfully generated labels for {ticker}.")

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

    # Combine all the labeled dataframes into one
    if all_labeled_data:
        final_df = pd.concat(all_labeled_data)
        final_df.to_csv(output_file)
        print(f"\nSuccessfully combined and saved all labeled data to {output_file}")
        print(f"Total labeled data points: {len(final_df)}")
    else:
        print("\nNo data was processed. The output file was not created.")
