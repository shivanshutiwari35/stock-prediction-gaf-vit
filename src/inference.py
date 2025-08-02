import torch
import yfinance as yf
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from pyts.image import GramianAngularField
from transformers import ViTForImageClassification
import os

def fetch_live_data(ticker, days=33):
    """Fetches the last N historical data days for a given ticker."""
    # We fetch a bit more to ensure we get N trading days
    df = yf.download(ticker, period=f"{days+10}d", progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return df.tail(days)

def create_inference_image(df_window, image_size=30):
    """Converts a 30-day window of closing prices into a GAF image."""
    window_np = df_window['Close'].to_numpy()
    
    scaler = np.max(np.abs(window_np))
    window_scaled = window_np / scaler if scaler != 0 else window_np

    gaf = GramianAngularField(image_size=image_size, method='summation')
    gaf_image = gaf.fit_transform(window_scaled.reshape(1, -1))
    
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots()
    ax.imshow(gaf_image[0], cmap='rainbow', origin='lower')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    pil_image = Image.open(buf).convert("RGB")
    return pil_image

def predict_and_verify(ticker, model_path, window_size=30, image_size=224):
    """
    Fetches the last 33 days, predicts on the first 30, 
    and verifies against the final 3.
    """
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 2
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Fetch and prepare data
    print(f"Fetching latest 33 trading days of data for {ticker}...")
    data_33_days = fetch_live_data(ticker, days=33)
    if len(data_33_days) < 33:
        raise ValueError(f"Not enough data. Need 33 days, got {len(data_33_days)}.")

    inference_window = data_33_days.head(window_size)
    verification_window = data_33_days.tail(3)
    
    day_30_date = inference_window.index[-1].date()
    day_33_date = verification_window.index[-1].date()

    print(f"Input data for model: {inference_window.index[0].date()} to {day_30_date}")
    print(f"Verification data:      {verification_window.index[0].date()} to {day_33_date}")

    # 3. Create GAF image and preprocess
    gaf_image = create_inference_image(inference_window, image_size=window_size)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(gaf_image).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    prediction = predicted_class.item()
    confidence_score = confidence.item()

    # 5. Verify
    price_day_30 = inference_window['Close'].iloc[-1]
    price_day_33 = verification_window['Close'].iloc[-1]
    actual_result = 1 if price_day_33 > price_day_30 else 0

    # 6. Display results
    print("\n--- Prediction ---")
    print(f"The model predicts the price will go {'UP' if prediction == 1 else 'DOWN'} in the next 3 days.")
    print(f"Confidence: {confidence_score:.2%}")
    print("--- Verification ---")
    print(f"Actual price on {day_30_date}: {price_day_30:.2f}")
    print(f"Actual price on {day_33_date}: {price_day_33:.2f}")
    print(f"The actual price went {'UP' if actual_result == 1 else 'DOWN'}.")
    print("--- Result ---")
    if prediction == actual_result:
        print("✅ Correct Prediction")
    else:
        print("❌ Incorrect Prediction")
    print("--------------------")

if __name__ == '__main__':
    TICKER_TO_PREDICT = 'AAPL' # You can change this to MSFT, TSLA, etc.
    MODEL_PATH = 'D:/GOLDMAN SACHS PROJECT/gaf-vitrade/models/vit_model.pth' 
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
    else:
        try:
            predict_and_verify(TICKER_TO_PREDICT, MODEL_PATH)
        except Exception as e:
            print(f"An error occurred: {e}")