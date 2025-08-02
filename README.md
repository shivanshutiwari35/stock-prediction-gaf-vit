# GAF-ViTrade: Stock Market Prediction using Gramian Angular Fields and Vision Transformers

## Overview
This project implements a novel approach to stock market prediction by combining Gramian Angular Fields (GAF) with Vision Transformers (ViT). The system converts time series stock data into 2D images using GAF and then uses a Vision Transformer to predict stock price movements.

## Features
- **Gramian Angular Fields (GAF)**: Converts 1D time series data into 2D images
- **Vision Transformer (ViT)**: Deep learning model for image classification
- **Stock Data Processing**: Handles multiple stock datasets
- **Model Training**: Complete training pipeline with evaluation
- **Streamlit Web App**: Interactive web interface for predictions

## Project Structure
```
gaf-vitrade/
├── data/                   # Stock data files
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── dataset.py         # Dataset classes
│   ├── fetch_data.py      # Data fetching utilities
│   ├── generate_gaf.py    # GAF image generation
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── models/                 # Trained model files
├── gaf_images/            # Generated GAF images
├── streamlit_app/         # Streamlit web application
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gaf-vitrade
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place your stock data CSV files in the `data/` directory
2. Run the data preprocessing script:
```bash
python src/fetch_data.py
```

### Generate GAF Images
```bash
python src/generate_gaf.py
```

### Train the Model
```bash
python src/train.py
```

### Run the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

## Model Architecture
- **Input**: GAF images (224x224 pixels)
- **Backbone**: Vision Transformer (ViT)
- **Output**: Binary classification (up/down movement)

## Results
The model achieves competitive accuracy on stock price movement prediction using the GAF + ViT approach.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.

## Acknowledgments
- Vision Transformer paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Gramian Angular Fields: "Imaging Time-Series to Improve Classification and Imputation"
