# Gender Voice Classification

A machine learning project for classifying gender from voice recordings using audio feature extraction and multiple classification algorithms.

## Project Structure

```
KaggleVoice/
├── data/
│   ├── raw/           # Raw audio files
│   └── processed/     # Processed data
├── models/            # Trained models
├── src/
│   ├── preprocessing/ # Audio preprocessing
│   ├── features/      # Feature extraction
│   ├── models/        # Model training
│   └── evaluation/    # Visualization & evaluation
├── notebooks/         # Jupyter notebooks
├── main.py           # Main pipeline script
├── requirements.txt  # Dependencies
└── README.md
```

## Features

### Audio Preprocessing
- **Trimming**: Remove silence from audio
- **Noise Reduction**: Reduce background noise
- **Normalization**: Consistent volume levels
- **Resampling**: Standard sampling rate (16kHz)
- **Padding/Truncating**: Uniform audio length

### Feature Extraction
- **MFCC**: Mel-Frequency Cepstral Coefficients (13 features)
- **Spectral Rolloff**: Frequency rolloff point
- **Zero Crossing Rate**: Signal sign changes
- **Spectral Centroid**: Frequency center of mass
- **RMS Energy**: Root mean square energy

### Models
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Naive Bayes**

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female
```

### Making Predictions

```bash
python main.py --predict audio.wav --model logistic_regression --load-models
```

### With Visualizations

```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female --visualize
```

## Data Organization

Organize your audio files as follows:
```
data/raw/
├── male/
│   ├── male_voice1.wav
│   ├── male_voice2.wav
│   └── ...
└── female/
    ├── female_voice1.wav
    ├── female_voice2.wav
    └── ...
```

## Example Usage

```python
from src.data_loader import DataLoader
from src.models.train_models import ModelTrainer
from src.evaluation.visualization import DataVisualizer

# Load and preprocess data
data_loader = DataLoader()
df = data_loader.load_and_preprocess_data('data/raw/male', 'data/raw/female')

# Train models
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
models = trainer.train_all_models()

# Evaluate models
results = trainer.evaluate_all_models()

# Make predictions
result = trainer.predict_single_file('test_audio.wav', 'logistic_regression')
print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
```

## Model Performance

The pipeline typically achieves:
- **Logistic Regression**: ~85-90% accuracy
- **Random Forest**: ~85-90% accuracy
- **XGBoost**: ~85-90% accuracy
- **Naive Bayes**: ~80-85% accuracy

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Librosa
- Matplotlib
- Seaborn
- Noisereduce

## License

This project is based on the Kaggle notebook for gender voice classification and is intended for educational purposes.