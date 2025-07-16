# Setup Guide for Gender Voice Classification

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Installation
```bash
python demo.py
```

### 3. Check if Everything Works
```bash
python main.py --help
```

## Data Setup

### Option 1: Use Your Own Audio Data
1. Create directories:
```bash
mkdir -p data/raw/male data/raw/female
```

2. Place your audio files:
   - Male voice files → `data/raw/male/`
   - Female voice files → `data/raw/female/`

3. Run training:
```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female
```

### Option 2: Download Sample Dataset
You can use the Common Voice dataset or any other voice dataset:
1. Download and extract audio files
2. Organize them by gender
3. Run the training pipeline

## Usage Examples

### Train Models
```bash
# Basic training
python main.py --male-folder data/raw/male --female-folder data/raw/female

# With visualizations
python main.py --male-folder data/raw/male --female-folder data/raw/female --visualize

# Sequential processing (if parallel fails)
python main.py --male-folder data/raw/male --female-folder data/raw/female --no-parallel
```

### Make Predictions
```bash
# Single file prediction
python main.py --predict path/to/audio.wav --load-models

# Use specific model
python main.py --predict path/to/audio.wav --load-models --model random_forest
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure you're running from the project root directory
   - Check that all dependencies are installed

2. **Audio File Issues**
   - Supported formats: WAV, MP3, FLAC, etc.
   - Ensure audio files are not corrupted

3. **Memory Issues**
   - Reduce batch size or use sequential processing
   - Process smaller datasets first

4. **Model Loading Issues**
   - Make sure models are trained first
   - Check that models/ directory exists

### Testing Individual Components

```bash
# Test preprocessing
python -c "from src.preprocessing.audio_preprocessing import AudioPreprocessor; print('OK')"

# Test feature extraction
python -c "from src.features.feature_extraction import FeatureExtractor; print('OK')"

# Test model training
python -c "from src.models.train_models import ModelTrainer; print('OK')"
```

## Project Structure After Setup

```
KaggleVoice/
├── data/
│   ├── raw/
│   │   ├── male/     # Your male voice files
│   │   └── female/   # Your female voice files
│   └── processed/
├── models/           # Trained models saved here
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── naive_bayes.pkl
│   └── scaler.pkl
├── src/             # Source code
└── main.py          # Main script
```

## Performance Tips

1. **Use Parallel Processing** (default): Faster for large datasets
2. **Start Small**: Test with ~100 files first
3. **Monitor Memory**: Large datasets may need more RAM
4. **GPU Support**: XGBoost can use GPU if available

## Expected Results

- **Logistic Regression**: ~85-90% accuracy
- **Random Forest**: ~85-90% accuracy  
- **XGBoost**: ~85-90% accuracy
- **Naive Bayes**: ~80-85% accuracy

Results depend on your dataset quality and size.