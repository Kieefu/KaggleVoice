# Quick Start Guide

## ✅ 1. Test Installation
```bash
python demo.py
```
This will test all components and create a sample model.

## ✅ 2. Basic Usage (No Data Required)
The demo already created a working model! You can test predictions:

```bash
# Test prediction with the sample audio file
python main.py --predict data/sample/test_audio.wav --load-models --model logistic_regression
```

## 📁 3. Using Your Own Data

### Step 1: Organize Your Audio Files
```bash
mkdir -p data/raw/male data/raw/female
```

Place your audio files:
- Male voices → `data/raw/male/`
- Female voices → `data/raw/female/`

### Step 2: Train Models
```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female
```

### Step 3: Make Predictions
```bash
python main.py --predict your_audio.wav --load-models
```

## 🔧 Troubleshooting Commands

### Check Components
```bash
# Test each component
python -c "from src.preprocessing.audio_preprocessing import AudioPreprocessor; print('✅ Preprocessing OK')"
python -c "from src.features.feature_extraction import FeatureExtractor; print('✅ Feature extraction OK')"
python -c "from src.models.train_models import ModelTrainer; print('✅ Model training OK')"
python -c "from src.evaluation.visualization import DataVisualizer; print('✅ Visualization OK')"
```

### Test Individual Features
```bash
# Test feature extraction on sample file
python -c "
from src.features.feature_extraction import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_all_features('data/sample/test_audio.wav')
print(f'✅ Extracted {len(features)} features')
"
```

### Check Models
```bash
# List saved models
ls -la models/
```

## 📊 Expected Output

### Training Output:
```
Processing X audio files...
Original data shape: (X, 17)
After removing duplicates: (X, 17)
After removing outliers: (X, 17)
Training set size: X
Testing set size: X
Training Logistic Regression...
Training Random Forest...
Training XGBoost...
Training Naive Bayes...
```

### Prediction Output:
```
Prediction: MALE
Confidence: 0.85
Probabilities: Female: 0.15, Male: 0.85
```

## 📈 Model Performance
- **Logistic Regression**: Usually best for voice classification
- **Random Forest**: Good robustness
- **XGBoost**: Often highest accuracy
- **Naive Bayes**: Fastest training

## 🎯 Tips for Best Results

1. **Audio Quality**: Use clear, good quality audio files
2. **Balanced Data**: Equal number of male/female samples
3. **File Format**: WAV files work best
4. **Duration**: 2-5 seconds of speech is optimal
5. **Sample Size**: Start with 100-200 files per gender

## 🚀 Next Steps

1. **Collect More Data**: The more diverse your dataset, the better
2. **Experiment with Models**: Try different models to see which works best
3. **Visualize Results**: Use `--visualize` flag to see performance charts
4. **Fine-tune**: Adjust preprocessing parameters for your specific data