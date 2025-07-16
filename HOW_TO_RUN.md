# How to Run Gender Voice Classification

## 🎯 Your Project is Ready!

Everything is set up and working. Here's exactly how to use it:

## 🚀 Immediate Usage (No Setup Required)

### 1. Test the Demo
```bash
python demo.py
```
This creates sample data and trains a model.

### 2. Make a Prediction
```bash
python main.py --predict data/sample/test_audio.wav --load-models
```

## 📂 Using Your Own Audio Data

### Step 1: Organize Your Files
```bash
# Create directories
mkdir -p data/raw/male data/raw/female

# Copy your audio files
cp /path/to/male_voices/* data/raw/male/
cp /path/to/female_voices/* data/raw/female/
```

### Step 2: Train Models
```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female
```

### Step 3: Make Predictions
```bash
python main.py --predict your_audio.wav --load-models
```

## 🎛️ Advanced Options

### Training with Visualizations
```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female --visualize
```

### Different Models
```bash
# Train all models, then predict with specific one
python main.py --predict audio.wav --load-models --model random_forest
python main.py --predict audio.wav --load-models --model xgboost
python main.py --predict audio.wav --load-models --model naive_bayes
```

### Sequential Processing (if parallel fails)
```bash
python main.py --male-folder data/raw/male --female-folder data/raw/female --no-parallel
```

## 📊 What to Expect

### Training Output:
```
Processing 1000 audio files...
Original data shape: (1000, 17)
After removing duplicates: (995, 17)
After removing outliers: (950, 17)
Training set size: 760
Testing set size: 190

Training Logistic Regression...
Training Random Forest...
Training XGBoost...
Training Naive Bayes...

LOGISTIC_REGRESSION: 0.8947
RANDOM_FOREST: 0.9000
XGBOOST: 0.9053
NAIVE_BAYES: 0.8684

Best Model: XGBOOST with accuracy 0.9053
```

### Prediction Output:
```
Prediction: FEMALE
Confidence: 0.856
Probabilities: Female: 0.856, Male: 0.144
```

## 📁 File Structure After Running

```
KaggleVoice/
├── data/
│   ├── raw/
│   │   ├── male/         # Your male audio files
│   │   └── female/       # Your female audio files
│   └── sample/
│       └── test_audio.wav # Demo audio file
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── naive_bayes.pkl
│   └── scaler.pkl
└── [other files...]
```

## 🔧 Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the right directory
pwd  # Should show: /Users/kieefu/PycharmProjects/KaggleVoice

# Test imports
python -c "from src.data_loader import DataLoader; print('OK')"
```

### If audio processing fails:
```bash
# Check audio file
python -c "import librosa; y, sr = librosa.load('your_audio.wav'); print(f'Length: {len(y)}, Sample rate: {sr}')"
```

### If models don't exist:
```bash
# Check if models were saved
ls -la models/

# If empty, run training first
python main.py --male-folder data/raw/male --female-folder data/raw/female
```

## 🎯 Pro Tips

1. **Audio Quality Matters**: Use clear, noise-free audio
2. **Balanced Dataset**: Equal male/female samples work best
3. **File Format**: WAV files are preferred
4. **Duration**: 2-5 seconds of speech is optimal
5. **Start Small**: Test with 50-100 files per gender first

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| No audio files found | Check folder paths and file extensions |
| Memory error | Use smaller dataset or add more RAM |
| Low accuracy | Need more/better quality training data |
| Import errors | Run from project root directory |
| Prediction fails | Ensure models are trained first |

## ✅ Your Project is Complete!

You now have a fully functional gender voice classification system. The code is organized, documented, and ready to use with your own data!