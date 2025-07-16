# Machine Learning Models in Voice Classification Project

## Overview

This document explains how logistic regression and other machine learning models are implemented and used in the KaggleVoice gender classification project. The system processes audio files to extract acoustic features and classify speakers by gender using multiple ML algorithms.

## Core Machine Learning Models

### 1. Logistic Regression (Primary Model)

**Implementation**: `src/models/train_models.py:40-45`
- **Algorithm**: Scikit-learn's `LogisticRegression`
- **Configuration**: 
  - `max_iter=1000` for convergence
  - `random_state=42` for reproducibility
- **Purpose**: Binary classification (male=1, female=0)
- **Training**: Fits on standardized features using `StandardScaler`
- **Output**: Saved as `models/logistic_regression.pkl`

**Why Logistic Regression**:
- Excellent baseline for binary classification
- Provides interpretable coefficients
- Fast training and prediction
- Probabilistic outputs for confidence scoring

### 2. Random Forest Classifier

**Implementation**: `src/models/train_models.py:47-51`
- **Algorithm**: Scikit-learn's `RandomForestClassifier`
- **Configuration**: Default parameters with `random_state=42`
- **Advantages**: 
  - Handles feature interactions naturally
  - Provides feature importance rankings
  - Robust to overfitting
- **Output**: Saved as `models/random_forest.pkl`

### 3. XGBoost Classifier

**Implementation**: `src/models/train_models.py:53-59`
- **Algorithm**: XGBoost's `XGBClassifier`
- **Configuration**:
  - `use_label_encoder=False`
  - `eval_metric='logloss'`
  - `random_state=42`
- **Advantages**: 
  - State-of-the-art gradient boosting
  - Excellent performance on tabular data
  - Built-in regularization
- **Output**: Saved as `models/xgboost.pkl`

### 4. Naive Bayes (Gaussian)

**Implementation**: `src/models/train_models.py:61-64`
- **Algorithm**: Scikit-learn's `GaussianNB`
- **Advantages**:
  - Fast training and prediction
  - Works well with continuous features
  - Good baseline performance
- **Output**: Saved as `models/naive_bayes.pkl`

## Feature Engineering Pipeline

### Audio Feature Extraction

**Location**: `src/features/feature_extraction.py`

#### 1. MFCC Features (Mel-Frequency Cepstral Coefficients)
- **Count**: 13 coefficients
- **Purpose**: Captures timbral and phonetic characteristics
- **Implementation**: `librosa.feature.mfcc(n_mfcc=13, hop_length=512, n_fft=2048)`
- **Processing**: Mean values across time frames

#### 2. Spectral Features
- **Spectral Rolloff**: Frequency containing 85% of signal energy
- **Spectral Centroid**: "Center of mass" of frequency spectrum  
- **Zero Crossing Rate**: Rate of signal sign changes
- **RMS Energy**: Root mean square energy of signal

#### 3. Feature Selection
- **Initial Features**: 17 total (13 MFCC + 4 spectral)
- **Removed**: `rms`, `spectral_centroid`, `zero_crossing_rate` (high correlation)
- **Final Features**: 14 features for model training

### Audio Preprocessing Pipeline

**Location**: `src/preprocessing/audio_preprocessing.py`

1. **Audio Loading**:
   - Sample rate: 16kHz (standard for speech)
   - Formats: WAV, MP3, FLAC support

2. **Silence Trimming**:
   - Method: `librosa.effects.trim(top_db=35)`
   - Removes leading/trailing silence

3. **Noise Reduction**:
   - Library: noisereduce
   - Removes background noise and artifacts

4. **Normalization**:
   - Method: `librosa.util.normalize()`
   - Ensures consistent volume levels

5. **Length Standardization**:
   - Target: 3 seconds (48,000 samples at 16kHz)
   - Method: Padding with zeros or truncation

## Model Training Process

### Data Pipeline

**Location**: `src/data_loader.py`

1. **Data Loading**:
   - Parallel processing for feature extraction
   - Progress tracking with tqdm
   - Support for folder-based organization

2. **Data Preprocessing**:
   - Outlier removal using IQR method
   - Label encoding (male=1, female=0)
   - Data shuffling for training

3. **Train-Test Split**:
   - 80/20 split ratio
   - Stratified sampling to maintain class balance

4. **Feature Scaling**:
   - `StandardScaler` for normalization
   - Saved as `models/scaler.pkl` for inference

### Training Workflow

**Location**: `src/models/train_models.py`

```python
class ModelTrainer:
    def train_all_models(self):
        # Train each model on scaled features
        # Save models and performance metrics
        # Generate evaluation reports
```

## Model Evaluation

### Metrics Used
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, Recall, F1-Score per class
- **Confusion Matrix**: Error analysis
- **Cross-validation**: Model robustness assessment

### Performance Comparison
- Automated evaluation across all 4 models
- Ranking by accuracy and other metrics
- Detailed analysis of strengths/weaknesses

## Practical Usage

### Training Models
```bash
python main.py --male-folder data/male --female-folder data/female
```

### Making Predictions
```bash
python main.py --predict audio.wav --model logistic_regression --load-models
```

### Prediction API
```python
def predict_single_file(file_path, model_name='logistic_regression'):
    # Extract features from audio
    # Scale using saved scaler
    # Predict with specified model
    # Return prediction + confidence
```

## Visualization and Analysis

**Location**: `src/evaluation/visualization.py`

Available visualizations:
- Audio waveform plots
- Spectrograms and mel-spectrograms
- MFCC coefficient heatmaps
- Feature distribution analysis
- Model performance comparisons
- Confusion matrices with accuracy metrics

## Model Selection Rationale

### Why Multiple Models?
1. **Ensemble Potential**: Different algorithms capture different patterns
2. **Performance Comparison**: Identify best approach for this dataset
3. **Robustness**: Multiple models reduce risk of overfitting
4. **Research Value**: Educational comparison of approaches

### Model Characteristics

| Model | Strengths | Use Cases |
|-------|-----------|-----------|
| **Logistic Regression** | Fast, interpretable, probabilistic | Baseline, feature analysis |
| **Random Forest** | Handles interactions, feature importance | Robust classification |
| **XGBoost** | High performance, regularization | Competition/production |
| **Naive Bayes** | Fast training, simple | Quick prototyping |

## Gender-Specific Acoustic Patterns

### Key Distinguishing Features
1. **Fundamental Frequency**: Males typically have lower F0 (pitch)
2. **Formant Frequencies**: Vocal tract differences affect resonances
3. **Spectral Rolloff**: Energy distribution across frequencies
4. **MFCC Patterns**: Different phonetic characteristics

### Feature Importance
Based on model analysis:
- MFCC coefficients 1-5: Most discriminative
- Spectral rolloff: Strong gender indicator
- Combined features: Better than individual measures

## Technical Dependencies

### Core Libraries
- **Audio Processing**: librosa, noisereduce
- **Machine Learning**: scikit-learn, xgboost
- **Data Science**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib, tqdm

### System Requirements
- Python 3.7+
- Audio file format support (FFmpeg)
- Sufficient RAM for model training
- CPU/GPU for XGBoost acceleration (optional)

## Best Practices Implemented

1. **Reproducibility**: Fixed random seeds across all models
2. **Data Leakage Prevention**: Proper train-test separation
3. **Feature Scaling**: Standardization for algorithm compatibility
4. **Model Persistence**: Serialized models for production use
5. **Comprehensive Evaluation**: Multiple metrics and visualizations
6. **Modular Design**: Separate preprocessing, training, and evaluation
7. **Error Handling**: Robust file processing and validation

## Future Improvements

### Potential Enhancements
1. **Deep Learning**: CNN/RNN models for sequence learning
2. **Feature Engineering**: Prosodic features, pitch tracking
3. **Data Augmentation**: Noise addition, time stretching
4. **Ensemble Methods**: Voting classifiers, stacking
5. **Real-time Processing**: Streaming audio classification
6. **Transfer Learning**: Pre-trained audio models

This documentation provides a comprehensive overview of the machine learning implementation in the voice classification project, covering both the technical details and practical applications of each model.