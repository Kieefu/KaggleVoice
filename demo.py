#!/usr/bin/env python3
"""
Demo script for Gender Voice Classification
"""

import numpy as np
import librosa
import os
from src.preprocessing.audio_preprocessing import AudioPreprocessor
from src.features.feature_extraction import FeatureExtractor
from src.models.train_models import ModelTrainer
from src.evaluation.visualization import DataVisualizer


def create_sample_audio():
    """Create a sample audio file for testing."""
    # Generate a simple sine wave as sample audio
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, audio_signal.shape)
    audio_signal += noise
    
    # Save the audio file
    import soundfile as sf
    os.makedirs('data/sample', exist_ok=True)
    sf.write('data/sample/test_audio.wav', audio_signal, sample_rate)
    print("Sample audio file created: data/sample/test_audio.wav")
    
    return 'data/sample/test_audio.wav'


def demo_preprocessing():
    """Demonstrate audio preprocessing."""
    print("="*50)
    print("DEMO: Audio Preprocessing")
    print("="*50)
    
    # Create sample audio if soundfile is available
    try:
        audio_file = create_sample_audio()
    except ImportError:
        print("soundfile not available. Using librosa to create sample.")
        # Create sample using librosa
        sample_rate = 16000
        duration = 3
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Save using librosa
        os.makedirs('data/sample', exist_ok=True)
        librosa.output.write_wav('data/sample/test_audio.wav', audio_signal, sample_rate)
        audio_file = 'data/sample/test_audio.wav'
        print("Sample audio file created: data/sample/test_audio.wav")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Preprocess the audio
    processed_audio, sr = preprocessor.preprocess_audio(audio_file)
    
    if processed_audio is not None:
        print(f"Original audio length: {len(processed_audio)} samples")
        print(f"Sample rate: {sr} Hz")
        print("Audio preprocessing completed successfully!")
    else:
        print("Failed to preprocess audio")
    
    return audio_file


def demo_feature_extraction():
    """Demonstrate feature extraction."""
    print("\n" + "="*50)
    print("DEMO: Feature Extraction")
    print("="*50)
    
    # Create or use existing sample audio
    audio_file = 'data/sample/test_audio.wav'
    if not os.path.exists(audio_file):
        audio_file = demo_preprocessing()
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    features = extractor.extract_all_features(audio_file)
    
    if features is not None:
        print(f"Extracted {len(features)} features:")
        feature_names = extractor.get_feature_names()
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"  {name}: {value:.4f}")
        print("Feature extraction completed successfully!")
    else:
        print("Failed to extract features")
    
    return features


def demo_model_training_simple():
    """Demonstrate model training with synthetic data."""
    print("\n" + "="*50)
    print("DEMO: Model Training (with synthetic data)")
    print("="*50)
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Get actual feature names from extractor
    from src.features.feature_extraction import FeatureExtractor
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    n_features = len(feature_names)
    
    print(f"Using {n_features} features: {feature_names}")
    
    # Generate synthetic features
    # Male voices (label 1) - higher spectral features
    male_features = np.random.normal(0.5, 0.2, (n_samples//2, n_features))
    male_labels = np.ones(n_samples//2)
    
    # Female voices (label 0) - lower spectral features  
    female_features = np.random.normal(-0.5, 0.2, (n_samples//2, n_features))
    female_labels = np.zeros(n_samples//2)
    
    # Combine data
    X = np.vstack([male_features, female_features])
    y = np.hstack([male_labels, female_labels])
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(X, columns=feature_names)
    df['gender'] = y
    
    print(f"Created synthetic dataset with {len(df)} samples")
    print(f"Features: {feature_names}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    
    # Train models
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train just one model for demo
    print("\nTraining Logistic Regression model...")
    lr_model = trainer.train_logistic_regression()
    
    # Evaluate
    result = trainer.evaluate_model('logistic_regression', lr_model)
    print(f"Model accuracy: {result['accuracy']:.3f}")
    
    # Save model
    trainer.save_models()
    print("Model saved successfully!")
    
    return trainer


def main():
    """Run all demos."""
    print("Gender Voice Classification - Demo")
    print("This demo shows how to use the various components of the project.")
    
    # Demo 1: Audio preprocessing
    demo_preprocessing()
    
    # Demo 2: Feature extraction
    demo_feature_extraction()
    
    # Demo 3: Model training
    demo_model_training_simple()
    
    print("\n" + "="*50)
    print("DEMO COMPLETED!")
    print("="*50)
    print("\nNext steps:")
    print("1. Organize your audio files in data/raw/male/ and data/raw/female/")
    print("2. Run: python main.py --male-folder data/raw/male --female-folder data/raw/female")
    print("3. For predictions: python main.py --predict audio.wav --load-models")


if __name__ == "__main__":
    main()