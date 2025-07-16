#!/usr/bin/env python3
"""
Test the gender voice classification model on various audio files
"""

import os
import random
from src.models.train_models import ModelTrainer


def test_random_samples():
    """Test model on random samples from the dataset."""
    print("🎯 Testing Gender Voice Classification Model")
    print("="*50)
    
    # Load trained model
    trainer = ModelTrainer()
    trainer.load_models()
    
    # Get some sample files
    male_files = [f for f in os.listdir('data/subset/male') if f.endswith('.wav')]
    female_files = [f for f in os.listdir('data/subset/female') if f.endswith('.wav')]
    
    # Test 5 random male samples
    print("\n🧔 Testing Male Voices:")
    print("-" * 30)
    for i in range(min(5, len(male_files))):
        file = random.choice(male_files)
        file_path = f"data/subset/male/{file}"
        result = trainer.predict_single_file(file_path, 'random_forest')
        
        if result:
            correct = "✅" if result['prediction'] == 'male' else "❌"
            print(f"{correct} {file[:20]:<20} → {result['prediction'].upper()} ({result['confidence']:.3f})")
    
    # Test 5 random female samples  
    print("\n👩 Testing Female Voices:")
    print("-" * 30)
    for i in range(min(5, len(female_files))):
        file = random.choice(female_files)
        file_path = f"data/subset/female/{file}"
        result = trainer.predict_single_file(file_path, 'random_forest')
        
        if result:
            correct = "✅" if result['prediction'] == 'female' else "❌"
            print(f"{correct} {file[:20]:<20} → {result['prediction'].upper()} ({result['confidence']:.3f})")


def test_single_file(file_path):
    """Test model on a single file."""
    print(f"\n🎯 Testing: {file_path}")
    print("="*50)
    
    trainer = ModelTrainer()
    trainer.load_models()
    
    # Test with all models
    models = ['logistic_regression', 'random_forest', 'xgboost', 'naive_bayes']
    
    for model_name in models:
        try:
            result = trainer.predict_single_file(file_path, model_name)
            if result:
                print(f"{model_name.upper():<20} → {result['prediction'].upper()} ({result['confidence']:.3f})")
            else:
                print(f"{model_name.upper():<20} → Failed to process")
        except Exception as e:
            print(f"{model_name.upper():<20} → Error: {str(e)[:30]}...")


def test_accuracy_on_samples():
    """Test accuracy on a batch of samples."""
    print("\n📊 Testing Accuracy on Sample Files")
    print("="*50)
    
    trainer = ModelTrainer()
    trainer.load_models()
    
    # Test on subset files
    male_files = [f for f in os.listdir('data/subset/male') if f.endswith('.wav')][:10]
    female_files = [f for f in os.listdir('data/subset/female') if f.endswith('.wav')][:10]
    
    correct = 0
    total = 0
    
    print("Testing 10 male + 10 female samples...")
    
    # Test male files
    for file in male_files:
        file_path = f"data/subset/male/{file}"
        result = trainer.predict_single_file(file_path, 'random_forest')
        if result:
            total += 1
            if result['prediction'] == 'male':
                correct += 1
    
    # Test female files
    for file in female_files:
        file_path = f"data/subset/female/{file}"
        result = trainer.predict_single_file(file_path, 'random_forest')
        if result:
            total += 1
            if result['prediction'] == 'female':
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📈 Accuracy: {correct}/{total} = {accuracy:.1%}")


def main():
    """Run various tests."""
    import sys
    
    if len(sys.argv) > 1:
        # Test specific file
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            test_single_file(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    else:
        # Run all tests
        test_random_samples()
        test_accuracy_on_samples()
        
        print("\n💡 Usage Tips:")
        print("- Test specific file: python test_predictions.py /path/to/audio.wav")
        print("- Test your recording: python test_predictions.py my_voice.wav")
        print("- Supported formats: WAV, MP3, FLAC")


if __name__ == "__main__":
    main()