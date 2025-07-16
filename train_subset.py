#!/usr/bin/env python3
"""
Train models on a subset of data for faster testing
"""

import os
import shutil
from src.data_loader import DataLoader
from src.models.train_models import ModelTrainer
from src.evaluation.visualization import DataVisualizer


def create_subset(male_folder, female_folder, subset_size=200):
    """Create a smaller subset for testing."""
    print(f"Creating subset with {subset_size} files per gender...")
    
    # Create subset directories
    os.makedirs('data/subset/male', exist_ok=True)
    os.makedirs('data/subset/female', exist_ok=True)
    
    # Copy subset of male files
    male_files = [f for f in os.listdir(male_folder) if f.endswith('.wav')][:subset_size]
    for i, file in enumerate(male_files):
        src = os.path.join(male_folder, file)
        dst = os.path.join('data/subset/male', file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        if (i + 1) % 50 == 0:
            print(f"Copied {i + 1}/{subset_size} male files")
    
    # Copy subset of female files
    female_files = [f for f in os.listdir(female_folder) if f.endswith('.wav')][:subset_size]
    for i, file in enumerate(female_files):
        src = os.path.join(female_folder, file)
        dst = os.path.join('data/subset/female', file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        if (i + 1) % 50 == 0:
            print(f"Copied {i + 1}/{subset_size} female files")
    
    print(f"Subset created: {len(male_files)} male + {len(female_files)} female files")
    return 'data/subset/male', 'data/subset/female'


def main():
    # Create subset for testing
    male_subset, female_subset = create_subset('data/raw/male', 'data/raw/female', subset_size=200)
    
    print("\n" + "="*60)
    print("GENDER VOICE CLASSIFICATION - SUBSET TRAINING")
    print("="*60)
    
    # Initialize components
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing subset data...")
    df = data_loader.load_and_preprocess_data(male_subset, female_subset, use_parallel=True)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    
    # Step 2: Prepare data for training
    print("\n2. Preparing data for training...")
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(df)
    
    # Step 3: Train models
    print("\n3. Training models...")
    models = model_trainer.train_all_models()
    
    # Step 4: Evaluate models
    print("\n4. Evaluating models...")
    results = model_trainer.evaluate_all_models()
    
    # Step 5: Save models
    print("\n5. Saving models...")
    model_trainer.save_models()
    
    print("\n" + "="*60)
    print("SUBSET TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test predictions: python main.py --predict audio.wav --load-models")
    print("2. For full dataset: python main.py --male-folder data/raw/male --female-folder data/raw/female")


if __name__ == "__main__":
    main()