#!/usr/bin/env python3
"""
Gender Voice Classification - Main Script

This script demonstrates the complete pipeline for gender voice classification
based on the Kaggle notebook implementation.
"""

import os
import sys
import argparse
from src.data_loader import DataLoader
from src.models.train_models import ModelTrainer
from src.evaluation.visualization import DataVisualizer


def main():
    parser = argparse.ArgumentParser(description='Gender Voice Classification Pipeline')
    parser.add_argument('--male-folder', type=str, help='Path to male audio folder')
    parser.add_argument('--female-folder', type=str, help='Path to female audio folder')
    parser.add_argument('--predict', type=str, help='Path to audio file for prediction')
    parser.add_argument('--model', type=str, default='logistic_regression', 
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'naive_bayes'],
                       help='Model to use for prediction')
    parser.add_argument('--load-models', action='store_true', help='Load pre-trained models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization report')
    parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Initialize components
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()
    
    # Single file prediction
    if args.predict:
        if args.load_models:
            print("Loading pre-trained models...")
            model_trainer.load_models()
        else:
            print("Error: No models loaded. Train models first or use --load-models flag.")
            return
        
        print(f"Predicting gender for: {args.predict}")
        result = model_trainer.predict_single_file(args.predict, args.model)
        
        if result:
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: Female: {result['probabilities']['female']:.3f}, Male: {result['probabilities']['male']:.3f}")
        else:
            print("Failed to process audio file")
        return
    
    # Training pipeline
    if args.male_folder and args.female_folder:
        print("="*60)
        print("GENDER VOICE CLASSIFICATION PIPELINE")
        print("="*60)
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        df = data_loader.load_and_preprocess_data(
            args.male_folder, 
            args.female_folder, 
            use_parallel=args.parallel
        )
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Features: {df.columns[:-1].tolist()}")
        
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
        
        # Step 6: Generate visualizations
        if args.visualize:
            print("\n6. Generating visualizations...")
            visualizer.create_comprehensive_report(df, results)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    else:
        print("Error: Please provide both --male-folder and --female-folder for training")
        print("Or use --predict with a single audio file for prediction")


def demo_with_sample_data():
    """Demo function to show how to use the pipeline with sample data."""
    print("="*60)
    print("DEMO: Gender Voice Classification")
    print("="*60)
    
    # This is a demo showing how to use the pipeline
    # Replace these paths with your actual data paths
    male_folder = "data/raw/male"
    female_folder = "data/raw/female"
    
    # Check if data folders exist
    if not os.path.exists(male_folder) or not os.path.exists(female_folder):
        print("Demo data folders not found. Please organize your data as follows:")
        print("data/raw/male/    - Place male voice files here")
        print("data/raw/female/  - Place female voice files here")
        return
    
    # Run the pipeline
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = data_loader.load_and_preprocess_data(male_folder, female_folder)
    
    # Train models
    print("Training models...")
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(df)
    models = model_trainer.train_all_models()
    
    # Evaluate models
    print("Evaluating models...")
    results = model_trainer.evaluate_all_models()
    
    # Save models
    print("Saving models...")
    model_trainer.save_models()
    
    # Generate visualizations
    print("Generating visualizations...")
    visualizer.create_comprehensive_report(df, results)
    
    print("Demo completed!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("Gender Voice Classification Pipeline")
        print("Usage examples:")
        print("  python main.py --male-folder data/male --female-folder data/female")
        print("  python main.py --predict audio.wav --model logistic_regression --load-models")
        print("  python main.py --help")
        print("\nTo run demo with sample data structure:")
        print("  python -c \"from main import demo_with_sample_data; demo_with_sample_data()\"")
    else:
        main()