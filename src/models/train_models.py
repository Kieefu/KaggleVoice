import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


class ModelTrainer:
    """Train and evaluate multiple models for voice classification."""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, df):
        """Prepare data for training."""
        # Separate features and target
        X = df.drop(columns='gender')
        y = df['gender']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model."""
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr
        return lr
    
    def train_random_forest(self):
        """Train Random Forest model."""
        print("Training Random Forest...")
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        return rf
    
    def train_xgboost(self):
        """Train XGBoost model."""
        print("Training XGBoost...")
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state
        )
        xgb.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb
        return xgb
    
    def train_naive_bayes(self):
        """Train Naive Bayes model."""
        print("Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb
        return nb
    
    def train_all_models(self):
        """Train all models."""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_naive_bayes()
        
        print(f"Trained {len(self.models)} models successfully!")
        return self.models
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model."""
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n{model_name.upper()} - Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models."""
        results = {}
        
        for model_name, model in self.models.items():
            results[model_name] = self.evaluate_model(model_name, model)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        for model_name, result in results.items():
            print(f"{model_name.upper()}: {result['accuracy']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0].upper()} with accuracy {best_model[1]['accuracy']:.4f}")
        
        return results
    
    def save_models(self, save_dir='models'):
        """Save trained models and scaler."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f'{model_name}.pkl'))
        
        print(f"Models saved to {save_dir} directory")
    
    def load_models(self, load_dir='models'):
        """Load trained models and scaler."""
        # Load scaler
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        
        # Load models
        model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'xgboost.pkl', 'naive_bayes.pkl']
        for model_file in model_files:
            model_path = os.path.join(load_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                self.models[model_name] = joblib.load(model_path)
        
        print(f"Loaded {len(self.models)} models from {load_dir}")
        return self.models
    
    def predict_single_file(self, file_path, model_name='logistic_regression'):
        """Predict gender for a single audio file."""
        from src.features.feature_extraction import FeatureExtractor
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(file_path)
        
        if features is None:
            return None
        
        # Scale features - match the training feature set
        import pandas as pd
        feature_names = extractor.get_feature_names()
        features_df = pd.DataFrame(features.reshape(1, -1), columns=feature_names)
        
        # Drop features that were removed during training
        features_to_drop = ['rms', 'spectral_centroid', 'zero_crossing_rate']
        existing_features = [col for col in features_to_drop if col in features_df.columns]
        if existing_features:
            features_df = features_df.drop(existing_features, axis=1)
        
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.models[model_name].predict(features_scaled)[0]
        probability = self.models[model_name].predict_proba(features_scaled)[0]
        
        gender = 'male' if prediction == 1 else 'female'
        confidence = max(probability)
        
        return {
            'prediction': gender,
            'confidence': confidence,
            'probabilities': {
                'female': probability[0],
                'male': probability[1]
            }
        }