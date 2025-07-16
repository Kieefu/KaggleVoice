import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
from src.features.feature_extraction import FeatureExtractor


class DataLoader:
    """Load and process audio data for voice classification."""
    
    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor or FeatureExtractor()
    
    def process_file(self, file_path_and_label):
        """Process a single audio file and extract features."""
        file_path, label = file_path_and_label
        features = self.feature_extractor.extract_all_features(file_path)
        if features is not None:
            return (features, label)
        return None
    
    def load_data_from_folders(self, male_folder, female_folder, use_parallel=True):
        """Load data from male and female folders."""
        data = []
        labels = []
        
        # Prepare file paths with labels
        file_paths_with_labels = []
        
        # Add male files
        if os.path.exists(male_folder):
            male_files = [f for f in os.listdir(male_folder) if os.path.isfile(os.path.join(male_folder, f))]
            for f in male_files:
                file_paths_with_labels.append((os.path.join(male_folder, f), 'male'))
        
        # Add female files
        if os.path.exists(female_folder):
            female_files = [f for f in os.listdir(female_folder) if os.path.isfile(os.path.join(female_folder, f))]
            for f in female_files:
                file_paths_with_labels.append((os.path.join(female_folder, f), 'female'))
        
        print(f"Processing {len(file_paths_with_labels)} audio files...")
        
        if use_parallel:
            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self.process_file, file_paths_with_labels),
                    total=len(file_paths_with_labels)
                ))
        else:
            # Process files sequentially
            results = []
            for file_info in tqdm(file_paths_with_labels):
                results.append(self.process_file(file_info))
        
        # Collect valid results
        for result in results:
            if result is not None:
                features, label = result
                data.append(features)
                labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def create_dataframe(self, data, labels):
        """Create a DataFrame from features and labels."""
        # Create DataFrame with features
        df = pd.DataFrame(data)
        
        # Add labels
        df['gender'] = labels
        
        # Set column names
        feature_names = self.feature_extractor.get_feature_names()
        df.columns = feature_names + ['gender']
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def preprocess_dataframe(self, df):
        """Preprocess the DataFrame (remove outliers, encode labels, etc.)."""
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Handle outliers using IQR method
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        Q1 = df[numerical_columns].quantile(0.25)
        Q3 = df[numerical_columns].quantile(0.75)
        IQR = Q3 - Q1
        
        # Remove outliers
        df = df[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | 
                  (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        print(f"After removing outliers: {df.shape}")
        
        # Encode gender labels
        df['gender'] = df['gender'].map({'male': 1, 'female': 0})
        
        # Drop highly correlated features (based on notebook analysis)
        features_to_drop = ['rms', 'spectral_centroid', 'zero_crossing_rate']
        existing_features = [col for col in features_to_drop if col in df.columns]
        if existing_features:
            df = df.drop(existing_features, axis=1)
            print(f"Dropped features: {existing_features}")
        
        print(f"Final data shape: {df.shape}")
        return df
    
    def load_and_preprocess_data(self, male_folder, female_folder, use_parallel=True):
        """Complete data loading and preprocessing pipeline."""
        # Load data
        data, labels = self.load_data_from_folders(male_folder, female_folder, use_parallel)
        
        # Create DataFrame
        df = self.create_dataframe(data, labels)
        
        # Preprocess
        df = self.preprocess_dataframe(df)
        
        return df