import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import librosa
import librosa.display


class DataVisualizer:
    """Visualization utilities for voice classification project."""
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
        plt.style.use('default')
        
    def plot_audio_waveform(self, audio_signal, sr, title="Audio Waveform"):
        """Plot audio waveform."""
        plt.figure(figsize=self.figsize)
        time = np.linspace(0, len(audio_signal) / sr, len(audio_signal))
        plt.plot(time, audio_signal, linewidth=1)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_spectrogram(self, audio_signal, sr, title="Spectrogram"):
        """Plot spectrogram of audio signal."""
        plt.figure(figsize=self.figsize)
        D = librosa.stft(audio_signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()
    
    def plot_mel_spectrogram(self, audio_signal, sr, title="Mel Spectrogram"):
        """Plot mel spectrogram."""
        plt.figure(figsize=self.figsize)
        S = librosa.feature.melspectrogram(y=audio_signal, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()
    
    def plot_mfcc(self, audio_signal, sr, title="MFCC"):
        """Plot MFCC features."""
        plt.figure(figsize=self.figsize)
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(title)
        plt.show()
    
    def plot_spectral_features(self, audio_signal, sr):
        """Plot spectral features (centroid, rolloff, ZCR, RMS)."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)
        times = librosa.times_like(cent)
        axes[0, 0].plot(times, cent.T)
        axes[0, 0].set_title('Spectral Centroid')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Hz')
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr)
        axes[0, 1].plot(times, rolloff.T)
        axes[0, 1].set_title('Spectral Rolloff')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Hz')
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_signal)
        axes[1, 0].plot(times, zcr.T)
        axes[1, 0].set_title('Zero Crossing Rate')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Rate')
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_signal)
        axes[1, 1].plot(times, rms.T)
        axes[1, 1].set_title('RMS Energy')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Energy')
        
        plt.tight_layout()
        plt.show()
    
    def plot_gender_distribution(self, df):
        """Plot gender distribution."""
        plt.figure(figsize=(8, 6))
        gender_counts = df['gender'].value_counts()
        
        # Convert numeric labels back to strings for display
        gender_labels = ['Female' if x == 0 else 'Male' for x in gender_counts.index]
        
        plt.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Gender Distribution')
        plt.axis('equal')
        plt.show()
    
    def plot_feature_distributions(self, df):
        """Plot feature distributions."""
        feature_columns = df.columns[:-1]  # Exclude 'gender' column
        n_features = len(feature_columns)
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_columns):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7)
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.suptitle('Feature Distributions', y=1.02)
        plt.show()
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix."""
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_boxplots(self, df):
        """Plot boxplots for features by gender."""
        feature_columns = df.columns[:-1]  # Exclude 'gender' column
        n_features = len(feature_columns)
        
        rows = (n_features + 1) // 2
        cols = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()
        
        # Convert gender labels for display
        df_display = df.copy()
        df_display['gender'] = df_display['gender'].map({0: 'Female', 1: 'Male'})
        
        for i, col in enumerate(feature_columns):
            if i < len(axes):
                sns.boxplot(x='gender', y=col, data=df_display, ax=axes[i])
                axes[i].set_title(f'{col}')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names=['Female', 'Male']):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_model_comparison(self, results):
        """Plot model comparison results."""
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'{model_name} - Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Model {model_name} doesn't have feature_importances_ attribute")
    
    def create_comprehensive_report(self, df, results, save_path=None):
        """Create a comprehensive visualization report."""
        print("Generating comprehensive visualization report...")
        
        # 1. Data distribution plots
        self.plot_gender_distribution(df)
        self.plot_feature_distributions(df)
        self.plot_correlation_matrix(df)
        self.plot_feature_boxplots(df)
        
        # 2. Model comparison
        self.plot_model_comparison(results)
        
        # 3. Confusion matrices for each model
        for model_name, result in results.items():
            # Note: This requires y_test to be available
            # You might need to modify this based on your setup
            print(f"\nConfusion Matrix for {model_name.upper()}:")
            print(result['confusion_matrix'])
        
        print("Visualization report completed!")