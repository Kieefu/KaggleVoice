import numpy as np
import librosa
from src.preprocessing.audio_preprocessing import AudioPreprocessor


class FeatureExtractor:
    """Extract audio features for voice classification."""
    
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=512, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
    
    def extract_mfcc(self, audio_signal):
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio_signal, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return np.mean(mfcc, axis=1)
    
    def extract_spectral_rolloff(self, audio_signal):
        """Extract spectral rolloff feature."""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_signal,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )[0]
        return np.mean(rolloff)
    
    def extract_zero_crossing_rate(self, audio_signal):
        """Extract zero crossing rate feature."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio_signal,
            hop_length=self.hop_length
        )[0]
        return np.mean(zcr)
    
    def extract_spectral_centroid(self, audio_signal):
        """Extract spectral centroid feature."""
        centroid = librosa.feature.spectral_centroid(
            y=audio_signal,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )[0]
        return np.mean(centroid)
    
    def extract_rms_energy(self, audio_signal):
        """Extract RMS energy feature."""
        rms = librosa.feature.rms(
            y=audio_signal,
            hop_length=self.hop_length
        )[0]
        return np.mean(rms)
    
    def extract_all_features(self, file_path):
        """Extract all features from audio file."""
        # Preprocess audio
        audio_signal, sample_rate = self.preprocessor.preprocess_audio(file_path)
        if audio_signal is None:
            return None
        
        # Extract features
        mfcc_features = self.extract_mfcc(audio_signal)
        rolloff_mean = self.extract_spectral_rolloff(audio_signal)
        zcr_mean = self.extract_zero_crossing_rate(audio_signal)
        centroid_mean = self.extract_spectral_centroid(audio_signal)
        rms_mean = self.extract_rms_energy(audio_signal)
        
        # Combine all features
        combined_features = np.hstack([
            mfcc_features,
            rolloff_mean,
            zcr_mean,
            centroid_mean,
            rms_mean
        ])
        
        return combined_features
    
    def get_feature_names(self):
        """Get feature column names."""
        feature_names = [f"mfcc_{i+1}" for i in range(self.n_mfcc)]
        feature_names.extend([
            "spectral_rolloff",
            "zero_crossing_rate", 
            "spectral_centroid",
            "rms"
        ])
        return feature_names