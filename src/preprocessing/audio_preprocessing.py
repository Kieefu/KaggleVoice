import numpy as np
import librosa
import librosa.effects
import noisereduce as nr


class AudioPreprocessor:
    """Audio preprocessing utilities for voice classification."""
    
    def __init__(self, sample_rate=16000, duration=3, top_db=35):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_track = sample_rate * duration
        self.top_db = top_db
    
    def load_audio(self, file_path):
        """Load audio file with specified sample rate."""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def trim_silence(self, audio_signal):
        """Remove silence from beginning and end of audio."""
        trimmed_audio, _ = librosa.effects.trim(audio_signal, top_db=self.top_db)
        return trimmed_audio
    
    def reduce_noise(self, audio_signal, sample_rate):
        """Reduce background noise from audio signal."""
        denoised_audio = nr.reduce_noise(y=audio_signal, sr=sample_rate)
        return denoised_audio
    
    def normalize_audio(self, audio_signal):
        """Normalize audio signal to consistent volume level."""
        normalized_audio = librosa.util.normalize(audio_signal)
        return normalized_audio
    
    def resample_audio(self, audio_signal, original_sr, target_sr):
        """Resample audio to target sample rate."""
        resampled_audio = librosa.resample(audio_signal, orig_sr=original_sr, target_sr=target_sr)
        return resampled_audio
    
    def pad_or_truncate(self, audio_signal):
        """Pad or truncate audio to desired length."""
        if len(audio_signal) < self.samples_per_track:
            # Pad with zeros
            padded_audio = np.pad(audio_signal, (0, self.samples_per_track - len(audio_signal)))
            return padded_audio
        else:
            # Truncate to desired length
            return audio_signal[:self.samples_per_track]
    
    def preprocess_audio(self, file_path):
        """Complete preprocessing pipeline for audio file."""
        try:
            # Load audio
            y, sr = self.load_audio(file_path)
            if y is None:
                return None, None
            
            # Trim silence
            y = self.trim_silence(y)
            
            # Reduce noise
            y = self.reduce_noise(y, sr)
            
            # Normalize
            y = self.normalize_audio(y)
            
            # Pad or truncate
            y = self.pad_or_truncate(y)
            
            return y, sr
            
        except Exception as e:
            print(f"Failed to process '{file_path}': {e}")
            return None, None