# features.py
import librosa
import numpy as np
from config import config

def extract_mel_spectrogram_img(y):
    """
    Return mel spectrogram as 3D array (height, width, 1) for CNN.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmax=config.SAMPLE_RATE//2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to [-1, 1]
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return mel_norm[..., np.newaxis]   # (N_MELS, time, 1)

def extract_handcrafted(y):
    features = {}
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    features['mfcc_mean'] = mfcc.mean(axis=1)
    features['mfcc_std'] = mfcc.std(axis=1)
    features['mfcc_delta'] = librosa.feature.delta(mfcc).mean(axis=1)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=config.SAMPLE_RATE, n_chroma=12)
    features['chroma_mean'] = chroma.mean(axis=1)
    features['chroma_std'] = chroma.std(axis=1)
    # Mel-spectrogram mean
    mel = librosa.feature.melspectrogram(y=y, sr=config.SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features['mel_mean'] = mel_db.mean(axis=1)
    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=config.SAMPLE_RATE).mean()
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=config.SAMPLE_RATE).mean()
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=config.SAMPLE_RATE).mean()
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=config.SAMPLE_RATE)
    features['tempo'] = tempo
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = zcr.mean()
    features['zcr_std'] = zcr.std()
    # RMS
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = rms.mean()
    # Tonnetz
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=config.SAMPLE_RATE)
    features['tonnetz_mean'] = tonnetz.mean(axis=1)
    # Concatenate all
    vec = np.concatenate([np.atleast_1d(v).flatten() for v in features.values()])
    return vec