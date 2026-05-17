# # config.py
# import torch

# class Config:
#     # Paths
#     DATA_RAW = "data/raw"
#     DATA_GENRES = "data/genres"
#     DATA_PROCESSED = "data/processed"
#     MODEL_SAVE_PATH = "best_model.pth"

#     # Audio parameters
#     SAMPLE_RATE = 22050
#     DURATION = 30           # seconds
#     N_SAMPLES = SAMPLE_RATE * DURATION

#     # Feature extraction
#     N_MELS = 128
#     N_FFT = 2048
#     HOP_LENGTH = 512
#     N_MFCC = 40

#     # Training
#     BATCH_SIZE = 32
#     EPOCHS = 50
#     LEARNING_RATE = 3e-4
#     WEIGHT_DECAY = 1e-4
#     NUM_WORKERS = 2
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Model
#     N_CLASSES = 10          # GTZAN has 10 genres
#     USE_MEL_SPECTROGRAM = True   # if False, use handcrafted features

#     # Augmentation
#     USE_MIXUP = True
#     MIXUP_ALPHA = 0.4

# config = Config()


# config.py
import torch
from pathlib import Path

class Config:
    # Paths
    DATA_RAW = "data/raw"
    DATA_GENRES = "data/genres"
    DATA_PROCESSED = "data/processed"
    MODEL_SAVE_PATH = "best_model.pth"

    # Audio parameters
    SAMPLE_RATE = 22050
    DURATION = 30
    N_SAMPLES = SAMPLE_RATE * DURATION

    # Feature extraction
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MFCC = 40

    # Training - GIẢM TẢI
    BATCH_SIZE = 8           # giảm từ 32 xuống 8
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0          # TẮT đa luồng (quan trọng trên Windows)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    N_CLASSES = 10
    USE_MEL_SPECTROGRAM = True

    # Augmentation - TẮT mixup để tránh lỗi thêm
    USE_MIXUP = False
    MIXUP_ALPHA = 0.4

config = Config()