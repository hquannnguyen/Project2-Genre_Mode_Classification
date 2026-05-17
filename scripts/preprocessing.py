# preprocessing.py
import librosa
import numpy as np
from pathlib import Path
from config import config

def load_and_preprocess(filepath):
    try:
        y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
        # trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        # pad or truncate
        if len(y) < config.N_SAMPLES:
            y = np.pad(y, (0, config.N_SAMPLES - len(y)))
        else:
            y = y[:config.N_SAMPLES]
        # normalize peak
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y
    except Exception as e:
        print(f"  [WARN] Failed to load {filepath}: {e}")
        return None

def augment(y):
    augmented = [y]
    # Gaussian noise
    noise = y + 0.005 * np.random.randn(len(y))
    augmented.append(noise)
    # Pitch shift +2 semitones
    pitch_up = librosa.effects.pitch_shift(y, sr=config.SAMPLE_RATE, n_steps=2)
    augmented.append(pitch_up)
    # Time stretch 0.9
    stretch = librosa.effects.time_stretch(y, rate=0.9)[:len(y)]
    augmented.append(stretch)
    return augmented

def preprocess_all():
    root = Path(config.DATA_GENRES)
    output = Path(config.DATA_PROCESSED)
    output.mkdir(exist_ok=True)

    for genre_dir in root.iterdir():
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        out_dir = output / genre
        out_dir.mkdir(exist_ok=True)

        wav_files = list(genre_dir.glob("*.wav"))
        processed = 0
        for idx, wav_file in enumerate(wav_files):
            y = load_and_preprocess(str(wav_file))
            if y is None:
                continue   # bỏ qua file lỗi
            # save original
            np.save(out_dir / f"{wav_file.stem}.npy", y)
            # save augmented versions
            for i, aug in enumerate(augment(y)):
                if i == 0:   # original already saved
                    continue
                np.save(out_dir / f"{wav_file.stem}_aug{i}.npy", aug)
            processed += 1
            if (idx+1) % 50 == 0:
                print(f"  Processed {idx+1}/{len(wav_files)} in {genre}")

        print(f"Finished {genre}: {processed}/{len(wav_files)} files saved")

if __name__ == "__main__":
    preprocess_all()