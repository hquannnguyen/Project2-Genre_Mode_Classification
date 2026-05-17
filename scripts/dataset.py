# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from config import config
from features import extract_mel_spectrogram_img, extract_handcrafted

class AudioGenreDataset(Dataset):
    def __init__(self, root=config.DATA_PROCESSED, use_mel=True, transform=None):
        self.root = Path(root)
        self.use_mel = use_mel
        self.transform = transform
        self.samples = []
        self.labels = []
        genres = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        for genre in genres:
            for npy_file in (self.root / genre).glob("*.npy"):
                self.samples.append(str(npy_file))
                self.labels.append(self.genre_to_idx[genre])
        print(f"Loaded {len(self.samples)} samples, {len(genres)} genres")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y = np.load(self.samples[idx])
        if self.use_mel:
            spec = extract_mel_spectrogram_img(y)          # (H, W, 1)
            spec = torch.from_numpy(spec).permute(2,0,1).float()  # (1, H, W)
            if self.transform:
                spec = self.transform(spec)
            return spec, self.labels[idx]
        else:
            fv = extract_handcrafted(y)                    # 1D vector
            return torch.from_numpy(fv).float(), self.labels[idx]