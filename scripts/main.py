# main.py
import torch
from torch.utils.data import DataLoader, random_split
from config import config
from dataset import AudioGenreDataset
from model_cnn import AudioCNN
from train import train
from evaluate import evaluate_model

def main():
    # 1. Dataset
    full_dataset = AudioGenreDataset(root=config.DATA_PROCESSED, use_mel=config.USE_MEL_SPECTROGRAM)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 2. Model
    model = AudioCNN(n_classes=config.N_CLASSES)

    # 3. Train
    print("Starting training...")
    best_acc = train(model, train_loader, val_loader, config)
    print(f"Best validation accuracy: {best_acc:.4f}")

    # 4. Evaluate on test set
    print("Evaluating on test set...")
    test_acc = evaluate_model(config.MODEL_SAVE_PATH, test_loader, config.DEVICE)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()