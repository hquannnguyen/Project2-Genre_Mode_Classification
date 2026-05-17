# evaluate.py
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import config
from model_cnn import AudioCNN
from dataset import AudioGenreDataset
from torch.utils.data import DataLoader, random_split

def evaluate_model(model_path, test_loader, device):
    model = AudioCNN(n_classes=config.N_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    # Lấy danh sách genres từ thư mục đã xử lý
    processed_path = Path(config.DATA_PROCESSED)
    genres = sorted([d.name for d in processed_path.iterdir() if d.is_dir()])
    
    print(classification_report(all_labels, all_preds, target_names=genres, digits=3))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=genres)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Genre Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    
    return np.mean(np.array(all_preds) == np.array(all_labels))

if __name__ == "__main__":
    # Tạo dataset và split giống như trong main.py
    full_dataset = AudioGenreDataset(root=config.DATA_PROCESSED, use_mel=config.USE_MEL_SPECTROGRAM)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    _, _, test_ds = random_split(full_dataset, [n_train, n_val, n_test])
    
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    acc = evaluate_model(config.MODEL_SAVE_PATH, test_loader, config.DEVICE)
    print(f"Test accuracy: {acc:.4f}")