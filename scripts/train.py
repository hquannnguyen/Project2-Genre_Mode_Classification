# train.py
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from config import config
from sklearn.metrics import accuracy_score

def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=config.MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # For accuracy, approximate using y_a
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_a.cpu().numpy())
        else:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), acc

def train(model, train_loader, val_loader, config_obj):
    device = config_obj.DEVICE
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config_obj.LEARNING_RATE, weight_decay=config_obj.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config_obj.EPOCHS, eta_min=1e-6)
    best_val_acc = 0.0
    for epoch in range(1, config_obj.EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=config_obj.USE_MIXUP)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config_obj.MODEL_SAVE_PATH)
            print(f"  -> Best model saved (acc={val_acc:.4f})")
    return best_val_acc