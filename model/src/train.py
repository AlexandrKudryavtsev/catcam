import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import calculate_metrics

def train_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss = 0.0

    all_outputs = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Training")
    for (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_outputs.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    metrics = calculate_metrics(all_outputs, all_labels, threshold)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics

def evaluate(model, dataloader, device, criterion=None, threshold=0.5):
    model.eval()
    
    if (criterion):
        total_loss = 0.0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            outputs = model(inputs)

            if (criterion):
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    metrics = calculate_metrics(all_outputs, all_labels, threshold)

    if (criterion):
        metrics['loss'] = total_loss / len(dataloader)

    return metrics

def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
               device,
               epochs,
               early_stopping_patience,
               save_best=False):
    best_val_fbeta = 0.0
    best_val_loss = float('inf')
    no_improve = 0

    metrics_history = {
        'train' : defaultdict(list),
        'val' : defaultdict(list)
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, criterion)

        scheduler.step(val_metrics['loss'])

        print(f"Train Loss: {train_metrics['loss']:.4f} | Precision: {train_metrics['precision']:.4f} |\
        Recall {train_metrics['recall']:.4f} | ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f} | Precision: {val_metrics['precision']:.4f} |\
        Recall {val_metrics['recall']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")

        for metric_name in train_metrics.keys():
            metrics_history['train'][metric_name].append(train_metrics[metric_name])
            metrics_history['val'][metric_name].append(val_metrics[metric_name])

        # Сохранение лучшей модели
        if (save_best):
            if val_metrics['fbeta_score'] < best_val_fbeta:
                best_val_fbeta = val_metrics['fbeta']
                torch.save(model.state_dict(), "best_fbeta_model.pth")
                print("Model saved!")

        if (val_metrics["loss"] < best_val_loss):
            best_val_loss = val_metrics["loss"]
        else:
            no_improve += 1

            if (no_improve == early_stopping_patience):
                print("Early stopping triggered!")
                return metrics_history

    return metrics_history